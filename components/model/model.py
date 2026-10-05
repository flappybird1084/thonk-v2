"""The Thonk language model.

A standard pre-norm decoder-only transformer (residual attention + MLP, as
always), with three twists wired together:

1. Per-layer embedding tables (Gemma 3n's "PLE"): every block owns a small
   embedding table over the vocabulary, so each layer gets its own learned,
   layer-specific view of the raw input token — independent of whatever the
   residual stream has become by that depth. (In Gemma 3n these tables can be
   offloaded at inference since they're just lookups.)

2. adaLN-Zero (from DiT, Peebles & Xie 2022): instead of learnable norm gains,
   each sublayer's shift/scale and a residual gate are *predicted* from a
   conditioning signal, and the projection producing them starts at zero so
   every block begins as the identity function.

3. The two are combined: the per-layer embedding IS the conditioning signal.
   DiT conditions its modulation on timestep+class; a plain LM has no such
   global signal, so we condition on the per-layer token embedding instead —
   each block modulates itself per-position based on which token sits there.

The token mixer in each block is either MLA (full attention) or gated
DeltaNet (linear attention), interleaved per cfg.full_attn_every — see
config.py and deltanet.py.
"""

from dataclasses import fields

import torch
import torch.nn as nn
import torch.nn.functional as F

from components.model.attention import MultiHeadLatentAttention
from components.model.config import ThonkConfig
from components.model.deltanet import GatedDeltaNet
from components.model.memory import ProductKeyMemory
from components.model.mlp import SwiGLU
from components.model.moe import MoEMLP


class Block(nn.Module):
    def __init__(self, cfg: ThonkConfig, full_attn: bool = True, layer_idx: int = 0):
        super().__init__()
        self.ple = nn.Embedding(cfg.vocab_size, cfg.d_ple)
        # One projection produces all six modulation tensors:
        # (shift, scale, gate) for the attention sublayer and again for the MLP.
        self.ada = nn.Linear(cfg.d_ple, 6 * cfg.d_model)

        # No learnable gain on these norms — scale/shift come from `ada`.
        self.attn_norm = nn.RMSNorm(cfg.d_model, elementwise_affine=False)
        # Either mixer sits behind the same interface and the same adaLN
        # wrapper; the attribute keeps the name `attn` so train.py's
        # Muon/AdamW split rule applies to both.
        self.attn = MultiHeadLatentAttention(cfg) if full_attn else GatedDeltaNet(cfg)
        self.mlp_norm = nn.RMSNorm(cfg.d_model, elementwise_affine=False)
        # Feed-forward: dense SwiGLU by default; MoE when n_experts > 0; one
        # layer can swap in a product-key memory (see experiments/).
        if layer_idx == cfg.memory_layer:
            self.mlp = ProductKeyMemory(cfg)
        elif cfg.n_experts:
            self.mlp = MoEMLP(cfg)
        else:
            self.mlp = SwiGLU(cfg)
        if cfg.ple_add:
            # Gemma 3n's additive path: the per-layer embedding also feeds the
            # residual stream directly, not just the adaLN modulation.
            # Zero-init (in Thonk) so it starts silent like the adaLN gates.
            self.ple_proj = nn.Linear(cfg.d_ple, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
        ple = self.ple(ids)
        cond = self.ada(ple)  # [B, T, 6*d_model]
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = cond.chunk(6, dim=-1)
        if hasattr(self, "ple_proj"):
            x = x + self.ple_proj(ple)

        x = x + gate_a * self.attn(self.attn_norm(x) * (1 + scale_a) + shift_a)
        x = x + gate_m * self.mlp(self.mlp_norm(x) * (1 + scale_m) + shift_m)
        return x


class Thonk(nn.Module):
    def __init__(self, cfg: ThonkConfig):
        super().__init__()
        # Checkpoints pickle the config, and one saved before newer fields
        # existed restores without them — rebuild so defaults fill the gaps.
        cfg = ThonkConfig(
            **{f.name: getattr(cfg, f.name) for f in fields(ThonkConfig) if hasattr(cfg, f.name)}
        )
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        # Layer i is full attention when (i+1) % full_attn_every == 0, gated
        # DeltaNet otherwise — every=4 gives the Qwen3-Next 3:1 hybrid with
        # full attention closing each group; every=1 is all-MLA.
        self.blocks = nn.ModuleList(
            Block(cfg, full_attn=(i + 1) % cfg.full_attn_every == 0, layer_idx=i)
            for i in range(cfg.n_layers)
        )
        self.final_norm = nn.RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight

        self.apply(self._init_weights)
        # adaLN-Zero: modulation starts at exactly zero, so at init every
        # gate is closed and the whole network is the identity on the token
        # embeddings. Gates learn to open; this replaces careful residual
        # init scaling. Must happen after apply() or it would be overwritten.
        # The additive PLE projection starts silent for the same reason.
        for block in self.blocks:
            nn.init.zeros_(block.ada.weight)
            nn.init.zeros_(block.ada.bias)
            if hasattr(block, "ple_proj"):
                nn.init.zeros_(block.ple_proj.weight)

    @staticmethod
    def _init_weights(module: nn.Module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, ids: torch.Tensor, targets: torch.Tensor | None = None):
        """Returns (logits, loss); loss is None when no targets are given."""
        x = self.tok_emb(ids)
        for block in self.blocks:
            x = block(x, ids)
        return self.head(x, targets)

    def head(self, x: torch.Tensor, targets: torch.Tensor | None = None):
        """Final norm, unembedding, and the loss.

        Kept as one unit so a per-module compile (train_lightning.py) can fuse
        the loss into the logits: the fp32 logits and their log-softmax are
        the largest activations in the model, and fusing them is most of the
        memory torch.compile saves.
        """
        x = self.final_norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.cfg.vocab_size), targets.view(-1)
            )
        return logits, loss
