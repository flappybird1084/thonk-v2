"""Multi-head latent attention (MLA), NoPE variant.

MLA (DeepSeek-V2, 2024) compresses keys and values into one small shared
latent per token. At inference you cache only that latent — kv_latent_dim
floats per token instead of 2 * n_heads * head_dim — and up-project on the
fly (or fold the up-projections into the query/output projections, the
"absorption" trick). During training we just materialize full k/v so we can
use flash attention like everyone else; DeepSeek does the same.

Because this model is NoPE (no positional encoding anywhere — see the NoPE
paper, Kazemnejad et al. 2023, and its use in SmolLM3 / Llama 4's interleaved
layers), we get to drop MLA's awkward "decoupled RoPE" side channel entirely:
no rotary keys, no extra head dims. Position information reaches the model
only through the causal mask, which is enough for a decoder-only LM.

Queries are a plain projection with no compression, as in DeepSeek-V2-Lite —
query compression only pays off at much larger scale.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from components.model.config import ThonkConfig


class MultiHeadLatentAttention(nn.Module):
    def __init__(self, cfg: ThonkConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        qkv_dim = cfg.n_heads * cfg.head_dim

        self.q_proj = nn.Linear(cfg.d_model, qkv_dim, bias=False)
        self.kv_down = nn.Linear(cfg.d_model, cfg.kv_latent_dim, bias=False)
        # DeepSeek normalizes the compressed latent before up-projecting
        # (their kv_a_layernorm); without it the latent scale drifts.
        self.kv_norm = nn.RMSNorm(cfg.kv_latent_dim)
        self.k_up = nn.Linear(cfg.kv_latent_dim, qkv_dim, bias=False)
        self.v_up = nn.Linear(cfg.kv_latent_dim, qkv_dim, bias=False)
        self.o_proj = nn.Linear(qkv_dim, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape

        q = self.q_proj(x)
        latent = self.kv_norm(self.kv_down(x))  # [B, T, kv_latent_dim]
        k = self.k_up(latent)
        v = self.v_up(latent)

        # [B, T, H*dh] -> [B, H, T, dh]
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        y = y.transpose(1, 2).reshape(B, T, self.n_heads * self.head_dim)
        return self.o_proj(y)
