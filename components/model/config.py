from dataclasses import dataclass


@dataclass
class ThonkConfig:
    """Architecture hyperparameters for the Thonk LM.

    Note there is no max sequence length here: the model is NoPE (no positional
    encoding at all), so nothing in the architecture fixes a context size. The
    training sequence length lives in train.py where it belongs.

    Defaults give ~434M params (~356M non-embedding).
    """

    # GPT-2 BPE has 50257 tokens; padded up to a multiple of 128 so the tied
    # embedding/lm_head matmul lands on tensor-core-friendly shapes. Ids 50257+
    # are simply never produced by the tokenizer.
    vocab_size: int = 50304

    d_model: int = 1024
    n_layers: int = 24

    # Attention (multi-head latent attention, DeepSeek-V2 style)
    n_heads: int = 16
    head_dim: int = 64
    kv_latent_dim: int = 512  # dimension keys/values are compressed down to

    # SwiGLU hidden size (~8/3 * d_model, the usual gated-MLP sizing)
    d_ff: int = 2816

    # Per-layer embedding width (Gemma 3n-style PLE); this is also the
    # conditioning input to each block's adaLN-Zero modulation.
    d_ple: int = 64

    # Every Nth layer is full attention (MLA), the rest are gated DeltaNet —
    # Qwen3-Next interleaves them 3:1 (full_attn_every = 4). 1 = all MLA.
    full_attn_every: int = 1

    # --- knowledge-retention experiments (see experiments/); defaults = off ---
    # Also add the PLE lookup into the residual stream through a zero-init
    # projection (Gemma 3n's actual design), not just adaLN conditioning.
    ple_add: bool = False
    # Mixture-of-experts MLPs: n_experts routed (0 = dense SwiGLU), top
    # n_active per token, plus one always-on shared expert of the same size.
    n_experts: int = 0
    n_active: int = 2
    expert_d_ff: int = 1408
    # Replace this layer's feed-forward with a product-key memory (-1 = none).
    memory_layer: int = -1
    memory_slots: int = 65536
    memory_topk: int = 32
    # Untie lm_head from the token embedding.
    tie_embeddings: bool = True
