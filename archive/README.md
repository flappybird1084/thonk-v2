# Archive: retired entrypoints and how we got here

## train_v1.py — the original mainline (Aug 30 – Sep 3, 2026)

The first Thonk trainer: all-MLA NoPE transformer (multi-head latent
attention, SwiGLU, per-layer embeddings conditioning adaLN-Zero, tied
lm_head), Muon + AdamW hybrid optimizer, streaming FineWeb-Edu, WSD schedule.

Its runs:

| run | config | result |
|---|---|---|
| full 20k steps, 1.31B tokens (~6h) | 24→8 layers during tuning; final run 8L/434M→ later 8L/179M variants | val loss 3.339, ppl 28.2; fact probe 20% (checkpoints/run1-20k/) |
| 65-min comparison vs the hybrid | 8L, 179M | val 3.861 @ step 3000 (checkpoints/run2-mla-1h/) |
| ablation anchor e0 (110 min) | 8L, 179M, pure FineWeb | val 3.744 @ step 5394 (experiments/checkpoints/e0_baseline/) |

Kept as a record, not a runnable script: it predates the loader rewrite of
Sep 4, 2026 (`StreamingTokens` now takes a parquet glob instead of a
`datasets` config name), so its data section no longer matches.

## train2.py — 3:1 gated DeltaNet : MLA hybrid (removed Sep 3, 2026)

Qwen3-Next-style layer mix (DDDADDDA). Head-to-head vs v1 on identical data:
**better per token** (matched-step val 3.983 vs 4.122 at step 1500) but 2.1×
slower wall-clock in our pure-PyTorch chunkwise scan, so it lost at fixed GPU
time. Shelved until fused kernels or longer contexts change the math; the
architecture remains available via `ThonkConfig(full_attn_every=4)` and
`components/model/deltanet.py`. Its checkpoint lives in checkpoints2/.

## What replaced v1 (see ../train.py)

The knowledge-retention ablation (experiments/, Sep 3) showed:

- **e1 data mixture**: 20% repeated-Wikipedia slice → retention of the slice
  went from loss 3.99 to 0.88 for a +0.10 FineWeb tax. The one clear win.
- **e2 embedding upgrades** (additive PLE, d_ple 256): +89M params for ~no
  measurable gain. Dropped.
- e3 MoE / e4 memory-layer / e5 untied were stopped before training; MoE and
  product-key memory remain untested config flags in components/model/.

So the new train.py = v1's architecture and optimizer, plus:

1. **Wiki mixture retuned for a full run**: ~3,500 articles (~26M tokens) at
   20% of the stream → ~10 exposures over 1.31B tokens. Breadth instead of
   e1's 300-article deep-memorization slice (which overfit: unseen-wiki loss
   rose from 4.02 to 5.06).
2. **Untied lm_head**: +51.5M params at zero FLOPs, and the init loss starts
   at ln(V) ≈ 10.8 instead of the tied-embedding artifact (~20.4), because
   the model no longer predicts the current token via e_x·norm(e_x).
