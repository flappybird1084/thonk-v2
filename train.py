"""Train Thonk from scratch: `uv run train.py`.

Second-generation mainline (the original and the results that shaped this
one are documented in archive/). Two changes over v1, both from the
knowledge-retention ablation in experiments/:

1. The train stream is 80% FineWeb-Edu + 20% of a ~135k-article Wikipedia
   slice cycled for ~10 exposures over the run (components/dataset/mixture.py)
   — repetition of knowledge-dense text is what moves facts into the weights.
   The FineWeb-Edu side runs in two phases: sample/10BT shards 0-2, then 3-7.
2. The lm_head is untied from the token embedding: +51.5M params at zero
   FLOPs, and the init loss starts at ln(V) instead of the tied-embedding
   self-prediction artifact.

Data streams straight from the HF hub — no prepare step, nothing lands on
disk but checkpoints. Expect several minutes of quiet startup while the
wiki slice downloads and tokenizes.

This is pretraining; chat_sft.py fine-tunes the final checkpoint for chat.

Optimization is the Muon + AdamW hybrid (as in the nanoGPT speedruns and
Moonlight/Kimi): Muon drives the transformer body matrices, AdamW handles
everything that isn't a body matrix — embeddings (token + per-layer tables),
the lm_head, norm gains, and the adaLN modulation projections. The adaLN
projections are deliberately kept on AdamW even though they are 2D: they
start at exactly zero, and Muon's updates are scale-invariant (orthogonalized
to unit singular values regardless of gradient size), which would blast the
zero-init gates open on step one instead of letting them ease open.
"""

import glob
import os
import time

import torch
from torch.utils.data import DataLoader

from components.dataset.loader import StreamingTokens, limit_malloc_arenas
from components.dataset.mixture import (
    FINEWEB,
    FINEWEB_EDU_PHASE1,
    FINEWEB_EDU_PHASE2,
    MixedTokens,
)
from components.model import Thonk, ThonkConfig
from components.optimizer.muon import Muon
from components.scheduler.lr import wsd_multiplier

# ---------------- hyperparameters ----------------
# model — see components/model/config.py for what each one means
vocab_size = 50304
d_model = 1024
n_layers = 8
n_heads = 16
head_dim = 64
kv_latent_dim = 512
d_ff = 2816
d_ple = 64
tie_embeddings = False  # untied head: knowledge headroom, cleaner init

# data — pretraining is two phases over FineWeb-Edu's sample/10BT shards,
# played back to back in this one run: shards 0-2 (~2.24B train tokens),
# then shards 3-7 (~3.72B). The switch lands near step 42.7k, in the flat
# stretch of the WSD schedule, so the LR doesn't notice it; the cooldown
# happens entirely inside phase 2.
fineweb_phases = (FINEWEB_EDU_PHASE1, FINEWEB_EDU_PHASE2)
wiki_docs = 135_000  # ≈147M tokens (past the long head articles, ~1k tokens each)
wiki_frac = 0.2  # exposures ≈ run tokens × wiki_frac / slice tokens ≈ 10
val_percent = 1  # 1 in 100 fineweb documents is held out for eval (hash-routed)

# training
seq_len = 1024
batch_size = 8  # per forward pass; fits a 16GB card with room to spare
grad_accum_steps = 8  # effective batch = batch_size * this
n_workers = 1  # one worker tokenizes far faster than the GPU consumes, and
# every extra worker duplicates the whole streaming footprint

# tokens/run = this * batch_size * grad_accum_steps * seq_len ≈ 7.3B. The two
# phases hold ~113.6k steps' worth of FineWeb at its 80% share; stopping ~1.4%
# short guarantees the stream never runs dry before the cooldown finishes.
total_steps = 112_000
warmup_steps = 200
cooldown_steps = 22_000  # final ~20% of the run decays linearly to zero

muon_lr = 0.02
muon_momentum = 0.95
adam_lr = 3e-4
adam_betas = (0.9, 0.95)
weight_decay = 0.01  # applied by both optimizers
grad_clip = 1.0

log_every = 20
eval_every = 500
eval_batches = 20
checkpoint_dir = "checkpoints"
max_checkpoints = 5  # older ones are deleted (finished runs live in subdirs)

compile_model = True  # torch.compile; the dry run absorbs the ~1 min compile
device = "cuda" if torch.cuda.is_available() else "cpu"
# -------------------------------------------------


def split_params(model: Thonk):
    """Muon gets the body matrices; AdamW gets the rest (see module docstring).

    Body matrices are exactly the 2D weights living inside attention and MLP
    modules — that naming test also keeps out the 1D norm gains inside them.
    Embeddings, the lm_head, adaLN projections, and the final norm never match.
    """
    muon_params, adamw_params = [], []
    for name, p in model.named_parameters():
        if p.ndim == 2 and (".attn." in name or ".mlp." in name):
            muon_params.append(p)
        else:
            adamw_params.append(p)
    return muon_params, adamw_params


def main():
    limit_malloc_arenas()  # workers get it via worker_init_fn below

    def make_loader(dataset, workers):
        return iter(
            DataLoader(
                dataset,
                batch_size=batch_size,
                num_workers=workers,
                pin_memory=(device == "cuda"),
                worker_init_fn=limit_malloc_arenas,
            )
        )

    # Kick off the train stream first — its worker downloads and tokenizes
    # the wiki slice and prefetches while we set up and dry-run the model.
    train_batches = make_loader(
        MixedTokens(seq_len, wiki_docs, wiki_frac, fineweb_phases), n_workers
    )

    cfg = ThonkConfig(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        head_dim=head_dim,
        kv_latent_dim=kv_latent_dim,
        d_ff=d_ff,
        d_ple=d_ple,
        tie_embeddings=tie_embeddings,
    )
    model = Thonk(cfg).to(device)

    n_total = sum(p.numel() for p in model.parameters())
    n_emb = sum(
        p.numel()
        for n, p in model.named_parameters()
        if "emb" in n or ".ple." in n or "lm_head" in n
    )
    print(
        f"params: {n_total / 1e6:.1f}M total, {(n_total - n_emb) / 1e6:.1f}M non-embedding"
    )

    muon_params, adamw_params = split_params(model)
    print(f"muon tensors: {len(muon_params)}, adamw tensors: {len(adamw_params)}")
    optimizers = [
        Muon(
            muon_params, lr=muon_lr, momentum=muon_momentum, weight_decay=weight_decay
        ),
        torch.optim.AdamW(
            adamw_params, lr=adam_lr, betas=adam_betas, weight_decay=weight_decay
        ),
    ]
    base_lrs = [[group["lr"] for group in opt.param_groups] for opt in optimizers]

    if compile_model:
        # In-place form of torch.compile(model): same graph, but the module
        # object is unchanged so checkpoint keys carry no `_orig_mod.` prefix.
        model.compile()

    autocast = torch.autocast(device_type=device, dtype=torch.bfloat16)

    # Fail fast on OOM, before we sit through minutes of dataset download:
    # run a full-size dummy micro-batch through forward/backward, step both
    # optimizers at lr 0 so they allocate their state buffers, then run a
    # second forward/backward with that state live. The second pass is the
    # real steady-state peak — the first one under-reports by the size of
    # the optimizer state (~2 GiB at this model size). State is cleared
    # after, so the dummy gradients never touch training (the real loop
    # re-sets lr from base_lrs every step).
    dummy = torch.zeros((batch_size, seq_len), dtype=torch.long, device=device)
    with autocast:
        _, loss = model(dummy, dummy)
    loss.backward()
    for opt in optimizers:
        for group in opt.param_groups:
            group["lr"] = 0.0
        opt.step()
    model.zero_grad(set_to_none=True)
    with autocast:
        _, loss = model(dummy, dummy)
    loss.backward()
    for opt in optimizers:
        opt.state.clear()
    model.zero_grad(set_to_none=True)
    if device == "cuda":
        peak = torch.cuda.max_memory_allocated() / 2**30
        print(f"dry run ok | peak GPU memory {peak:.1f} GiB")
    else:
        print("dry run ok")

    # Materialize one fixed val set up front (pure held-out fineweb — the
    # mixture never touches it): val docs are 1-in-100 in the stream, so
    # re-streaming every eval would waste downloads — and a fixed set keeps
    # val losses comparable across evals.
    print("materializing val set...")
    val_loader = make_loader(StreamingTokens(*FINEWEB, "val", seq_len, val_percent), 0)
    val_set = [next(val_loader) for _ in range(eval_batches)]
    del val_loader

    os.makedirs(checkpoint_dir, exist_ok=True)

    t0 = time.time()
    for step in range(total_steps):
        mult = wsd_multiplier(step, total_steps, warmup_steps, cooldown_steps)
        for opt, lrs in zip(optimizers, base_lrs):
            for group, base in zip(opt.param_groups, lrs):
                group["lr"] = base * mult

        loss_accum = 0.0
        for _ in range(grad_accum_steps):
            x, y = next(train_batches)
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with autocast:
                _, loss = model(x, y)
            (loss / grad_accum_steps).backward()
            loss_accum += loss.item() / grad_accum_steps

        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        for opt in optimizers:
            opt.step()
        model.zero_grad(set_to_none=True)

        if step % log_every == 0:
            dt = time.time() - t0
            t0 = time.time()
            tok_per_s = (
                log_every * grad_accum_steps * batch_size * seq_len / dt if step else 0
            )
            print(
                f"step {step:6d} | loss {loss_accum:.4f} | lr_mult {mult:.3f} | {tok_per_s:,.0f} tok/s"
            )

        if step % eval_every == 0 or step == total_steps - 1:
            model.eval()
            with torch.no_grad():
                val_loss = 0.0
                for x, y in val_set:
                    x, y = x.to(device), y.to(device)
                    with autocast:
                        _, loss = model(x, y)
                    val_loss += loss.item() / len(val_set)
            model.train()
            print(f"step {step:6d} | val loss {val_loss:.4f}")

            torch.save(
                {"model": model.state_dict(), "config": cfg, "step": step},
                os.path.join(checkpoint_dir, f"step_{step:06d}.pt"),
            )
            # zero-padded names sort chronologically; drop all but the newest
            for old in sorted(glob.glob(os.path.join(checkpoint_dir, "step_*.pt")))[
                :-max_checkpoints
            ]:
                os.remove(old)


if __name__ == "__main__":
    main()
