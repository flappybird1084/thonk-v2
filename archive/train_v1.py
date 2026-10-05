"""Train Thonk from scratch: `uv run train.py`.

Data streams straight from the HF hub (see components/dataset/loader.py) —
no prepare step, nothing lands on disk but checkpoints.

Optimization is the Muon + AdamW hybrid (as in the nanoGPT speedruns and
Moonlight/Kimi): Muon drives the transformer body matrices, AdamW handles
everything that isn't a body matrix — embeddings (token + per-layer tables),
the tied lm_head, norm gains, and the adaLN modulation projections. The adaLN
projections are deliberately kept on AdamW even though they are 2D: they start
at exactly zero, and Muon's updates are scale-invariant (orthogonalized to
unit singular values regardless of gradient size), which would blast the
zero-init gates open on step one instead of letting them ease open.
"""

import glob
import os
import time

import torch
from torch.utils.data import DataLoader

from components.dataset.loader import StreamingTokens, limit_malloc_arenas
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

# data
hf_dataset = "HuggingFaceFW/fineweb-edu"
hf_subset = "default"
val_percent = 1  # 1 in 100 documents is held out for eval (hash-routed)

# training
seq_len = 1024
batch_size = 8  # per forward pass; fits a 16GB card with room to spare
grad_accum_steps = 8  # effective batch = batch_size * this
n_workers = 1  # one worker tokenizes far faster than the GPU consumes, and
# every extra worker duplicates the whole streaming footprint

total_steps = 20_000  # tokens/run = this * batch_size * grad_accum_steps * seq_len
warmup_steps = 200
cooldown_steps = 4_000  # final 20% of the run decays linearly to zero

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
max_checkpoints = 5  # older ones are deleted

compile_model = False  # torch.compile; big speedup, slow first step
device = "cuda" if torch.cuda.is_available() else "cpu"
# -------------------------------------------------


def split_params(model: Thonk):
    """Muon gets the body matrices; AdamW gets the rest (see module docstring).

    Body matrices are exactly the 2D weights living inside attention and MLP
    modules — that naming test also keeps out the 1D norm gains inside them.
    Embeddings, adaLN projections, and the final norm never match.
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

    def make_loader(split, workers):
        return iter(
            DataLoader(
                StreamingTokens(hf_dataset, hf_subset, split, seq_len, val_percent),
                batch_size=batch_size,
                num_workers=workers,
                pin_memory=(device == "cuda"),
                worker_init_fn=limit_malloc_arenas,
            )
        )

    # Kick off the train stream first — its workers prefetch in the
    # background while we set up and dry-run the model.
    train_batches = make_loader("train", n_workers)

    cfg = ThonkConfig(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        head_dim=head_dim,
        kv_latent_dim=kv_latent_dim,
        d_ff=d_ff,
        d_ple=d_ple,
    )
    model = Thonk(cfg).to(device)

    n_total = sum(p.numel() for p in model.parameters())
    n_emb = sum(
        p.numel() for n, p in model.named_parameters() if "emb" in n or ".ple." in n
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
        model = torch.compile(model)

    autocast = torch.autocast(device_type=device, dtype=torch.bfloat16)

    # Fail fast on OOM: run one full-size dummy micro-batch through
    # forward/backward and step both optimizers at lr 0 so they allocate
    # their state buffers too — all before we sit through minutes of dataset
    # download. State is cleared after, so the dummy gradients never touch
    # training (the real loop re-sets lr from base_lrs every step).
    dummy = torch.zeros((batch_size, seq_len), dtype=torch.long, device=device)
    with autocast:
        _, loss = model(dummy, dummy)
    loss.backward()
    for opt in optimizers:
        for group in opt.param_groups:
            group["lr"] = 0.0
        opt.step()
        opt.state.clear()
    model.zero_grad(set_to_none=True)
    if device == "cuda":
        peak = torch.cuda.max_memory_allocated() / 2**30
        print(f"dry run ok | peak GPU memory {peak:.1f} GiB")
    else:
        print("dry run ok")

    # Materialize one fixed val set up front: val docs are 1-in-100 in the
    # stream, so re-streaming every eval would waste downloads — and a fixed
    # set keeps val losses comparable across evals.
    print("materializing val set...")
    val_loader = make_loader("val", 0)
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
