"""Additive knowledge-retention ablation. Run: uv run python -m experiments.ablation

Six sequential runs, each ~TRAIN_MINUTES on the GPU, each adding one change
on top of the last (main train.py is untouched — everything here goes
through config flags that default to off):

    e0_baseline  train.py's model on pure FineWeb-Edu (the anchor)
    e1_datamix   + 20% repeated-Wikipedia data mixture (experiments/mixdata.py)
    e2_ple       + additive PLE path into the residual stream, d_ple 64→256
    e3_moe       + MoE MLPs (8 routed experts + 1 shared, top-2)
    e4_memory    + product-key memory replacing the middle layer's MLP
    e5_untied    + untied lm_head

Every run: same seed, same effective batch (64 windows/step), same schedule
(warmup then flat; runs end at the wall-clock deadline), checkpoints and a
meta.json of metrics under experiments/checkpoints/<name>/. A run whose
meta.json exists is skipped, so the script is resumable. Score everything
afterwards with: uv run python -m experiments.evaluate
"""

import json
import os
import time

import torch
from torch.utils.data import DataLoader

from components.dataset.loader import StreamingTokens, limit_malloc_arenas
from components.model import Thonk, ThonkConfig
from components.optimizer.muon import Muon
from components.scheduler.lr import wsd_multiplier
from components.dataset.mixture import FINEWEB
from experiments.mixdata import MixedTokens

# ---------------- hyperparameters (mirroring train.py) ----------------
base_model = dict(
    vocab_size=50304, d_model=1024, n_layers=8, n_heads=16, head_dim=64,
    kv_latent_dim=512, d_ff=2816, d_ple=64,
)
seq_len = 1024
grad_accum_windows = 64  # effective batch, kept fixed across batch sizes
train_minutes = 110
warmup_steps = 200  # then flat LR until the deadline (no cooldown phase)
muon_lr, muon_momentum = 0.02, 0.95
adam_lr, adam_betas = 3e-4, (0.9, 0.95)
weight_decay, grad_clip = 0.01, 1.0
log_every, eval_every, eval_batches = 50, 500, 20
keep_checkpoints = 2
seed = 1234
# ----------------------------------------------------------------------

EXPERIMENTS = [
    ("e0_baseline", {}, "fineweb"),
    ("e1_datamix", {}, "mix"),
    ("e2_ple", dict(ple_add=True, d_ple=256), "mix"),
    ("e3_moe", dict(ple_add=True, d_ple=256, n_experts=8), "mix"),
    ("e4_memory", dict(ple_add=True, d_ple=256, n_experts=8, memory_layer=4), "mix"),
    (
        "e5_untied",
        dict(ple_add=True, d_ple=256, n_experts=8, memory_layer=4, tie_embeddings=False),
        "mix",
    ),
]


def split_params(model):
    """train.py's Muon/AdamW split, plus: the product-key memory's value
    table is a sparsely-updated embedding living under `.mlp.` — keep it
    (name contains `values`) on AdamW."""
    muon_params, adamw_params = [], []
    for name, p in model.named_parameters():
        if p.ndim == 2 and (".attn." in name or ".mlp." in name) and "values" not in name:
            muon_params.append(p)
        else:
            adamw_params.append(p)
    return muon_params, adamw_params


def run(name, overrides, data, batch_size):
    ckpt_dir = f"experiments/checkpoints/{name}"
    os.makedirs(ckpt_dir, exist_ok=True)
    grad_accum = grad_accum_windows // batch_size
    torch.manual_seed(seed)  # same init and same mixture RNG across experiments

    dataset = MixedTokens(seq_len) if data == "mix" else StreamingTokens(*FINEWEB, "train", seq_len)
    batches = iter(
        DataLoader(
            dataset, batch_size=batch_size, num_workers=1,
            pin_memory=True, worker_init_fn=limit_malloc_arenas,
        )
    )

    cfg = ThonkConfig(**{**base_model, **overrides})
    model = Thonk(cfg).cuda()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[{name}] {n_params / 1e6:.1f}M params | batch {batch_size}x{grad_accum}", flush=True)

    muon_params, adamw_params = split_params(model)
    optimizers = [
        Muon(muon_params, lr=muon_lr, momentum=muon_momentum, weight_decay=weight_decay),
        torch.optim.AdamW(adamw_params, lr=adam_lr, betas=adam_betas, weight_decay=weight_decay),
    ]
    base_lrs = [[g["lr"] for g in opt.param_groups] for opt in optimizers]
    autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16)

    # fail fast on OOM before touching the network (raises to the caller,
    # which retries at a smaller batch size)
    torch.cuda.reset_peak_memory_stats()
    dummy = torch.zeros((batch_size, seq_len), dtype=torch.long, device="cuda")
    with autocast:
        _, loss = model(dummy, dummy)
    loss.backward()
    for opt in optimizers:
        for g in opt.param_groups:
            g["lr"] = 0.0
        opt.step()
        opt.state.clear()
    model.zero_grad(set_to_none=True)
    print(f"[{name}] dry run ok | peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)

    # fixed fineweb val set, identical across experiments (deterministic stream)
    val_loader = iter(
        DataLoader(StreamingTokens(*FINEWEB, "val", seq_len), batch_size=8)
    )
    val_set = [next(val_loader) for _ in range(eval_batches)]
    del val_loader

    def evaluate():
        model.eval()
        total = 0.0
        with torch.no_grad():
            for x, y in val_set:
                with autocast:
                    _, l = model(x.cuda(), y.cuda())
                total += l.item() / len(val_set)
        model.train()
        return total

    def save(step):
        torch.save(
            {"model": model.state_dict(), "config": cfg, "step": step},
            os.path.join(ckpt_dir, f"step_{step:06d}.pt"),
        )
        import glob as g
        for old in sorted(g.glob(os.path.join(ckpt_dir, "step_*.pt")))[:-keep_checkpoints]:
            os.remove(old)

    deadline = time.time() + train_minutes * 60
    val_curve = []
    step = 0
    t_start = time.time()
    while step < 20_000 and time.time() < deadline:
        mult = wsd_multiplier(step, 20_000, warmup_steps, 1)
        for opt, lrs in zip(optimizers, base_lrs):
            for g, base in zip(opt.param_groups, lrs):
                g["lr"] = base * mult

        loss_accum = 0.0
        for _ in range(grad_accum):
            x, y = next(batches)
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            with autocast:
                _, loss = model(x, y)
            (loss / grad_accum).backward()
            loss_accum += loss.item() / grad_accum
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        for opt in optimizers:
            opt.step()
        model.zero_grad(set_to_none=True)

        if step % log_every == 0:
            print(f"[{name}] step {step:5d} | loss {loss_accum:.4f}", flush=True)
        if step % eval_every == 0:
            vl = evaluate()
            val_curve.append([step, round(vl, 4)])
            print(f"[{name}] step {step:5d} | fineweb val {vl:.4f}", flush=True)
            save(step)
        step += 1

    train_time = time.time() - t_start
    vl = evaluate()
    val_curve.append([step, round(vl, 4)])
    save(step)
    tokens = step * grad_accum_windows * seq_len
    meta = {
        "name": name, "overrides": overrides, "data": data,
        "params_m": round(n_params / 1e6, 1), "steps": step, "tokens": tokens,
        "tok_per_s": round(tokens / train_time), "batch_size": batch_size,
        "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 1),
        "minutes": round(train_time / 60, 1), "val_curve": val_curve,
    }
    with open(os.path.join(ckpt_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(f"[{name}] done: {step} steps, {tokens / 1e6:.0f}M tokens, final val {vl:.4f}", flush=True)


def main():
    limit_malloc_arenas()
    for name, overrides, data in EXPERIMENTS:
        if os.path.exists(f"experiments/checkpoints/{name}/meta.json"):
            print(f"[{name}] already done, skipping", flush=True)
            continue
        for batch_size in (8, 4):
            try:
                run(name, overrides, data, batch_size)
                break
            except torch.OutOfMemoryError:
                print(f"[{name}] OOM at batch {batch_size}, retrying smaller", flush=True)
                torch.cuda.empty_cache()
            except Exception as e:  # keep the chain alive overnight
                print(f"[{name}] FAILED: {type(e).__name__}: {e}", flush=True)
                break
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
