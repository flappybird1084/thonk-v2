"""Chat fine-tune a pretrained Thonk: `uv run chat_sft.py`.

The third stage after train.py's two pretraining phases: supervised
fine-tuning on smol-smoltalk + smoltalk + Synthetic-Persona-Chat, rendered
in the ChatML template with loss only on assistant turns (see
components/dataset/chat.py). Starts from the newest checkpoint in
checkpoints/ — the end of pretraining — and writes to checkpoints_chat/,
which infer.py recognizes as a chat model.

Same model, same Muon + AdamW split, same dry run as train.py; what changes
is the data and the schedule. Pretraining ended its cooldown at lr 0, so the
fine-tune warms back up briefly to a fraction of the pretraining LRs and
decays linearly to zero — a full-LR restart would undo too much of what
pretraining stored. Optimizer state starts fresh (checkpoints don't carry
it).
"""

import glob
import os
import time

import torch
from torch.utils.data import DataLoader

from components.dataset.chat import ChatWindows
from components.dataset.loader import limit_malloc_arenas
from components.model import Thonk
from components.optimizer.muon import Muon
from components.scheduler.lr import wsd_multiplier
from train import adam_betas, adam_lr, muon_lr, muon_momentum, split_params

# ---------------- hyperparameters ----------------
init_checkpoint_dir = "checkpoints"  # newest step_*.pt in here is the base model
init_checkpoint = ""  # or a specific file

# data — share of conversations drawn from each source. At 20k steps
# (~1.24 conversations per window) this is about one pass over smoltalk
# (1.04M) and smol-smoltalk (460k), and ~3 over the ~20k persona chats,
# which would be lost at their natural 1% share.
chat_mix = {"smol-smoltalk": 0.30, "smoltalk": 0.66, "persona": 0.04}
val_mix = {"smol-smoltalk-test": 1.0}

# training — the batch shape matches pretraining
seq_len = 1024
batch_size = 8
grad_accum_steps = 8

total_steps = 20_000
warmup_steps = 100
lr_scale = 0.2  # fraction of train.py's Muon/AdamW LRs
weight_decay = 0.01
grad_clip = 1.0

log_every = 20
eval_every = 500
eval_batches = 20
checkpoint_dir = "checkpoints_chat"
max_checkpoints = 5

compile_model = True
device = "cuda" if torch.cuda.is_available() else "cpu"
# -------------------------------------------------


def load_base_model() -> Thonk:
    path = init_checkpoint or max(
        glob.glob(os.path.join(init_checkpoint_dir, "step_*.pt")), default=None
    )
    if path is None:
        raise SystemExit(f"no checkpoints in {init_checkpoint_dir}/ — run train.py first")
    # weights_only=False: the checkpoint pickles our ThonkConfig dataclass.
    ckpt = torch.load(path, map_location=device, weights_only=False)
    # Older checkpoints, saved from a torch.compile'd model, prefix every key.
    state = {k.removeprefix("_orig_mod."): v for k, v in ckpt["model"].items()}
    model = Thonk(ckpt["config"]).to(device)
    model.load_state_dict(state)
    print(f"fine-tuning {path} (pretrained for {ckpt['step'] + 1} steps)")
    return model


def main():
    limit_malloc_arenas()

    def make_loader(dataset):
        return iter(
            DataLoader(
                dataset,
                batch_size=batch_size,
                num_workers=1,  # ChatWindows is single-worker by design
                pin_memory=(device == "cuda"),
                worker_init_fn=limit_malloc_arenas,
            )
        )

    train_batches = make_loader(ChatWindows(seq_len, chat_mix))

    model = load_base_model()
    cfg = model.cfg

    muon_params, adamw_params = split_params(model)
    optimizers = [
        Muon(
            muon_params,
            lr=muon_lr * lr_scale,
            momentum=muon_momentum,
            weight_decay=weight_decay,
        ),
        torch.optim.AdamW(
            adamw_params, lr=adam_lr * lr_scale, betas=adam_betas, weight_decay=weight_decay
        ),
    ]
    base_lrs = [[group["lr"] for group in opt.param_groups] for opt in optimizers]

    if compile_model:
        model.compile()

    autocast = torch.autocast(device_type=device, dtype=torch.bfloat16)

    # Fail fast on OOM — the same dry run as train.py (see there for why it
    # steps the optimizers and runs twice).
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
    print("dry run ok")

    # A fixed val set from smol-smoltalk's held-out test split.
    print("materializing val set...")
    val_loader = make_loader(ChatWindows(seq_len, val_mix))
    val_set = [next(val_loader) for _ in range(eval_batches)]
    del val_loader

    os.makedirs(checkpoint_dir, exist_ok=True)

    t0 = time.time()
    for step in range(total_steps):
        # WSD with the cooldown spanning the whole run: warmup, then linear
        # decay to zero.
        mult = wsd_multiplier(step, total_steps, warmup_steps, total_steps - warmup_steps)
        for opt, lrs in zip(optimizers, base_lrs):
            for group, base in zip(opt.param_groups, lrs):
                group["lr"] = base * mult

        loss_accum = 0.0
        for _ in range(grad_accum_steps):
            x, y = next(train_batches)
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with autocast:
                _, loss = model(x, y)  # mean over assistant tokens only
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

            # "chat": True tells infer.py to use the chat template.
            torch.save(
                {"model": model.state_dict(), "config": cfg, "step": step, "chat": True},
                os.path.join(checkpoint_dir, f"step_{step:06d}.pt"),
            )
            for old in sorted(glob.glob(os.path.join(checkpoint_dir, "step_*.pt")))[
                :-max_checkpoints
            ]:
                os.remove(old)


if __name__ == "__main__":
    main()
