"""Train Thonk with PyTorch Lightning: `uv run train_lightning.py`.

A clone of train.py — same hyperparameters, data, Muon + AdamW split, WSD
schedule, dry run, and checkpoint format — with the loop handed to
Lightning's Trainer. Keep the two in lockstep: a change to one is mirrored
in the other (see CLAUDE.md). The one deliberate difference is compilation:
train.py compiles the whole model as a single graph, this script compiles
each module on its own (every block, the final norm, the lm_head).

What Lightning owns here: device placement, bf16 autocast, the train /
validation cadence, and the DataLoader lifecycle. What stays hand-written,
via manual optimization: gradient accumulation, clipping, the schedule, and
stepping the two optimizers. Lightning's automatic optimization drives a
single optimizer, and the two-optimizer split is the point of this trainer.

Multi-GPU: set n_gpus (choose which ones with CUDA_VISIBLE_DEVICES) and
Lightning runs DDP. grad_accum_steps shrinks to match, so the effective
batch, the step count, and the schedule are all still train.py's. Each rank
trains on its own hash-routed share of FineWeb; validation runs the full val
set on every rank, so the val loss is exactly the single-GPU number.

Rough equivalences with train.py's loop, since Lightning counts in
micro-batches (one DataLoader batch) rather than optimizer steps:
    limit_train_batches = total_steps * grad_accum_steps
    val_check_interval  = eval_every  * grad_accum_steps
The sanity-check validation Lightning runs before training doubles as
train.py's step-0 eval (minus the checkpoint).
"""

import glob
import os
import time

import lightning as L
import torch
from lightning.pytorch.utilities import rank_zero_only
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

# ---------------- hyperparameters (mirror train.py) ----------------
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
n_gpus = 1  # DDP across this many GPUs; must divide 8
grad_accum_steps = 8 // n_gpus  # effective batch = batch_size * this * n_gpus = 64
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
checkpoint_dir = "checkpoints_lightning"  # its own dir; infer.py can point here
max_checkpoints = 5  # older ones are deleted (finished runs live in subdirs)

compile_model = True  # torch.compile, one graph per module (see compile_modules)
device = "cuda" if torch.cuda.is_available() else "cpu"
# -------------------------------------------------------------------

print0 = rank_zero_only(print)  # under DDP, only rank 0 prints


def split_params(model: Thonk):
    """Muon gets the body matrices; AdamW gets the rest (see train.py).

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


def compile_modules(model: Thonk):
    """Compile the model one module at a time instead of as a single graph.

    Each block is its own graph (all eight share it: the compiler treats
    module weights as inputs, not constants), and the head — final norm,
    lm_head, loss — is another. Smaller graphs compile faster and a
    recompile stays local. The head must be compiled as one unit: fusing
    the loss into the logits is where most of compile's memory saving comes
    from; compiling final_norm and lm_head separately would forfeit it. The
    embedding lookup is left alone — there is nothing to fuse.

    In-place `.compile()` keeps the module objects, so checkpoint keys stay
    clean. The head is a method rather than a module, so it gets the
    function form, rebound on the instance.
    """
    for block in model.blocks:
        block.compile()
    model.head = torch.compile(model.head)


def make_loader(dataset, workers):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=workers,
        pin_memory=(device == "cuda"),
        worker_init_fn=limit_malloc_arenas,
    )


class LitThonk(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.automatic_optimization = False  # two optimizers, hand-rolled step

        self.cfg = ThonkConfig(
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
        self.model = Thonk(self.cfg)
        if compile_model:
            compile_modules(self.model)

        n_total = sum(p.numel() for p in self.model.parameters())
        n_emb = sum(
            p.numel()
            for n, p in self.model.named_parameters()
            if "emb" in n or ".ple." in n or "lm_head" in n
        )
        print0(
            f"params: {n_total / 1e6:.1f}M total, {(n_total - n_emb) / 1e6:.1f}M non-embedding"
        )

        self.step_idx = 0  # optimizer steps completed — train.py's `step`
        self.loss_accum = 0.0
        self.val_losses = []
        self.last_eval_step = None

    # ---- data ----

    def train_dataloader(self):
        dataset = MixedTokens(
            seq_len, wiki_docs, wiki_frac, fineweb_phases,
            rank=self.global_rank, world_size=self.trainer.world_size,
        )
        return make_loader(dataset, n_workers)

    def val_dataloader(self):
        # One fixed val set (pure held-out fineweb — the mixture never touches
        # it), materialized once: val docs are 1-in-100 in the stream, so
        # re-streaming every eval would waste downloads, and a fixed set keeps
        # val losses comparable. Lightning calls this once and keeps the
        # loader. batch_size=None hands each pre-batched (x, y) through as-is.
        print0("materializing val set...")
        stream = iter(make_loader(StreamingTokens(*FINEWEB, "val", seq_len, val_percent), 0))
        val_set = [next(stream) for _ in range(eval_batches)]
        return DataLoader(val_set, batch_size=None)

    # ---- optimization ----

    def configure_optimizers(self):
        muon_params, adamw_params = split_params(self.model)
        print0(f"muon tensors: {len(muon_params)}, adamw tensors: {len(adamw_params)}")
        return [
            Muon(
                muon_params, lr=muon_lr, momentum=muon_momentum, weight_decay=weight_decay
            ),
            torch.optim.AdamW(
                adamw_params, lr=adam_lr, betas=adam_betas, weight_decay=weight_decay
            ),
        ]

    def on_fit_start(self):
        optimizers = self.trainer.optimizers  # the raw torch optimizers
        self.base_lrs = [[group["lr"] for group in opt.param_groups] for opt in optimizers]

        # Fail fast on OOM, before we sit through minutes of dataset download
        # (same recipe as train.py): dummy forward/backward, lr-0 steps so
        # both optimizers allocate their state, a second forward/backward
        # with that state live — the real steady-state peak — then clear.
        # This is also where the compile happens.
        dummy = torch.zeros((batch_size, seq_len), dtype=torch.long, device=self.device)
        autocast = torch.autocast(device_type=self.device.type, dtype=torch.bfloat16)
        with autocast:
            _, loss = self.model(dummy, dummy)
        loss.backward()
        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = 0.0
            opt.step()
        self.model.zero_grad(set_to_none=True)
        with autocast:
            _, loss = self.model(dummy, dummy)
        loss.backward()
        for opt in optimizers:
            opt.state.clear()
        self.model.zero_grad(set_to_none=True)
        if self.device.type == "cuda":
            peak = torch.cuda.max_memory_allocated() / 2**30
            print0(f"dry run ok | peak GPU memory {peak:.1f} GiB")
        else:
            print0("dry run ok")

    def on_train_start(self):
        self.t0 = time.time()

    def training_step(self, batch, batch_idx):
        # Lightning hands us one micro-batch (already on device, under bf16
        # autocast). We accumulate grad_accum_steps of them, then do what
        # train.py does once per step: schedule, clip, step both optimizers.
        x, y = batch
        _, loss = self.model(x, y)
        self.manual_backward(loss / grad_accum_steps)
        self.loss_accum += loss.item() / grad_accum_steps
        if (batch_idx + 1) % grad_accum_steps != 0:
            return

        step = self.step_idx
        mult = wsd_multiplier(step, total_steps, warmup_steps, cooldown_steps)
        for opt, lrs in zip(self.optimizers(), self.base_lrs):
            for group, base in zip(opt.param_groups, lrs):
                group["lr"] = base * mult

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
        for opt in self.optimizers():
            opt.step()
        self.model.zero_grad(set_to_none=True)

        if step % log_every == 0:
            dt = time.time() - self.t0
            self.t0 = time.time()
            tok_per_s = (
                log_every * grad_accum_steps * n_gpus * batch_size * seq_len / dt
                if step
                else 0
            )
            print0(
                f"step {step:6d} | loss {self.loss_accum:.4f} | lr_mult {mult:.3f} | {tok_per_s:,.0f} tok/s"
            )

        self.step_idx += 1
        self.loss_accum = 0.0

    # ---- evaluation + checkpoints ----

    def validation_step(self, batch, batch_idx):
        x, y = batch
        _, loss = self.model(x, y)
        self.val_losses.append(loss.item())

    def on_validation_epoch_end(self):
        val_loss = sum(self.val_losses) / len(self.val_losses)
        self.val_losses = []
        self.last_eval_step = self.step_idx
        print0(f"step {self.step_idx:6d} | val loss {val_loss:.4f}")
        if not self.trainer.sanity_checking:  # the pre-training look isn't worth saving
            self.save_checkpoint()

    def on_train_end(self):
        # train.py always evaluates and saves at the last step. Lightning only
        # validates on val_check_interval boundaries, so when total_steps
        # doesn't land on one, do the final eval by hand on the same val set.
        if self.last_eval_step == self.step_idx:
            return
        self.model.eval()
        losses = []
        with torch.no_grad(), torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            for x, y in self.trainer.val_dataloaders:
                _, loss = self.model(x.to(self.device), y.to(self.device))
                losses.append(loss.item())
        self.model.train()
        print0(f"step {self.step_idx:6d} | val loss {sum(losses) / len(losses):.4f}")
        self.save_checkpoint()

    def checkpoint_path(self):
        return os.path.join(checkpoint_dir, f"step_{self.step_idx:06d}.pt")

    def save_checkpoint(self):
        # Under multi-GPU every rank holds the same weights; if each one wrote
        # and pruned, they would race on the same files.
        if not self.trainer.is_global_zero:
            return
        os.makedirs(checkpoint_dir, exist_ok=True)
        torch.save(
            {"model": self.model.state_dict(), "config": self.cfg, "step": self.step_idx},
            self.checkpoint_path(),
        )
        # zero-padded names sort chronologically; drop all but the newest
        for old in sorted(glob.glob(os.path.join(checkpoint_dir, "step_*.pt")))[
            :-max_checkpoints
        ]:
            os.remove(old)


def main():
    limit_malloc_arenas()  # workers get it via worker_init_fn in make_loader

    trainer = L.Trainer(
        accelerator="gpu" if device == "cuda" else "cpu",
        devices=n_gpus,
        precision="bf16-mixed",  # fp32 weights, bf16 autocast — as in train.py
        max_epochs=1,  # the stream is endless; limit_train_batches ends the run
        limit_train_batches=total_steps * grad_accum_steps,
        val_check_interval=eval_every * grad_accum_steps,
        limit_val_batches=eval_batches,
        num_sanity_val_steps=eval_batches,
        # Lightning's own logging, checkpoints, and progress bar are off: the
        # print lines and checkpoint files match train.py's instead.
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        # Every rank evaluates the whole val set (it's only eval_batches
        # long) rather than a DistributedSampler slice of it.
        use_distributed_sampler=False,
    )
    trainer.fit(LitThonk())


if __name__ == "__main__":
    main()
