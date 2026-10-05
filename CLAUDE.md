# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

A from-scratch LLM built in Python, incorporating recent techniques published in open models (Qwen3, Qwen3-Next, and similar). Everything is implemented here rather than pulled from a modeling library.

## Design philosophy: "handwritten-like"

This is the most important rule in the repo and it overrides default habits:

- **Clarity beats micro-optimization.** If a trick saves 2% compute but costs 10% more lines, don't do it. Write the obvious version.
- **Explain code when necessary.** Non-obvious math, tensor-shape gymnastics, and technique-specific choices (e.g. why a particular attention variant or norm placement) deserve a short comment — ideally citing the paper/model the technique comes from. Obvious code gets no comment.
- Code should read like a person carefully wrote it to be understood, not like it was generated or golfed.

## Layout

Keep the project root clean — don't dump files there unless they genuinely belong at the top level.

- `components/model/` — model architecture (attention, MLP, norms, embeddings, etc.)
- `components/dataset/` — data loading and tokenization
- `components/scheduler/` — LR schedules and similar training machinery (add subdirectories under `components/` as needed, following this pattern)
- `train.py` — training entrypoint
- `train_lightning.py` — the same trainer on PyTorch Lightning, compiled per module. It is a clone of `train.py` and must stay one: any change to `train.py`'s hyperparameters, data, optimizer split, schedule, or checkpoint format is mirrored there in the same change (and vice versa)
- `chat_sft.py` — chat fine-tune of the pretrained model (smol-smoltalk, smoltalk, Synthetic-Persona-Chat), the stage after `train.py`; writes `checkpoints_chat/`
- `infer.py` — inference entrypoint (base or chat mode, picked by the checkpoint)
- `train2.py` etc. — experiment variants: full siblings of train.py with different hyperparams or architecture flags, each writing to its own checkpoint dir
- `experiments/` — multi-run ablations and their evaluation/results
- `archive/` — retired entrypoints, with a README documenting the runs and findings that retired them

Hyperparameters live as plain variables at the top of the entrypoint scripts — no argparse.

## Toolchain

Python 3.12 managed by uv, already initialized in this repo. Always go through uv:

- `uv run train.py` / `uv run infer.py` — run entrypoints
- `uv add <package>` — add a dependency (updates `pyproject.toml` and `uv.lock`)
- `uv sync` — sync the environment to the lockfile

Do not use pip, create a venv manually, or edit dependencies in `pyproject.toml` by hand.
