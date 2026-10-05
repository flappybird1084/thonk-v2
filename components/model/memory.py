"""Product-key memory layer (Lample et al. 2019; Meta's "Memory Layers at
Scale", 2024).

A huge learned key-value store standing in for an MLP: each token retrieves
its top-k of memory_slots value vectors and takes a softmax-weighted sum.
Nearly all the parameters are in the value table, and only k rows of it are
touched per token, so it adds knowledge capacity at almost no FLOPs — Meta
measured large factual-QA gains at fixed compute from exactly this trade.

The product-key trick makes the search over N slots affordable: keys are the
cartesian product of two sets of √N half-keys. Score both halves against the
two halves of the query (2·√N small dot products), take top-k in each, and
only the k×k candidate combinations need exact scoring. The search is
approximate but the paper shows it loses little.

The value table must NOT be trained with Muon (sparsely updated embedding
rows, like the token embedding) — the experiment runner's split rule keys on
the attribute name `values`.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from components.model.config import ThonkConfig


class ProductKeyMemory(nn.Module):
    def __init__(self, cfg: ThonkConfig):
        super().__init__()
        self.n_half = int(math.isqrt(cfg.memory_slots))
        assert self.n_half**2 == cfg.memory_slots, "memory_slots must be a square"
        self.topk = cfg.memory_topk
        half = cfg.d_model // 2

        self.q_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.keys1 = nn.Parameter(torch.randn(self.n_half, half) * half**-0.5)
        self.keys2 = nn.Parameter(torch.randn(self.n_half, half) * half**-0.5)
        self.values = nn.Embedding(cfg.memory_slots, cfg.d_model)
        # Output gate as in Meta's memory-layer variant; the block-level
        # adaLN gate already zero-inits the whole branch, so no special init.
        self.gate_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        q1, q2 = self.q_proj(x).chunk(2, dim=-1)

        s1, i1 = (q1 @ self.keys1.T).topk(self.topk, dim=-1)  # [B,T,k] each half
        s2, i2 = (q2 @ self.keys2.T).topk(self.topk, dim=-1)
        # all k×k combinations of the two half-scores, then the true top-k
        grid = s1.unsqueeze(-1) + s2.unsqueeze(-2)  # [B,T,k,k]
        scores, flat_idx = grid.flatten(-2).topk(self.topk, dim=-1)
        slot = (
            i1.gather(-1, flat_idx // self.topk) * self.n_half
            + i2.gather(-1, flat_idx % self.topk)
        )

        picked = self.values(slot)  # [B,T,k,D]
        out = (F.softmax(scores, dim=-1).unsqueeze(-1) * picked).sum(-2)
        return out * F.silu(self.gate_proj(x))
