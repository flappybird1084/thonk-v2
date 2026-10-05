"""Mixture-of-experts feed-forward (DeepSeek-V2/V3 and Qwen3 style).

Facts live overwhelmingly in MLP weights, so MoE multiplies that storage
without multiplying per-token compute: each token runs one always-on shared
expert plus its top n_active of n_experts routed experts, all narrower than
the dense MLP they replace.

Load balancing is DeepSeek-V3's auxiliary-loss-free trick: a non-learned
per-expert bias is added to the routing scores for SELECTION only (gate
weights still come from the raw scores), and after each training forward the
bias is nudged up for underloaded experts and down for overloaded ones.
"""

import torch
import torch.nn as nn

from components.model.config import ThonkConfig
from components.model.mlp import SwiGLU


class MoEMLP(nn.Module):
    def __init__(self, cfg: ThonkConfig):
        super().__init__()
        self.n_experts = cfg.n_experts
        self.n_active = cfg.n_active
        self.experts = nn.ModuleList(
            SwiGLU(cfg, d_ff=cfg.expert_d_ff) for _ in range(cfg.n_experts)
        )
        self.shared = SwiGLU(cfg, d_ff=cfg.expert_d_ff)
        self.router = nn.Linear(cfg.d_model, cfg.n_experts, bias=False)
        self.register_buffer("route_bias", torch.zeros(cfg.n_experts))
        self.bias_speed = 1e-3

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        flat = x.reshape(-1, D)

        scores = torch.sigmoid(self.router(flat))  # sigmoid scores, as in DeepSeek-V3
        picked = (scores + self.route_bias).topk(self.n_active, dim=-1).indices
        gates = scores.gather(-1, picked)
        gates = gates / (gates.sum(-1, keepdim=True) + 1e-9)

        out = self.shared(flat)
        for e in range(self.n_experts):
            rows, slot = torch.where(picked == e)
            if rows.numel():
                out[rows] += gates[rows, slot, None] * self.experts[e](flat[rows])

        if self.training:
            with torch.no_grad():
                load = torch.bincount(
                    picked.reshape(-1), minlength=self.n_experts
                ).float()
                load = load / load.sum()
                self.route_bias += self.bias_speed * torch.sign(load.mean() - load)

        return out.view(B, T, D)
