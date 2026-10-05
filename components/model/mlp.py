"""SwiGLU feed-forward block (Shazeer 2020, "GLU Variants Improve Transformer").

The standard gated MLP used by Llama/Qwen/DeepSeek: one branch is squashed
through SiLU and gates the other elementwise, then everything projects back
down. No biases, as is now conventional.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from components.model.config import ThonkConfig


class SwiGLU(nn.Module):
    def __init__(self, cfg: ThonkConfig, d_ff: int | None = None):
        super().__init__()
        d_ff = d_ff or cfg.d_ff  # MoE builds narrower expert copies
        self.gate_proj = nn.Linear(cfg.d_model, d_ff, bias=False)
        self.up_proj = nn.Linear(cfg.d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
