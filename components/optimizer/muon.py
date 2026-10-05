"""Muon — MomentUm Orthogonalized by Newton-Schulz (Keller Jordan et al., 2024).

The idea: take the SGD-momentum update for a weight *matrix*, and replace it
with the nearest (semi-)orthogonal matrix before applying it. Orthogonalizing
flattens the update's singular values to ~1, so every direction in the matrix
moves at the same rate instead of a few dominant directions hogging the step.
This trains transformer body matrices markedly faster than AdamW (nanoGPT
speedruns; Moonshot's Moonlight and Kimi K2 at scale).

Muon is only meant for 2D weight matrices in the model body. Embeddings, the
lm_head, norm gains, biases — anything that isn't really a "matrix" acting on
activations — should stay on AdamW. train.py does that split.
"""

import torch


def orthogonalize_via_newtonschulz(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Approximate UV^T where G = U S V^T, i.e. G with singular values -> 1.

    A quintic Newton-Schulz iteration. The coefficients are Jordan's, tuned so
    that ~5 iterations flatten the spectrum well even from a poor start; the
    iteration is stable enough to run entirely in bfloat16.
    """
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16()
    # The iteration needs the smaller dimension first (it multiplies X X^T,
    # so keep that square factor as small as possible); transpose back at the end.
    transposed = G.size(-2) > G.size(-1)
    if transposed:
        X = X.mT
    # Normalize so the spectral norm is <= 1, which the iteration requires.
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        X = a * X + (b * A + c * A @ A) @ X
    if transposed:
        X = X.mT
    return X.to(G.dtype)


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95, weight_decay=0.0, ns_steps=5):
        defaults = dict(
            lr=lr, momentum=momentum, weight_decay=weight_decay, ns_steps=ns_steps
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        # The optional closure is the torch.optim convention (re-evaluate the
        # loss before stepping); training frameworks like Lightning pass one.
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                assert p.ndim == 2, "Muon is for 2D weight matrices only"

                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p.grad)
                buf = state["momentum_buffer"]

                # Nesterov momentum: blend the fresh gradient back into the
                # updated buffer, which looks one step ahead.
                buf.lerp_(p.grad, 1 - group["momentum"])
                update = p.grad.lerp_(buf, group["momentum"])

                update = orthogonalize_via_newtonschulz(update, group["ns_steps"])

                if group["weight_decay"] > 0:
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                # Scale by sqrt(fan_out/fan_in) (when > 1) so tall matrices
                # take proportionally larger steps — keeps the per-neuron
                # update RMS consistent across differently-shaped layers.
                scale = max(1.0, p.size(-2) / p.size(-1)) ** 0.5
                p.add_(update, alpha=-group["lr"] * scale)
        return loss
