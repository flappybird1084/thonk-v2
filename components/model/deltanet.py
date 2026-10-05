"""Gated DeltaNet: linear attention with the gated delta rule.

(Yang et al. 2024, "Gated Delta Networks"; the linear-attention mixer in
Qwen3-Next, which interleaves it 3:1 with full attention.)

Instead of a KV cache that grows with context, each head carries a fixed-size
state matrix S ∈ R^{dv×dk} — a small associative memory mapping keys to
values. Per token:

    S_t = a_t * S_{t-1} + β_t (v_t − a_t S_{t-1} k_t) k_tᵀ
    o_t = S_t q_t

That is the classic delta rule — overwrite whatever the memory currently
returns for key k_t with v_t, at write strength β_t ∈ (0,1) — plus a
per-token forget gate a_t ∈ (0,1] (Mamba2-style decay). With L2-normalized
keys the update is contractive, so the recurrence is stable at any length.
No positional encoding is needed: the recurrence is ordered by construction,
which suits this NoPE model.

gated_delta_recurrent() below IS that definition, one token at a time, and is
kept as the reference the fast path is tested against. gated_delta_scan() is
what training uses: within a chunk of C tokens the recurrence collapses into
one C×C triangular solve (the WY-representation trick from the DeltaNet
parallelization paper), so the sequential loop runs T/C times instead of T
times. The scan runs in fp32 regardless of autocast.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from components.model.config import ThonkConfig


def gated_delta_recurrent(q, k, v, beta, g):
    """The definition, one token at a time. Sequential and slow — tests only.

    q, k, v: [B, T, H, d];  beta, g: [B, T, H], β write strength in (0,1),
    g log-decay ≤ 0. Returns o: [B, T, H, d].
    """
    B, T, H, d = q.shape
    S = q.new_zeros(B, H, v.shape[-1], d, dtype=torch.float32)
    outs = []
    for t in range(T):
        S = g[:, t].exp().float()[..., None, None] * S
        kt, vt, qt = k[:, t].float(), v[:, t].float(), q[:, t].float()
        bt = beta[:, t].float()
        returned = (S @ kt.unsqueeze(-1)).squeeze(-1)  # memory's current value for k_t
        S = S + (bt.unsqueeze(-1) * (vt - returned)).unsqueeze(-1) * kt.unsqueeze(-2)
        outs.append((S @ qt.unsqueeze(-1)).squeeze(-1))
    return torch.stack(outs, dim=1).to(q.dtype)


def gated_delta_scan(q, k, v, beta, g, chunk_size=64):
    """Chunkwise gated delta rule; same signature/result as the recurrent form.

    Derivation sketch: define pseudo-values u_t = β_t(v_t − a_t S_{t-1} k_t);
    then the update is plain gated linear attention, S_t = a_t S_{t-1} + u_t k_tᵀ.
    Substituting the unrolled state into u's definition gives, per chunk with
    entry state S₀ and inclusive cumulative log-decay G_t:

        (I + D) U = β V − β e^{G} K S₀ᵀ,   D[t,s] = β_t e^{G_t−G_s} (k_tᵀ k_s), s<t
        o_t = e^{G_t} S₀ q_t + Σ_{s≤t} e^{G_t−G_s} (q_tᵀ k_s) u_s
        S_C = e^{G_C} S₀ + Σ_s e^{G_C−G_s} u_s k_sᵀ
    """
    B, T, H, d = q.shape
    out_dtype = q.dtype
    with torch.autocast(device_type=q.device.type, enabled=False):
        q, k, v = (t.float().transpose(1, 2) for t in (q, k, v))  # [B, H, T, d]
        beta, g = beta.float().transpose(1, 2), g.float().transpose(1, 2)

        # Pad T to a chunk multiple with no-op tokens: β=0 writes nothing,
        # g=0 decays nothing; the padded outputs are sliced off at the end.
        pad = -T % chunk_size
        if pad:
            q, k, v = (F.pad(t, (0, 0, 0, pad)) for t in (q, k, v))
            beta, g = F.pad(beta, (0, pad)), F.pad(g, (0, pad))
        C = chunk_size
        N = (T + pad) // C
        q, k, v = (t.view(B, H, N, C, d) for t in (q, k, v))
        beta = beta.view(B, H, N, C)
        G = g.view(B, H, N, C).cumsum(-1)

        # decay[t,s] = e^{G_t−G_s}: how much what token s wrote has decayed by
        # the time token t reads it. Zeroed (via exp(-inf)) where s is in the
        # future; the strict variant also zeroes the diagonal.
        diff = G.unsqueeze(-1) - G.unsqueeze(-2)  # [B, H, N, C, C]
        below = torch.tril(torch.ones(C, C, dtype=torch.bool, device=q.device))
        decay = diff.masked_fill(~below, float("-inf")).exp()
        decay_strict = diff.masked_fill(~below.tril(-1), float("-inf")).exp()

        # One triangular solve per chunk covers both right-hand sides — the
        # data term and the coefficient of S₀ — so the sequential loop below
        # is only a couple of matmuls per chunk. unitriangular: the solver
        # takes D's diagonal as 1, i.e. it solves against (I + D).
        D = beta.unsqueeze(-1) * decay_strict * (k @ k.mT)
        rhs = torch.cat(
            [beta.unsqueeze(-1) * v, (beta * G.exp()).unsqueeze(-1) * k], dim=-1
        )
        sol = torch.linalg.solve_triangular(D, rhs, upper=False, unitriangular=True)
        U_v, W = sol.split([v.shape[-1], d], dim=-1)  # U = U_v − W S₀ᵀ

        M = decay * (q @ k.mT)  # within-chunk reads of pseudo-values

        S = q.new_zeros(B, H, v.shape[-1], d)
        outs = []
        for i in range(N):
            U = U_v[:, :, i] - W[:, :, i] @ S.mT
            outs.append(
                G[:, :, i].exp().unsqueeze(-1) * (q[:, :, i] @ S.mT) + M[:, :, i] @ U
            )
            g_end = G[:, :, i, -1]
            S = g_end.exp()[..., None, None] * S + (
                U * (g_end.unsqueeze(-1) - G[:, :, i]).exp().unsqueeze(-1)
            ).mT @ k[:, :, i]
        o = torch.cat(outs, dim=2)[:, :, :T]
    return o.transpose(1, 2).to(out_dtype)


class GatedDeltaNet(nn.Module):
    def __init__(self, cfg: ThonkConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        hd = cfg.n_heads * cfg.head_dim

        self.q_proj = nn.Linear(cfg.d_model, hd, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, hd, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, hd, bias=False)
        # Short causal depthwise convs give the recurrence a local view a few
        # tokens wide (both the paper and Qwen3-Next include them).
        self.conv_q = nn.Conv1d(hd, hd, 4, groups=hd, padding=3, bias=False)
        self.conv_k = nn.Conv1d(hd, hd, 4, groups=hd, padding=3, bias=False)
        self.conv_v = nn.Conv1d(hd, hd, 4, groups=hd, padding=3, bias=False)

        self.beta_proj = nn.Linear(cfg.d_model, cfg.n_heads, bias=False)
        # Forget gate a_t = exp(−A · softplus(a_proj(x) + dt_bias)), per head —
        # Mamba2's discretization and its init: A in [1, 16], base timestep
        # log-uniform in [0.001, 0.1], so heads start spread across timescales.
        self.a_proj = nn.Linear(cfg.d_model, cfg.n_heads, bias=False)
        self.A_log = nn.Parameter(torch.empty(cfg.n_heads).uniform_(1, 16).log())
        dt = torch.exp(
            torch.empty(cfg.n_heads).uniform_(torch.log(torch.tensor(0.001)), torch.log(torch.tensor(0.1)))
        )
        self.dt_bias = nn.Parameter(torch.log(torch.expm1(dt)))  # inverse softplus

        # Output: per-head RMSNorm, gated by silu(gate) — the paper's gated
        # output norm; it sets the magnitude the recurrence itself doesn't.
        self.out_norm = nn.RMSNorm(cfg.head_dim)
        self.gate_proj = nn.Linear(cfg.d_model, hd, bias=False)
        self.o_proj = nn.Linear(hd, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        H, dh = self.n_heads, self.head_dim

        # conv1d wants [B, channels, T]; slice off the causal padding tail
        q = F.silu(self.conv_q(self.q_proj(x).mT)[..., :T].mT)
        k = F.silu(self.conv_k(self.k_proj(x).mT)[..., :T].mT)
        v = F.silu(self.conv_v(self.v_proj(x).mT)[..., :T].mT)
        q = F.normalize(q.view(B, T, H, dh), dim=-1)  # unit keys/queries keep
        k = F.normalize(k.view(B, T, H, dh), dim=-1)  # the delta rule contractive
        v = v.view(B, T, H, dh)

        beta = torch.sigmoid(self.beta_proj(x))
        g = -self.A_log.exp() * F.softplus(self.a_proj(x) + self.dt_bias)

        o = gated_delta_scan(q, k, v, beta, g)
        o = self.out_norm(o) * F.silu(self.gate_proj(x)).view(B, T, H, dh)
        return self.o_proj(o.reshape(B, T, H * dh))
