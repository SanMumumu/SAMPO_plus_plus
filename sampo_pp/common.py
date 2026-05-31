"""Shared low-level building blocks for the SAMPO++ ``sampo_pp`` package.

Kept dependency-light (torch only) and compatible with torch>=1.12, so the
package is importable and unit-testable on a CPU-only box without ``nn.RMSNorm``
(which only ships with torch>=2.4).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Root-mean-square layer norm.

    Falls back to a manual implementation so the package runs on torch<2.4.
    We deliberately use RMSNorm everywhere in the continuous flow path: it has
    no mean-subtraction, which keeps the velocity field's zero-action fixed
    point exactly at the origin and avoids the additive drift that plain
    LayerNorm injects into long ODE rollouts.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        norm = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (norm * self.weight.float()).to(dtype)


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """AdaLN-style affine modulation. ``x``: [B, N, D]; ``shift``/``scale``: [B, D]."""
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 10000.0) -> torch.Tensor:
    """Sinusoidal embedding for a continuous flow-time scalar ``t`` in [0, 1]."""
    if t.ndim == 0:
        t = t[None]
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / max(half, 1)
    )
    args = t.float()[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb.to(t.dtype if t.is_floating_point() else torch.float32)


class MLP(nn.Module):
    """Standard transformer feed-forward block."""

    def __init__(self, dim: int, mlp_ratio: float = 4.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


def zero_module(module: nn.Module) -> nn.Module:
    """Zero out the parameters of a module (identity-preserving residual init)."""
    for p in module.parameters():
        nn.init.zeros_(p)
    return module
