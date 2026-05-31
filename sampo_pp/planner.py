"""Inter-frame causal temporal planner (the "Predict" stage).

Phase-1 contribution. The legacy planner consumed only the finest-scale latent
and emitted a single global plan vector, contradicting the paper's multi-scale
claim (Reviewer 1, point 9). Here the planner:

  * consumes the FULL latent pyramid of every history frame (a per-scale encoder
    that reads each grid distinctly, so coarse layout and fine texture enter the
    causal state through separate pathways), and
  * emits one plan vector PER SCALE via S scale-specific readout heads on a
    shared causal backbone (``multi_scale=True``). The coarse readout drives
    low-frequency / global dynamics, the fine readout high-frequency detail.

``multi_scale=False`` reproduces the legacy single-vector planner exactly, so the
single-vs-multi-scale ablation is a one-flag change.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import MLP, RMSNorm
from .config import TemporalPlannerConfig


class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]                      # [B, H, T, hd]
        attn = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        attn = attn.masked_fill(mask, float('-inf'))
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, T, D)
        return self.proj(out)


class TemporalBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = CausalSelfAttention(dim, num_heads)
        self.norm2 = RMSNorm(dim)
        self.mlp = MLP(dim, mlp_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class TemporalPlanner(nn.Module):
    def __init__(self, config: TemporalPlannerConfig):
        super().__init__()
        if not isinstance(config, TemporalPlannerConfig):
            config = TemporalPlannerConfig(**config)
        self.config = config
        self.scales = tuple(config.latent_scales)
        self.num_scales = len(self.scales)
        self.multi_scale = config.multi_scale
        D = config.hidden_size
        C = config.latent_channels

        # per-scale input encoder: each grid s x s read distinctly (multi-scale input)
        self.scale_in = nn.ModuleList([nn.Linear(C * s * s, D) for s in self.scales])
        self.scale_embed = nn.Embedding(self.num_scales, D)
        self.frame_pos = nn.Embedding(config.max_frames, D)
        self.action_proj = nn.Linear(config.action_dim, D) if config.action_dim > 0 else None

        self.blocks = nn.ModuleList(
            [TemporalBlock(D, config.num_heads, config.mlp_ratio) for _ in range(config.num_layers)]
        )
        self.norm = RMSNorm(D)

        n_read = self.num_scales if self.multi_scale else 1
        self.readout = nn.Parameter(torch.randn(n_read, D) * 0.02)
        self.plan_head = nn.Linear(D, config.plan_size)

    # ------------------------------------------------------------------ #
    def _frame_tokens(self, fine_latents: torch.Tensor, actions: Optional[torch.Tensor]) -> torch.Tensor:
        """[B, T, C, H, W] (+ optional actions) -> per-frame tokens [B, T, D]."""
        B, T, C, H, W = fine_latents.shape
        flat = fine_latents.reshape(B * T, C, H, W)
        tok = 0.0
        for i, s in enumerate(self.scales):
            pooled = F.adaptive_avg_pool2d(flat, s) if s != H else flat
            pooled = pooled.reshape(B, T, C * s * s)
            tok = tok + self.scale_in[i](pooled) + self.scale_embed.weight[i]
        if self.action_proj is not None and actions is not None:
            tok = tok + self.action_proj(actions[:, :T])
        positions = torch.arange(T, device=fine_latents.device)
        tok = tok + self.frame_pos(positions).unsqueeze(0)
        return tok

    def _backbone(self, tokens: torch.Tensor) -> torch.Tensor:
        x = tokens
        for block in self.blocks:
            x = block(x)
        return self.norm(x)

    def _readout_plans(self, h: torch.Tensor) -> torch.Tensor:
        """h: [B, L, D] -> plans [B, L, S, P] (multi) or [B, L, P] (single)."""
        if self.multi_scale:
            mixed = h.unsqueeze(2) + self.readout.view(1, 1, self.num_scales, -1)   # [B, L, S, D]
            return self.plan_head(mixed)
        return self.plan_head(h + self.readout[0])

    # ------------------------------------------------------------------ #
    def forward_teacher_forcing(self, fine_latents: torch.Tensor, actions: Optional[torch.Tensor],
                                context_length: int) -> torch.Tensor:
        """Returns plans for the future frames: [B, T-context, (S,) P].

        The plan that generates future frame t is read from the causal hidden
        state at position t-1 (it has seen frames <t and actions <t)."""
        tokens = self._frame_tokens(fine_latents, actions)
        h = self._backbone(tokens)
        T = fine_latents.shape[1]
        h_future = h[:, context_length - 1: T - 1]              # [B, T-context, D]
        return self._readout_plans(h_future)

    def step(self, history_fine: torch.Tensor, past_actions: Optional[torch.Tensor],
             cache: Optional[Dict] = None) -> Tuple[torch.Tensor, Dict]:
        """One autoregressive step: plan for the next frame from the history.

        Returns ``(plan, cache)`` with plan ``[B, (S,) P]``. The plan is read
        from the last causal position, so it is invariant to whether a cache is
        supplied (KV-cache parity)."""
        tokens = self._frame_tokens(history_fine, past_actions)
        h = self._backbone(tokens)
        plan = self._readout_plans(h[:, -1:])[:, 0]            # [B, (S,) P]
        new_cache = {'length': history_fine.shape[1]}
        return plan, new_cache
