"""Intra-frame scale-wise flow renderer (the "Refine" stage).

A Scale-aware DiT parameterizes the flow-matching velocity field v_theta that
transports noise to the scale-s latent, conditioned on the coarser scales, the
per-scale plan, the flow timestep, and the action. The action enters through one
of several mechanisms selected by ``action_mode`` so each row of the
action-conditioning ablation is a one-flag change; the default ``acvf`` routes
the action through the Action-Controlled Velocity Field residual.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .acvf import ACVFBlock
from .common import MLP, RMSNorm, modulate, timestep_embedding
from .config import FlowRendererConfig
from .pcd_rope import build_spatial_rope


def sincos_2d(n_h: int, n_w: int, dim: int, device) -> torch.Tensor:
    """Deterministic 2D sin-cos positional embedding, [n_h*n_w, dim]."""
    half = dim // 2
    yy, xx = torch.meshgrid(
        torch.arange(n_h, device=device, dtype=torch.float32),
        torch.arange(n_w, device=device, dtype=torch.float32),
        indexing='ij',
    )
    omega = torch.arange(half // 2, device=device, dtype=torch.float32) / max(half // 2, 1)
    omega = 1.0 / (10000 ** omega)
    ey = torch.cat([torch.sin(yy.flatten()[:, None] * omega), torch.cos(yy.flatten()[:, None] * omega)], dim=-1)
    ex = torch.cat([torch.sin(xx.flatten()[:, None] * omega), torch.cos(xx.flatten()[:, None] * omega)], dim=-1)
    emb = torch.cat([ey, ex], dim=-1)
    if emb.shape[-1] < dim:
        emb = F.pad(emb, (0, dim - emb.shape[-1]))
    return emb[:, :dim]


class SpatialAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, rope):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.rope = rope

    def forward(self, x: torch.Tensor, scale_index: int, H: int, W: int) -> torch.Tensor:
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]                         # [B, H, N, hd]
        q, k = self.rope(q, k, scale_index, H, W)
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, D)
        return self.proj(out)


class CrossAttention(nn.Module):
    """Query=tokens, key/value=single action token (for the 'crossattn' ablation)."""

    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, 2 * dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        kv = self.kv(cond).reshape(B, 1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, D)
        return self.proj(out)


class DiTBlock(nn.Module):
    def __init__(self, config: FlowRendererConfig, rope):
        super().__init__()
        dim = config.hidden_size
        self.action_mode = config.action_mode
        self.norm1 = RMSNorm(dim)
        self.attn = SpatialAttention(dim, config.num_heads, rope)
        self.norm2 = RMSNorm(dim)
        self.mlp = MLP(dim, config.mlp_ratio)
        # passive (action-free) structural + timestep modulation
        self.adaln = nn.Linear(dim, 6 * dim)
        nn.init.zeros_(self.adaln.weight)
        nn.init.zeros_(self.adaln.bias)

        if self.action_mode == 'acvf':
            self.acvf = ACVFBlock(dim, config.action_dim, config.max_scales, config.gamma_max)
        elif self.action_mode in ('adm', 'adaln'):
            self.act_mod = nn.Linear(dim, 2 * dim)
            nn.init.zeros_(self.act_mod.weight)
            nn.init.zeros_(self.act_mod.bias)
        elif self.action_mode == 'crossattn':
            self.cross = CrossAttention(dim, config.num_heads)

    def forward(self, x, c_struct, h_s, action_emb, action_raw, scale_index, H, W):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaln(c_struct).chunk(6, dim=-1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), scale_index, H, W)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))

        control = None
        if self.action_mode == 'acvf' and action_raw is not None:
            control = self.acvf.control_residual(x, h_s, action_raw, scale_index)
            x = x + control
        elif self.action_mode in ('adm', 'adaln') and action_emb is not None:
            gamma, beta = self.act_mod(action_emb).chunk(2, dim=-1)
            x = (1.0 + gamma.unsqueeze(1)) * x + beta.unsqueeze(1)
        elif self.action_mode == 'crossattn' and action_emb is not None:
            x = x + self.cross(x, action_emb)
        return x, control


class ScaleAwareFlowRenderer(nn.Module):
    def __init__(self, config: FlowRendererConfig):
        super().__init__()
        if not isinstance(config, FlowRendererConfig):
            config = FlowRendererConfig(**config)
        self.config = config
        dim = config.hidden_size
        C = config.latent_channels
        self.action_mode = config.action_mode

        self.x_embed = nn.Linear(C, dim)
        self.coarse_embed = nn.Linear(C, dim)
        self.t_mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.scale_embed = nn.Embedding(config.max_scales, dim)
        self.frame_embed = nn.Embedding(config.max_frames, dim)
        self.plan_proj = nn.Linear(config.plan_size, dim)
        if config.action_dim > 0 and config.action_mode in ('concat', 'crossattn', 'adm', 'adaln'):
            self.action_mlp = nn.Linear(config.action_dim, dim)
        else:
            self.action_mlp = None

        rope = build_spatial_rope(config.rope_mode, dim // config.num_heads, config.max_scales)
        self.blocks = nn.ModuleList([DiTBlock(config, rope) for _ in range(config.num_layers)])

        self.final_norm = RMSNorm(dim)
        self.final_adaln = nn.Linear(dim, 2 * dim)
        nn.init.zeros_(self.final_adaln.weight)
        nn.init.zeros_(self.final_adaln.bias)
        self.out_proj = nn.Linear(dim, C)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        self.last_control_energy = torch.zeros(())

    def forward(self, x, t, coarse, plan, action, frame_index, scale_index, accumulate_control=False):
        B, C, H, W = x.shape
        N = H * W
        dim = self.config.hidden_size
        tokens = self.x_embed(x.flatten(2).transpose(1, 2))                 # [B, N, D]
        tokens = tokens + sincos_2d(H, W, dim, x.device).unsqueeze(0)
        if coarse is not None:
            coarse_up = F.interpolate(coarse, size=(H, W), mode='bilinear', align_corners=False)
            tokens = tokens + self.coarse_embed(coarse_up.flatten(2).transpose(1, 2))

        scale_idx = torch.tensor(min(scale_index, self.config.max_scales - 1), device=x.device)
        frame_idx = torch.tensor(min(int(frame_index), self.config.max_frames - 1), device=x.device)
        plan_h = self.plan_proj(plan)                                       # [B, D] - planner state h_t^s
        c_struct = self.t_mlp(timestep_embedding(t, dim)) + self.scale_embed(scale_idx) \
            + self.frame_embed(frame_idx) + plan_h

        action_emb = None
        if action is not None and self.action_mlp is not None:
            action_emb = self.action_mlp(action)
            if self.action_mode == 'adm':
                action_emb = action_emb + self.scale_embed(scale_idx)
            if self.action_mode == 'concat':
                tokens = tokens + action_emb.unsqueeze(1)
                action_emb = None

        control_energy = x.new_zeros(())
        h = tokens
        for block in self.blocks:
            h, control = block(h, c_struct, plan_h, action_emb, action, scale_idx, H, W)
            if accumulate_control and control is not None:
                control_energy = control_energy + control.pow(2).mean()
        self.last_control_energy = control_energy

        shift, scale = self.final_adaln(c_struct).chunk(2, dim=-1)
        h = modulate(self.final_norm(h), shift, scale)
        v = self.out_proj(h).transpose(1, 2).reshape(B, C, H, W)
        return v


class SILoss:
    """Stochastic-interpolant / conditional flow-matching loss with a linear
    (optimal-transport) path. Returns a per-sample loss vector [B]."""

    def __init__(self, path_type: str = 'linear'):
        self.path_type = path_type

    def interpolate(self, x0, x1, t):
        t_ = t.view(-1, *([1] * (x1.ndim - 1)))
        x_t = (1 - t_) * x0 + t_ * x1
        target_v = x1 - x0
        return x_t, target_v

    def __call__(self, renderer, target, coarse, plan, action, frame_index, scale_index,
                 accumulate_control=False):
        B = target.shape[0]
        t = torch.rand(B, device=target.device)
        x0 = torch.randn_like(target)
        x_t, target_v = self.interpolate(x0, target, t)
        v = renderer(x_t, t, coarse, plan, action, frame_index, scale_index,
                     accumulate_control=accumulate_control)
        return (v - target_v).pow(2).flatten(1).mean(1)
