"""Pyramid-Consistent Dynamic RoPE (PCD-RoPE) and positional-encoding routing.

Two fixes over a naive 4D (time / scale / space) rotary extension (the legacy
SA-RoPE flagged by Reviewer 1, points 4 & 11):

  1. Resolution-normalized PHYSICAL coordinates: token (i, j) at scale s maps to
     ((i+0.5)/H_s, (j+0.5)/W_s), so coarse (1x1) and fine (16x16) tokens share
     ONE continuous coordinate frame instead of scale-dependent integer indices.
     The same physical location attends consistently across the pyramid.
  2. Scale-specific FREQUENCY BANDS  Omega_s = M_s * Omega: coarse scales keep
     only low frequencies; finer scales progressively open higher ones. This
     resolves the "identical rotary frequency on a 1x1 and a 256x256 grid"
     physical implausibility.

``build_spatial_rope`` exposes the same interface for every ablation mode so
switching positional encodings is a one-flag change.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).reshape_as(x)


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: [..., N, D]; cos/sin: [N, D] broadcast over the leading dims."""
    return x * cos + rotate_half(x) * sin


class SpatialRoPE(nn.Module):
    """Spatial rotary position embedding with selectable mode.

    mode:
      * 'pcd'      - PCD-RoPE: normalized physical coords + scale-specific bands.
      * 'sarope4d' - legacy SA-RoPE: integer coords, identical band on every scale.
      * 'rope2d'   - vanilla 2D RoPE: integer coords, full band (no scale mask).
      * 'spacetime'- alias of 'rope2d' for the spatial axes (time handled by the
                     frame embedding outside attention).
      * 'learned'  - no rotary (the renderer falls back to its learned 2D table).
    """

    def __init__(self, dim_axis: int, num_scales: int, mode: str = 'pcd', base: float = 10000.0):
        super().__init__()
        self.mode = mode
        self.num_scales = num_scales
        # rotary channels per spatial axis = dim_axis; total spatial rotary dim = 2*dim_axis
        inv = 1.0 / (base ** (torch.arange(0, dim_axis, 2).float() / dim_axis))   # [dim_axis/2]
        self.register_buffer('inv_freq', inv, persistent=False)
        masks = []
        for s in range(num_scales):
            k = max(1, round(len(inv) * (s + 1) / num_scales))
            m = torch.zeros(len(inv))
            m[:k] = 1.0
            masks.append(m)
        self.register_buffer('band', torch.stack(masks), persistent=False)         # [S, dim_axis/2]

    def _coords(self, n: int, mode_normalized: bool, device):
        idx = torch.arange(n, device=device).float()
        if mode_normalized:
            return (idx + 0.5) / max(n, 1)        # resolution-normalized physical coords in [0, 1)
        return idx                                # raw integer indices (legacy)

    def _angles(self, s: int, H_s: int, W_s: int, device) -> torch.Tensor:
        normalized = self.mode == 'pcd'
        yi = self._coords(H_s, normalized, device)
        xj = self._coords(W_s, normalized, device)
        gy, gx = torch.meshgrid(yi, xj, indexing='ij')
        f = self.inv_freq.to(device)
        if self.mode == 'pcd':
            s_idx = min(s, self.num_scales - 1)
            f = f * self.band[s_idx].to(device)   # scale-specific band-limited frequencies
        ay = torch.outer(gy.flatten(), f)         # [N, dim_axis/2]
        ax = torch.outer(gx.flatten(), f)
        ang = torch.cat([ay, ax], dim=-1)         # [N, dim_axis]
        return torch.repeat_interleave(ang, 2, dim=-1)   # [N, 2*dim_axis] interleaved pairs

    def forward(self, q: torch.Tensor, k: torch.Tensor, s: int, H_s: int, W_s: int):
        if self.mode == 'learned':
            return q, k
        ang = self._angles(s, H_s, W_s, q.device)
        cos, sin = ang.cos().to(q.dtype), ang.sin().to(q.dtype)
        return apply_rotary(q, cos, sin), apply_rotary(k, cos, sin)


def build_spatial_rope(rope_mode: str, head_dim: int, num_scales: int) -> SpatialRoPE:
    """Factory: total spatial rotary dim = head_dim = 2 * dim_axis."""
    if head_dim % 4 != 0:
        # rotary needs an even split across (y, x) and interleaved pairs.
        raise ValueError(f'head_dim={head_dim} must be divisible by 4 for spatial RoPE')
    dim_axis = head_dim // 2
    return SpatialRoPE(dim_axis=dim_axis, num_scales=num_scales, mode=rope_mode)


# The reference implementation exported this under the name ``PCDRoPE``.
PCDRoPE = SpatialRoPE
