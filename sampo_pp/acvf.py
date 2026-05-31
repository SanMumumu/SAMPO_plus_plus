"""Action-Controlled Velocity Field (ACVF) - replaces the legacy ADM block.

The scale-s flow velocity is decomposed as

    v = v_drift  +  m ⊙ ( g ⊙ control(u, B_s ψ(a)) )

  * v_drift : passive evolution (NO action) - structural + timestep conditioning,
              supplied by the surrounding DiT block.
  * control : action-induced residual; scale-specific basis B_s; spatial mask m;
              tanh-bounded channel gain g; zero-initialized output projection.

Two construction guarantees make the model a stable passive integrator that
learns control as an identity-preserving residual:

  (i)  ψ has NO bias  ->  ψ(0)=0  ->  the residual is EXACTLY zero for the no-op
       action a=0 at every point during training (not just at init). This is the
       structural form of the L_noop objective and directly answers the
       "action conditioning is decorative / unstable" criticism.
  (ii) the output projection is zero-initialized  ->  the residual is exactly
       zero at initialization for ANY action, so the velocity field starts as a
       pure passive flow and the ODE solver is stable from step 0.

The scale-specific basis ``B_s`` is small-random initialized (not zero) so that
gradient reaches it through the zero-initialized projection.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .common import RMSNorm


class ACVFBlock(nn.Module):
    def __init__(self, dim: int, action_dim: int, num_scales: int, gamma_max: float = 1.0):
        super().__init__()
        self.gamma_max = gamma_max
        self.norm = RMSNorm(dim)

        # action-control branch
        self.psi = nn.Linear(action_dim, dim, bias=False)              # ψ(a); ψ(0)=0 exactly
        self.B = nn.Parameter(torch.randn(num_scales, dim, dim) * (dim ** -0.5))   # scale basis B_s
        self.gain = nn.Linear(dim, dim)                                # pre-tanh channel gain from h_s
        self.mask = nn.Sequential(                                     # spatial action-influence mask
            nn.Linear(2 * dim, dim), nn.SiLU(), nn.Linear(dim, 1)
        )
        # control output projection: NO bias so that control(a=0)=proj(0)=0 holds
        # exactly at all times; zero-init weight so the residual is also 0 at init.
        self.proj = nn.Linear(dim, dim, bias=False)
        nn.init.zeros_(self.proj.weight)

    def control_residual(self, u: torch.Tensor, h_s: torch.Tensor, a: torch.Tensor, s: int) -> torch.Tensor:
        """Δv_act(u, h_s, a, s). Exactly 0 for the no-op action a=0 (ψ has no bias)."""
        un = self.norm(u)                                             # [B, N, D]
        s_idx = min(s, self.B.shape[0] - 1)
        a_emb = self.psi(a) @ self.B[s_idx]                           # B_s ψ(a)        [B, D]
        ctrl = un * a_emb.unsqueeze(1)                                # inject action into features
        g = self.gamma_max * torch.tanh(self.gain(h_s)).unsqueeze(1)  # bounded gain    [B, 1, D]
        m = torch.sigmoid(self.mask(torch.cat(                        # spatial mask    [B, N, 1]
            [un, h_s.unsqueeze(1).expand(-1, un.size(1), -1)], dim=-1)))
        return m * (g * self.proj(ctrl))                             # zero at init / at no-op


def noop_consistency_loss(residual: torch.Tensor) -> torch.Tensor:
    """L_noop: the accumulated control residual under the null action must vanish."""
    return residual.pow(2).mean()


def counterfactual_velocity_ranking(v_true: torch.Tensor, v_wrong: torch.Tensor,
                                    v_target: torch.Tensor, margin: float = 0.1) -> torch.Tensor:
    """L_cf: the velocity under the TRUE action fits the flow-matching target
    (z_t^s - eps) strictly better than under a sampled WRONG action."""
    d_true = (v_true - v_target).pow(2).flatten(1).mean(1)
    d_wrong = (v_wrong - v_target).pow(2).flatten(1).mean(1)
    return torch.relu(margin + d_true - d_wrong).mean()


@torch.no_grad()
def scale_action_energy(v_action: torch.Tensor, v_noop: torch.Tensor) -> torch.Tensor:
    """Per-scale action energy  E_s(a) = || v(.,a) - v(.,0) ||_2  (measurement only).

    Expectation for the "scale-aware control" evidence figure: a coarse-scale
    action (e.g. 'move arm') has high energy on coarse/mid scales, a fine action
    (e.g. 'gripper close') on mid/fine scales, and the no-op near zero on all.
    """
    return (v_action - v_noop).flatten(1).norm(dim=1).mean()
