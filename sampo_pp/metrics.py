"""World-model-native evaluation metrics.

FVD measures perceptual realism, not whether a *world model* is controllable or
physically stable. These metrics probe the properties a world model is actually
supposed to have:

  * action_alignment     - does the action-induced velocity point along the true
                           latent displacement? (controllability direction)
  * counterfactual_accuracy - does the TRUE action explain the observed transition
                           better than a wrong action? (action causality)
  * noop_stability       - does the scene stay still under the no-op action?
  * rollout_drift        - imagined-vs-GT latent divergence as a function of the
                           horizon, plus its linear slope (long-horizon stability).
  * object_permanence_proxy - does predicted structure persist over the rollout?
  * action_energy_profile- per-scale action energy E_s(a)=||v(.,a)-v(.,0)|| (the
                           scale-decoupling evidence figure).

All functions are torch-only and run on a dummy batch without external deps.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F

from .acvf import scale_action_energy


@torch.no_grad()
def rollout_drift(predicted_fine: torch.Tensor, target_fine: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Per-step latent L2 divergence and its slope. Inputs: [B, T, C, H, W]."""
    d = (predicted_fine - target_fine).flatten(2).norm(dim=2).mean(0)        # [T]
    t = torch.arange(d.shape[0], device=d.device, dtype=d.dtype)
    tc, dc = t - t.mean(), d - d.mean()
    slope = (tc * dc).sum() / (tc.pow(2).sum() + 1e-8)
    return {'per_step': d, 'slope': slope, 'final': d[-1]}


@torch.no_grad()
def _sample_flow_state(target: torch.Tensor):
    tau = torch.rand(target.shape[0], device=target.device)
    x0 = torch.randn_like(target)
    tau_ = tau.view(-1, *([1] * (target.ndim - 1)))
    return (1 - tau_) * x0 + tau_ * target, tau, target - x0


@torch.no_grad()
def counterfactual_accuracy(model, prepared: Dict, actions: torch.Tensor) -> torch.Tensor:
    """Fraction of (frame, scale) cells where the true action fits the flow target
    strictly better than a batch-shuffled wrong action."""
    fine = prepared['fine_latents']
    context_length = int(prepared.get('context_length', model.context_length))
    plans = model.planner.forward_teacher_forcing(fine, actions, context_length)
    correct, total = 0.0, 0
    for f in range(fine.shape[1] - context_length):
        t_global = context_length + f
        a_true = actions[:, t_global]
        a_wrong = a_true[torch.randperm(a_true.shape[0], device=a_true.device)]
        for si, s in enumerate(model.scales):
            target = prepared[f'pyramid_{s}'][:, t_global]
            coarse = prepared[f'pyramid_{model.scales[si - 1]}'][:, t_global] if si > 0 else None
            plan_s = plans[:, f, si] if model.multi_scale else plans[:, f]
            x_t, tau, target_v = _sample_flow_state(target)
            v_true = model.renderer(x_t, tau, coarse, plan_s, a_true, t_global, si)
            v_wrong = model.renderer(x_t, tau, coarse, plan_s, a_wrong, t_global, si)
            d_true = (v_true - target_v).flatten(1).pow(2).mean(1)
            d_wrong = (v_wrong - target_v).flatten(1).pow(2).mean(1)
            correct += (d_true < d_wrong).float().sum().item()
            total += d_true.shape[0]
    return torch.tensor(correct / max(total, 1))


@torch.no_grad()
def noop_stability(model, adapter, context_frames: torch.Tensor, num_future_frames: int = 8,
                   num_flow_steps: int = 10) -> torch.Tensor:
    """Mean squared frame-to-frame change of the predicted rollout under the
    no-op action. Lower = more stable (the world stays put when not acted on)."""
    action_dim = model.planner.config.action_dim
    actions = None
    if action_dim > 0:
        total = context_frames.shape[1] + num_future_frames
        actions = torch.zeros(context_frames.shape[0], total, action_dim, device=context_frames.device)
    out = model.rollout(adapter, context_frames, actions=actions,
                        num_future_frames=num_future_frames, num_flow_steps=num_flow_steps)
    pred = out['frames'][:, context_frames.shape[1]:]
    if pred.shape[1] < 2:
        return torch.zeros((), device=pred.device)
    return (pred[:, 1:] - pred[:, :-1]).flatten(1).pow(2).mean()


@torch.no_grad()
def action_energy_profile(model, prepared: Dict, actions: torch.Tensor,
                          frame: int = 0) -> Optional[torch.Tensor]:
    """Per-scale action energy E_s for a single transition: [S]."""
    if model.action_mode != 'acvf':
        return None
    fine = prepared['fine_latents']
    context_length = int(prepared.get('context_length', model.context_length))
    plans = model.planner.forward_teacher_forcing(fine, actions, context_length)
    t_global = context_length + frame
    a_true = actions[:, t_global]
    a_zero = torch.zeros_like(a_true)
    energies = []
    for si, s in enumerate(model.scales):
        target = prepared[f'pyramid_{s}'][:, t_global]
        coarse = prepared[f'pyramid_{model.scales[si - 1]}'][:, t_global] if si > 0 else None
        plan_s = plans[:, frame, si] if model.multi_scale else plans[:, frame]
        x_t, tau, _ = _sample_flow_state(target)
        v_a = model.renderer(x_t, tau, coarse, plan_s, a_true, t_global, si)
        v_0 = model.renderer(x_t, tau, coarse, plan_s, a_zero, t_global, si)
        energies.append(scale_action_energy(v_a, v_0))
    return torch.stack(energies)


@torch.no_grad()
def action_alignment(model, prepared: Dict, actions: torch.Tensor, frame: int = 0) -> torch.Tensor:
    """Cosine similarity between the action-induced velocity $v(\\cdot,a)-v(\\cdot,0)$
    and the true latent displacement $z_t-z_{t-1}$, averaged over scales. A value
    near $1$ means the action steers the field toward the observed change; near $0$
    means the action does not move the dynamics in the right direction."""
    fine = prepared['fine_latents']
    context_length = int(prepared.get('context_length', model.context_length))
    plans = model.planner.forward_teacher_forcing(fine, actions, context_length)
    t_global = context_length + frame
    a_true = actions[:, t_global]
    a_zero = torch.zeros_like(a_true)
    aligns = []
    for si, s in enumerate(model.scales):
        target = prepared[f'pyramid_{s}'][:, t_global]
        prev = prepared[f'pyramid_{s}'][:, t_global - 1]
        coarse = prepared[f'pyramid_{model.scales[si - 1]}'][:, t_global] if si > 0 else None
        plan_s = plans[:, frame, si] if model.multi_scale else plans[:, frame]
        x_t, tau, _ = _sample_flow_state(target)
        dv = (model.renderer(x_t, tau, coarse, plan_s, a_true, t_global, si)
              - model.renderer(x_t, tau, coarse, plan_s, a_zero, t_global, si))
        disp = target - prev
        aligns.append(F.cosine_similarity(dv.flatten(1), disp.flatten(1), dim=1).mean())
    return torch.stack(aligns).mean()


@torch.no_grad()
def object_permanence_proxy(predicted_fine: torch.Tensor, target_fine: torch.Tensor) -> torch.Tensor:
    """Mean cosine similarity between predicted and ground-truth latents per frame,
    a proxy for whether scene structure persists over the rollout (1 = retained,
    0 = collapsed). Inputs: [B, T, C, H, W]."""
    p = predicted_fine.flatten(2)
    g = target_fine.flatten(2)
    return F.cosine_similarity(p, g, dim=2).mean()
