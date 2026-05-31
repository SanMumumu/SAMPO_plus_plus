"""SampoPlusModel: the unified temporal-AR + scale-wise-flow world model.

Ties the multi-scale planner (Predict) and the ACVF flow renderer (Refine) into
one module with a training objective and an autoregressive rollout. The training
loss is

    L = sum_s w_s * L_Flow(s)  +  lambda_noop * L_noop  +  lambda_cf * L_cf

where L_noop enforces that the action-control residual vanishes under the no-op
action and L_cf enforces that the true action explains the observed transition
strictly better than a counterfactual action - turning the action channel from a
decorative condition into a causally-grounded control input.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from .acvf import counterfactual_velocity_ranking
from .config import FlowRendererConfig, SampoPlusConfig, TemporalPlannerConfig
from .planner import TemporalPlanner
from .renderer import ScaleAwareFlowRenderer


class SampoPlusModel(nn.Module):
    def __init__(self, config: SampoPlusConfig):
        super().__init__()
        if isinstance(config, dict):
            config = SampoPlusConfig.from_dict(config)
        self.config = config
        self.scales = tuple(config.latent_scales)
        self.num_scales = len(self.scales)
        self.context_length = config.context_length

        planner_cfg = TemporalPlannerConfig(**dict(config.planner)) if config.planner else TemporalPlannerConfig()
        planner_cfg.latent_scales = self.scales
        planner_cfg.__post_init__()
        self.planner = TemporalPlanner(planner_cfg)
        self.multi_scale = planner_cfg.multi_scale

        renderer_cfg = FlowRendererConfig(**dict(config.renderer)) if config.renderer else FlowRendererConfig()
        self.renderer = ScaleAwareFlowRenderer(renderer_cfg)
        self.action_mode = renderer_cfg.action_mode

        self.flow_loss_weights = tuple(config.flow_loss_weights)
        self.lambda_noop = config.lambda_noop
        self.lambda_cf = config.lambda_cf
        self.cf_margin = config.cf_margin
        self.rollout_aware_p = getattr(config, 'rollout_aware_p', 0.0)
        self.rollout_k = getattr(config, 'rollout_k', 4)
        self.rollout_aware_warmup = getattr(config, 'rollout_aware_warmup', 0)
        self.lambda_rollout = getattr(config, 'lambda_rollout', 1.0)

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #
    def compute_loss(self, prepared: Dict, actions: Optional[torch.Tensor] = None,
                     global_step: Optional[int] = None) -> Tuple[torch.Tensor, Dict]:
        fine = prepared['fine_latents']
        context_length = int(prepared.get('context_length', self.context_length))
        T = fine.shape[1]
        n_future = T - context_length
        plans = self.planner.forward_teacher_forcing(fine, actions, context_length)

        use_action = actions is not None and self.action_mode == 'acvf'
        flow_terms, noop_terms, cf_terms = [], [], []

        for f in range(n_future):
            t_global = context_length + f
            action_t = actions[:, t_global] if actions is not None else None
            wrong_action = None
            if use_action:
                perm = torch.randperm(action_t.shape[0], device=action_t.device)
                wrong_action = action_t[perm]

            for si, s in enumerate(self.scales):
                target = prepared[f'pyramid_{s}'][:, t_global]
                coarse = prepared[f'pyramid_{self.scales[si - 1]}'][:, t_global] if si > 0 else None
                plan_s = plans[:, f, si] if self.multi_scale else plans[:, f]
                w = self.flow_loss_weights[si]

                tau = torch.rand(target.shape[0], device=target.device)
                x0 = torch.randn_like(target)
                tau_ = tau.view(-1, 1, 1, 1)
                x_t = (1 - tau_) * x0 + tau_ * target
                target_v = target - x0

                v_true = self.renderer(x_t, tau, coarse, plan_s, action_t, t_global, si,
                                       accumulate_control=use_action)
                flow_terms.append(w * (v_true - target_v).pow(2).flatten(1).mean(1).mean())

                if use_action and self.lambda_noop > 0:
                    zero_action = torch.zeros_like(action_t)
                    self.renderer(x_t, tau, coarse, plan_s, zero_action, t_global, si, accumulate_control=True)
                    noop_terms.append(self.renderer.last_control_energy)
                if use_action and self.lambda_cf > 0:
                    v_wrong = self.renderer(x_t, tau, coarse, plan_s, wrong_action, t_global, si)
                    cf_terms.append(counterfactual_velocity_ranking(v_true, v_wrong, target_v, self.cf_margin))

        loss_flow = torch.stack(flow_terms).mean()
        loss_noop = torch.stack(noop_terms).mean() if noop_terms else loss_flow.new_zeros(())
        loss_cf = torch.stack(cf_terms).mean() if cf_terms else loss_flow.new_zeros(())

        loss_rollout = loss_flow.new_zeros(())
        if self._rollout_active(global_step):
            loss_rollout = self._rollout_consistency(prepared, actions, context_length)

        loss = (loss_flow + self.lambda_noop * loss_noop + self.lambda_cf * loss_cf
                + self.lambda_rollout * loss_rollout)

        metrics = {
            'loss': loss.detach(),
            'loss_flow': loss_flow.detach(),
            'loss_noop': loss_noop.detach(),
            'loss_cf': loss_cf.detach(),
            'loss_rollout': loss_rollout.detach(),
        }
        return loss, metrics

    # ------------------------------------------------------------------ #
    # Rollout-aware training (scheduled self-forcing + K-step consistency)
    # ------------------------------------------------------------------ #
    def _rollout_active(self, global_step: Optional[int]) -> bool:
        if self.rollout_aware_p <= 0:
            return False
        p = self.rollout_aware_p
        if global_step is not None and self.rollout_aware_warmup > 0:
            p = p * min(1.0, float(global_step) / float(self.rollout_aware_warmup))
        return bool(torch.rand(()) < p)

    def _rollout_consistency(self, prepared: Dict, actions: Optional[torch.Tensor],
                             context_length: int, num_flow_steps: int = 4) -> torch.Tensor:
        """K-step imagined rollout in latent space with self-forcing; penalizes
        divergence from the ground-truth latent trajectory. Gradients flow through
        the model's own predictions, aligning training with the rollout distribution."""
        fine = prepared['fine_latents']
        B, T, C = fine.shape[0], fine.shape[1], fine.shape[2]
        k = min(self.rollout_k, T - context_length)
        if k <= 0:
            return fine.new_zeros(())
        history = fine[:, :context_length]
        hist_actions = actions[:, :context_length] if actions is not None else None
        cache, terms = None, []
        for f in range(k):
            t_global = context_length + f
            plan, cache = self.planner.step(history, hist_actions, cache)
            action_t = actions[:, t_global] if actions is not None else None
            z_prev = None
            for si, s in enumerate(self.scales):
                plan_s = plan[:, si] if self.multi_scale else plan
                x = torch.randn(B, C, s, s, device=fine.device)
                dt = 1.0 / num_flow_steps
                for step in range(num_flow_steps):
                    tau = torch.full((B,), step * dt, device=fine.device)
                    x = x + dt * self.renderer(x, tau, z_prev, plan_s, action_t, t_global, si)
                z_prev = x
            terms.append((z_prev - fine[:, t_global]).pow(2).mean())
            history = torch.cat([history, z_prev.unsqueeze(1)], dim=1)   # self-forcing
            if hist_actions is not None and action_t is not None:
                hist_actions = torch.cat([hist_actions, action_t.unsqueeze(1)], dim=1)
        return torch.stack(terms).mean()

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def rollout(self, adapter, context_frames: torch.Tensor, actions: Optional[torch.Tensor] = None,
                num_future_frames: int = 1, num_flow_steps: int = 10, use_rectification: bool = False) -> Dict:
        if use_rectification:
            raise NotImplementedError(
                'Flow rectification is a reserved inference-only module; it is gated off in this revision. '
                'Use rollout-aware training (Phase 5) for principled long-horizon stabilization.'
            )
        device = next(self.parameters()).device
        context_frames = context_frames.to(device)
        B, Tc = context_frames.shape[:2]
        context_fine = adapter.encode_frames(context_frames)
        history_fine = context_fine
        if actions is not None:
            actions = actions.to(device)
            history_actions = actions[:, :Tc]
        else:
            history_actions = None

        predicted, plans_out = [], []
        cache = None
        for f in range(num_future_frames):
            t_global = Tc + f
            plan, cache = self.planner.step(history_fine, history_actions, cache)
            plans_out.append(plan)
            if actions is not None:
                idx = min(t_global, actions.shape[1] - 1)
                action_t = actions[:, idx]
            else:
                action_t = None

            z_prev = None
            for si, s in enumerate(self.scales):
                plan_s = plan[:, si] if self.multi_scale else plan
                x = torch.randn(B, adapter.latent_channels, s, s, device=device)
                dt = 1.0 / num_flow_steps
                for step in range(num_flow_steps):
                    tau = torch.full((B,), step * dt, device=device)
                    v = self.renderer(x, tau, z_prev, plan_s, action_t, t_global, si)
                    x = x + dt * v
                z_prev = x
            predicted.append(z_prev)
            history_fine = torch.cat([history_fine, z_prev.unsqueeze(1)], dim=1)
            if history_actions is not None and action_t is not None:
                history_actions = torch.cat([history_actions, action_t.unsqueeze(1)], dim=1)

        predicted_fine = torch.stack(predicted, dim=1)
        full_fine = torch.cat([context_fine, predicted_fine], dim=1)
        frames = adapter.decode_frames(full_fine)
        plans = torch.stack(plans_out, dim=1)
        return {'frames': frames, 'predicted_fine': predicted_fine, 'plans': plans}

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def save_pretrained(self, save_dir):
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        (save_dir / 'config.json').write_text(json.dumps(self.config.to_dict()), encoding='utf-8')
        torch.save(self.state_dict(), save_dir / 'model.pt')

    @classmethod
    def from_pretrained(cls, load_dir) -> 'SampoPlusModel':
        load_dir = Path(load_dir)
        config = SampoPlusConfig.from_dict(json.loads((load_dir / 'config.json').read_text(encoding='utf-8')))
        model = cls(config)
        state_path = load_dir / 'model.pt'
        if state_path.exists():
            model.load_state_dict(torch.load(state_path, map_location='cpu'), strict=False)
        return model


def build_sampo_plus(adapter, context_length: int = 2, latent_scales=(1, 2, 4, 8, 16),
                     flow_loss_weights=None, action_dim: int = 0, plan_size: int = 512,
                     planner_hidden_size: int = 512, planner_num_layers: int = 4, planner_num_heads: int = 8,
                     renderer_hidden_size: int = 512, renderer_num_layers: int = 6, renderer_num_heads: int = 8,
                     max_frames: int = 32, max_scales: int = 8, multi_scale: bool = True,
                     action_mode: str = 'acvf', rope_mode: str = 'pcd',
                     lambda_noop: float = 0.1, lambda_cf: float = 0.1, cf_margin: float = 0.1,
                     rollout_aware_p: float = 0.0, rollout_k: int = 4, rollout_aware_warmup: int = 0,
                     lambda_rollout: float = 1.0, mlp_ratio: float = 4.0) -> SampoPlusModel:
    scales = tuple(sorted(int(s) for s in latent_scales))
    if flow_loss_weights is None:
        flow_loss_weights = tuple(1.0 for _ in scales)
    planner = TemporalPlannerConfig(
        latent_channels=adapter.latent_channels, hidden_size=planner_hidden_size, plan_size=plan_size,
        action_dim=action_dim, num_layers=planner_num_layers, num_heads=planner_num_heads, mlp_ratio=mlp_ratio,
        max_frames=max_frames, latent_scales=scales, multi_scale=multi_scale,
    )
    renderer = FlowRendererConfig(
        latent_channels=adapter.latent_channels, plan_size=plan_size, action_dim=action_dim,
        hidden_size=renderer_hidden_size, num_layers=renderer_num_layers, num_heads=renderer_num_heads,
        mlp_ratio=mlp_ratio, max_frames=max_frames, max_scales=max(max_scales, len(scales) + 2),
        action_mode=action_mode, rope_mode=rope_mode,
    )
    config = SampoPlusConfig(
        context_length=context_length, latent_scales=scales, flow_loss_weights=flow_loss_weights,
        planner=planner.to_dict(), renderer=renderer.to_dict(),
        lambda_noop=lambda_noop, lambda_cf=lambda_cf, cf_margin=cf_margin,
        action_conditioned=action_dim > 0,
        rollout_aware_p=rollout_aware_p, rollout_k=rollout_k,
        rollout_aware_warmup=rollout_aware_warmup, lambda_rollout=lambda_rollout,
    )
    return SampoPlusModel(config)
