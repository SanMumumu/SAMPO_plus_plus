"""Adapters for using SAMPO++ as a rollout simulator for pi0-style policies.

The module intentionally does not import a pi0 implementation.  Different pi0
checkpoints expose action chunks through slightly different APIs, but the common
contract is a tensor-like action sequence.  This file turns those action chunks
into the action tensor expected by :class:`SampoPlusModel.rollout`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Sequence, Union

import torch


TensorLike = Union[torch.Tensor, Sequence[float], Sequence[Sequence[float]]]


@dataclass
class Pi0ActionSpec:
    """Shape and normalization contract for pi0 action chunks.

    Args:
        action_dim: Number of scalar action dimensions expected by SAMPO++.
        horizon: Optional future horizon.  If set, action chunks are padded or
            truncated to this length before rollout.
        normalization: ``identity``, ``standardize``, or ``minmax``.
        action_mean/action_std: Per-dimension statistics for ``standardize``.
        action_low/action_high: Per-dimension bounds for ``minmax``; maps to
            approximately ``[-1, 1]``.
        no_op_action: Per-dimension no-op action used for context padding and
            horizon padding.  Defaults to zeros.
    """

    action_dim: int
    horizon: Optional[int] = None
    normalization: str = "identity"
    action_mean: Optional[Sequence[float]] = None
    action_std: Optional[Sequence[float]] = None
    action_low: Optional[Sequence[float]] = None
    action_high: Optional[Sequence[float]] = None
    no_op_action: Optional[Sequence[float]] = None

    def no_op(self, *, device=None, dtype=None) -> torch.Tensor:
        if self.no_op_action is None:
            return torch.zeros(self.action_dim, device=device, dtype=dtype or torch.float32)
        return torch.as_tensor(self.no_op_action, device=device, dtype=dtype or torch.float32)


def _stat_tensor(values: Optional[Sequence[float]], name: str, spec: Pi0ActionSpec,
                 *, device, dtype) -> torch.Tensor:
    if values is None:
        raise ValueError(f"{name} is required for {spec.normalization!r} normalization")
    tensor = torch.as_tensor(values, device=device, dtype=dtype)
    if tensor.numel() != spec.action_dim:
        raise ValueError(f"{name} has {tensor.numel()} values, expected {spec.action_dim}")
    return tensor.view(1, 1, spec.action_dim)


def prepare_pi0_actions(actions: TensorLike, spec: Pi0ActionSpec, *,
                        batch_size: Optional[int] = None, horizon: Optional[int] = None,
                        device=None, dtype=torch.float32) -> torch.Tensor:
    """Convert a pi0 action chunk into ``[B, T, D]`` SAMPO++ actions."""
    out = torch.as_tensor(actions, device=device, dtype=dtype)
    if out.ndim == 1:
        out = out.view(1, 1, -1)
    elif out.ndim == 2:
        out = out.unsqueeze(0)
    elif out.ndim != 3:
        raise ValueError(f"pi0 actions must have shape [D], [T,D], or [B,T,D], got {tuple(out.shape)}")

    if out.shape[-1] != spec.action_dim:
        raise ValueError(f"action_dim mismatch: got {out.shape[-1]}, expected {spec.action_dim}")

    if batch_size is not None:
        if out.shape[0] == 1 and batch_size > 1:
            out = out.repeat(batch_size, 1, 1)
        elif out.shape[0] != batch_size:
            raise ValueError(f"batch mismatch: got {out.shape[0]}, expected {batch_size}")

    target_horizon = horizon if horizon is not None else spec.horizon
    if target_horizon is not None:
        target_horizon = int(target_horizon)
        if out.shape[1] > target_horizon:
            out = out[:, :target_horizon]
        elif out.shape[1] < target_horizon:
            pad = spec.no_op(device=out.device, dtype=out.dtype).view(1, 1, spec.action_dim)
            pad = pad.repeat(out.shape[0], target_horizon - out.shape[1], 1)
            out = torch.cat([out, pad], dim=1)

    if spec.normalization == "identity":
        return out
    if spec.normalization == "standardize":
        mean = _stat_tensor(spec.action_mean, "action_mean", spec, device=out.device, dtype=out.dtype)
        std = _stat_tensor(spec.action_std, "action_std", spec, device=out.device, dtype=out.dtype)
        return (out - mean) / std.clamp_min(1e-6)
    if spec.normalization == "minmax":
        low = _stat_tensor(spec.action_low, "action_low", spec, device=out.device, dtype=out.dtype)
        high = _stat_tensor(spec.action_high, "action_high", spec, device=out.device, dtype=out.dtype)
        return 2.0 * (out - low) / (high - low).clamp_min(1e-6) - 1.0
    raise ValueError(f"Unknown normalization mode: {spec.normalization}")


class Pi0SampoInterface:
    """Run SAMPO++ rollouts from pi0-style action chunks."""

    def __init__(self, model, adapter, action_spec: Pi0ActionSpec, *, device: Optional[torch.device] = None):
        self.model = model
        self.adapter = adapter
        self.action_spec = action_spec
        self.device = device if device is not None else next(model.parameters()).device

    def build_rollout_actions(self, context_frames: torch.Tensor, pi0_actions: TensorLike,
                              *, horizon: Optional[int] = None) -> torch.Tensor:
        """Prepend no-op context actions and return ``[B, T_context + T_future, D]``."""
        batch_size, context_length = context_frames.shape[:2]
        future = prepare_pi0_actions(
            pi0_actions,
            self.action_spec,
            batch_size=batch_size,
            horizon=horizon,
            device=self.device,
            dtype=context_frames.dtype,
        )
        no_op = self.action_spec.no_op(device=self.device, dtype=future.dtype).view(1, 1, -1)
        prefix = no_op.repeat(batch_size, context_length, 1)
        return torch.cat([prefix, future], dim=1)

    @torch.no_grad()
    def rollout(self, context_frames: torch.Tensor, pi0_actions: TensorLike, *,
                num_future_frames: Optional[int] = None, num_flow_steps: int = 25) -> Dict:
        """Roll out SAMPO++ under a single pi0 action chunk."""
        context_frames = context_frames.to(self.device)
        future_horizon = num_future_frames if num_future_frames is not None else self.action_spec.horizon
        actions = self.build_rollout_actions(context_frames, pi0_actions, horizon=future_horizon)
        if num_future_frames is None:
            num_future_frames = actions.shape[1] - context_frames.shape[1]
        return self.model.rollout(
            self.adapter,
            context_frames=context_frames,
            actions=actions,
            num_future_frames=int(num_future_frames),
            num_flow_steps=num_flow_steps,
        )

    @torch.no_grad()
    def rollout_candidates(self, context_frames: torch.Tensor, candidate_actions: torch.Tensor, *,
                           num_flow_steps: int = 25) -> Dict:
        """Roll out a set of candidate action sequences.

        ``candidate_actions`` may be ``[N,T,D]`` for a single context batch or
        ``[B,N,T,D]``.  The output contains one rollout per flattened candidate.
        """
        if candidate_actions.ndim == 3:
            candidates = candidate_actions.unsqueeze(0)
        elif candidate_actions.ndim == 4:
            candidates = candidate_actions
        else:
            raise ValueError("candidate_actions must be [N,T,D] or [B,N,T,D]")

        batch_size, num_candidates = candidates.shape[:2]
        if context_frames.shape[0] == 1 and batch_size > 1:
            context = context_frames.repeat(batch_size, 1, 1, 1, 1)
        elif context_frames.shape[0] == batch_size:
            context = context_frames
        else:
            raise ValueError(f"context batch {context_frames.shape[0]} is incompatible with candidates {batch_size}")

        flat_context = context[:, None].repeat(1, num_candidates, 1, 1, 1, 1).flatten(0, 1)
        flat_actions = candidates.flatten(0, 1)
        return self.rollout(flat_context, flat_actions, num_future_frames=flat_actions.shape[1],
                            num_flow_steps=num_flow_steps)

    @torch.no_grad()
    def rank_action_sequences(self, context_frames: torch.Tensor, candidate_actions: torch.Tensor,
                              score_fn: Callable[[Dict], torch.Tensor], *,
                              num_flow_steps: int = 25) -> torch.Tensor:
        """Return candidate indices sorted by descending ``score_fn(rollout)``."""
        rollout = self.rollout_candidates(context_frames, candidate_actions, num_flow_steps=num_flow_steps)
        scores = score_fn(rollout)
        if scores.ndim != 1:
            scores = scores.flatten()
        return torch.argsort(scores, descending=True)
