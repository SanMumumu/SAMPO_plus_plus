"""Configuration dataclasses for the ``sampo_pp`` package.

Every behavioural change introduced for the TPAMI revision sits behind a config
flag whose default reproduces the intended SAMPO++ behaviour, while alternative
values reproduce the ablation corners (single-scale planner, AdaLN/ADM action
conditioning, legacy SA-RoPE, etc.). This keeps the 2x2 / ablation tables a
one-flag change rather than a code fork.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, Optional, Tuple


def parse_scales(value) -> Tuple[int, ...]:
    """Parse ``'1,2,4,8,16'`` (or an iterable) into a sorted tuple of ints."""
    if value is None:
        return (1, 2, 4, 8, 16)
    if isinstance(value, str):
        scales = tuple(int(item.strip()) for item in value.split(',') if item.strip())
    else:
        scales = tuple(int(item) for item in value)
    if not scales:
        raise ValueError('latent_scales must be non-empty')
    return tuple(sorted(scales))


@dataclass
class AdapterConfig:
    """Continuous VAE adapter / latent-pyramid configuration."""

    backend: str = 'debug'                       # {'debug', 'lightningdit_vavae'}
    ckpt_path: Optional[str] = None
    repo_path: Optional[str] = None
    input_resolution: int = 64
    model_resolution: int = 256
    latent_resolution: int = 16
    latent_channels: int = 16
    latent_scales: Tuple[int, ...] = (1, 2, 4, 8, 16)
    use_variational: bool = False

    def __post_init__(self):
        self.latent_scales = parse_scales(self.latent_scales)
        self.latent_resolution = max(self.latent_resolution, max(self.latent_scales))

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class TemporalPlannerConfig:
    """Inter-frame causal planner.

    ``multi_scale=True`` is the SAMPO++ contribution: the planner consumes the
    full latent pyramid and emits one plan vector *per scale* (S readout heads
    on a shared causal backbone). ``multi_scale=False`` reproduces the legacy
    single-vector planner that the reviewers flagged as self-contradictory.
    """

    latent_channels: int = 16
    hidden_size: int = 512
    plan_size: int = 512
    action_dim: int = 0
    num_layers: int = 4
    num_heads: int = 8
    mlp_ratio: float = 4.0
    max_frames: int = 32
    latent_scales: Tuple[int, ...] = (1, 2, 4, 8, 16)
    multi_scale: bool = True
    recurrence: str = 'compact_state'            # {'compact_state', 'full_latent'}

    def __post_init__(self):
        self.latent_scales = parse_scales(self.latent_scales)

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class FlowRendererConfig:
    """Intra-frame scale-wise flow renderer (the velocity field v_theta).

    ``action_mode`` selects how the action conditions the velocity field and is
    the axis of the action-conditioning ablation table:
      * 'concat'    - action concatenated to the input tokens (passive).
      * 'crossattn' - action injected via cross-attention (passive).
      * 'adaln'     - action-only AdaLN (vanilla conditional norm).
      * 'adm'       - scale-aware AdaLN (the legacy SAMPO++ ADM block).
      * 'acvf'      - Action-Controlled Velocity Field (default, the new core).
    ``rope_mode`` selects the positional encoding (PCD-RoPE by default).
    """

    latent_channels: int = 16
    plan_size: int = 512
    action_dim: int = 0
    hidden_size: int = 512
    num_layers: int = 6
    num_heads: int = 8
    mlp_ratio: float = 4.0
    max_frames: int = 32
    max_scales: int = 8
    action_mode: str = 'acvf'
    rope_mode: str = 'pcd'                        # {'learned','rope2d','spacetime','sarope4d','pcd'}
    gamma_max: float = 1.0

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class SampoPlusConfig:
    """Top-level world-model configuration."""

    context_length: int = 2
    latent_scales: Tuple[int, ...] = (1, 2, 4, 8, 16)
    flow_loss_weights: Tuple[float, ...] = (1.0, 1.0, 1.0, 1.0, 1.0)
    planner: Dict = field(default_factory=dict)
    renderer: Dict = field(default_factory=dict)
    # Causal action objectives (Phase 2). Set the lambdas to 0 to ablate.
    lambda_noop: float = 0.1
    lambda_cf: float = 0.1
    cf_margin: float = 0.1
    action_conditioned: bool = True
    # Rollout-aware training (Phase 5): scheduled self-forcing + K-step
    # rollout-consistency. rollout_aware_p=0 reproduces pure teacher forcing.
    rollout_aware_p: float = 0.0
    rollout_k: int = 4
    rollout_aware_warmup: int = 0
    lambda_rollout: float = 1.0

    def __post_init__(self):
        self.latent_scales = parse_scales(self.latent_scales)
        if len(self.flow_loss_weights) != len(self.latent_scales):
            self.flow_loss_weights = tuple(1.0 for _ in self.latent_scales)

    def to_dict(self) -> Dict:
        return {
            'context_length': self.context_length,
            'latent_scales': list(self.latent_scales),
            'flow_loss_weights': list(self.flow_loss_weights),
            'planner': dict(self.planner),
            'renderer': dict(self.renderer),
            'lambda_noop': self.lambda_noop,
            'lambda_cf': self.lambda_cf,
            'cf_margin': self.cf_margin,
            'action_conditioned': self.action_conditioned,
            'rollout_aware_p': self.rollout_aware_p,
            'rollout_k': self.rollout_k,
            'rollout_aware_warmup': self.rollout_aware_warmup,
            'lambda_rollout': self.lambda_rollout,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'SampoPlusConfig':
        return cls(**data)
