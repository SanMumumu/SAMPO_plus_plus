"""SAMPO++ continuous-latent world model (``sampo_pp``).

A self-contained refactor of the SAMPO++ core into a flag-driven package built
around three contributions, each gated so the legacy behaviour and every
ablation corner are a one-flag change:

  * multi-scale temporal planner  (planner.multi_scale)
  * Action-Controlled Velocity Field renderer  (renderer.action_mode='acvf')
  * Pyramid-Consistent Dynamic RoPE  (renderer.rope_mode='pcd')

Public API mirrors the interfaces expected by ``train_var.py``,
``inference/predict.py`` and the test-suite shape contracts.
"""
from .config import (
    AdapterConfig,
    FlowRendererConfig,
    SampoPlusConfig,
    TemporalPlannerConfig,
    parse_scales,
)
from .adapter import ContinuousLatentAdapter
from .planner import TemporalPlanner
from .renderer import ScaleAwareFlowRenderer, SILoss
from .acvf import ACVFBlock, counterfactual_velocity_ranking, noop_consistency_loss, scale_action_energy
from .pcd_rope import PCDRoPE, SpatialRoPE, build_spatial_rope
from .model import SampoPlusModel, build_sampo_plus

__all__ = [
    'AdapterConfig',
    'TemporalPlannerConfig',
    'FlowRendererConfig',
    'SampoPlusConfig',
    'parse_scales',
    'ContinuousLatentAdapter',
    'TemporalPlanner',
    'ScaleAwareFlowRenderer',
    'SILoss',
    'ACVFBlock',
    'noop_consistency_loss',
    'counterfactual_velocity_ranking',
    'scale_action_energy',
    'SpatialRoPE',
    'PCDRoPE',
    'build_spatial_rope',
    'SampoPlusModel',
    'build_sampo_plus',
]
