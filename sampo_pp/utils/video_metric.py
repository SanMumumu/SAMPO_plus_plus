"""Compatibility shim: video metrics (FVD / frame quality) for ``sampo_pp``.

The FVD/I3D evaluator is NOT part of the refactored ``sampo_pp`` contributions;
world-model-native metrics live in ``sampo_pp.metrics``. This module forwards to
the original evaluator so ``train_var.py`` can import uniformly from
``sampo_pp``. Replace with the real module if you merge the core package.
"""
try:
    from SAMPO.utils.video_metric import Evaluator, FeatureStats  # noqa: F401
except ImportError as exc:  # pragma: no cover - depends on the core package
    raise ImportError(
        'sampo_pp.utils.video_metric forwards to the original FVD/frame-quality '
        'evaluator, which is not available in this checkout. Provide the real '
        'module before enabling --use_fvd / --use_frame_metrics.'
    ) from exc
