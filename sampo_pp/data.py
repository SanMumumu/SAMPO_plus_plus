"""Compatibility shim: dataset loaders for ``sampo_pp``.

The OXE / robotic data pipeline is NOT part of the refactored ``sampo_pp``
contributions (multi-scale planner / ACVF / PCD-RoPE). It forwards to the
original core implementation so entry scripts can import uniformly from
``sampo_pp``.

If you merge the core package into ``sampo_pp`` on the training server, replace
this file with the real ``data.py``; the names re-exported below must remain
importable.
"""
try:
    from SAMPO.data import DATASET_NAMED_MIXES, SimpleRoboticDataLoaderv2  # noqa: F401
except ImportError as exc:  # pragma: no cover - depends on the core package
    raise ImportError(
        'sampo_pp.data forwards to the original data loaders, which are not '
        'available in this checkout. Provide the real data module (e.g. merge '
        'the core package into sampo_pp) before running training/inference.'
    ) from exc
