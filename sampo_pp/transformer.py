"""Compatibility shim: legacy discrete-path transformer head for ``sampo_pp``.

``HeadModelWithAction`` is the action-conditioned head of the legacy discrete
SAMPO path (used by ``vp/`` and ``mbrl/``). The continuous ``sampo_pp`` core
implements its own attention / positional encoding (see ``sampo_pp.pcd_rope``
and ``sampo_pp.renderer``); this module only forwards the legacy head so the
downstream discrete modules import uniformly from ``sampo_pp``.
"""
try:
    from SAMPO.transformer import HeadModelWithAction  # noqa: F401
except ImportError as exc:  # pragma: no cover - depends on the core package
    raise ImportError(
        'sampo_pp.transformer forwards to the legacy discrete-path transformer '
        'head, which is not available in this checkout. Provide the real module '
        'to use the discrete SAMPO path (vp/ , mbrl/).'
    ) from exc
