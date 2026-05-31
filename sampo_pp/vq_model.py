"""Compatibility shim: discrete VQ tokenizer for the legacy SAMPO path.

The discrete VQ model (``CompressiveVQModel``, ``LPIPS``) belongs to the legacy
discrete SAMPO path used by ``vp/`` and ``mbrl/``. It is intentionally NOT part
of the continuous ``sampo_pp`` core (which must never modify the tokenizer), and
is forwarded here so those downstream modules import uniformly from ``sampo_pp``.
"""
try:
    from SAMPO.vq_model import CompressiveVQModel, LPIPS  # noqa: F401
except ImportError as exc:  # pragma: no cover - depends on the core package
    raise ImportError(
        'sampo_pp.vq_model forwards to the legacy discrete VQ tokenizer, which '
        'is not available in this checkout. Provide the real vq_model module to '
        'use the discrete SAMPO path (vp/ , mbrl/).'
    ) from exc
