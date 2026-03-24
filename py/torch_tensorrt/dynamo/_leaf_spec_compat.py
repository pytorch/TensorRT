"""
Compatibility shim for a PyTorch 2.11 bug where ``LeafSpec`` (frozen dataclass
with ``slots=True``) inherits the ``type`` slot from ``TreeSpec`` but never
initialises it, leaving the slot empty.  This causes

    AttributeError: 'LeafSpec' object has no attribute 'type'

inside ``ExportedProgram.run_decompositions()`` when a model returns a single
tensor (i.e. the output pytree spec is a leaf rather than a list/tuple).

The fix is applied once at import time and is a no-op on versions that already
set the attribute correctly.

Upstream fix: https://github.com/pytorch/pytorch/issues/<TBD>
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def _apply_leaf_spec_patch() -> None:
    """Patch ``LeafSpec`` so its inherited ``type`` slot is always set to ``None``.

    Safe to call multiple times; the patch is idempotent.
    """
    try:
        from torch.utils._pytree import _LEAF_SPEC, LeafSpec
    except ImportError:
        return  # too old / too new, nothing to do

    # Check whether the bug is present on the singleton instance
    try:
        _ = _LEAF_SPEC.type  # noqa: F841
        return  # attribute accessible — no patch needed
    except AttributeError:
        pass

    logger.debug(
        "torch_tensorrt: applying LeafSpec.type compatibility patch "
        "(PyTorch bug: frozen-dataclass slot not initialised in subclass)"
    )

    # Fix the pre-existing singleton that all pytree leaf specs share
    object.__setattr__(_LEAF_SPEC, "type", None)
    object.__setattr__(_LEAF_SPEC, "_context", None)
    object.__setattr__(_LEAF_SPEC, "_children", [])

    # Patch __post_init__ so any new LeafSpec() instances are also fixed
    _orig_post_init = LeafSpec.__post_init__

    def _post_init_with_type(self: LeafSpec) -> None:
        _orig_post_init(self)
        object.__setattr__(self, "type", None)
        object.__setattr__(self, "_context", None)
        object.__setattr__(self, "_children", [])

    LeafSpec.__post_init__ = _post_init_with_type
