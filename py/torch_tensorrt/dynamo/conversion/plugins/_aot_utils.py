"""Helpers for writing AOT QDP plugins backed by Triton kernels."""

from typing import Any, Sequence


def _has_triton_scratch_params(compiled_kernel: Any) -> bool:
    md = getattr(compiled_kernel, "metadata", None)
    if md is None:
        return False
    return hasattr(md, "global_scratch_size") and hasattr(md, "profile_scratch_size")


def make_aot_extra_args(
    user_args: Sequence[Any],
    *,
    compiled_kernel: Any = None,
) -> Any:
    """Build a ``trtp.SymIntExprs`` for an AOT plugin's ``extra_args`` return.

    When ``compiled_kernel`` is a Triton-compiled kernel, four trailing
    ``SymInt32(0)`` are appended to cover the two ``.param .u64 .ptr`` slots
    (``global_scratch``, ``profile_scratch``) that Triton >= 3.x always emits
    in PTX even when their sizes are zero. TRT's AOT plugin path does not
    plumb those slots through, so without padding ``enqueueV3`` reads stale
    register state for them and segfaults on the first call.
    """
    import tensorrt.plugin as trtp

    pad = 4 if _has_triton_scratch_params(compiled_kernel) else 0
    total = len(user_args) + pad
    out = trtp.SymIntExprs(total)
    for i, arg in enumerate(user_args):
        out[i] = arg
    zero = trtp.SymInt32(0)
    for i in range(len(user_args), total):
        out[i] = zero
    return out
