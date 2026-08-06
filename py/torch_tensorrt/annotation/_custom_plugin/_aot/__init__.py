"""AOT kernel backend implementations for the TTA custom plugin system.

Each sub-module implements the ``aot_impl_<backend>`` function that drives a
backend-specific compile pipeline and returns the QDP AOT 4-tuple::

    (kernel_name, code_bytes, KernelLaunchParams, SymIntExprs)

Backends
--------
_triton   — Triton JIT kernels compiled to PTX via triton.compile()
_cutile   — cuTILE programs compiled to CUBIN via cuda.tile compile_tile()
_cutedsl  — CuTe DSL @cute.jit kernels compiled via cutlass.cute.compile()

All three backends share ``_qdp_utils`` helpers for sandboxing, launch-arg
analysis, artifact dumping, and the unified ``AOTMetadata`` result type, plus
the PTX-text helpers below for cross-backend post-processing.
"""

import re
from typing import Tuple

_PTX_VERSION_RE = re.compile(r"\.version\s+(\d+)\.(\d+)")
PTX_MAX_SUPPORTED_VERSION: Tuple[int, int] = (9, 0)


def downgrade_ptx_version(
    ptx: str,
    max_version: Tuple[int, int] = PTX_MAX_SUPPORTED_VERSION,
) -> str:
    """Lower the PTX ``.version`` directive to ``max_version`` when it exceeds it.

    Triton on CUDA 13.x and cuda-tile's tileiras both emit ``.version 9.1`` but
    the TRT QDP PTX loader (and CUDA 13.0 drivers) only accept up to ``9.0``.
    Patching the header is safe as long as the body doesn't use ISA features
    introduced after ``max_version`` — neither backend currently does.

    Args:
        ptx:          PTX source as a string.
        max_version:  ``(major, minor)`` ceiling; defaults to ``(9, 0)``.

    Returns:
        The PTX string with the ``.version`` line capped at ``max_version``,
        or the original string when no change is needed.
    """
    m = _PTX_VERSION_RE.search(ptx)
    if not m:
        return ptx
    major, minor = int(m.group(1)), int(m.group(2))
    if (major, minor) <= max_version:
        return ptx
    replacement = f".version {max_version[0]}.{max_version[1]}"
    return ptx[: m.start()] + replacement + ptx[m.end() :]
