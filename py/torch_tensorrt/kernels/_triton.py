from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

_LOGGER = logging.getLogger(__name__)

# PTX ISA versions to probe, highest first. The driver accepts a version iff it
# is <= the toolchain the driver ships with; we cap emitted PTX to the highest
# version the running driver actually loads.
_PTX_VERSION_CANDIDATES: Tuple[Tuple[int, int], ...] = (
    (9, 5),
    (9, 4),
    (9, 3),
    (9, 2),
    (9, 1),
    (9, 0),
    (8, 8),
    (8, 7),
    (8, 5),
)


def _ptx_version_to_int(major: int, minor: int) -> int:
    """``(9, 1) -> 91`` — the integer form Triton's ``ptx_version`` option uses."""
    return major * 10 + minor


_driver_max_ptx: Optional[Tuple[int, int]] = None
_driver_max_ptx_probed = False


def _parse_entry_params(
    ptx: str, kernel_name: str
) -> Tuple[Optional[re.Match[str]], List[str]]:
    """Locate the ``.visible .entry <kernel_name>( ... )`` param list.

    Returns the regex match (groups: prefix, param-block, close-paren) and the
    list of individual ``.param`` declaration strings, in order.
    """
    pattern = re.compile(
        r"(\.visible\s+\.entry\s+" + re.escape(kernel_name) + r"\s*\()([^)]*)(\))",
        re.DOTALL,
    )
    match = pattern.search(ptx)
    if match is None:
        return match, []
    params = [p.strip() for p in match.group(2).split(",") if p.strip()]
    return match, params


def _strip_trailing_scratch_params(ptx: str, kernel_name: str, keep: int) -> str:
    """Drop Triton's trailing scratch params so the entry matches TRT's launch.

    Triton (3.x) unconditionally appends ``global_scratch`` and
    ``profile_scratch`` pointer parameters after the user's kernel arguments.
    TensorRT's AOT QDP launcher only passes the declared tensor + extra-scalar
    arguments, so the extra params make the kernel's parameter count exceed what
    TRT supplies — the launch is misaligned and fails at ``onShapeChange``.

    When those scratch buffers are zero-sized (the common case) the params are
    unused in the kernel body, so removing them from the ``.entry`` signature is
    a safe, purely-syntactic fix. ``keep`` is the number of real runtime
    parameters (i.e. ``len(signature)``).
    """
    match, params = _parse_entry_params(ptx, kernel_name)
    if match is None or len(params) <= keep:
        return ptx
    kept = params[:keep]
    new_block = "\n\t" + ",\n\t".join(kept) + "\n"
    return (
        ptx[: match.start()]
        + match.group(1)
        + new_block
        + match.group(3)
        + ptx[match.end() :]
    )


def _driver_max_ptx_version() -> Optional[Tuple[int, int]]:
    """Probe the highest PTX ``.version`` the running CUDA driver will load.

    A driver older than the CUDA toolkit that produced the PTX rejects the
    newer ISA with ``CUDA_ERROR_UNSUPPORTED_PTX_VERSION``. We load a trivial
    module at descending ISA versions and return the first the driver accepts.
    The result is memoized. Returns ``None`` if probing is unavailable, in which
    case no capping is applied.
    """
    global _driver_max_ptx, _driver_max_ptx_probed
    if _driver_max_ptx_probed:
        return _driver_max_ptx
    _driver_max_ptx_probed = True
    try:
        from cuda.bindings import driver as cuda

        cuda.cuInit(0)
        dev = cuda.cuDeviceGet(0)[1]
        ctx = cuda.cuDevicePrimaryCtxRetain(dev)[1]
        cuda.cuCtxSetCurrent(ctx)
        probe = (
            "//\n.version {maj}.{minr}\n.target sm_90\n.address_size 64\n"
            ".visible .entry _ttk_probe(){{ret;}}\n"
        )
        for maj, minr in _PTX_VERSION_CANDIDATES:
            res = cuda.cuModuleLoadData(probe.format(maj=maj, minr=minr).encode())
            if int(res[0]) == 0:
                try:
                    cuda.cuModuleUnload(res[1])
                except Exception:
                    pass
                _driver_max_ptx = (maj, minr)
                break
    except Exception as exc:  # pragma: no cover - environment dependent
        _LOGGER.debug("Could not probe driver PTX version: %s", exc)
        _driver_max_ptx = None
    return _driver_max_ptx


def _parse_ptx_version(ptx: str) -> Optional[Tuple[int, int]]:
    """Read the ``.version <major>.<minor>`` ISA header from PTX text."""
    match = re.search(r"\.version (\d+)\.(\d+)", ptx)
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


def _triton_import() -> Any:
    """Import triton, raising an actionable error if it is not installed."""
    try:
        import triton
        import triton.compiler  # noqa: F401  (ensures ASTSource is importable)

        return triton
    except ImportError:
        raise ImportError(
            "triton is required for triton_op plugins. "
            "Install it with: pip install triton"
        )


def compile_triton_to_ptx(
    kernel: Any,
    signature: Dict[str, str],
    constexprs: Dict[str, Any],
) -> Tuple[bytes, str, int, int]:
    """Compile a ``@triton.jit`` kernel to PTX ahead of time.

    This mirrors what ``examples/dynamo/aot_plugin.py`` does by hand inside its
    ``@trtp.aot_impl`` body, but performs it once so callers don't have to.

    Args:
        kernel: the ``@triton.jit`` kernel function.
        signature: Triton signature mapping *non-constexpr* parameter names to
            Triton type strings, in kernel-declaration order — e.g.
            ``{"x_ptr": "*fp32", "n_elements": "i32", "y_ptr": "*fp32"}``.
            Pointer args use ``*<dtype>``; scalar args use the bare dtype.
        constexprs: compile-time ``tl.constexpr`` values baked into the PTX,
            e.g. ``{"BLOCK_SIZE": 256}``. These must NOT appear in ``signature``.

    Returns:
        ``(ptx_bytes, kernel_name, num_warps, shared_mem_bytes)`` — the PTX to
        embed in the TRT engine, the entry symbol inside it, and the launch
        metadata needed to build ``trtp.KernelLaunchParams``.
    """
    triton = _triton_import()

    def _compile(ptx_version: Optional[int] = None) -> Any:
        src = triton.compiler.ASTSource(
            fn=kernel,
            signature=dict(signature),
            constexprs=dict(constexprs),
        )
        options = {"ptx_version": ptx_version} if ptx_version is not None else None
        return triton.compile(src, options=options)

    compiled = _compile()
    ptx_text: str = compiled.asm["ptx"]

    # Cap the emitted PTX ISA to what the running driver supports. Triton derives
    # the ISA from its bundled ptxas, which can be newer than the installed
    # driver; the driver then rejects the PTX at load time with
    # CUDA_ERROR_UNSUPPORTED_PTX_VERSION (for AOT plugins this surfaces as a TRT
    # ``onShapeChange`` failure at engine runtime). Ask Triton to emit a lower
    # ISA directly via ``ptx_version`` rather than rewriting the PTX text. Only
    # lower, never raise, so we never exceed what the toolchain can assemble.
    driver_max = _driver_max_ptx_version()
    emitted = _parse_ptx_version(ptx_text)
    if driver_max is not None and emitted is not None and emitted > driver_max:
        target = _ptx_version_to_int(*driver_max)
        _LOGGER.debug(
            "PTX ISA %d.%d exceeds driver max %d.%d; recompiling '%s' with "
            "ptx_version=%d",
            emitted[0],
            emitted[1],
            driver_max[0],
            driver_max[1],
            kernel.__name__ if hasattr(kernel, "__name__") else "<kernel>",
            target,
        )
        compiled = _compile(ptx_version=target)
        ptx_text = compiled.asm["ptx"]

    kernel_name = compiled.metadata.name
    num_warps = compiled.metadata.num_warps
    shared_mem = compiled.metadata.shared

    # Triton appends trailing global/profile scratch pointer params that TRT's
    # AOT launcher does not supply. If they are zero-sized we strip them so the
    # kernel's parameter list matches exactly what TRT passes; otherwise the
    # kernel genuinely needs scratch the AOT QDP path cannot provide.
    global_scratch = int(getattr(compiled.metadata, "global_scratch_size", 0) or 0)
    profile_scratch = int(getattr(compiled.metadata, "profile_scratch_size", 0) or 0)
    _, params = _parse_entry_params(ptx_text, kernel_name)
    num_runtime_params = len(signature)
    if len(params) > num_runtime_params:
        if global_scratch != 0 or profile_scratch != 0:
            raise RuntimeError(
                f"Triton kernel '{kernel_name}' requires non-zero scratch memory "
                f"(global={global_scratch}, profile={profile_scratch}), which the "
                "TensorRT AOT QDP launch path cannot provide. Rewrite the kernel "
                "to avoid Triton scratch buffers to use triton_op."
            )
        ptx_text = _strip_trailing_scratch_params(
            ptx_text, kernel_name, num_runtime_params
        )
        _LOGGER.debug(
            "Stripped %d trailing scratch param(s) from triton kernel '%s'",
            len(params) - num_runtime_params,
            kernel_name,
        )

    ptx_bytes = ptx_text.encode("utf-8")

    _LOGGER.debug(
        "Compiled triton kernel '%s' -> PTX (%d bytes, num_warps=%d, shared=%d)",
        kernel_name,
        len(ptx_bytes),
        num_warps,
        shared_mem,
    )
    return ptx_bytes, kernel_name, num_warps, shared_mem
