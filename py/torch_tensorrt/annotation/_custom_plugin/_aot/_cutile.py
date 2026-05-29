"""cuTILE AOT backend: source → sandbox recording → CUBIN/PTX → KernelLaunchParams.

AOT compilation pipeline for cuTILE kernels
============================================
1. **Sandbox execution** — A patched ``cuda.tile.launch`` shim is injected alongside
   ``CuTileLaunchRecorder`` proxies for each cuTILE program object found in the
   user's module globals.  ``launch_fn`` is called with ``SymbolicTensor`` arguments
   so that the recorded grid and tensor bindings are captured symbolically.

2. **Argument analysis** — ``analyze_launch_args`` splits recorded call arguments into
   *pointer binding indices* and *scalar SymInt32 expressions* that TRT evaluates at
   runtime.

3. **CUBIN compilation** — ``compile_tile(pyfunc, example_tensors, CompilerOptions())``
   from ``cuda.tile._compile`` produces a CUBIN file containing an embedded PTX
   debug section (``.nv_debug_ptx_txt``).

4. **PTX extraction and parameter reordering** — ``_extract_ptx_from_cubin`` recovers
   the PTX from the CUBIN.  The cuTILE kernel ABI uses per-tensor groups of
   ``(ptr, extents..., strides...)`` parameters in manual (kernel-source) order.
   The QDP runtime passes ``(input_ptrs, extra_args, output_ptrs)`` in a different
   order.  ``_reorder_cutile_ptx_for_trt`` reorders the ``.param`` declarations to
   match the runtime expectation and downgrades the ``.version`` directive if the
   CUDA driver is older than the PTX version emitted by ``tileiras``.

5. **Tactic uniquification** — Identical to the Triton backend: a config-derived
   suffix is appended to the kernel name so TRT registers separate PTX/cubin entries
   per tactic.

The public entry-point is ``aot_impl_cutile``, which returns the QDP AOT 4-tuple
``(kernel_name_bytes, code_bytes, KernelLaunchParams, SymIntExprs)``.
``compile_cutile_program`` is a thin tactic-manager wrapper.
"""
from __future__ import annotations

import logging
import os
import re
import tempfile
from typing import Any, Dict, List, Mapping, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    import tensorrt as trt
    import tensorrt.plugin as trtp
except ImportError as e:
    raise ImportError(
        "TensorRT with plugin support is required for cuTILE AOT compilation."
    ) from e


def _trt_dtype_to_ct(td_dtype: Any) -> Any:
    """Map a ``trt.DataType`` value to the corresponding ``cuda.tile.DType``.

    Mirrors :func:`_qdp_utils.td_dtype_to_torch`; the supported set is the same
    one cuTile / TRT QDP can handle.
    """
    import cuda.tile as ct  # type: ignore[import]

    if td_dtype == trt.float32:
        return ct.float32
    if td_dtype == trt.float16:
        return ct.float16
    if td_dtype == trt.bfloat16:
        return ct.bfloat16
    if td_dtype == trt.int32:
        return ct.int32
    raise RuntimeError(
        f"unsupported TRT dtype {td_dtype!r} for cuTile;"
        " supported: float32, float16, bfloat16, int32"
    )

from .._qdp_utils import (
    AOTMetadata,
    TTAPluginError,
    _as_symint32,
    _assign_recorded_grid,
    _launch_params_from_trt,
    _safe_dim,
    _shape_expr_to_ints,
    _sym_dim,
    analyze_launch_args,
    dump_code_artifact,
    is_cutile_program,
    run_kernel_sandbox,
    td_dtype_to_torch,
)
from ..._recorders import CuTileLaunchRecorder
from ..._specs import CuTileSpec
from .._symbolic import SymbolicTensor, TensorRole


def _params_per_tensor_for_rank(rank: int) -> int:
    """CuTile kernel ABI: ptr + extent per dim + stride per dim = 1 + 2*rank."""
    return 1 + 2 * max(1, rank)


def _cutile_trt_order_for_io(
    num_inputs: int,
    num_outputs: int,
    num_scalars: int = 1,
    rank: int = 1,
) -> Tuple[int, Tuple[int, ...]]:
    """Compute (num_params, runtime_order) for CuTile kernel.

    Per-tensor params: ptr (1) + extent per dim (rank) + stride per dim (rank) = 1 + 2*rank.
    Manual (kernel) order: for each input [ptr, ext..., str...]; for each output [ptr, ext..., str...]; scalars.
    QDP runtime passes: input_ptrs, extra_args (extent/stride + scalars), output_ptrs.
    Returns permutation so that reordered[physical_idx] = manual_param_index.
    """
    n_in = num_inputs
    n_out = num_outputs
    ppt = _params_per_tensor_for_rank(rank)
    num_params = (n_in + n_out) * ppt + num_scalars
    manual_ptr_in = [i * ppt for i in range(n_in)]
    manual_ptr_out = [n_in * ppt + j * ppt for j in range(n_out)]
    extent_stride_indices = list(range(1, ppt))
    manual_extent_stride_in = [
        i * ppt + k for i in range(n_in) for k in extent_stride_indices
    ]
    manual_extent_stride_out = [
        n_in * ppt + j * ppt + k
        for j in range(n_out)
        for k in extent_stride_indices
    ]
    manual_scalars = [(n_in + n_out) * ppt + s for s in range(num_scalars)]
    trt_order = (
        list(manual_ptr_in)
        + manual_extent_stride_in
        + manual_extent_stride_out
        + manual_scalars
        + list(manual_ptr_out)
    )
    return num_params, tuple(trt_order)


_ELF_MIN_SIZE = 64

_PTX_ENTRY_PARAM_RE = re.compile(
    r"(\.visible\s+\.entry\s+(\w+)\s*\()([^)]*)\)",
    re.DOTALL,
)

_PTX_VERSION_RE = re.compile(r"\.version\s+(\d+)\.(\d+)")
_PTX_REQNTID_RE = re.compile(r"\.reqntid\s+(\d+)")
_PTX_MAX_SUPPORTED_VERSION = (9, 0)


def _extract_ptx_from_cubin(cubin_bytes: bytes) -> Optional[str]:
    """Extract embedded PTX from CUBIN (.nv_debug_ptx_txt section).

    The CuTile compiler embeds PTX as null-separated strings in this debug section.
    We scan for '.version' to find the start, then track brace depth to find the
    matching closing '}' of the kernel function body.  A simple find(b'}') is wrong
    because PTX kernels that use vector-register syntax (e.g. mov.b64 {%r1, %r2})
    contain inline '}' characters before the actual function-closing '}'.
    Returns formatted PTX string or None.
    """
    if len(cubin_bytes) < _ELF_MIN_SIZE or cubin_bytes[:4] != b"\x7fELF":
        return None
    idx = cubin_bytes.find(b".version")
    if idx < 0:
        return None
    # Find the opening '{' of the kernel function body (the first '{' after .version).
    open_brace = cubin_bytes.find(b"{", idx)
    if open_brace < 0:
        return None
    # Track brace depth to find the matching closing '}', handling inline '{...}'
    # pairs inside instructions (e.g. mov.b64 {%r1, %r2}, %rd0).
    depth = 0
    end = -1
    scan_limit = min(open_brace + 500_000, len(cubin_bytes))
    for i in range(open_brace, scan_limit):
        c = cubin_bytes[i]
        if c == ord("{"):
            depth += 1
        elif c == ord("}"):
            depth -= 1
            if depth == 0:
                end = i
                break
    if end < 0:
        return None
    raw = cubin_bytes[idx : end + 1]
    ptx = raw.replace(b"\x00", b"\n").decode("utf-8", errors="replace")
    lines = [l for l in ptx.splitlines() if l.strip()]
    return "\n".join(lines) + "\n"


# LIMITATION (short-term workaround): PTX version downgrade is a best-effort
# text substitution.  CUDA 13.1 tileiras emits .version 9.1 but the driver in
# our container (CUDA 13.0) only JIT-compiles up to PTX 9.0.  Lowering the
# declared version by editing the .version line is safe only if the PTX body
# uses no ISA features introduced after the target version.  This will silently
# produce incorrect PTX if a future tileiras version emits instructions that
# require a higher PTX version than the declared one after downgrade.
# Long-term fix: align the driver and CUDA toolkit versions so no downgrade is
# needed.
def _downgrade_ptx_version(ptx: str) -> str:
    """Downgrade .version directive if it exceeds what the runtime driver supports.

    CUDA 13.1 tileiras emits .version 9.1 but a CUDA 13.0 driver only JIT-compiles
    up to 9.0.  Lowering the declared version is safe as long as the PTX body doesn't
    use features introduced after the target version (CuTile-generated PTX doesn't).
    """
    m = _PTX_VERSION_RE.search(ptx)
    if not m:
        return ptx
    major, minor = int(m.group(1)), int(m.group(2))
    max_major, max_minor = _PTX_MAX_SUPPORTED_VERSION
    if (major, minor) <= (max_major, max_minor):
        return ptx
    replacement = f".version {max_major}.{max_minor}"
    return ptx[: m.start()] + replacement + ptx[m.end() :]


def _parse_reqntid(ptx: str) -> Optional[int]:
    """Extract the .reqntid value (required threads per CTA) from PTX.

    CuTile may vectorise (e.g. f32x2) and use fewer threads than the tile size.
    The kernel MUST be launched with exactly this many threads.
    """
    m = _PTX_REQNTID_RE.search(ptx)
    return int(m.group(1)) if m else None


def _count_ptx_params(ptx: str) -> int:
    """Count the number of .param arguments in the PTX .entry declaration."""
    m = _PTX_ENTRY_PARAM_RE.search(ptx)
    if not m:
        return 0
    param_block = m.group(3)
    params_raw = [p.strip().rstrip(",") for p in param_block.split(",") if p.strip()]
    return len(params_raw)


def _reorder_ptx_params(ptx: str, runtime_order: Tuple[int, ...]) -> str:
    """Reorder .param declarations in PTX entry to match runtime_order permutation.

    runtime_order[i] = manual_param_index: physical slot i holds manual param runtime_order[i].
    So the new declaration order is: for each i in 0..n-1, place the param that was originally at index runtime_order[i].
    """
    m = _PTX_ENTRY_PARAM_RE.search(ptx)
    if not m:
        return ptx
    prefix = m.group(1)
    param_block = m.group(3)
    params_raw = [p.strip().rstrip(",") for p in param_block.split(",") if p.strip()]
    if len(params_raw) != len(runtime_order):
        return ptx
    reordered = [params_raw[runtime_order[i]] for i in range(len(runtime_order))]
    new_param_block = ",\n    ".join(reordered)
    new_entry = prefix + "\n    " + new_param_block + "\n)"
    return ptx[: m.start()] + new_entry + ptx[m.end() :]


# LIMITATION (fragile PTX rewrite): _reorder_cutile_ptx_for_trt extracts PTX
# embedded in the CuTile CUBIN via a regex on the raw ELF bytes, then rewrites
# .param declarations to match the TRT plugin runtime argument order (inputs,
# scalars, outputs).  This is brittle for several reasons:
#   1. The PTX extraction scans raw ELF bytes for the PTX section header — it
#      can silently return wrong bytes if the CUBIN internal layout changes.
#   2. The .param reorder uses a regex on the textual PTX entry signature; any
#      change to how tileiras serialises the .param block (whitespace, line
#      breaks, inline comments) will cause the reorder to silently no-op and
#      return the original CUBIN, producing incorrect kernel argument binding.
#   3. If tileiras starts emitting multiple .entry kernels per CUBIN, the regex
#      matches only the first one.
# The proper fix is for tileiras / TRT CuTile integration to agree on a
# canonical argument order without requiring a post-processing rewrite.
def _reorder_cutile_ptx_for_trt(
    cubin_bytes: bytes,
    kernel_name: str,
    num_inputs: int = 1,
    num_outputs: int = 1,
    num_scalars: int = 1,
    rank: int = 1,
) -> bytes:
    """Extract PTX from CuTile CUBIN and reorder .param declarations to CuTile plugin runtime order.

    Returns reordered PTX as bytes (accepted by runtime and JIT-compiled to the
    right arch), or original cubin_bytes unchanged on failure.
    """
    if len(cubin_bytes) < _ELF_MIN_SIZE or cubin_bytes[:4] != b"\x7fELF":
        return cubin_bytes
    ptx = _extract_ptx_from_cubin(cubin_bytes)
    if not ptx:
        return cubin_bytes
    expected_n, runtime_order = _cutile_trt_order_for_io(
        num_inputs, num_outputs, num_scalars, rank=rank
    )
    if expected_n < 1:
        return cubin_bytes
    reordered_ptx = _reorder_ptx_params(ptx, runtime_order)
    if reordered_ptx == ptx:
        return cubin_bytes
    reordered_ptx = _downgrade_ptx_version(reordered_ptx)
    return reordered_ptx.encode("utf-8")


def _infer_cutile_extra_args(
    inp_descs: List[Any],
    out_descs: List[Any],
    block_size: int,
    scalar_symints: Optional[List[Any]] = None,
    rank: int = 1,
) -> Any:
    """Build SymIntExprs: (extent per dim, stride per dim) per tensor then scalar(s).

    Uses symbolic SymInt32 expressions for dynamic dimensions so that TRT can
    evaluate them at runtime with actual input shapes.
    """
    num_scalars = 1
    if scalar_symints:
        num_scalars = len(scalar_symints)
    n_tensors = len(inp_descs) + len(out_descs)
    r = max(1, rank)
    n = n_tensors * 2 * r + num_scalars
    extra_args = trtp.SymIntExprs(n)
    idx = 0
    for td in list(inp_descs) + list(out_descs):
        try:
            shape = list(td.shape_expr)  # list of int or SymInt32
            if r == 1:
                # numel = symbolic product of all dims; wrap result back to SymInt32
                numel: Any = trtp.SymInt32(1)
                for d in shape:
                    numel = numel * _sym_dim(d)
                extra_args[idx] = _as_symint32(numel)
                extra_args[idx + 1] = trtp.SymInt32(1)
            else:
                use_shape = shape[:r] if len(shape) >= r else (shape + [1] * (r - len(shape)))
                # extents (symbolic)
                for i in range(r):
                    d = use_shape[i] if i < len(use_shape) else 1
                    extra_args[idx + i] = _as_symint32(_sym_dim(d))
                # row-major strides (symbolic products)
                for i in range(r):
                    s: Any = trtp.SymInt32(1)
                    for j in range(i + 1, len(use_shape)):
                        s = s * _sym_dim(use_shape[j] if j < len(use_shape) else 1)
                    extra_args[idx + r + i] = _as_symint32(s)
        except (AttributeError, TypeError, RuntimeError):
            # Broad shape_expr access may fail for mock/stub TensorDescs used in
            # tests; fall back to SymInt32(1) for all extent/stride slots.
            for i in range(2 * r):
                extra_args[idx + i] = trtp.SymInt32(1)
        idx += 2 * r
    for i in range(num_scalars):
        if scalar_symints and i < len(scalar_symints):
            extra_args[idx + i] = _as_symint32(scalar_symints[i])
        else:
            extra_args[idx + i] = trtp.SymInt32(block_size)
    return extra_args


def aot_impl_cutile(
    *,
    qdp_symbol: str,
    spec: CuTileSpec,
    cfg: Mapping[str, Any],
    launch_fn: Any,
    host_args: List[SymbolicTensor],
    inp_descs: List[Any],
    out_descs: List[Any],
    attrs: Optional[Dict[str, Any]] = None,
) -> Tuple[bytes, bytes, Any, Any]:
    """cuTILE AOT implementation: sandbox → record → compile → CUBIN.

    Steps:
    1. Find cuTILE program objects in launch_fn's module.
    2. Replace with CuTileLaunchRecorder proxies.
    3. Run sandboxed launch_fn(*host_args, **merged_kwargs).
    4. Analyse recorded args → param_binding_indices + scalar SymInts.
    5. Compile to CUBIN using cuTILE internal APIs.
    6. Return (kernel_name, cubin, KernelLaunchParams, SymIntExprs).

    Returns:
        (kernel_name_bytes, cubin_bytes, KernelLaunchParams, SymIntExprs)
    """
    import torch

    backend = "cutile"

    try:
        import cuda.tile  # type: ignore[import]  # noqa: F401
    except ImportError as exc:
        raise TTAPluginError(
            op=qdp_symbol,
            stage="aot_impl",
            backend=backend,
            msg=f"cuTILE module (cuda.tile) not available for op '{qdp_symbol}': {exc}",
        ) from exc

    # 1-2. Locate module, find cuTILE programs, sandbox and run.
    #      Build a patched ct.launch so that ct.launch(stream, grid, prog, args)
    #      correctly sets recorder.grid on a CuTileLaunchRecorder proxy.
    #      strict=True: raise on module-not-found and no-program-found.
    import cuda.tile as _ct  # type: ignore[import]
    import types as _types

    # Sentinel list: closure trick to capture the recorder after creation.
    _recorder_ref: List[Any] = []

    def _sandbox_launch(stream, grid, kernel, kernel_args):
        if isinstance(kernel, CuTileLaunchRecorder):
            kernel.grid = grid
            kernel(*kernel_args)
        else:
            _ct.launch(stream, grid, kernel, kernel_args)

    _ct_sandbox = _types.ModuleType("cuda.tile.sandbox")
    for _k, _v in vars(_ct).items():
        setattr(_ct_sandbox, _k, _v)
    _ct_sandbox.launch = _sandbox_launch

    merged_kwargs = dict(cfg)
    if attrs:
        merged_kwargs.update(attrs)

    # Pass the patched ct module as extra_override; harmless if module doesn't use it.
    # Broad catch is necessary: the sandbox executes arbitrary user launch_fn code
    # and can raise any exception type (ImportError for missing deps, TypeError
    # for shape mismatches, AttributeError from proxy gaps, etc.).
    try:
        used_recorder, prog_recorders = run_kernel_sandbox(
            launch_fn=launch_fn,
            host_args=host_args,
            is_kernel_fn=is_cutile_program,
            recorder_factory=lambda obj: CuTileLaunchRecorder(real_prog=obj),
            host_kwargs=merged_kwargs,
            extra_overrides={"ct": _ct_sandbox},
            strict=True,
            op=qdp_symbol,
            backend=backend,
        )
    except TTAPluginError:
        raise
    except Exception as exc:
        raise TTAPluginError(
            op=qdp_symbol,
            stage="aot_impl",
            backend=backend,
            msg=f"sandbox failed for op '{qdp_symbol}': {exc}",
        ) from exc

    # 3. Exactly one cuTILE program must have been called.
    used = [rec for rec in prog_recorders.values() if rec.args is not None]
    if len(used) != 1:
        raise TTAPluginError(
            op=qdp_symbol,
            stage="aot_impl",
            backend=backend,
            msg=f"expected exactly 1 cuTILE program launch; got {len(used)}",
        )

    recorder = used[0]
    args = recorder.args  # guaranteed non-None

    # 4. Analyze recorded args → pointer binding indices + scalar SymInts.
    num_inputs = len(inp_descs)
    num_outputs = len(out_descs)

    _, scalar_symints = analyze_launch_args(
        args=args,
        num_inputs=num_inputs,
        num_outputs=num_outputs,
        op=qdp_symbol,
        backend=backend,
    )

    def _symint_to_int(v: Any, default: int) -> int:
        """Extract concrete int from a scalar that may be Python int or SymInt32."""
        if isinstance(v, int):
            return v
        # Try the same fallbacks as _shape_expr_to_ints
        try:
            return int(v)
        except (TypeError, ValueError):
            pass
        if hasattr(v, "max"):
            try:
                return int(v.max())
            except (TypeError, ValueError, AttributeError):
                pass
        if hasattr(v, "constant_value") and not getattr(v, "is_fake", True):
            try:
                cv = v.constant_value
                return int(cv() if callable(cv) else cv)
            except (TypeError, ValueError, AttributeError, RuntimeError):
                pass
        return default

    block_size = int(cfg.get("BLOCK", 256))
    if scalar_symints:
        block_size = _symint_to_int(scalar_symints[0], block_size)

    try:
        import cuda.tile as ct  # type: ignore[import]
        from cuda.tile.compilation import (  # type: ignore[import]
            ArrayConstraint,
            CallingConvention,
            ConstantConstraint,
            KernelSignature,
            export_kernel,
        )
    except ImportError as exc:
        raise TTAPluginError(
            op=qdp_symbol,
            stage="aot_impl",
            backend=backend,
            msg=f"cuTILE compilation API (cuda.tile.compilation) not available for op '{qdp_symbol}': {exc}",
        ) from exc

    # Verify that tileiras is on PATH — it ships with the cuda-tile package.
    # Users must ensure their PATH includes the cuda-tile bin directory
    # (e.g. add <site-packages>/nvidia/cu13/bin to PATH).
    import shutil as _shutil
    if not _shutil.which("tileiras"):
        raise TTAPluginError(
            op=qdp_symbol,
            stage="aot_impl",
            backend=backend,
            msg=(
                "tileiras compiler not found on PATH.  "
                "Install cuda-tile (`pip install cuda-tile`) and ensure that "
                "the package's bin directory (e.g. <site-packages>/nvidia/cu13/bin) "
                "is on your PATH before importing torch_tensorrt.annotation."
            ),
        )

    all_descs = list(inp_descs) + list(out_descs)
    num_inputs = len(inp_descs)
    num_outputs = len(out_descs)

    if scalar_symints:
        scalar_ints = [_symint_to_int(s, block_size) for s in scalar_symints]
    else:
        scalar_ints = [block_size]

    def _build_signature(flatten_all: bool) -> KernelSignature:
        # Match the kernel ABI: arrays in (inputs..., outputs...) order,
        # followed by scalar attributes as compile-time constants.
        params: List[Any] = []
        for td in all_descs:
            natural_ndim = len(td.shape_expr) or 1
            ndim = 1 if (flatten_all or natural_ndim > 2) else natural_ndim
            params.append(
                ArrayConstraint(
                    dtype=_trt_dtype_to_ct(td.dtype),
                    ndim=ndim,
                    index_dtype=ct.int32,
                    # cuTile rejects negative strides by default; the TRT
                    # tensors we receive always have non-negative strides.
                    stride_lower_bound_incl=0,
                    alias_groups=(),
                    may_alias_internally=False,
                )
            )
        for v in scalar_ints:
            params.append(ConstantConstraint(int(v)))
        return KernelSignature(
            parameters=tuple(params),
            calling_convention=CallingConvention.cutile_python_v1(),
            symbol=None,
        )

    real_prog = recorder.real_prog

    # ``export_kernel`` writes the compiled artefact to a file or file-like
    # object.  Use BytesIO to keep the cubin in memory.
    import io as _io

    major, minor = torch.cuda.get_device_capability()
    gpu_code = f"sm_{major}{minor}"

    def _compile_to_cubin(flatten_all: bool) -> bytes:
        buf = _io.BytesIO()
        export_kernel(
            real_prog,
            [_build_signature(flatten_all=flatten_all)],
            buf,
            gpu_code=gpu_code,
            output_format="cubin",
        )
        return buf.getvalue()

    try:
        cubin_bytes = _compile_to_cubin(flatten_all=False)
    except Exception as exc:
        exc_str = str(exc)
        if "Index size" in exc_str and "array rank" in exc_str:
            try:
                cubin_bytes = _compile_to_cubin(flatten_all=True)
            except Exception as retry_exc:
                raise TTAPluginError(
                    op=qdp_symbol,
                    stage="aot_impl",
                    backend=backend,
                    msg=f"cuTILE compilation failed (retry with flattened shapes): {retry_exc}",
                ) from retry_exc
        else:
            raise TTAPluginError(
                op=qdp_symbol,
                stage="aot_impl",
                backend=backend,
                msg=f"cuTILE compilation failed: {exc}",
            ) from exc

    kernel_name_str = (
        getattr(real_prog, "kernel_name", None)
        or getattr(real_prog, "__name__", None)
        or "cutile_kernel"
    )

    num_scalars = len(scalar_symints) if scalar_symints else 1
    rank = max(1, max(len(td.shape_expr) for td in all_descs))
    ptx_raw = _extract_ptx_from_cubin(cubin_bytes)
    num_ptx_params = _count_ptx_params(ptx_raw) if ptx_raw else 0
    n_io = num_inputs + num_outputs
    if n_io > 0 and num_ptx_params >= num_scalars:
        remainder = (num_ptx_params - num_scalars) % n_io
        if remainder == 0:
            ppt = (num_ptx_params - num_scalars) // n_io
            if ppt >= 1:
                effective_rank = (ppt - 1) // 2
                if effective_rank < 1:
                    effective_rank = 1
            else:
                effective_rank = 1
        else:
            effective_rank = 1 if rank > 2 else rank
    else:
        effective_rank = 1 if rank > 2 else rank
    if rank > 1 and ptx_raw:
        expected_n, _ = _cutile_trt_order_for_io(
            num_inputs, num_outputs, num_scalars, rank=effective_rank
        )
        # Dump diagnostic PTX for rank>1 kernels and log the path.
        diag_filename = f"cutile_rank_gt1_{kernel_name_str}_nparam_{num_ptx_params}_expected_{expected_n}.ptx"
        _default_dump_dir = os.path.join(tempfile.gettempdir(), "torch_tensorrt_cutile_dump")
        dump_code_artifact("CUTILE_DUMP_DIR", diag_filename, ptx_raw, default_dir=_default_dump_dir)
        ptx_path = os.path.join(
            os.environ.get("CUTILE_DUMP_DIR") or _default_dump_dir, diag_filename
        )
        logger.info(
            "cutile rank>1: kernel=%s num_ptx_params=%d expected_n=%d effective_rank=%d (num_inputs=%d num_outputs=%d num_scalars=%d) ptx_dumped=%s",
            kernel_name_str,
            num_ptx_params,
            expected_n,
            effective_rank,
            num_inputs,
            num_outputs,
            num_scalars,
            ptx_path,
        )
    code_out = _reorder_cutile_ptx_for_trt(
        cubin_bytes,
        kernel_name_str,
        num_inputs=num_inputs,
        num_outputs=num_outputs,
        num_scalars=num_scalars,
        rank=effective_rank,
    )

    if ptx_raw is not None:
        dump_code_artifact(
            "CUTILE_DUMP_DIR",
            f"cutile_rank{effective_rank}_{kernel_name_str}_orig.ptx",
            ptx_raw,
            default_dir="/tmp/cutile_dump",
        )
        dump_code_artifact(
            "CUTILE_DUMP_DIR",
            f"cutile_rank{effective_rank}_{kernel_name_str}_orig.cubin",
            cubin_bytes,
            default_dir="/tmp/cutile_dump",
        )
        if b".version" in code_out[:64]:
            dump_code_artifact(
                "CUTILE_DUMP_DIR",
                f"cutile_rank{effective_rank}_{kernel_name_str}_reordered.ptx",
                code_out.decode("utf-8", errors="replace"),
                default_dir="/tmp/cutile_dump",
            )

    # CuTile may vectorise and use fewer threads than tile_size (.reqntid).
    is_ptx = b".version" in code_out[:64]
    if is_ptx:
        ptx_str = code_out.decode("utf-8", errors="replace")
        reqntid = _parse_reqntid(ptx_str)
        actual_block = reqntid if reqntid else block_size
    else:
        actual_block = block_size

    # Follow the same pattern as _triton_aot.py: use recorder.grid directly.
    # The grid recorded during sandbox execution is always correct — it's whatever
    # the launch function computed from the input shapes (concrete ints for static
    # shapes, SymInt32 expressions for dynamic shapes).
    launch = trtp.KernelLaunchParams()
    recorded_grid = getattr(recorder, "grid", None)
    if recorded_grid is not None and len(recorded_grid) >= 1:
        _assign_recorded_grid(launch, recorded_grid)
    else:
        # Fallback: 1D grid from output numel / block_size.
        # Broad catch is necessary: _shape_expr_to_ints raises RuntimeError for
        # unbounded dynamic dims and may raise AttributeError for stub TensorDescs.
        try:
            out_ints = _shape_expr_to_ints(out_descs[0].shape_expr)
            numel = 1
            for x in out_ints:
                numel *= x
            launch.grid_x = trtp.SymInt32((numel + block_size - 1) // block_size)
        except (RuntimeError, AttributeError, TypeError):
            launch.grid_x = trtp.SymInt32(1)
        launch.grid_y = trtp.SymInt32(1)
        launch.grid_z = trtp.SymInt32(1)
    launch.block_x = actual_block
    launch.shared_mem = 0

    extra_args = _infer_cutile_extra_args(
        inp_descs, out_descs, block_size, scalar_symints, rank=effective_rank
    )

    ext = ".ptx" if is_ptx else ".cubin"
    dump_code_artifact(
        "CUTILE_DUMP_DIR",
        kernel_name_str + ext,
        code_out,
        default_dir="/tmp/cutile_dump",
    )

    # Uniquify kernel name per config so TRT registers separate PTX/cubin per tactic.
    # Without this, two tactics with the same kernel function but different tile sizes
    # (e.g. BLOCK=128 vs BLOCK=256) share the same name and TRT applies wrong launch
    # params to one of them, causing ~50% element mismatches.
    if cfg:
        suffix = "_".join(f"{k}{v}" for k, v in sorted(cfg.items()))
        unique_name = f"{kernel_name_str}_{suffix}"
        # If code_out is still a cubin but we have ptx_raw available, convert to PTX
        # so we can do a safe string-replace for the kernel name.
        if not is_ptx and ptx_raw is not None:
            downgraded = _downgrade_ptx_version(ptx_raw)
            code_out = downgraded.encode("utf-8")
            is_ptx = True
        if is_ptx:
            ptx_str_out = code_out.decode("utf-8", errors="replace")
            ptx_str_out = ptx_str_out.replace(kernel_name_str, unique_name)
            code_out = ptx_str_out.encode("utf-8")
            kernel_name_str = unique_name
        else:
            # Cubin without embedded PTX: patch null-terminated name in-place.
            # Only safe when the new name fits in the original allocation.
            old_b = kernel_name_str.encode("utf-8") + b"\x00"
            new_b = unique_name.encode("utf-8") + b"\x00"
            if len(new_b) <= len(old_b):
                code_out = code_out.replace(old_b, new_b + b"\x00" * (len(old_b) - len(new_b)))
                kernel_name_str = unique_name
            else:
                logger.warning(
                    "cutile aot: cannot uniquify cubin kernel name %r → %r (new name is longer); "
                    "tactic name collision may occur",
                    kernel_name_str,
                    unique_name,
                )

    return kernel_name_str.encode("utf-8"), code_out, launch, extra_args


def compile_cutile_program(spec: CuTileSpec, config: Dict[str, Any]) -> AOTMetadata:
    """Compile a CuTileSpec into unified AOTMetadata.

    This is the tactic-manager entry-point.  It constructs synthetic 1-input /
    1-output TensorDesc stubs (shape [256], float32) to drive the sandbox run,
    delegates to ``aot_impl_cutile`` for the full pipeline, and wraps the result
    in ``AOTMetadata``.

    Args:
        spec:   CuTileSpec carrying ``launch_fn``, optional ``configs``, etc.
        config: Single tactic configuration dict (e.g. ``{"BLOCK": 256}``).

    Returns:
        AOTMetadata with ``backend="cutile"``, compiled PTX/CUBIN bytes, and launch params.
    """
    inp_descs = [trtp.TensorDesc(dtype=trt.float32, shape_expr=[256])]
    out_descs = [trtp.TensorDesc(dtype=trt.float32, shape_expr=[256])]
    sym_inputs = [
        SymbolicTensor(td=inp_descs[0], role=TensorRole.INPUT, index=0)
    ]
    sym_outputs = [
        SymbolicTensor(td=out_descs[0], role=TensorRole.OUTPUT, index=0)
    ]
    host_args = sym_inputs + sym_outputs
    kernel_name_bytes, code_bytes, launch, extra_args = aot_impl_cutile(
        qdp_symbol="cutile_compile",
        spec=spec,
        cfg=config,
        launch_fn=spec.launch_fn,
        host_args=host_args,
        inp_descs=inp_descs,
        out_descs=out_descs,
    )
    program_name = kernel_name_bytes.decode("utf-8") if isinstance(kernel_name_bytes, bytes) else kernel_name_bytes
    launch_params = _launch_params_from_trt(launch, extra_args, num_inputs=1, num_outputs=1)
    return AOTMetadata(binary=code_bytes, kernel_name=program_name, launch_params=launch_params, backend="cutile")
