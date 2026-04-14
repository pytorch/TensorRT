"""Triton AOT backend: source → sandbox recording → PTX → KernelLaunchParams.

AOT compilation pipeline for Triton kernels
============================================
1. **Sandbox execution** — The user's ``launch_fn`` is run with ``SymbolicTensor``
   proxies in place of real tensors and ``TritonLaunchRecorder`` objects injected
   over the real ``@triton.jit`` kernels in the module globals.  This lets us
   capture the kernel call without executing on a GPU.

2. **Argument analysis** — ``analyze_launch_args`` separates the recorded call
   arguments into *pointer binding indices* (which TRT tensor buffer maps to
   which kernel parameter) and *scalar SymInt32 expressions* (grid or shape
   scalars that TRT evaluates at runtime).

3. **PTX compilation** — ``triton.compile(ASTSource(fn, signature, constexprs))``
   produces both PTX and CUBIN.  We extract the PTX from ``compiled.asm["ptx"]``
   because TRT's QDP runtime JIT-compiles PTX for the current GPU architecture,
   matching the official ``aot_plugin`` example and avoiding cubin arch mismatch.

4. **Parameter reordering** — Triton emits `.param` declarations in Python/kernel
   order (pointers first, then scalars).  The QDP runtime passes arguments in
   *runtime order* (input pointers, scalars, output pointers).  ``_fix_triton_ptx_for_trt``
   rewrites the `.param` block and all references in the PTX body to match.

5. **Tactic uniquification** — When multiple tactics compile the same kernel
   function with different ``constexprs`` (e.g. ``BLOCK_M=16`` vs ``BLOCK_M=32``),
   a config-derived suffix is appended to the kernel name so TRT registers
   separate PTX entries per tactic.

The public entry-point for the descriptor system is ``aot_impl_triton``, which
returns the QDP AOT 4-tuple ``(kernel_name, ptx_bytes, KernelLaunchParams, SymIntExprs)``.
``compile_triton_kernel`` is a thin wrapper used by the tactic manager.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Mapping, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    import tensorrt as trt
    import tensorrt.plugin as trtp
except ImportError as e:
    raise ImportError(
        "TensorRT with plugin support is required for Triton AOT compilation."
    ) from e

from .._qdp_utils import (
    AOTMetadata,
    TTAPluginError,
    _as_symint32,
    _assign_recorded_grid,
    _launch_params_from_trt,
    analyze_launch_args,
    dump_code_artifact,
    is_triton_kernel,
    run_kernel_sandbox,
)
from ..._recorders import TritonLaunchRecorder
from ..._specs import TritonSpec
from .._symbolic import SymbolicTensor, TensorRole
from torch_tensorrt._enums import dtype as _dtype  # torch_tensorrt's shared cross-framework dtype enum

# Triton's ASTSource signature dict requires string element-type tokens for
# pointer arguments (e.g. "*fp16").  These tokens are Triton-ABI-specific and
# have no counterpart in tensorrt or numpy, so they cannot be expressed as
# torch_tensorrt.dtype values directly.  We map through torch_tensorrt.dtype
# (the shared cross-framework enum) to stay consistent with the rest of the
# library's dtype handling, and express the Triton-specific token as a final
# per-backend step.
_DTYPE_TO_TRITON_PTR: Dict[_dtype, str] = {
    _dtype.f16: "fp16",
    _dtype.bf16: "bf16",
    _dtype.f32: "fp32",
    _dtype.i32: "i32",
}


def _trt_dtype_to_triton_ptr(trt_dtype: Any, qdp_symbol: str) -> str:
    """Map a TensorRT DataType to a Triton pointer element-type string.

    Converts via ``torch_tensorrt.dtype`` (the shared cross-framework enum)
    so the mapping stays consistent with the rest of the library's dtype
    handling.  The final Triton ABI string is Triton-specific and lives only
    in ``_DTYPE_TO_TRITON_PTR``.
    """
    try:
        tta_dtype = _dtype._from(trt_dtype)
    except TypeError:
        tta_dtype = None
    ptr_type = _DTYPE_TO_TRITON_PTR.get(tta_dtype) if tta_dtype is not None else None
    if ptr_type is None:
        raise TTAPluginError(
            op=qdp_symbol,
            stage="aot_impl",
            backend="triton",
            msg=f"unsupported tensor dtype {trt_dtype} for Triton kernel signature",
        )
    return ptr_type


# LIMITATION (fragile PTX rewrite): the three _ptx_* helpers below rewrite .param
# declarations and their references inside the PTX body using line-by-line text
# scanning.  Triton compiles parameters in Python/kernel order (pointers first,
# then constexprs), while the TRT plugin runtime expects (inputs, scalars,
# outputs).  The rewrite is fragile because:
#   1. It identifies param references by matching the prefix ``{kernel_name}_param_``
#      as a plain string — any change to Triton's param naming convention silently
#      produces incorrect PTX.
#   2. It scans for ``.entry {kernel_name}(`` as a literal string; multi-line or
#      differently-formatted entry declarations will not be recognised.
#   3. Unused trailing params (constexprs not referenced in the body) are stripped
#      by counting references — this is correct only if Triton doesn't reuse param
#      indices in non-obvious ways.
# The proper fix is for TRT's QDP AOT API to expose a parameter-order remapping
# mechanism so that post-compilation PTX rewriting is not needed.


def _ptx_downgrade_version(ptx: str) -> str:
    """Downgrade the PTX ``.version`` line from 9.x to 9.0.

    Triton on CUDA 13.x emits ``.version 9.1`` (set by LLVM's NVPTX backend).
    TRT's QDP PTX loader caps at 9.0 — kernels with a higher version silently
    fail to load, producing a spurious ``onShapeChange`` error at runtime.

    Why not pin ``ptx_version=90`` in ``triton.compile``?  Requesting 9.0 makes
    LLVM emit ``.version 9.0``, which then fails Triton's ``make_cubin`` step
    because the bundled ``ptxas`` (v8.7) cannot assemble PTX 9.0.  Post-compilation
    header patching is therefore the only viable workaround until TRT's PTX loader
    is updated to accept 9.1.  The ``.target`` line (e.g. ``sm_120a``) is unchanged.
    """
    lines = ptx.split("\n")
    result = []
    for line in lines:
        if line.startswith(".version "):
            line = re.sub(r"^(\.version\s+)9\.([1-9]\d*)", r"\g<1>9.0", line)
        result.append(line)
    return "\n".join(result)


def _ptx_reorder_and_strip_params(
    ptx: str,
    kernel_name: str,
    trt_order: List[int],
    num_used_params: int,
) -> str:
    """Reorder ``.param`` declarations and body references; strip trailing params.

    Two rewrites combined into one pass because they both operate on the same
    param-block region of the PTX:

    * **Reorder**: rearrange the ``.param`` declaration lines inside the
      ``.entry`` block to match ``trt_order`` (TRT runtime order:
      inputs → scalars → outputs).  Body references (``{kernel}_param_N``)
      are also renamed to match.

    * **Strip**: Triton appends internal params (``printf_buffer``, ``prevGrid``)
      beyond the user-declared arguments.  TRT passes exactly ``num_used_params``
      args; the extras are dropped and the trailing comma is fixed.

    Args:
        ptx:            PTX text (after version downgrade).
        kernel_name:    Entry-point name matching the ``.entry`` directive.
        trt_order:      Permutation list mapping new param index → original index.
                        ``trt_order[i]`` is the original position of the param
                        that should appear at position ``i`` in the TRT call.
        num_used_params: Number of params TRT will pass (= len(positional_names)).
    """
    needs_reorder = trt_order != list(range(num_used_params))
    pfx = f"{kernel_name}_param_"
    lines = ptx.split("\n")
    result: List[str] = []
    in_entry = False
    param_lines: List[str] = []

    for line in lines:
        if f".entry {kernel_name}(" in line:
            in_entry = True
            param_lines = []
            result.append(line)
            continue

        if in_entry and ".param" in line and pfx in line:
            param_lines.append(line)
            continue

        if in_entry and ")" in line and ".param" not in line:
            in_entry = False
            reordered = (
                [param_lines[i] for i in trt_order if i < len(param_lines)]
                if needs_reorder
                else param_lines[:num_used_params]
            )
            for i, pline in enumerate(reordered):
                pline = pline.rstrip().rstrip(",")
                if i < len(reordered) - 1:
                    pline += ","
                result.append(pline)
            result.append(line)
            continue

        result.append(line)

    if needs_reorder:
        old_to_new = {old: new for new, old in enumerate(trt_order)}
        joined = "\n".join(result)
        width = len(str(num_used_params - 1)) if num_used_params > 0 else 1
        for old_idx in range(num_used_params - 1, -1, -1):
            joined = joined.replace(f"{pfx}{old_idx}", f"{pfx}TEMP{old_idx:0{width}d}")
        for old_idx, new_idx in old_to_new.items():
            joined = joined.replace(f"{pfx}TEMP{old_idx:0{width}d}", f"{pfx}{new_idx}")
        result = joined.split("\n")

    return "\n".join(result)


def _fix_triton_ptx_for_trt(
    ptx: str,
    kernel_name: str,
    num_used_params: int,
    param_binding_indices: List[int],
    num_inputs: int,
    num_scalars: int,
) -> str:
    """Apply all PTX mutations needed for TRT QDP compatibility.

    Delegates each discrete rewrite to a named helper:

    1. :func:`_ptx_downgrade_version` — cap ``.version`` at 9.0.
    2. :func:`_ptx_reorder_and_strip_params` — reorder param declarations and
       body references to TRT runtime order (inputs → scalars → outputs) and
       strip Triton's internal trailing params.
    """
    # Compute TRT runtime parameter order from the recorded binding indices.
    num_ptrs = len(param_binding_indices)
    input_params = sorted(
        ((binding, orig) for orig, binding in enumerate(param_binding_indices) if binding < num_inputs)
    )
    output_params = sorted(
        ((binding, orig) for orig, binding in enumerate(param_binding_indices) if binding >= num_inputs)
    )
    scalar_params = list(range(num_ptrs, num_ptrs + num_scalars))
    trt_order = (
        [orig for _, orig in input_params]
        + scalar_params
        + [orig for _, orig in output_params]
    )

    ptx = _ptx_downgrade_version(ptx)
    ptx = _ptx_reorder_and_strip_params(ptx, kernel_name, trt_order, num_used_params)
    return ptx


def _specialize_ptx_kernel_name(
    ptx: str,
    kernel_name: str,
    cfg: Mapping[str, Any],
) -> Tuple[str, str]:
    """Append a config-derived suffix to the PTX kernel name.

    When two tactics compile the same ``@triton.jit`` function with different
    constexprs (e.g. ``BLOCK_M=16`` vs ``BLOCK_M=32``), both return the same
    ``kernel_name``.  TRT identifies kernels by name, so it uses whichever PTX
    was registered last for *all* tactics sharing that name — but still applies
    each tactic's launch params, causing a mismatch (wrong grid dimensions for
    the baked-in tile sizes).  Append a short config suffix to give every tactic
    a distinct name.

    Args:
        ptx:         Raw PTX text from ``compiled.asm["ptx"]``.
        kernel_name: Current kernel entry-point name (from ``compiled.metadata.name``).
        cfg:         Tactic config dict (e.g. ``{"BLOCK_M": 32}``).  Empty dict
                     means no suffix is needed.

    Returns:
        ``(new_ptx, new_kernel_name)`` — the rewritten PTX and updated name.
        If *cfg* is empty both values are returned unchanged.
    """
    if not cfg:
        return ptx, kernel_name
    suffix = "_".join(f"{k}{v}" for k, v in sorted(cfg.items()))
    unique_name = f"{kernel_name}_{suffix}"
    ptx = ptx.replace(kernel_name, unique_name)
    return ptx, unique_name


def _make_triton_launch_params(
    grid: Any,
    num_warps: int,
    shared_mem: int,
    scalar_symints: List[Any],
) -> Tuple[Any, Any]:
    """Build ``trtp.KernelLaunchParams`` and ``trtp.SymIntExprs`` for a Triton kernel.

    Centralises the repeated pattern of converting a recorded grid tuple into
    ``KernelLaunchParams.grid_{x,y,z}`` SymInt32 values and populating
    ``extra_args`` from the scalar SymInts captured during sandbox execution.

    Args:
        grid:          Grid value captured by :class:`TritonLaunchRecorder`.
                       May be a tuple of ints/SymInt32 or a single value.
        num_warps:     ``compiled.metadata.num_warps`` from ``triton.compile``.
        shared_mem:    ``compiled.metadata.shared`` from ``triton.compile``.
        scalar_symints: Scalar SymInt expressions from ``analyze_launch_args``.

    Returns:
        ``(launch, extra_args)`` — a ``trtp.KernelLaunchParams`` with grid/block/shared
        populated and a ``trtp.SymIntExprs`` with one slot per scalar.
    """
    launch = trtp.KernelLaunchParams()
    _assign_recorded_grid(launch, grid)
    launch.block_x = num_warps * 32
    launch.block_y = 1
    launch.block_z = 1
    launch.shared_mem = shared_mem

    extra_args = trtp.SymIntExprs(len(scalar_symints))
    for idx, val in enumerate(scalar_symints):
        extra_args[idx] = _as_symint32(val)

    return launch, extra_args


def aot_impl_triton(
    *,
    qdp_symbol: str,
    spec: TritonSpec,
    cfg: Mapping[str, Any],
    launch_fn: Any,
    host_args: List[SymbolicTensor],
    inp_descs: List[Any],
    out_descs: List[Any],
    attrs: Optional[Dict[str, Any]] = None,
) -> Tuple[str, bytes, Any, Any]:
    """Triton AOT implementation: sandbox → record → compile → PTX.

    Steps:
    1. Find @triton.jit kernels in launch_fn's module.
    2. Replace with TritonLaunchRecorder proxies.
    3. Run sandboxed launch_fn(*host_args, **merged_kwargs).
    4. Analyse recorded args → param_binding_indices + scalar SymInts.
    5. Build per-tensor-dtype Triton signature, triton.compile() → PTX.
    6. Fix PTX: reorder params to runtime order, strip unused trailing params.
    7. Return (kernel_name, ptx, KernelLaunchParams, SymIntExprs).

    Returns:
        (kernel_name, ptx, KernelLaunchParams, SymIntExprs)
    """
    import triton

    backend = "triton"

    merged_kwargs = dict(cfg)
    if attrs:
        merged_kwargs.update(attrs)

    # 1-2. Locate module, find @triton.jit kernels, sandbox and run.
    #      strict=True: propagate module-not-found and no-kernel-found as errors.
    #      host_kwargs: cfg (tactic constexprs) + attrs (plugin compile-time constants).
    # Broad catch is necessary: the sandbox executes arbitrary user launch_fn code
    # and can raise any exception type (ImportError for missing deps, TypeError
    # for shape mismatches, AttributeError from proxy gaps, etc.).
    #
    # WHY the sandbox recorder is needed before calling ASTSource:
    #   * Which kernel — launch_fn may import several @triton.jit functions; the
    #     recorder tells us exactly which one was launched (``used[0].real_kernel``).
    #   * Argument order — the recorder captures positional args in the order the
    #     launch_fn passes them.  We use this to derive ``param_binding_indices``
    #     (which arg index maps to which TRT input/output descriptor) and the
    #     Triton ``signature`` dict.  ASTSource cannot provide this information.
    #   * Grid expression — ``recorder.grid`` captures the symbolic or concrete grid
    #     value to populate ``KernelLaunchParams.grid_{x,y,z}`` at plugin build time.
    try:
        used_recorder, kernel_recorders = run_kernel_sandbox(
            launch_fn=launch_fn,
            host_args=host_args,
            is_kernel_fn=is_triton_kernel,
            recorder_factory=lambda obj: TritonLaunchRecorder(real_kernel=obj),
            host_kwargs=merged_kwargs,
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

    # 3. Exactly one Triton kernel must have been launched.
    used = [rec for rec in kernel_recorders.values() if rec.grid is not None]
    if len(used) != 1:
        raise TTAPluginError(
            op=qdp_symbol,
            stage="aot_impl",
            backend=backend,
            msg=f"expected exactly 1 Triton kernel launch; got {len(used)}",
        )

    recorder = used[0]
    grid = recorder.grid
    args = recorder.args
    kwargs = recorder.kwargs or {}

    num_inputs = len(inp_descs)
    num_outputs = len(out_descs)

    def _to_int(x: Any) -> Any:
        try:
            return int(x)
        except (TypeError, ValueError):
            return x

    constexprs = {k: _to_int(v) for k, v in cfg.items()}
    if attrs:
        constexprs.update(attrs)
    kernel_arg_names = list(recorder.real_kernel.arg_names)
    positional_names = [n for n in kernel_arg_names if n not in constexprs]

    full_args: List[Any] = []
    for i, name in enumerate(positional_names):
        if i < len(args):
            full_args.append(args[i])
        elif name in kwargs:
            full_args.append(kwargs[name])
        else:
            raise TTAPluginError(
                op=qdp_symbol,
                stage="aot_impl",
                backend=backend,
                msg=f"missing kernel argument '{name}' (pass positionally or by keyword)",
            )

    # 4. Analyze recorded args → pointer binding indices + scalar SymInts.
    param_binding_indices, scalar_symints = analyze_launch_args(
        args=full_args,
        num_inputs=num_inputs,
        num_outputs=num_outputs,
        op=qdp_symbol,
        backend=backend,
    )

    if not param_binding_indices:
        raise TTAPluginError(
            op=qdp_symbol,
            stage="aot_impl",
            backend=backend,
            msg="no pointer arguments recorded for Triton kernel",
        )

    # 5. Build Triton signature dict {param_name: type_str} for non-constexpr args.
    all_descs = list(inp_descs) + list(out_descs)

    ptr_idx = 0
    scalar_idx = 0
    signature: Dict[str, str] = {}
    for name in positional_names:
        if ptr_idx < len(param_binding_indices):
            b = param_binding_indices[ptr_idx]
            dtype_str = _trt_dtype_to_triton_ptr(all_descs[b].dtype, qdp_symbol)
            signature[name] = f"*{dtype_str}"
            ptr_idx += 1
        else:
            signature[name] = "i32"
            scalar_idx += 1

    expected_num_scalars = len(positional_names) - len(param_binding_indices)
    if len(scalar_symints) != expected_num_scalars:
        raise TTAPluginError(
            op=qdp_symbol,
            stage="aot_impl",
            backend=backend,
            msg=(
                f"scalar count mismatch: launch passed {len(scalar_symints)} scalars "
                f"but kernel has {expected_num_scalars} scalar args (depends on descriptor ranks for this invocation)"
            ),
        )

    # triton.compile raises a mix of Exception subclasses (CompilationError,
    # subprocess.CalledProcessError, RuntimeError) depending on the failure mode;
    # a broad catch is necessary here to give a structured diagnostic in all cases.
    try:
        compiled = triton.compile(
            triton.compiler.ASTSource(
                fn=recorder.real_kernel,
                signature=signature,
                constexprs=constexprs,
            )
        )
    except Exception as exc:
        raise TTAPluginError(
            op=qdp_symbol,
            stage="aot_impl",
            backend=backend,
            msg=f"triton.compile failed for kernel '{recorder.real_kernel.__name__}': {exc}",
        ) from exc

    # 6. Build KernelLaunchParams (grid can be symbolic; QDP evaluates at runtime).
    launch, extra_args = _make_triton_launch_params(
        grid=grid,
        num_warps=compiled.metadata.num_warps,
        shared_mem=compiled.metadata.shared,
        scalar_symints=scalar_symints,
    )

    # 7. Extract PTX, reorder params to runtime order, strip unused trailing params.
    kernel_name_str: str = compiled.metadata.name
    ptx: str = compiled.asm["ptx"]
    if isinstance(ptx, bytes):
        ptx = ptx.decode("utf-8")

    dump_code_artifact("TTA_DUMP_TRITON_PTX", f"{kernel_name_str}_raw.ptx", ptx)

    ptx, kernel_name_str = _specialize_ptx_kernel_name(ptx, kernel_name_str, cfg)

    num_used = len(positional_names)
    ptx = _fix_triton_ptx_for_trt(
        ptx=ptx,
        kernel_name=kernel_name_str,
        num_used_params=num_used,
        param_binding_indices=param_binding_indices,
        num_inputs=num_inputs,
        num_scalars=len(scalar_symints),
    )
    ptx_bytes = ptx.encode("utf-8")

    dump_code_artifact("TTA_DUMP_TRITON_PTX", f"{kernel_name_str}_fixed.ptx", ptx)

    return kernel_name_str, ptx_bytes, launch, extra_args


def compile_triton_kernel(spec: TritonSpec, config: Dict[str, Any]) -> AOTMetadata:
    """Compile a TritonSpec into unified AOTMetadata.

    This is the tactic-manager entry-point.  It constructs synthetic 1-input /
    1-output TensorDesc stubs (shape [256], float32) to drive the sandbox run,
    delegates to ``aot_impl_triton`` for the full pipeline, and wraps the result
    in ``AOTMetadata``.

    Args:
        spec:   TritonSpec carrying ``launch_fn``, optional ``configs``, etc.
        config: Single tactic configuration dict (e.g. ``{"BLOCK_M": 32}``).

    Returns:
        AOTMetadata with ``backend="triton"``, compiled PTX bytes, and launch params.
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
    kernel_name_str, ptx_bytes, launch, extra_args = aot_impl_triton(
        qdp_symbol="triton_compile",
        spec=spec,
        cfg=config,
        launch_fn=spec.launch_fn,
        host_args=host_args,
        inp_descs=inp_descs,
        out_descs=out_descs,
    )
    launch_params = _launch_params_from_trt(launch, extra_args, num_inputs=1, num_outputs=1)
    return AOTMetadata(binary=ptx_bytes, kernel_name=kernel_name_str, launch_params=launch_params, backend="triton")
