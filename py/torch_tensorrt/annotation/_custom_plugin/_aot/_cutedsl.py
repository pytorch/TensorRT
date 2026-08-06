"""CuTe DSL AOT backend: @cute.jit sandbox → cute.compile → PTX → KernelLaunchParams.

AOT compilation pipeline for CuTe DSL kernels
===============================================
1. **Grid sandbox** — The user's ``launch_fn`` (a ``@cute.jit``-decorated function)
   is unwrapped to its raw Python body via ``_get_jit_raw_fn``.  The raw body is
   executed in a sandbox with ``SymbolicTensor`` proxies and ``CuTeDSLKernelRecorder``
   objects injected over ``@cute.kernel`` callables in the module globals.  This
   captures the symbolic grid expression (``recorded_grid``) without running on a GPU.
   If the sandbox fails (``strict=False``), the grid falls back to ``(1, 1, 1)`` and
   TRT launches with a single CTA.

2. **CUDA tensor construction** — Dummy ``torch.zeros`` tensors matching the shapes
   and dtypes described by the ``inp_descs`` / ``out_descs`` TensorDescs are placed
   on the GPU.  They are converted to CuTe tensors via ``from_dlpack`` for the
   ``cute.compile`` call.

3. **AOT compilation** — ``cute.compile(launch_fn, *cute_tensors, options=..., **cfg)``
   compiles the kernel with config constants baked in as compile-time values.  The
   ``--dump-dir`` / ``--keep-ptx`` options cause PTX to be written to a temp directory
   *and* attached to ``compiled.artifacts.PTX``.

4. **PTX extraction** — PTX is read from ``compiled.artifacts.PTX``.  The kernel name
   is taken from ``compiled.kernel_info`` (first key).

5. **KernelLaunchParams construction** — The recorded symbolic grid and block dimensions
   are stored in a ``trtp.KernelLaunchParams`` so TRT evaluates grid size at runtime.
   ``extra_args`` is empty (CuTe DSL encodes shape information differently from Triton).

The public entry-point is ``aot_impl_cutedsl``, which returns the QDP AOT 4-tuple
``(kernel_name, ptx_bytes, KernelLaunchParams, SymIntExprs)``.
``compile_cutedsl_kernel`` is a thin tactic-manager wrapper.
"""
from __future__ import annotations

import tempfile
from typing import Any, Dict, List, Mapping, Optional, Tuple

try:
    import tensorrt as trt
    import tensorrt.plugin as trtp
except ImportError as e:
    raise ImportError(
        "TensorRT with plugin support is required for CuTe DSL AOT compilation."
    ) from e

import torch

from .._qdp_utils import (
    AOTMetadata,
    TTAPluginError,
    _assign_recorded_grid,
    _launch_params_from_trt,
    _safe_dim,
    is_cute_kernel,
    run_kernel_sandbox,
    td_dtype_to_torch,
)
from ..._recorders import CuTeDSLKernelRecorder
from ..._specs import CuTeDSLSpec
from .._symbolic import SymbolicTensor, TensorRole


def _get_jit_raw_fn(fn: Any) -> Optional[Any]:
    """Try to extract the raw Python function from a @cute.jit decorated object.

    @cute.jit may preserve the original function via __wrapped__ (functools.wraps)
    or a backend-specific attribute.  Check __wrapped__ BEFORE __code__ because
    @cute.jit objects may have __code__ on the wrapper (calling it triggers JIT
    compilation), while __wrapped__ points to the raw Python body.
    """
    for attr in ("__wrapped__", "_fn", "fn", "_func", "func"):
        raw = getattr(fn, attr, None)
        if raw is not None and raw is not fn and callable(raw) and hasattr(raw, "__code__"):
            return raw
    if hasattr(fn, "__code__"):
        return fn
    return None


def aot_impl_cutedsl(
    *,
    qdp_symbol: str,
    spec: CuTeDSLSpec,
    cfg: Mapping[str, Any],
    launch_fn: Any,
    host_args: List[SymbolicTensor],
    inp_descs: List[Any],
    out_descs: List[Any],
    attrs: Optional[Dict[str, Any]] = None,
) -> Tuple[str, bytes, Any, Any]:
    """CuTe DSL AOT implementation: sandbox → record grid → compile → extract PTX.

    Steps:
    1. Find @cute.kernel objects in launch_fn's module.
    2. Run sandboxed launch_fn body with SymbolicTensor proxies to record the grid.
    3. Create dummy GPU tensors matching inp/out TensorDescs.
    4. Convert to CuTe tensors via from_dlpack.
    5. Call cute.compile(launch_fn, *cute_tensors) to get compiled object.
    6. Extract PTX from compiled.artifacts.PTX.
    7. Get kernel name from compiled.kernel_info.
    8. Return (kernel_name, ptx, KernelLaunchParams, SymIntExprs).

    Returns:
        (kernel_name_bytes, ptx_bytes, KernelLaunchParams, SymIntExprs)
    """
    backend = "cutedsl"

    try:
        import cutlass.cute as cute
        from cutlass.cute.runtime import from_dlpack
    except ImportError as exc:
        raise TTAPluginError(
            op=qdp_symbol,
            stage="aot_impl",
            backend=backend,
            msg=f"cutlass.cute not available: {exc}",
        ) from exc

    # 1. Sandbox run: find @cute.kernel objects and record the symbolic grid.
    #    strict=False: if @cute.jit doesn't expose __code__ or the sandbox fails,
    #    we fall back to a (1, 1, 1) grid rather than raising.
    recorded_grid: Optional[Tuple[Any, ...]] = None
    recorded_block: Optional[Tuple[Any, ...]] = None

    raw_fn = _get_jit_raw_fn(launch_fn)
    used_recorder, _ = run_kernel_sandbox(
        launch_fn=launch_fn,
        host_args=host_args,
        is_kernel_fn=is_cute_kernel,
        recorder_factory=lambda obj: CuTeDSLKernelRecorder(real_kernel=obj),
        raw_fn=raw_fn,
        host_kwargs=dict(cfg, **(attrs or {})) if cfg or attrs else None,
        strict=False,
    )
    if used_recorder is not None:
        recorded_grid = used_recorder.grid
        recorded_block = used_recorder.block

    # 2. Create dummy GPU tensors for cute.compile.
    all_descs = list(inp_descs) + list(out_descs)
    dummy_tensors = []
    for td in all_descs:
        shape = [_safe_dim(d) for d in td.shape_expr]
        torch_dtype = td_dtype_to_torch(td.dtype)
        dummy_tensors.append(torch.zeros(shape, dtype=torch_dtype, device="cuda"))

    cute_tensors = [from_dlpack(t) for t in dummy_tensors]

    # 3. Compile.
    # TemporaryDirectory cleans up the dump files on exit, even on exception.
    # compiled.artifacts.PTX is an in-memory string, so it remains accessible
    # after the context manager exits.
    with tempfile.TemporaryDirectory(prefix="cutedsl_aot_") as dump_dir:
        compile_opts = f"--dump-dir={dump_dir} --keep-cubin --keep-ptx"
        # Pass cfg as kwargs so config constants (e.g. BLOCK_SIZE) are compile-time
        # values in the PTX, enabling nvvm.reqntid and correct per-tactic block sizes.
        compile_kwargs = dict(cfg) if cfg else {}
        # cute.compile can raise RuntimeError, subprocess.CalledProcessError, or
        # arbitrary Exception subclasses from NVCC/NVVM; a broad catch is necessary
        # to wrap all failures in a structured diagnostic.
        try:
            compiled = cute.compile(
                launch_fn, *cute_tensors, options=compile_opts, **compile_kwargs
            )
        except Exception as exc:
            raise TTAPluginError(
                op=qdp_symbol,
                stage="aot_impl",
                backend=backend,
                msg=f"cute.compile failed for op '{qdp_symbol}': {exc}",
            ) from exc

    if not hasattr(compiled, "artifacts") or not hasattr(compiled.artifacts, "PTX"):
        raise TTAPluginError(
            op=qdp_symbol,
            stage="aot_impl",
            backend=backend,
            msg=(
                f"compiled object (type {type(compiled).__name__}) has no "
                f"artifacts.PTX attribute"
            ),
        )

    ptx_str = compiled.artifacts.PTX
    if not ptx_str or ".entry" not in ptx_str:
        raise TTAPluginError(
            op=qdp_symbol,
            stage="aot_impl",
            backend=backend,
            msg="compiled PTX is empty or has no .entry kernel",
        )

    kernel_names = list(compiled.kernel_info.keys())
    if not kernel_names:
        raise TTAPluginError(
            op=qdp_symbol,
            stage="aot_impl",
            backend=backend,
            msg="compiled object has no kernel names in kernel_info",
        )
    kernel_name_str = kernel_names[0]

    # 4. Build KernelLaunchParams using the recorded (symbolic) grid.
    launch = trtp.KernelLaunchParams()
    _assign_recorded_grid(launch, recorded_grid)

    if recorded_block is not None and len(recorded_block) >= 1:
        launch.block_x = recorded_block[0] if isinstance(recorded_block[0], int) else 1
        launch.block_y = recorded_block[1] if len(recorded_block) > 1 and isinstance(recorded_block[1], int) else 1
        launch.block_z = recorded_block[2] if len(recorded_block) > 2 and isinstance(recorded_block[2], int) else 1
    else:
        launch.block_x = 1
        launch.block_y = 1
        launch.block_z = 1
    launch.shared_mem = 0

    extra_args = trtp.SymIntExprs(0)

    ptx_bytes = ptx_str.encode("utf-8")
    return kernel_name_str, ptx_bytes, launch, extra_args


def compile_cutedsl_kernel(
    spec: CuTeDSLSpec, config: Dict[str, Any]
) -> AOTMetadata:
    """Compile a CuTeDSLSpec into unified AOTMetadata.

    This is the tactic-manager entry-point.  It constructs synthetic 1-input /
    1-output TensorDesc stubs (shape [256], float32) to drive the sandbox run,
    delegates to ``aot_impl_cutedsl`` for the full pipeline, and wraps the result
    in ``AOTMetadata``.

    Args:
        spec:   CuTeDSLSpec carrying ``launch_fn`` (a ``@cute.jit`` function),
                optional ``configs``, etc.
        config: Single tactic configuration dict (e.g. ``{"BLOCK_SIZE": 128}``).

    Returns:
        AOTMetadata with ``backend="cutedsl"``, compiled PTX bytes, and launch params.
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
    kernel_name_bytes, ptx_bytes, launch, extra_args = aot_impl_cutedsl(
        qdp_symbol="cutedsl_compile",
        spec=spec,
        cfg=config,
        launch_fn=spec.launch_fn,
        host_args=host_args,
        inp_descs=inp_descs,
        out_descs=out_descs,
    )
    kernel_name = (
        kernel_name_bytes.decode("utf-8")
        if isinstance(kernel_name_bytes, bytes)
        else kernel_name_bytes
    )
    launch_params = _launch_params_from_trt(launch, extra_args, num_inputs=1, num_outputs=1)
    return AOTMetadata(binary=ptx_bytes, kernel_name=kernel_name, launch_params=launch_params, backend="cutedsl")
