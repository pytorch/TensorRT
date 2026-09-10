"""Public entry points for ``torch_tensorrt.kernels``.

Three functions, three paths into the same registration funnel:

* :func:`cuda_kernel_op` — declarative entry for CUDA C++ source. Reads
  a :class:`KernelSpec` and derives meta / eager / aot / schema, with
  override keyword arguments for cases outside the DSL.
* :func:`ptx_op` — escape hatch for pre-compiled PTX bytes (Triton output,
  cached NVRTC artifact). User supplies meta / eager / aot directly.
* :func:`cutile_op` — declarative entry for a ``@ct.kernel`` cuTile program.
  Compiles the kernel to PTX for you, reorders its parameters into TensorRT's
  launch order, and derives the AOT launch.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Mapping, Optional, Sequence, Union

from torch_tensorrt._features import ENABLED_FEATURES
from torch_tensorrt.dynamo.conversion._ConverterRegistry import ConverterPriority
from torch_tensorrt.kernels import _derive, _validation
from torch_tensorrt.kernels._cuda_python_spec import (
    CudaPythonSpec,
    _default_cuda_include_paths,
)
from torch_tensorrt.kernels._dsl import KernelSpec, ScalarInput

_LOGGER = logging.getLogger(__name__)


def _require_qdp_plugin() -> None:
    """Raise unless the installed TensorRT exposes Quick Deployable Plugins."""
    if not ENABLED_FEATURES.qdp_plugin:
        raise RuntimeError(
            "TensorRT QDP plugins are not available. "
            "Requires TensorRT >= 10.7.0 (and not 10.14.x)."
        )


def cuda_kernel_op(
    op_name: str,
    spec: KernelSpec,
    *,
    meta_fn: Optional[Callable[..., Any]] = None,
    eager_fn: Optional[Callable[..., Any]] = None,
    aot_fn: Optional[Callable[..., Any]] = None,
    schema: Optional[str] = None,
    supports_dynamic_shapes: bool = True,
    requires_output_allocator: bool = False,
    priority: ConverterPriority = ConverterPriority.STANDARD,
    capability_validator: Optional[Callable[..., Any]] = None,
) -> None:
    """Register a CUDA kernel as a TensorRT QDP plugin end-to-end.

    Two paths share one entry point:

    * **Declarative** — pass a fully-populated :class:`KernelSpec` and the
      meta fn, eager fn, AOT fn, and PyTorch schema are all derived for you.
      Covers Elementwise / Reduction kernels out of the box.
    * **Override** — pass any of ``meta_fn`` / ``eager_fn`` / ``aot_fn`` /
      ``schema`` as keyword arguments and the corresponding ``KernelSpec``
      fields become optional. Use this for shape-changing kernels,
      multi-output kernels, or anything outside the declarative DSL.

    Override rules (validated at registration time):

    * ``meta_fn`` provided → ``spec.outputs`` may be omitted.
    * ``eager_fn`` and ``aot_fn`` both provided → ``spec.geometry`` may be omitted.
    * ``schema`` provided → falls back to inferring from ``spec.inputs`` /
      ``spec.outputs`` if both exist, else from ``meta_fn`` type hints.

    The kernel must follow the calling convention
    ``(input_ptrs..., scalar_inputs..., extras..., output_ptrs...)``.
    """
    _require_qdp_plugin()

    # Late import to avoid circular imports and keep the decorator cheap.
    from torch_tensorrt.kernels._register import register_cuda_python_plugin

    _validation._validate_spec(
        spec,
        has_meta_fn=meta_fn is not None,
        has_eager_fn=eager_fn is not None,
        has_aot_fn=aot_fn is not None,
    )

    # Module-qualified call so tests can monkeypatch ``_derive._compile_kernel``.
    ptx, device, kernel_obj = _derive._compile_kernel(spec)

    final_meta = meta_fn if meta_fn is not None else _derive._make_meta_fn(spec)
    final_eager = (
        eager_fn
        if eager_fn is not None
        else _derive._make_eager_fn(spec, kernel_obj, device)
    )
    final_aot = aot_fn if aot_fn is not None else _derive._make_aot_fn(spec)

    if schema is not None:
        final_schema: Optional[str] = schema
    elif spec.inputs and spec.outputs:
        final_schema = _derive._build_schema(spec)
    else:
        # Let register_cuda_python_plugin fall back to _infer_schema(meta_fn).
        final_schema = None

    cuda_spec = CudaPythonSpec(
        kernel_source=spec.kernel_source,
        kernel_name=spec.kernel_name,
        aot_fn=final_aot,
        eager_fn=final_eager,
        include_paths=(
            list(spec.include_paths)
            if spec.include_paths is not None
            else _default_cuda_include_paths()
        ),
        compile_std=spec.compile_std,
        arch_override=spec.arch_override,
    )

    # A user-supplied aot_fn takes ownership of the AOT path; only the
    # auto-derived path needs the ScalarInput → JIT fallback because QDP
    # AOT extras don't currently support runtime float / bool scalars.
    if aot_fn is not None:
        use_aot = True
    else:
        use_aot = not any(
            isinstance(input_spec, ScalarInput) for input_spec in (spec.inputs or [])
        )

    register_cuda_python_plugin(
        op_name=op_name,
        spec=cuda_spec,
        meta_fn=final_meta,
        supports_dynamic_shapes=supports_dynamic_shapes,
        requires_output_allocator=requires_output_allocator,
        priority=priority,
        capability_validator=capability_validator,
        register_torch_op=True,
        schema=final_schema,
        precompiled_ptx=ptx,
        use_aot_if_available=use_aot,
    )
    _LOGGER.info("cuda_kernel_op '%s' registered (schema: %s)", op_name, final_schema)


def ptx_op(
    op_name: str,
    ptx: bytes,
    kernel_name: str,
    meta_fn: Callable[..., Any],
    eager_fn: Optional[Callable[..., Any]],
    aot_fn: Callable[..., Any],
    *,
    supports_dynamic_shapes: bool = False,
    requires_output_allocator: bool = False,
    priority: ConverterPriority = ConverterPriority.STANDARD,
    capability_validator: Optional[Callable[..., Any]] = None,
    schema: Optional[str] = None,
) -> None:
    """Register a pre-compiled PTX kernel as a TensorRT QDP plugin.

    Use this when the PTX comes from an external compiler (Triton, a cached
    NVRTC output, etc.) and NVRTC compilation should be skipped.
    """
    _require_qdp_plugin()

    from torch_tensorrt.kernels._register import register_cuda_python_plugin

    spec = CudaPythonSpec(
        kernel_source="",
        kernel_name=kernel_name,
        aot_fn=aot_fn,
        eager_fn=eager_fn,
    )
    register_cuda_python_plugin(
        op_name=op_name,
        spec=spec,
        meta_fn=meta_fn,
        supports_dynamic_shapes=supports_dynamic_shapes,
        requires_output_allocator=requires_output_allocator,
        priority=priority,
        capability_validator=capability_validator,
        register_torch_op=True,
        schema=schema,
        precompiled_ptx=ptx,
    )


def cutile_op(
    op_name: str,
    kernel: Any,
    signature: Mapping[str, Any],
    meta_fn: Callable[..., Any],
    *,
    grid: Optional[Callable[..., Any]] = None,
    constants: Optional[Mapping[str, Union[bool, int, float]]] = None,
    ndim: int = 1,
    block_size: Optional[Union[int, Sequence[int]]] = None,
    aot_fn: Optional[Callable[..., Any]] = None,
    eager_fn: Optional[Callable[..., Any]] = None,
    arch_override: Optional[str] = None,
    max_ptx_version: Optional[int] = None,
    supports_dynamic_shapes: bool = True,
    requires_output_allocator: bool = False,
    priority: ConverterPriority = ConverterPriority.STANDARD,
    capability_validator: Optional[Callable[..., Any]] = None,
    schema: Optional[str] = None,
) -> None:
    """Register a ``@ct.kernel`` cuTile program as a TensorRT AOT QDP plugin.

    The cuTile analogue of :func:`cuda_kernel_op`: compiles the kernel once with
    ``cuda.tile.compilation.export_kernel``, permutes the compiled PTX into
    TensorRT's launch order, and hands the result to :func:`ptx_op`.

    A cuTile kernel declares its *array* parameters first, inputs then outputs,
    followed by its ``ct.Constant`` parameters::

        @ct.kernel
        def relu(x, out, tile_size: ct.Constant[int]): ...

    ``signature`` names the arrays in that order; ``constants`` supplies the
    ``ct.Constant`` values. The cuTile entry-point documentation explains why
    the PTX has to be permuted.

    Args:
        op_name: qualified op name ``"ns::name"``. After registration
            ``torch.ops.ns.name`` exists and is lowered to the QDP plugin
            during ``torch_tensorrt.compile``.
        kernel: the ``@ct.kernel`` program object.
        signature: the kernel's array parameters in declaration order, inputs
            then outputs, mapped to their element type — e.g.
            ``{"x": "fp32", "out": "fp32"}``. Values may be a
            :class:`torch.dtype` or its name (``"float32"``, ``"fp32"``).
        meta_fn: the fake / meta kernel used for shape+dtype inference. The
            PyTorch schema is inferred from its type hints unless ``schema`` is
            passed.
        grid: ``callable(inputs, outputs) -> int | tuple`` returning the launch
            grid in tiles, where ``inputs`` / ``outputs`` are ``trtp.TensorDesc``
            objects (use ``.shape_expr`` for symbolic dims). Up to three dims
            become ``grid_x`` / ``grid_y`` / ``grid_z``. Required unless
            ``aot_fn`` is given, which replaces it.
        constants: ``ct.Constant`` parameter values, in declaration order,
            baked into the compiled symbol — e.g. ``{"tile_size": 256}``. The
            AOT launch path cannot supply runtime scalars, so every non-array
            parameter must be a constant.
        ndim: the rank each array is compiled for. Defaults to 1, matching
            kernels written against a flattened view; a rank-1 array's extent is
            the tensor's element count, so such an op accepts any input shape.
        block_size: one to three block dimensions. Defaults to the ``.reqntid``
            the compiled kernel declares, which is authoritative. Pass this only
            for kernels that declare none and only with the derived ``grid`` path.
        aot_fn: optional replacement for the derived AOT launch
            (``callable(inputs, outputs, tactic) -> (KernelLaunchParams,
            extra_args)``), used instead of ``grid``. The PTX is still permuted,
            so the override must emit extra arguments in cuTile ABI order:
            every input array's extents and strides, then every output's, in a
            ``trtp.SymIntExprs`` container. The wrapper checks the container,
            item type, and count, but the caller owns the values and order. Its
            block dimensions must be static and exactly match the compiled
            kernel's ``.reqntid`` when one is declared; dynamic shared memory
            must be zero.
        eager_fn: optional CUDA eager implementation registered on the torch
            op. Omit if the op is only used through ``torch_tensorrt.compile``.
        arch_override: target architecture such as ``"sm_100"``. Defaults to the
            current device's compute capability. An override for a different
            architecture is a cross-compile, so its PTX cannot be driver-checked
            locally and must be verified on the target device.
        max_ptx_version: explicit ISA ceiling for the embedded PTX, as a
            ``90``-style int (``.version 9.0``). The compiler's header is left
            unchanged by default. A capped candidate is verified by the driver.
        capability_validator: optional extra predicate gating conversion. It is
            combined with the dtype check derived from ``signature`` — both must
            pass for the op to be lowered to the plugin.

    Raises:
        ValueError: if ``signature`` disagrees with ``meta_fn``'s arity, names a
            dtype cuTile cannot be compiled for, overlaps ``constants``, or if
            neither / both of ``grid`` and ``aot_fn`` are given.
        RuntimeError: if the compiled kernel's PTX parameter list does not match
            the signature — most often a rank mismatch or a runtime scalar the
            AOT launch path cannot supply.

    .. note::
        Compiles a single PTX for the dtypes in ``signature`` and the values in
        ``constants``. Inputs of other dtypes are declined at conversion time and
        left to PyTorch. Multi-config autotuning is follow-up work.
    """
    _require_qdp_plugin()

    from torch_tensorrt.kernels import _cutile
    from torch_tensorrt.kernels._register import _torch_op_already_registered

    schema_info = _cutile.analyze_tensor_schema(op_name, meta_fn, schema)
    raw_constants = {} if constants is None else constants

    if eager_fn is not None and not callable(eager_fn):
        raise ValueError(f"cutile_op '{op_name}' eager_fn must be callable.")
    if capability_validator is not None and not callable(capability_validator):
        raise ValueError(
            f"cutile_op '{op_name}' capability_validator must be callable."
        )

    def _ensure_name_available() -> None:
        if _torch_op_already_registered(op_name):
            raise ValueError(
                f"cutile_op '{op_name}' is already registered with PyTorch; "
                "choose a unique op_name."
            )

    # Validate before compiling: nothing here needs the kernel built, and every
    # rule it enforces would otherwise surface as wrong numbers, not an error.
    layout = _cutile.validate_cutile_config(
        op_name,
        signature,
        raw_constants,
        (schema_info.num_inputs, schema_info.num_outputs),
        default_ndim=ndim,
        input_names=schema_info.input_names,
    )
    constants_dict = dict(raw_constants)

    if aot_fn is None:
        if not callable(grid):
            raise ValueError(
                f"cutile_op '{op_name}' needs a callable grid= to derive the "
                "launch, or a callable aot_fn= to replace it."
            )
    else:
        if not callable(aot_fn):
            raise ValueError(f"cutile_op '{op_name}' aot_fn must be callable.")
        if grid is not None:
            raise ValueError(
                f"cutile_op '{op_name}' was given both grid= and aot_fn=; the "
                "custom aot_fn builds the whole launch, so grid would be ignored."
            )
        if block_size is not None:
            raise ValueError(
                f"cutile_op '{op_name}' block_size is only used with the derived "
                "grid launch and would be ignored by the custom aot_fn."
            )

    # Fail closed instead of letting the shared legacy registrar treat an
    # unrelated existing Torch op as an idempotent registration.
    _ensure_name_available()
    ptx, kernel_name, reqntid = _cutile.compile_cutile_to_ptx(
        op_name, kernel, layout, constants_dict, arch_override, max_ptx_version
    )

    if aot_fn is None:
        assert grid is not None
        aot_fn = _cutile.make_aot_fn(
            op_name,
            layout,
            grid,
            _cutile.resolve_block_dims(op_name, kernel_name, reqntid, block_size),
        )
    else:
        aot_fn = _cutile.make_checked_aot_fn(
            op_name, kernel_name, layout, reqntid, aot_fn
        )

    # Compilation can execute arbitrary toolchain code, so check again in case
    # it registered the name re-entrantly before handing off to ``ptx_op``.
    _ensure_name_available()

    # Everything past this point is "register pre-compiled PTX", which is
    # exactly what ptx_op is; the only cuTile-specific addition is the dtype
    # gate derived from the signature.
    ptx_op(
        op_name,
        ptx,
        kernel_name,
        meta_fn=meta_fn,
        eager_fn=eager_fn,
        aot_fn=aot_fn,
        supports_dynamic_shapes=supports_dynamic_shapes,
        requires_output_allocator=requires_output_allocator,
        priority=priority,
        capability_validator=_cutile.make_dtype_capability_validator(
            op_name, layout, capability_validator
        ),
        schema=schema_info.schema,
    )
    _LOGGER.info("cutile_op '%s' registered (kernel: %s)", op_name, kernel_name)
