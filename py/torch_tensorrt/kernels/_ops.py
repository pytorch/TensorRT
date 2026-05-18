"""Public entry points for ``torch_tensorrt.kernels``.

Two functions, two paths into the same registration funnel:

* :func:`cuda_kernel_op` — declarative entry for CUDA C++ source. Reads
  a :class:`KernelSpec` and derives meta / eager / aot / schema, with
  override keyword arguments for cases outside the DSL.
* :func:`ptx_op` — escape hatch for pre-compiled PTX bytes (Triton output,
  cached NVRTC artifact). User supplies meta / eager / aot directly.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

from torch_tensorrt._features import ENABLED_FEATURES
from torch_tensorrt.dynamo.conversion._ConverterRegistry import ConverterPriority
from torch_tensorrt.kernels import _derive, _validation
from torch_tensorrt.kernels._cuda_python_spec import (
    CudaPythonSpec,
    _default_cuda_include_paths,
)
from torch_tensorrt.kernels._dsl import KernelSpec, ScalarInput

_LOGGER = logging.getLogger(__name__)


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
    if not ENABLED_FEATURES.qdp_plugin:
        raise RuntimeError(
            "TensorRT QDP plugins are not available. "
            "Requires TensorRT >= 10.7.0 (and not 10.14.x)."
        )

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
    eager_fn: Callable[..., Any],
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
    if not ENABLED_FEATURES.qdp_plugin:
        raise RuntimeError(
            "TensorRT QDP plugins are not available. "
            "Requires TensorRT >= 10.7.0 (and not 10.14.x)."
        )

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
