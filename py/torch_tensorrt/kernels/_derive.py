"""Derivation engine behind :func:`cuda_kernel_op`.

Given a :class:`KernelSpec`, this module builds the runtime artifacts that
:func:`cuda_kernel_op` (in ``_ops``) consumes:

* a PyTorch meta/fake impl from each :class:`OutputDecl.shape`,
* a PyTorch eager impl that launches the compiled kernel via cuda-python,
* a TensorRT AOT impl with symbolic launch params and extras,
* a PyTorch op schema string,
* NVRTC-compiled PTX + a loadable kernel handle.
"""

from __future__ import annotations

import logging
import textwrap
from typing import Any, Callable, Dict, List, Sequence, Tuple

import torch

from torch_tensorrt.kernels._cuda_python_spec import _default_cuda_include_paths
from torch_tensorrt.kernels._dsl import (
    Custom,
    DimSize,
    Elementwise,
    ExtraArg,
    InputDecl,
    KernelSpec,
    Numel,
    OutputDecl,
    ReduceDims,
    Reduction,
    SameAs,
    ScalarInput,
)

_LOGGER = logging.getLogger(__name__)


# =============================================================================
# ShapeRel evaluation (torch + TRT variants)
# =============================================================================


def _resolve_axis(axis: int, ndim: int) -> int:
    return axis if axis >= 0 else ndim + axis


def _tensor_input_decls(inputs: Sequence[Any]) -> List[InputDecl]:
    """Return only the tensor inputs, preserving order."""
    return [d for d in inputs if isinstance(d, InputDecl)]


def _resolve_input_ref(
    ref: Any, tensor_decls: Sequence[InputDecl], *, where: str
) -> int:
    """Resolve a ``SameAs`` / ``ReduceDims`` input reference to an integer
    position into the tensor-only input list.

    Accepts either an int index or the name of a tensor input.
    """
    if isinstance(ref, int) and not isinstance(ref, bool):
        if ref < 0 or ref >= len(tensor_decls):
            raise ValueError(
                f"{where}: input_idx={ref} out of range; spec has "
                f"{len(tensor_decls)} tensor inputs"
            )
        return ref
    if isinstance(ref, str):
        for i, d in enumerate(tensor_decls):
            if d.name == ref:
                return i
        names = [d.name for d in tensor_decls]
        raise ValueError(
            f"{where}: input_idx={ref!r} is not a tensor input. "
            f"Known tensor inputs: {names}"
        )
    raise TypeError(f"{where}: input_idx must be int or str, got {type(ref).__name__}")


def _torch_output_shape_dtype(
    decl: OutputDecl,
    input_tensors: List[torch.Tensor],
    input_decls: List[InputDecl],
) -> Tuple[Tuple[int, ...], torch.dtype]:
    """Return (shape, dtype) for one output given concrete input tensors.

    ``input_decls`` and ``input_tensors`` must be aligned lists of tensor-only
    inputs; scalar inputs are filtered out by the caller.
    """
    rel = decl.shape
    name_to_idx = {d.name: i for i, d in enumerate(input_decls)}

    if isinstance(rel, SameAs):
        idx = _resolve_input_ref(
            rel.input_idx, input_decls, where=f"OutputDecl {decl.name!r}"
        )
        src = input_tensors[idx]
        dtype = (
            input_tensors[name_to_idx[decl.dtype_from]].dtype
            if decl.dtype_from is not None
            else src.dtype
        )
        return tuple(src.shape), dtype

    if isinstance(rel, ReduceDims):
        idx = _resolve_input_ref(
            rel.input_idx, input_decls, where=f"OutputDecl {decl.name!r}"
        )
        src = input_tensors[idx]
        dims = {_resolve_axis(d, src.ndim) for d in rel.dims}
        new_shape = []
        for i, s in enumerate(src.shape):
            if i in dims:
                if rel.keepdim:
                    new_shape.append(1)
            else:
                # Preserve SymInts under FakeTensorMode; do not call int(s)
                # which would concretize a SymInt to its hint value.
                new_shape.append(s)
        dtype = (
            input_tensors[name_to_idx[decl.dtype_from]].dtype
            if decl.dtype_from is not None
            else src.dtype
        )
        return tuple(new_shape), dtype

    raise TypeError(f"Unsupported ShapeRel: {rel!r}")


# =============================================================================
# Launch geometry: concrete (eager) + symbolic (aot)
# =============================================================================


def _cdiv_int(a: int, b: int) -> int:
    return (a + b - 1) // b


def _compute_eager_launch(
    geom: Any,
    outputs: List[torch.Tensor],
    inputs: List[torch.Tensor],
) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
    """Return (grid, block) both padded to 3-tuples for cuda.core.LaunchConfig."""
    if isinstance(geom, Elementwise):
        out = outputs[0]
        if geom.layout == "flat":
            n = int(out.numel())
            bx = int(geom.block[0])
            grid = (_cdiv_int(n, bx), 1, 1)
            block = (bx, 1, 1)
            return grid, block

        # layout == "nd"
        block_dims = tuple(int(b) for b in geom.block)
        out_shape = tuple(int(s) for s in out.shape)
        b_ndim = len(block_dims)
        if len(out_shape) < b_ndim:
            raise ValueError(
                f"Elementwise(layout='nd', block={block_dims}) needs output ndim >= "
                f"{b_ndim}, got output shape {out_shape}"
            )
        # block[0] -> innermost axis; block[i] -> axis out_shape[-(i+1)]
        inner_axes = [out_shape[-(i + 1)] for i in range(b_ndim)]
        grid_xyz = [_cdiv_int(axis, block_dims[i]) for i, axis in enumerate(inner_axes)]
        while len(grid_xyz) < 3:
            grid_xyz.append(1)
        outer_axes = out_shape[: len(out_shape) - b_ndim]
        if outer_axes:
            prod = 1
            for s in outer_axes:
                prod *= s
            grid_xyz[2] *= prod
        block_padded = tuple(block_dims) + (1,) * (3 - b_ndim)
        return tuple(grid_xyz), block_padded  # type: ignore[return-value]

    if isinstance(geom, Reduction):
        # One block per "row" = numel(input[0]) / product(reduce_dim_sizes).
        # This is independent of whether the output is smaller (sum/max) or
        # same-shape (softmax/layernorm): the row count is a property of the
        # *input*'s non-reduced axes.
        src = inputs[0]
        reduce_axes = {_resolve_axis(d, src.ndim) for d in geom.reduce_dims}
        rows = 1
        for i, s in enumerate(src.shape):
            if i not in reduce_axes:
                rows *= int(s)
        grid = (rows, 1, 1)
        block = (int(geom.block_size), 1, 1)
        return grid, block

    if isinstance(geom, Custom):
        raise RuntimeError(
            "Custom geometry has no eager equivalent. Pass eager_fn= to "
            "cuda_kernel_op() instead of using Custom geometry."
        )

    raise TypeError(f"Unsupported Geometry: {geom!r}")


def _compute_aot_launch(
    geom: Any,
    inputs_desc: Any,
    outputs_desc: Any,
) -> Any:
    """Return a ``trtp.KernelLaunchParams`` matching the eager launch."""
    import tensorrt.plugin as trtp

    params = trtp.KernelLaunchParams()
    params.shared_mem = 0

    if isinstance(geom, Elementwise):
        out = outputs_desc[0]
        if geom.layout == "flat":
            n = out.shape_expr.numel()
            bx = int(geom.block[0])
            params.grid_x = trtp.cdiv(n, bx)
            params.block_x = bx
            return params

        block_dims = tuple(int(b) for b in geom.block)
        out_shape = out.shape_expr
        out_ndim = len(out_shape)
        b_ndim = len(block_dims)
        if out_ndim < b_ndim:
            raise ValueError(
                f"Elementwise(layout='nd', block={block_dims}) needs output ndim >= "
                f"{b_ndim}, got ndim={out_ndim}"
            )

        # Map block[0] -> last axis, etc. Fill grid_x/y/z as available.
        grid_axes = [out_shape[out_ndim - 1 - i] for i in range(b_ndim)]
        if b_ndim >= 1:
            params.grid_x = trtp.cdiv(grid_axes[0], block_dims[0])
            params.block_x = block_dims[0]
        if b_ndim >= 2:
            params.grid_y = trtp.cdiv(grid_axes[1], block_dims[1])
            params.block_y = block_dims[1]
        if b_ndim >= 3:
            params.grid_z = trtp.cdiv(grid_axes[2], block_dims[2])
            params.block_z = block_dims[2]

        outer_ndim = out_ndim - b_ndim
        if outer_ndim > 0:
            prod = out_shape[0]
            for i in range(1, outer_ndim):
                prod = prod * out_shape[i]
            if b_ndim < 3:
                params.grid_z = prod
            else:
                params.grid_z = params.grid_z * prod
        return params

    if isinstance(geom, Reduction):
        src = inputs_desc[0]
        src_shape = src.shape_expr
        src_ndim = len(src_shape)
        reduce_axes = {_resolve_axis(d, src_ndim) for d in geom.reduce_dims}
        rows = None
        for i in range(src_ndim):
            if i in reduce_axes:
                continue
            rows = src_shape[i] if rows is None else rows * src_shape[i]
        params.grid_x = rows if rows is not None else 1
        params.block_x = int(geom.block_size)
        return params

    if isinstance(geom, Custom):
        raise RuntimeError("Custom geometry routed through dedicated path")

    raise TypeError(f"Unsupported Geometry: {geom!r}")


# =============================================================================
# Extras packing
# =============================================================================


def _pack_extras_eager(
    extras: List[ExtraArg],
    input_by_name: Dict[str, torch.Tensor],
) -> List[int]:
    out = []
    for e in extras:
        src = input_by_name[e.input_name]
        if isinstance(e, Numel):
            out.append(int(src.numel()))
        elif isinstance(e, DimSize):
            axis = _resolve_axis(e.axis, src.ndim)
            out.append(int(src.shape[axis]))
        else:
            raise TypeError(f"Unsupported ExtraArg: {e!r}")
    return out


def _pack_extras_aot(extras: List[ExtraArg], input_desc_by_name: Dict[str, Any]) -> Any:
    import tensorrt.plugin as trtp

    if not extras:
        return trtp.SymIntExprs(0)

    sym_exprs = trtp.SymIntExprs(len(extras))
    for i, e in enumerate(extras):
        src = input_desc_by_name[e.input_name]
        if isinstance(e, Numel):
            sym_exprs[i] = trtp.SymInt32(src.shape_expr.numel())
        elif isinstance(e, DimSize):
            axis = _resolve_axis(e.axis, len(src.shape_expr))
            sym_exprs[i] = trtp.SymInt32(src.shape_expr[axis])
        else:
            raise TypeError(f"Unsupported ExtraArg: {e!r}")
    return sym_exprs


# =============================================================================
# Schema + dynamic function builders
# =============================================================================


_PYTYPE_TO_SCHEMA = {float: "float", int: "int", bool: "bool"}


def _schema_type_for_input(decl: Any) -> str:
    if isinstance(decl, ScalarInput):
        if decl.py_type not in _PYTYPE_TO_SCHEMA:
            raise ValueError(
                f"ScalarInput {decl.name!r}: py_type must be float, int, or bool, "
                f"got {decl.py_type!r}"
            )
        return _PYTYPE_TO_SCHEMA[decl.py_type]
    return "Tensor"


def _annot_type_for_input(decl: Any) -> Any:
    if isinstance(decl, ScalarInput):
        return decl.py_type
    return torch.Tensor


def _build_schema(spec: KernelSpec) -> str:
    assert spec.inputs is not None and spec.outputs is not None
    args = ", ".join(f"{_schema_type_for_input(d)} {d.name}" for d in spec.inputs)
    if len(spec.outputs) == 1:
        ret = "Tensor"
    else:
        ret = "(" + ", ".join("Tensor" for _ in spec.outputs) + ")"
    return f"({args}) -> {ret}"


def _make_positional_fn(
    fn: Callable[..., Any],
    input_decls: Sequence[Any],
    *,
    num_outputs: int = 1,
) -> Callable[..., Any]:
    """Wrap ``fn`` so it carries a proper positional signature.

    ``torch.library.Library.impl`` and ``register_fake`` both introspect the
    wrapped callable; a generic ``*args`` function wouldn't satisfy them.
    The synthesized wrapper types each parameter per its :class:`InputDecl` /
    :class:`ScalarInput` kind so the torch dispatcher matches the schema.

    ``num_outputs`` controls the return annotation: ``1`` produces
    ``-> torch.Tensor`` (matching a single-output schema), >1 produces
    ``-> Tuple[torch.Tensor, ...]`` (matching a multi-output schema).
    """
    param_names = [d.name for d in input_decls]
    sig_pieces = []
    annotations = {}
    for d in input_decls:
        if isinstance(d, ScalarInput):
            sig_pieces.append(f"{d.name}: '{d.py_type.__name__}'")
            annotations[d.name] = d.py_type
        else:
            sig_pieces.append(f"{d.name}: 'torch.Tensor'")
            annotations[d.name] = torch.Tensor
    sig_src = ", ".join(sig_pieces)
    if num_outputs == 1:
        return_src = "'torch.Tensor'"
        return_annot: Any = torch.Tensor
    else:
        return_src = "'typing.Tuple[torch.Tensor, ...]'"
        return_annot = Tuple[torch.Tensor, ...]
    body = textwrap.dedent(f"""
        def _wrapper({sig_src}) -> {return_src}:
            return _fn({", ".join(param_names)})
        """)
    ns: Dict[str, Any] = {"_fn": fn, "torch": torch, "typing": __import__("typing")}
    exec(compile(body, "<cuda_kernel_op>", "exec"), ns)
    wrapper: Callable[..., Any] = ns["_wrapper"]
    wrapper.__annotations__ = dict(annotations)
    wrapper.__annotations__["return"] = return_annot
    return wrapper


# =============================================================================
# Kernel compilation (wraps cuda-python)
# =============================================================================


def _compile_kernel(spec: KernelSpec) -> Tuple[bytes, Any, Any]:
    """Compile spec.kernel_source to PTX + get a loadable kernel object."""
    try:
        from cuda.core import (
            Device,
            Program,
            ProgramOptions,
        )
    except ImportError:
        from cuda.core.experimental import (
            Device,
            Program,
            ProgramOptions,
        )

    device = Device()
    device.set_current()
    arch = spec.arch_override if spec.arch_override else f"sm_{device.arch}"
    include_paths = (
        list(spec.include_paths)
        if spec.include_paths is not None
        else _default_cuda_include_paths()
    )
    options = ProgramOptions(
        std=spec.compile_std, arch=arch, include_path=include_paths
    )
    program = Program(spec.kernel_source, code_type="c++", options=options)
    module = program.compile("ptx", name_expressions=(spec.kernel_name,))
    ptx: bytes = module.code
    kernel = module.get_kernel(spec.kernel_name)
    return ptx, device, kernel


# =============================================================================
# Eager / meta / aot function factories
# =============================================================================


def _split_inputs(
    all_args: Sequence[Any], input_specs: Sequence[Any]
) -> Tuple[List[torch.Tensor], List[InputDecl], Dict[str, object]]:
    """Partition positional args into (tensor_inputs, scalar_values) aligned
    with the tensor InputDecls and ScalarInputs respectively.
    """
    tensors: List[torch.Tensor] = []
    tensor_decls: List[InputDecl] = []
    scalars: Dict[str, object] = {}
    for decl, val in zip(input_specs, all_args):
        if isinstance(decl, ScalarInput):
            scalars[decl.name] = val
        else:
            tensors.append(val)
            tensor_decls.append(decl)
    return tensors, tensor_decls, scalars


def _make_meta_fn(spec: KernelSpec) -> Callable[..., Any]:
    assert spec.inputs is not None and spec.outputs is not None
    input_specs = list(spec.inputs)
    output_decls = list(spec.outputs)

    def _meta(*args: Any) -> Any:
        tensors, tensor_decls, _scalars = _split_inputs(args, input_specs)
        device = tensors[0].device if tensors else torch.device("cuda")
        outs = []
        for odecl in output_decls:
            shape, dtype = _torch_output_shape_dtype(odecl, tensors, tensor_decls)
            outs.append(torch.empty(shape, dtype=dtype, device=device))
        return outs[0] if len(outs) == 1 else tuple(outs)

    return _make_positional_fn(_meta, input_specs, num_outputs=len(output_decls))


def _make_eager_fn(
    spec: KernelSpec, kernel_obj: Any, device: Any
) -> Callable[..., Any]:
    try:
        from cuda.core import LaunchConfig
        from cuda.core import launch as cuda_launch
    except ImportError:
        from cuda.core.experimental import LaunchConfig
        from cuda.core.experimental import launch as cuda_launch

    assert spec.inputs is not None and spec.outputs is not None
    input_specs = list(spec.inputs)
    output_decls = list(spec.outputs)
    extras = list(spec.extras)

    class _PTStream:
        def __cuda_stream__(self) -> Tuple[int, int]:  # noqa: D401
            return (0, torch.cuda.current_stream().cuda_stream)

    def _eager(*args: Any) -> Any:
        tensors, tensor_decls, _scalars = _split_inputs(args, input_specs)

        outs: List[torch.Tensor] = []
        for odecl in output_decls:
            shape, dtype = _torch_output_shape_dtype(odecl, tensors, tensor_decls)
            outs.append(torch.empty(shape, dtype=dtype, device=tensors[0].device))

        grid, block = _compute_eager_launch(spec.geometry, outs, tensors)

        input_by_name = {d.name: t for d, t in zip(tensor_decls, tensors)}
        extra_vals = _pack_extras_eager(extras, input_by_name)

        # Kernel arg order: inputs in declaration order (ptr for tensors,
        # value for scalars), then extras, then output pointers.
        arg_list: List[Any] = []
        for decl, val in zip(input_specs, args):
            if isinstance(decl, ScalarInput):
                arg_list.append(_coerce_scalar(val, decl.py_type))
            else:
                arg_list.append(val.data_ptr())
        arg_list.extend(extra_vals)
        arg_list.extend(t.data_ptr() for t in outs)

        stream = device.create_stream(_PTStream())
        cuda_launch(stream, LaunchConfig(grid=grid, block=block), kernel_obj, *arg_list)
        return outs[0] if len(outs) == 1 else tuple(outs)

    return _make_positional_fn(_eager, input_specs, num_outputs=len(output_decls))


def _coerce_scalar(value: Any, py_type: Any) -> Any:
    """Convert a Python scalar to the ctypes type that cuda.core needs to
    forward it by value to the kernel.
    """
    import ctypes

    if py_type is float:
        return ctypes.c_float(float(value))
    if py_type is int:
        return ctypes.c_int32(int(value))
    if py_type is bool:
        return ctypes.c_bool(bool(value))
    raise TypeError(f"Unsupported ScalarInput.py_type: {py_type!r}")


def _make_aot_fn(spec: KernelSpec) -> Callable[..., Any]:
    assert spec.inputs is not None
    tensor_input_decls = _tensor_input_decls(spec.inputs)
    extras = list(spec.extras)

    def _aot(inputs: Any, outputs: Any, tactic: Any) -> Any:
        # TRT plugin inputs are the tensor-typed args only. The trtp layer
        # slots them in by (tensor) arg order, so our tensor_input_decls list
        # aligns with inputs positionally.
        if isinstance(spec.geometry, Custom):
            return spec.geometry.fn(inputs, outputs, tactic)
        params = _compute_aot_launch(spec.geometry, inputs, outputs)
        input_desc_by_name = {d.name: td for d, td in zip(tensor_input_decls, inputs)}
        extra_exprs = _pack_extras_aot(extras, input_desc_by_name)
        return params, extra_exprs

    return _aot
