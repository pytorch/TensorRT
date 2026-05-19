import logging
import typing
import uuid
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import numpy.typing as npt
import tensorrt as trt
import torch
from torch.fx.node import Argument, Node, Target

from torch_tensorrt._features import needs_qdp_plugin
from torch_tensorrt.dynamo._settings import CompilationSettings
from torch_tensorrt.dynamo.conversion._ConversionContext import ConversionContext
from torch_tensorrt.dynamo.conversion._ConverterRegistry import (
    DYNAMO_CONVERTERS,
    ConverterPriority,
    DynamoConverterImplSignature,
    dynamo_tensorrt_converter,
)
from torch_tensorrt.dynamo.conversion.converter_utils import get_trt_tensor

_LOGGER: logging.Logger = logging.getLogger(__name__)


def _coerce_scalar_plugin_attr(value: Any, arg_type: torch._C.Type) -> Any:
    """Convert FX/Numpy scalar constants to Python values for QDP attributes."""
    if arg_type.isSubtypeOf(torch._C.FloatType.get()):
        return float(_unwrap_scalar_attr(value))
    if arg_type.isSubtypeOf(torch._C.IntType.get()):
        return int(_unwrap_scalar_attr(value))
    if arg_type.isSubtypeOf(torch._C.BoolType.get()):
        return bool(_unwrap_scalar_attr(value))
    if arg_type.isSubtypeOf(torch._C.StringType.get()):
        return str(_unwrap_scalar_attr(value))
    return value


def _coerce_plugin_attr_for_qdp(value: Any, attr_annotation: Any) -> Any:
    """Convert Python scalars to the serialized type expected by QDP."""
    if _is_numpy_attr_annotation(attr_annotation):
        return np.asarray(
            _unwrap_scalar_attr(value), dtype=_numpy_attr_dtype(attr_annotation)
        )
    return value


_PYTHON_SCALAR_TO_NUMPY_DTYPE = {
    float: np.float64,
    int: np.int64,
    bool: np.bool_,
}


def _patch_trtp_scalar_attr_roundtrip() -> None:
    """Patch ``tensorrt.plugin``'s scalar-attribute reconstruction.

    ``_TemplatePluginCreator.create_plugin`` rebuilds a Python scalar attr via
    ``attr_type_annot(f.data)`` (e.g. ``float(f.data)``). The serialize path
    stores Python scalars as ``np.array([value])`` (1-d size-1) and TRT's
    C++ PluginField construction further promotes any input ndarray to 1-d,
    so ``f.data`` is always 1-d on read. ``float(np.array([0.2]))`` raises
    "only 0-dimensional arrays can be converted to Python scalars", which
    surfaces as a converter failure on any plugin that declares scalar attrs
    via ``@trtp.register`` (e.g. ``b: float, a: int``).

    PluginField data is also immutable on the Python side, so we can't fix
    the shape before the unpatched code reads it. Instead: route around the
    broken ``attr_type_annot(f.data)`` line by temporarily promoting the
    scalar annotation to ``npt.NDArray[<corresponding dtype>]`` — that
    branches into ``.astype()``, which handles 1-d arrays fine — then
    restore the annotation and unwrap the resulting 1-d arrays back to the
    Python scalar types the descriptor/impl expects. Applied once, no-op if
    the upstream bindings are ever fixed.
    """
    try:
        from tensorrt_bindings.plugin import _lib as _trtp_lib
        from tensorrt_bindings.plugin._utils import _is_numpy_array
    except ImportError:
        return

    creator_cls = getattr(_trtp_lib, "_TemplatePluginCreator", None)
    if creator_cls is None or getattr(creator_cls, "_torch_trt_scalar_patched", False):
        return

    orig_create_plugin = creator_cls.create_plugin

    def _patched_create_plugin(
        self: Any,
        name: str,
        namespace: str,
        fc: Any,
        phase: Any,
        qpcr: Any = None,
    ) -> Any:
        from tensorrt_bindings.plugin._lib import QDP_REGISTRY

        desc = QDP_REGISTRY.get(f"{namespace}::{name}")
        if desc is None:
            return orig_create_plugin(self, name, namespace, fc, phase, qpcr)

        scalar_attrs: dict[str, type] = {}
        for f in fc:
            ann = desc.input_attrs.get(f.name)
            if ann is None or _is_numpy_array(ann):
                continue
            if not isinstance(ann, type):
                continue
            if ann in _PYTHON_SCALAR_TO_NUMPY_DTYPE:
                scalar_attrs[f.name] = ann

        if not scalar_attrs:
            return orig_create_plugin(self, name, namespace, fc, phase, qpcr)

        saved_annotations = {n: desc.input_attrs[n] for n in scalar_attrs}
        for n, ann in scalar_attrs.items():
            # mypy reads ``npt.NDArray[X]`` as a static type form, but X here
            # is a runtime value pulled from the dtype lookup table.
            desc.input_attrs[n] = npt.NDArray[_PYTHON_SCALAR_TO_NUMPY_DTYPE[ann]]  # type: ignore[valid-type]
        try:
            plg = orig_create_plugin(self, name, namespace, fc, phase, qpcr)
        finally:
            for n, ann in saved_annotations.items():
                desc.input_attrs[n] = ann

        # Unwrap the 1-d size-1 ndarrays the promoted path produced back to
        # the Python scalar types the descriptor's annotations declared, so
        # the user's `@trtp.register` / `@trtp.impl` bodies receive what
        # they signed up for.
        for n, ann in scalar_attrs.items():
            value = plg.attrs.get(n)
            if isinstance(value, np.ndarray) and value.size == 1:
                plg.attrs[n] = ann(value.reshape(()).item())

        return plg

    creator_cls.create_plugin = _patched_create_plugin
    creator_cls._torch_trt_scalar_patched = True


_patch_trtp_scalar_attr_roundtrip()


def _is_numpy_attr_annotation(annotation: Any) -> bool:
    return annotation is np.ndarray or typing.get_origin(annotation) is np.ndarray


def _numpy_attr_dtype(annotation: Any) -> np.dtype:
    if annotation is np.ndarray:
        return np.dtype(object)
    dtype_arg = typing.get_args(annotation)[1]
    dtype_args = typing.get_args(dtype_arg)
    if not dtype_args:
        raise TypeError(f"Could not infer NumPy dtype from annotation {annotation!r}")
    return np.dtype(dtype_args[0])


def _unwrap_scalar_attr(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if value.size != 1:
            raise ValueError(
                "Expected scalar plugin attribute, got ndarray with shape"
                f" {value.shape}"
            )
        return value.reshape(()).item()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _generate_plugin_converter(
    namespace: str,
    op_name: str,
    overload: Optional[str] = None,
    capability_validator: Optional[Callable[[Node, CompilationSettings], bool]] = None,
    priority: ConverterPriority = ConverterPriority.STANDARD,
    supports_dynamic_shapes: bool = False,
    requires_output_allocator: bool = False,
    use_aot_if_available: bool = True,
) -> DynamoConverterImplSignature:
    try:
        import tensorrt.plugin as trtp

    except ImportError as e:
        raise RuntimeError(
            "Unable to import TensorRT plugin. TensorRT version must be 10.7.0 or"
            " higher to support for Triton based TensorRT plugins"
        )
    from tensorrt.plugin._lib import QDP_REGISTRY

    torch_target = getattr(getattr(torch.ops, namespace), op_name)
    overload_str = overload if overload else ""
    overload_name = overload_str if overload else "default"
    torch_overload = getattr(torch_target, overload_name)
    assert f"{namespace}::{op_name}" in QDP_REGISTRY, (
        f"Could not find a tensorrt plugin registered for op {namespace}::{op_name},"
        " unable to generate converter"
    )
    torch_schema = torch_target._schemas[overload_str]

    use_aot_plugin = use_aot_if_available

    if use_aot_if_available:
        desc = QDP_REGISTRY[f"{namespace}::{op_name}"]
        if desc.aot_impl_func is None:
            use_aot_plugin = False
            _LOGGER.debug(
                f"AOT impl func not found for {namespace}::{op_name}, use JIT plugin"
                " instead"
            )

    def custom_kernel_converter(
        ctx: ConversionContext,
        target: Target,
        args: Tuple[Argument, ...],
        kwargs: Dict[str, Argument],
        name: str,
    ) -> Union[trt.ITensor, Sequence[trt.ITensor]]:

        plugin = getattr(getattr(trtp.op, namespace), op_name)

        tensor_inputs = plugin.input_tensor_names
        tensor_args = args[0 : len(tensor_inputs)]

        unique_id = uuid.uuid4()
        itensor_args = [
            get_trt_tensor(ctx, t, f"{t_name}_{unique_id}")
            for (t, t_name) in zip(tensor_args, tensor_inputs)
        ]

        # Assuming TensorRT preserves kwargs order like PyTorch does
        non_tensor_inputs = plugin.input_attrs

        kwargs = {}

        for arg in torch_schema.arguments:
            if arg.default_value is not None:
                kwargs[arg.name] = arg.default_value

        non_tensor_args = args[len(tensor_inputs) :]
        non_tensor_kwargs = dict(zip(list(non_tensor_inputs.keys()), non_tensor_args))

        # Update kwargs with non_tensor_kwargs, adding new keys or overwriting existing ones
        kwargs.update(non_tensor_kwargs)

        arg_types = {arg.name: arg.type for arg in torch_schema.arguments}
        for k, v in list(kwargs.items()):
            if isinstance(v, torch.fx.immutable_collections.immutable_list):
                kwargs[k] = np.array(v)
            kwargs[k] = _coerce_scalar_plugin_attr(kwargs[k], arg_types[k])
            kwargs[k] = _coerce_plugin_attr_for_qdp(kwargs[k], non_tensor_inputs[k])

        layer = ctx.net.add_plugin(plugin(*itensor_args, **kwargs), aot=use_aot_plugin)
        assert layer, f"{namespace}::{name} plugin layer was not able to be created"
        _LOGGER.debug(
            f"Adding generated plugin for {namespace}::{name} to tensorrt network"
        )
        layer.name = f"[{target}]-[{name}]"

        # The QDP plugin populates `aliased_map` (output_idx -> input_idx, with
        # -1 meaning no alias) during `add_plugin` when TRT invokes the
        # descriptor's `get_output_data_types`. Any non-negative entry means
        # the engine build needs the aliased plugin I/O preview feature
        # enabled. `plugin(*args)` itself is just a creation closure — the
        # populated `aliased_map` lives on the layer's `plugin` attribute.
        # JIT plugins: `layer.plugin` returns the Python `_TemplateJITPlugin`
        # instance, whose `aliased_map` (output_idx -> input_idx, -1 means
        # none) is populated when TRT invokes the descriptor during
        # `add_plugin`. AOT plugins: `layer.plugin` returns a bare
        # `trt.IPluginV3` C++ wrapper that doesn't expose the Python
        # attribute. We read the JIT map when we can, and otherwise enable
        # the aliased-I/O preview feature unconditionally for AOT — it's
        # dormant in TRT when no plugin actually declares `.aliased()`, so
        # this only adds the flag (which is needed when aliasing *is*
        # declared) without changing semantics when it isn't.
        layer_plugin = getattr(layer, "plugin", None)
        aliased_map = getattr(layer_plugin, "aliased_map", None)
        if aliased_map and any(v != -1 for v in aliased_map.values()):
            ctx.requires_aliased_plugin_io = True
        elif use_aot_plugin:
            ctx.requires_aliased_plugin_io = True
        # Single-output ops expect a bare ITensor; multi-output ops expect a
        # tuple so the downstream ``getitem`` converter can unpack it.
        num_outputs = len(torch_schema.returns)
        if num_outputs == 1:
            return layer.get_output(0)
        return tuple(layer.get_output(i) for i in range(num_outputs))

    custom_kernel_converter = dynamo_tensorrt_converter(
        torch_overload,
        capability_validator=capability_validator,
        priority=priority,
        supports_dynamic_shapes=supports_dynamic_shapes,
        requires_output_allocator=requires_output_allocator,
    )(custom_kernel_converter)
    assert torch_overload in DYNAMO_CONVERTERS, (
        f"Generated dynamo converter for {namespace}::{op_name} did not get properly"
        " registered in the converter registry"
    )
    return custom_kernel_converter


@needs_qdp_plugin
def generate_plugin_converter(
    plugin_id: str,
    capability_validator: Optional[Callable[[Node, CompilationSettings], bool]] = None,
    priority: ConverterPriority = ConverterPriority.STANDARD,
    supports_dynamic_shapes: bool = False,
    requires_output_allocator: bool = False,
    use_aot_if_available: bool = True,
) -> DynamoConverterImplSignature:
    plugin_ns, plugin_name = plugin_id.split("::")
    return _generate_plugin_converter(
        plugin_ns,
        plugin_name,
        capability_validator=capability_validator,
        priority=priority,
        supports_dynamic_shapes=supports_dynamic_shapes,
        requires_output_allocator=requires_output_allocator,
        use_aot_if_available=use_aot_if_available,
    )
