import inspect
import logging
from enum import IntEnum
from types import FunctionType
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np

# Seems like a bug in TensorRT
import tensorrt_bindings.plugin as trtp
from tensorrt_bindings.plugin._lib import QDP_REGISTRY
import torch
from torch._guards import detect_fake_mode
from torch._library.custom_ops import CustomOpDef, device_types_t
from torch._ops import OpOverload
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.node import Argument, Node, Target, _get_qualified_name
from torch_tensorrt._enums import dtype
from torch_tensorrt.dynamo._settings import CompilationSettings
from torch_tensorrt.dynamo.conversion._ConversionContext import ConversionContext
from torch_tensorrt.dynamo.conversion._ConverterRegistry import (
    DYNAMO_CONVERTERS,
    ConverterPriority,
    ConverterSupport,
    dynamo_tensorrt_converter,
)
from torch_tensorrt.dynamo.conversion.converter_utils import get_trt_tensor

import tensorrt as trt

_LOGGER: logging.Logger = logging.getLogger(__name__)


class Tactic(IntEnum):
    TORCH = 1
    TRITON = 2


def custom_op(
    name: str,
    fn: Optional[Callable] = None,
    /,
    *,
    mutates_args: Union[str, Iterable[str]],
    device_types: device_types_t = None,
    schema: Optional[str] = None,
    capability_validator: Optional[Callable[[Node, CompilationSettings], bool]] = None,
    priority: ConverterPriority = ConverterPriority.STANDARD,
    supports_dynamic_shapes: bool = False,
) -> Callable:
    def inner(fn):
        torch_custom_op_def = torch.library.custom_op(
            name, mutates_args=mutates_args, device_types=device_types, schema=schema
        )(fn)

        def _tensorrt_plugin_desc(args) -> Tuple[trtp.TensorDesc]:
            print(args)
            return (args[0].like(),)

        def tensorrt_plugin_desc(
            in0: trtp.TensorDesc, in1: trtp.TensorDesc
        ) -> Tuple[trtp.TensorDesc]:
            return in0.like()

        tensorrt_plugin_reg = trtp.register(name)(tensorrt_plugin_desc)
        print(tensorrt_plugin_reg)

        def _tensorrt_plugin_impl(args) -> None:
            print(args)

        @trtp.impl(name)
        def tensorrt_plugin_impl(
            in0: trtp.Tensor, in1: trtp.Tensor, outputs: Tuple[trtp.Tensor], stream: int
        ) -> None:
            # This should be based on Torch schema
            in_tensors = [
                torch.as_tensor(i, device="cuda") for i in (in0, in1)
            ]  # What is the right device??
            dest_tensors = [torch.as_tensor(o, device="cuda") for o in outputs]

            stream = torch.cuda.ExternalStream(stream)
            with torch.cuda.stream(stream):
                out_tensors = torch_custom_op_def._opoverload(*in_tensors)
                [d.copy_(o) for (d, o) in zip(dest_tensors, out_tensors)]

        op_converter = generate_torch_op_converter(
            torch_custom_op_def, capability_validator, priority, supports_dynamic_shapes
        )

        torch_custom_op_def._tensorrt_plugin_desc = tensorrt_plugin_desc
        torch_custom_op_def._tensorrt_plugin_impl = tensorrt_plugin_impl
        torch_custom_op_def._torch_tensorrt_converter = op_converter

        print(torch_custom_op_def._schema)
        return torch_custom_op_def

    if fn is None:
        return inner

    return inner(fn)

def _generate_plugin(
    namespace: str,
    op_name: str,
):
    @trtp.register(f"{namespace}::{op_name}")
    def add_plugin_desc(x: trtp.TensorDesc, y: trtp.TensorDesc, b: float, a: int) -> Tuple[trtp.TensorDesc]:
        from torch._subclasses.fake_tensor import FakeTensorMode
        from torch.fx.experimental.symbolic_shapes import ShapeEnv
        from sympy import lambdify
        shape_env = ShapeEnv()
        fake_mode = FakeTensorMode(shape_env=shape_env)
        sample_x = {f"x{i}": 5 for i in range(x.ndim)}
        sample_y = {f"y{i}": 5 for i in range(y.ndim)}
        syms_x = [mksym(shape_env, v, LocalSource(k), DimDynamic.DYNAMIC) for k,v in sample_x.items()]
        syms_y = [mksym(shape_env, v, LocalSource(k), DimDynamic.DYNAMIC) for k,v in sample_y.items()]
        with FakeTensorMode() as fake_mode:
            fake_x = torch.randn(syms_x)
            fake_y = torch.randn(syms_y)
            z = torch.ops.torchtrt_ex.elementwise_mul(fake_x, fake_y, b, a)

        shape_calc_fns = [None] * x.ndim
        for i in range(x.ndim):
            shape_calc_fns[i] = lambdify((syms_x[i].node.expr, syms_y[i].node.expr), z.shape[i].node.expr, "math")

        out_desc = x.like()
        for i in range(out_desc.ndim):
            out_desc.shape_expr[i] = shape_calc_fns[i](x.shape_expr[i], y.shape_expr[i])


    # Type annotations can be omitted for autotune and impl definitions, but will be checked for consistency if added
    @trtp.autotune(f"{namespace}::{op_name}")
    def add_plugin_autotune(
        inp0: trtp.TensorDesc, block_size: int, outputs: Tuple[trtp.TensorDesc]
    ) -> List[trtp.AutoTuneCombination]:
        return [trtp.AutoTuneCombination("FP32|FP16, FP32|FP16", "LINEAR", [1, 2])]


    @trtp.impl(f"{namespace}::{op_name}")
    def add_plugin_impl(x: trtp.Tensor, y: trtp.Tensor, b: float, a: int, outputs: Tuple[trtp.Tensor], stream: int):
        # This should be based on Torch schema
        in_tensors = [
            torch.as_tensor(i, device="cuda") for i in (x, y)
        ]  # What is the right device??
        dest_tensors = [torch.as_tensor(o, device="cuda") for o in outputs]

        stream = torch.cuda.ExternalStream(stream)
        with torch.cuda.stream(stream):
            out_tensors = torch.ops.torchtrt_ex.elementwise_mul(*in_tensors, b, a)
            [d.copy_(o) for (d, o) in zip(dest_tensors, out_tensors)]


            
def generate_plugin(
    plugin_id: str,
    # capability_validator: Optional[Callable[[Node, CompilationSettings], bool]] = None,

):
    plugin_ns, plugin_name = plugin_id.split("::")
    _generate_plugin(
        plugin_ns,
        plugin_name,
    )

def _generate_plugin_converter(
    namespace: str,
    op_name: str,
    overload: Optional[str] = None,
    capability_validator: Optional[Callable[[Node, CompilationSettings], bool]] = None,
    priority: ConverterPriority = ConverterPriority.STANDARD,
    supports_dynamic_shapes: bool = False,
):
    torch_target = getattr(getattr(torch.ops, namespace), op_name)
    overload_str = overload if overload else ""
    overload_name = overload_str if overload else "default"
    torch_overload = getattr(torch_target, overload_name)
    assert f"{namespace}::{op_name}" in QDP_REGISTRY, f"Could not find a tensorrt plugin registered for op {namespace}::{op_name}, unable to generate converter"
    torch_schema = torch_target._schemas[overload_str]

    def custom_kernel_converter(
        ctx: ConversionContext,
        target: Target,
        args: Tuple[Argument, ...],
        kwargs: Dict[str, Argument],
        name: str,
    ):
        plugin = getattr(getattr(trtp.op, namespace), op_name)
        tensor_inputs = plugin.input_tensor_names
        tensor_args = args[0:len(tensor_inputs)]
        itensor_args = [get_trt_tensor(ctx, t, f"{t_name}") for (t, t_name) in zip(tensor_args, tensor_inputs)]

        # Assuming TensorRT preserves kwargs order like PyTorch does
        non_tensor_inputs = plugin.input_attrs

        non_tensor_args = args[len(tensor_inputs):]
        non_tensor_kwargs = {k:v for k, v in zip(list(non_tensor_inputs.keys()), non_tensor_args)}
        for (k,v) in non_tensor_kwargs.items():
            if isinstance(v, torch.fx.immutable_collections.immutable_list):
                non_tensor_kwargs[k] = np.array(v)

        layer = ctx.net.add_plugin(plugin(*itensor_args, **non_tensor_kwargs))
        assert (
            layer
        ), f"{namespace}::{name} plugin layer was not able to be created"
        _LOGGER.debug(
            f"Adding generated plugin for {namespace}::{name} to tensorrt network"
        )
        layer.name = f"[{target}]-[{name}]"
        return layer.get_output(0)

    custom_kernel_converter = dynamo_tensorrt_converter(
        torch_overload,
        capability_validator=capability_validator,
        priority=priority,
        supports_dynamic_shapes=supports_dynamic_shapes,
    )(
        custom_kernel_converter
    )  # type: ignore
    assert torch_overload in DYNAMO_CONVERTERS, f"Generated dynamo converter for {namespace}::{name} did not get properly registered in the converter registry"
    return custom_kernel_converter


def generate_torch_op_converter(
    op_reg: CustomOpDef,
    capability_validator: Optional[Callable[[Node, CompilationSettings], bool]] = None,
    priority: ConverterPriority = ConverterPriority.STANDARD,
    supports_dynamic_shapes: bool = False,
):
    return _generate_plugin_converter(
        op_reg._namespace,
        op_reg._name,
        capability_validator=capability_validator,
        priority=priority,
        supports_dynamic_shapes=supports_dynamic_shapes
    )



def generate_plugin_converter(
    plugin_id: str,
    capability_validator: Optional[Callable[[Node, CompilationSettings], bool]] = None,
    priority: ConverterPriority = ConverterPriority.STANDARD,
    supports_dynamic_shapes: bool = False,
):
    plugin_ns, plugin_name = plugin_id.split("::")
    return _generate_plugin_converter(
        plugin_ns,
        plugin_name,
        capability_validator=capability_validator,
        priority=priority,
        supports_dynamic_shapes=supports_dynamic_shapes
    )
