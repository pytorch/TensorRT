from typing import Any, Dict, Optional, Sequence, Tuple

import torch
import torch._custom_ops as library
from torch.fx.node import Argument, Target
from torch_tensorrt.dynamo.lowering._pre_aot_lowering import register_substitution
from torch_tensorrt.fx.converter_registry import tensorrt_converter
from torch_tensorrt.fx.converters.converter_utils import set_layer_name
from torch_tensorrt.fx.types import TRTNetwork, TRTTensor

library.custom_op(
    "tensorrt::einsum",
    "(str equation, Tensor[] tensors) -> Tensor",
)


@library.impl("tensorrt::einsum")  # type: ignore[misc]
@library.impl_abstract("tensorrt::einsum")  # type: ignore[misc]
def einsum_generic(
    *args: Any,
    **kwargs: Any,
) -> Any:
    # Defines a converter implementation for AOT Autograd to use for shape analysis/propagation
    return torch.einsum(
        *args,
        **kwargs,
    )


# TODO: @gs-olive Port to dynamo converter
@tensorrt_converter(torch.ops.tensorrt.einsum.default)  # type: ignore[misc]
def aten_ops_einsum(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> TRTTensor:
    # Defines converter replacing the default operator for this function
    assert isinstance(args[1], Sequence)
    for input_trt in args[1]:
        if not isinstance(input_trt, TRTTensor):
            raise RuntimeError(f"Einsum received non-TRTTensor input: {input_trt}")

    einsum_layer = network.add_einsum(inputs=args[1], equation=args[0])

    set_layer_name(einsum_layer, target, name)
    return einsum_layer.get_output(0)


@register_substitution(torch.einsum, torch.ops.tensorrt.einsum)  # type: ignore[misc]
def einsum_insertion_fn(
    gm: torch.fx.GraphModule,
    node: torch.fx.Node,
    submodule: Optional[torch.nn.Module] = None,
) -> torch.fx.Node:
    equation = node.args[0]

    # Ensure inputs is a list of (Tensor) arguments
    if isinstance(node.args[1], (tuple, list)):
        inputs = node.args[1]
    else:
        inputs = node.args[1:]

    assert (
        1 <= len(inputs) <= 2
    ), f"TRT Einsum currently only supports 1 or 2 Tensors, got {len(inputs)} Tensors"

    # Ensure the input is formatted as an equation and
    new_node: torch.fx.Node = gm.graph.call_function(
        torch.ops.tensorrt.einsum,
        args=(equation, inputs),
        kwargs=node.kwargs,
    )

    return new_node
