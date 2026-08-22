"""
.. _converter_overloading:

Overloading Torch-TensorRT Converters with Custom Converters
===================================================================

If for some reason you want to change the conversion behavior of a specific PyTorch operation to TensorRT, you can do so by writing a custom converter and overloading Torch-TensorRT's.
This may be for reasons like wanting to use a custom kernel instead of TensorRT's kernels or because you want to use a different implementation of a layer in TensorRT than the one
Torch-TensorRT would normally use.

In this tutorial, we will demonstrate how to overload Torch-TensorRT's conversion of the `torch.nn.functional.gelu` operation to TensorRT with a custom converter that uses a different implementation
of the GeLU layer.

"""

import logging
import sys

import torch
import torch_tensorrt

# %% GeLU Operator in PyTorch
#
# GeLU has 2 modes in PyTorch, one using the ``erf`` function and the other using the ``tanh`` approximation.
# TensorRT natively supports both implementations as an activation layer, but suppose we want to use a custom implementation of GeLU in TensorRT only for ``tanh`` mode.


class GeLU(torch.nn.Module):
    def __init__(self, mode="tanh"):
        super().__init__()
        self.mode = mode

    def forward(self, x):
        return torch.nn.functional.gelu(x, approximate=self.mode)


my_mod = GeLU(mode="tanh")
ex_input = torch.randn(2, 5).to("cuda")


# %%
# As a baseline, we can use the standard Torch-TensorRT GeLU converter (in tanh approximation mode) with our module.
my_standard_gelu = torch_tensorrt.compile(
    my_mod, arg_inputs=(ex_input,), min_block_size=1
)
print(my_standard_gelu.graph)
print(my_standard_gelu(ex_input))

# %%
# Writing a Custom Converter
# --------------------------
#
# Converters are functions that take a specific instance of a PyTorch operation in a PyTorch graph and convert it to an equivalent set TensorRT operations in an under-construction TensorRT graph.
# They are registered with Torch-TensorRT using the ``@torch_tensorrt.dynamo.conversion.dynamo_tensorrt_converter`` decorator.
# At a code level, converter takes the current conversion state (``ConversionCtx``), the next operator in the graph to convert, and the arguments to that node
# and returns the placeholder outputs for that operation, while as side-effect inserting the necessary TensorRT layers into the TensorRT network.
#

from typing import Dict, Sequence, Tuple, Union

import tensorrt as trt
from torch.fx.node import Argument, Node, Target
from torch_tensorrt.dynamo import CompilationSettings
from torch_tensorrt.dynamo.conversion import ConversionContext

# %%
# Converter Metadata
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^


@torch_tensorrt.dynamo.conversion.dynamo_tensorrt_converter(
    # The PyTorch operation to convert, when this operation is encountered, this converter will be called
    torch.ops.aten.gelu.default,
    # Validators are functions that determine that given a specific node, if it can be converted by the converter
    capability_validator=lambda node, settings: (
        "approximate" in node.kwargs and node.kwargs["approximate"] == "tanh"
    ),
    # Can this converter be used in cases where the input shapes are dynamic
    supports_dynamic_shapes=True,
    # Set the priority of the converter to supersede the default one
    priority=torch_tensorrt.dynamo.conversion.ConverterPriority.HIGH,
    # Whether the converter requires a dynamic output allocator to run (e.g. data dependent ops)
    requires_output_allocator=True,
)

# %%
# For the decorator defining a converter, there is one required argument and a few optional ones.
# All converters need a target operator they will run against, the idea being that when there is an instance of ``torch.ops.aten.gelu.default`` in the graph, this converter will be called.
#
# Following the target operator, you can provide additional metadata that defines the capabilities of the converter and the priority of the converter verses other possible converters for the target in question
#
# The primary tool for defining the capabilities of a converter is the ``capability_validator`` argument,
# which is a lambda function that takes a specific node in the graph as well as the user compilation settings and returns a boolean indicating if the converter can be used for that node.
# This validator function gets run prior to the graph partitioning phase against each instance of the converter target op. Nodes where there are no converters with validators that pass during this phase, will be executed in PyTorch at runtime.
# This is useful for cases where you want to use a custom converter only in specific cases, like in our case where we only want to use our converter when ``approximate == "tanh"``.
#
# Distinct to the validator is the ``supports_dynamic_shapes`` argument, which is a boolean indicating if the converter can be used in cases where the input shapes are dynamic.
# If this is set to ``False``, in cases where the inputs provided by the user are dynamic, this converter will be disabled. If there are no alternatives that support dynamic shape, the operation will be run in PyTorch.
#
# Finally there is the ``priority`` argument, which is an enum from the ``torch_tensorrt.dynamo.conversion.ConverterPriority`` class that defines the priority of the converter. The two options are ``HIGH`` and ``STANDARD``.
# Converters registered with ``STANDARD`` will be appended to the converter list for a given operation, while converters registered with ``HIGH`` will be prepended to the list.
# Candidate converters are evalated for their suitability in this priority order and the first converter that passes the validator is used.


# %%
# Converter Implementation
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^
# The converter function itself takes the following arguments: the current conversion context, the target operator, the arguments to the target operator, the keyword arguments to the target operator, and the name of the target operator.
# Arguments can either any of python primitives, ``torch.Tensor``, ``np.Arrays`` or ``ITensor`` objects.
# The converter function should return the outputs of the target operator in terms of TensorRT ``ITensor`` primarily. These inputs and outputs should correspond to the schema
# of the target PyTorch operator which can be found here `https://pytorch.org/docs/main/torch.compiler_ir.html <https://pytorch.org/docs/main/torch.compiler_ir.html>`_.
#
# Since Torch-TensorRT covers the core ATen opset, it has already abstracted many of the common low-level operations into helper functions that can be used to build up the TensorRT network.
# This allows developers to avoid the boilerplate of creating the TensorRT layers directly and instead focus on the high-level logic of the conversion.
# The helper functions are located in the ``torch_tensorrt.dynamo.conversion.impl`` module and are designed to be composable and interoperable with raw-TensorRT implementations.
# In this case, we will use the Torch-TensorRT ``mul``, ``add`` and ``tanh`` functions from ``impl`` to implement our alternative GeLU layer.


def aten_ops_gelu(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[trt.ITensor, Sequence[trt.ITensor]]:
    # The schema for torch.ops.aten.gelu.default is gelu(Tensor self, *, str approximate=’none’) -> Tensor

    from torch_tensorrt.dynamo import SourceIR
    from torch_tensorrt.dynamo.conversion import impl

    # Cheap way to allow layer names to be unqiue
    op_count = 0

    def get_op_count():
        nonlocal op_count
        op_count += 1
        return op_count

    mul = lambda x, y: impl.elementwise.mul(
        ctx,
        target,
        name=f"mul_{get_op_count()}",
        source_ir=SourceIR.ATEN,
        lhs_val=x,
        rhs_val=y,
    )
    add = lambda x, y: impl.elementwise.add(
        ctx,
        target,
        name=f"add_{get_op_count()}",
        source_ir=SourceIR.ATEN,
        lhs_val=x,
        rhs_val=y,
    )
    tanh = lambda x: impl.activation.tanh(
        ctx, target, name=f"tanh_{get_op_count()}", source_ir=SourceIR.ATEN, input_val=x
    )

    # So we know that our custom converter is being run instead of the standard one
    print("\n\n---------------------------")
    print("Using custom GeLU converter")
    print("---------------------------\n\n")

    x_7 = mul(args[0], 0.5)
    x_8 = mul(args[0], 0.79788456080000003)
    x_9 = mul(args[0], 0.044714999999999998)
    x_10 = mul(x_9, args[0])
    x_11 = add(x_10, 1.0)
    x_12 = mul(x_8, x_11)
    x_13 = tanh(x_12)
    x_14 = add(x_13, 1.0)
    x_15 = mul(x_7, x_14)

    return x_15


# %%
# Using our Custom Converter
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# We can now recompile and see that our custom converter is being called to convert GeLU to TensorRT.
my_custom_gelu = torch_tensorrt.compile(
    my_mod, arg_inputs=(ex_input,), min_block_size=1
)

print(my_custom_gelu.graph)
print(my_custom_gelu(ex_input))

# %%
#
# We can verify that our implementation matches the TensorRT implementation for the ``tanh`` approximation.
print(
    f"tanh approximations are close: {torch.allclose(my_standard_gelu(ex_input), my_custom_gelu(ex_input))}"
)


# %%
#
# Finally, we want to verify that in the case that the ``approximate`` argument is not set to ``tanh``, our custom converter is not used.

my_mod_erf = GeLU(mode="none")
my_gelu_erf = torch_tensorrt.compile(
    my_mod_erf, arg_inputs=(ex_input,), min_block_size=1
)

# %%
#
# Notice that we don't see the print statement from our custom converter, indicating that it was not used. However, looking at the graph, we can still see that a TensorRT engine was created to run the GeLU operation.
# In this case, the validator for our custom converter returned ``False``, so the conversion system moved on to the next converter in the list, the standard GeLU converter and used that one to convert the operation.

print(my_gelu_erf.graph)
print(my_gelu_erf(ex_input))
