"""
.. _cudagraphs_wrapper_example:

Wrapped runtime module for cuda graphs
======================================

If Torch-TensorRT encounters unsupported operations during compilation, it can fall back to using
PyTorch's native implementation for those specific operations. This fallback mechanism allows the
rest of the model to be executed using TensorRT, while only the unsupported parts are handled by PyTorch.
This fallback results in a graph break, which can reduce the overall performance benefits of using
TensorRT because it introduces additional overhead from switching between TensorRT and PyTorch execution contexts

Applying CUDA Graphs to a PyTorch module that contains graph breaks can enhance performance by leveraging
the benefits of CUDA Graphs even in the presence of these breaks. Torch-TensorRT provides
wrapper runtime module with CUDA Graphs for modules that have graph breaks allows you to mitigate the
inefficiencies introduced by these breaks
"""

# %%
# Imports and Model Definition
# ----------------------------------

import torch
import torch_tensorrt


class SampleModel(torch.nn.Module):
    def forward(self, x):
        return torch.relu((x + 2) * 0.5)


model = SampleModel().eval().cuda()
input = torch.randn((1, 3, 224, 224)).to("cuda")

# %%
# Compiler options
# ----------------------------------
#
# The 'torch_executed_ops' compiler option is used to demonstrate graph breaks for this example.
# debug=True compiler option provides detailed insights into the compilation process and helps
# pinpoint where graph breaks occur

# Create a TensorRT-compiled model
trt_model = torch_tensorrt.compile(
    model,
    ir="dynamo",
    inputs=[input],
    min_block_size=1,
    pass_through_build_failures=True,
    debug=True,
    torch_executed_ops={"torch.ops.aten.mul.Tensor"},
)

# %%
# Compiler log
# ----------------------------------
#
# This compiler log indicates torch.ops.aten.mul.Tensor operator is executed by PyTorch.
# Peformance of this module can be enhanced by using wrapped module.

##############################################################################
# .. code-block:: none
#
#        ++++++++++++++ Dry-Run Results for Graph +++++++++++++++++
#
#        The graph consists of 3 Total Operators, of which 2 operators are supported, 66.67% coverage
#
#        The following ops are currently unsupported or excluded from conversion, and are listed with their op-count in the graph:
#        torch.ops.aten.mul.Tensor: 1
#
#        The following nodes are currently set to run in Torch:
#        Node: torch.ops.aten.mul.Tensor, with layer location: /mul
#        Note: Some of the above nodes may be supported, but were not included in a TRT graph by the partitioner

# %%
# Running wrapped module with cuda graphs
# ----------------------------------
#
# Please note that initializing with wrapper module involve warm-up phase where the module
# is executed several times. This ensures that memory allocations and initializations are
# not recorded in CUDA Graphs.
# When using the TensorRT module within a CUDA Graph context manager, a wrapped_module is returned.
# This module captures the execution graph, allowing for efficient replay during subsequent
# inferences by reducing kernel launch overheads and improving performance.
with torch_tensorrt.runtime.enable_cudagraphs(trt_model) as wrapped_module:
    wrapped_module(input)
