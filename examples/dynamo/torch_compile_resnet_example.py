"""
.. _torch_compile_resnet:

Compiling ResNet with dynamic shapes using the `torch.compile` backend
==========================================================

This interactive script is intended as a sample of the Torch-TensorRT workflow with `torch.compile` on a ResNet model.
"""

# %%
# Imports and Model Definition
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

import torch
import torch_tensorrt
import torchvision.models as models

# %%

# Initialize model with half precision and sample inputs
model = models.resnet18(pretrained=True).half().eval().to("cuda")
inputs = [torch.randn((1, 3, 224, 224)).to("cuda").half()]

# %%
# Optional Input Arguments to `torch_tensorrt.compile`
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# Enabled precision for TensorRT optimization
enabled_precisions = {torch.half}

# Whether to print verbose logs
debug = True

# Workspace size for TensorRT
workspace_size = 20 << 30

# Maximum number of TRT Engines
# (Lower value allows more graph segmentation)
min_block_size = 7

# Operations to Run in Torch, regardless of converter support
torch_executed_ops = {}

# %%
# Compilation with `torch_tensorrt.compile`
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# Build and compile the model with torch.compile, using Torch-TensorRT backend
optimized_model = torch_tensorrt.compile(
    model,
    ir="torch_compile",
    inputs=inputs,
    enabled_precisions=enabled_precisions,
    debug=debug,
    workspace_size=workspace_size,
    min_block_size=min_block_size,
    torch_executed_ops=torch_executed_ops,
)

# %%
# Equivalently, we could have run the above via the torch.compile frontend, as so:
# `optimized_model = torch.compile(model, backend="torch_tensorrt", options={"enabled_precisions": enabled_precisions, ...}); optimized_model(*inputs)`

# %%
# Inference
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# Does not cause recompilation (same batch size as input)
new_inputs = [torch.randn((1, 3, 224, 224)).half().to("cuda")]
new_outputs = optimized_model(*new_inputs)

# %%

# Does cause recompilation (new batch size)
new_batch_size_inputs = [torch.randn((8, 3, 224, 224)).half().to("cuda")]
new_batch_size_outputs = optimized_model(*new_batch_size_inputs)

# %%
# Avoid recompilation by specifying dynamic shapes before Torch-TRT compilation
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# The following code illustrates the workflow using ir=torch_compile (which uses torch.compile under the hood)
inputs_bs8 = torch.randn((8, 3, 224, 224)).half().to("cuda")
# This indicates dimension 0 of inputs_bs8 is dynamic whose range of values is [2, 16]
torch._dynamo.mark_dynamic(inputs_bs8, 0, min=2, max=16)
optimized_model = torch_tensorrt.compile(
    model,
    ir="torch_compile",
    inputs=inputs_bs8,
    enabled_precisions=enabled_precisions,
    debug=debug,
    workspace_size=workspace_size,
    min_block_size=min_block_size,
    torch_executed_ops=torch_executed_ops,
)
outputs_bs8 = optimized_model(inputs_bs8)

# No recompilation happens for batch size = 12
inputs_bs12 = torch.randn((12, 3, 224, 224)).half().to("cuda")
outputs_bs12 = optimized_model(inputs_bs12)

# The following code illustrates the workflow using ir=dynamo (which uses torch.export APIs under the hood)
# dynamic shapes for any inputs are specified using torch_tensorrt.Input API
compile_spec = {
    "inputs": [
        torch_tensorrt.Input(
            min_shape=(1, 3, 224, 224),
            opt_shape=(8, 3, 224, 224),
            max_shape=(16, 3, 224, 224),
            dtype=torch.half,
        )
    ],
    "enabled_precisions": enabled_precisions,
    "ir": "dynamo",
}
trt_model = torch_tensorrt.compile(model, **compile_spec)

# No recompilation happens for batch size = 12
inputs_bs12 = torch.randn((12, 3, 224, 224)).half().to("cuda")
outputs_bs12 = trt_model(inputs_bs12)
