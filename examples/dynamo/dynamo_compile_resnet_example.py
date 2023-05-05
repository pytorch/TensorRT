"""
Dynamo Compile ResNet Example
=========================

This interactive script is intended as a sample of the `torch_tensorrt.dynamo.compile` workflow on a ResNet model."""

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
# Optional Input Arguments to `torch_tensorrt.dynamo.compile`
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# Enabled precision for TensorRT optimization
enabled_precisions = {torch.half}

# Whether to print verbose logs
debug = True

# Workspace size for TensorRT
workspace_size = 20 << 30

# Maximum number of TRT Engines
# (Lower value allows more graph segmentation)
min_block_size = 3

# Operations to Run in Torch, regardless of converter support
torch_executed_ops = {}

# %%
# Compilation with `torch_tensorrt.dynamo.compile`
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# Build and compile the model with torch.compile, using Torch-TensorRT backend
optimized_model = torch_tensorrt.dynamo.compile(
    model,
    inputs,
    enabled_precisions=enabled_precisions,
    debug=debug,
    workspace_size=workspace_size,
    min_block_size=min_block_size,
    torch_executed_ops=torch_executed_ops,
)

# %%
# Equivalently, we could have run the above via the convenience frontend, as so:
# `torch_tensorrt.compile(model, ir="dynamo_compile", inputs=inputs, ...)`

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
# Cleanup
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# Finally, we use Torch utilities to clean up the workspace
torch._dynamo.reset()

with torch.no_grad():
    torch.cuda.empty_cache()
