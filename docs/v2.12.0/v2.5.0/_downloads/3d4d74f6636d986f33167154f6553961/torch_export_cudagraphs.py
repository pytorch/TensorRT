"""
.. _torch_export_cudagraphs:

Torch Export with Cudagraphs
======================================================

This interactive script is intended as an overview of the process by which the Torch-TensorRT Cudagraphs integration can be used in the `ir="dynamo"` path. The functionality works similarly in the `torch.compile` path as well."""

# %%
# Imports and Model Definition
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

import torch
import torchvision.models as models

import torch_tensorrt

# %%
# Compilation with `torch_tensorrt.compile` Using Default Settings
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# We begin by defining and initializing a model
model = models.resnet18(pretrained=True).eval().to("cuda")

# Define sample inputs
inputs = torch.randn((16, 3, 224, 224)).cuda()

# %%

# Next, we compile the model using torch_tensorrt.compile
# We use the `ir="dynamo"` flag here, and `ir="torch_compile"` should
# work with cudagraphs as well.
opt = torch_tensorrt.compile(
    model,
    ir="dynamo",
    inputs=torch_tensorrt.Input(
        min_shape=(1, 3, 224, 224),
        opt_shape=(8, 3, 224, 224),
        max_shape=(16, 3, 224, 224),
        dtype=torch.float,
        name="x",
    ),
)

# %%
# Inference using the Cudagraphs Integration
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# We can enable the cudagraphs API with a context manager
with torch_tensorrt.runtime.enable_cudagraphs():
    out_trt = opt(inputs)

# Alternatively, we can set the cudagraphs mode for the session
torch_tensorrt.runtime.set_cudagraphs_mode(True)
out_trt = opt(inputs)

# We can also turn off cudagraphs mode and perform inference as normal
torch_tensorrt.runtime.set_cudagraphs_mode(False)
out_trt = opt(inputs)

# %%

# If we provide new input shapes, cudagraphs will re-record the graph
inputs_2 = torch.randn((8, 3, 224, 224)).cuda()
inputs_3 = torch.randn((4, 3, 224, 224)).cuda()

with torch_tensorrt.runtime.enable_cudagraphs():
    out_trt_2 = opt(inputs_2)
    out_trt_3 = opt(inputs_3)
