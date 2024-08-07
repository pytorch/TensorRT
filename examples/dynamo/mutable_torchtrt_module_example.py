"""
.. _refit_engine_example:

Refit  TenorRT Graph Module with Torch-TensorRT
===================================================================

We are going to demonstrate how a compiled TensorRT Graph Module can be refitted with updated weights.

In many cases, we frequently update the weights of models, such as applying various LoRA to Stable Diffusion or constant A/B testing of AI products.
That poses challenges for TensorRT inference optimizations, as compiling the TensorRT engines takes significant time, making repetitive compilation highly inefficient.
Torch-TensorRT supports refitting TensorRT graph modules without re-compiling the engine, considerably accelerating the workflow.

In this tutorial, we are going to walk through
1. Compiling a PyTorch model to a TensorRT Graph Module
2. Save and load a graph module
3. Refit the graph module
"""

# %%
# Standard Workflow
# -----------------------------

# %%
# Imports and model definition
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

import numpy as np
import torch
import torch.nn.functional as F
import torch_tensorrt as torch_trt
import torchvision.models as models
from torch import nn

np.random.seed(0)
torch.manual_seed(0)
inputs = [torch.rand((1, 3, 224, 224)).to("cuda")]

# %%
# Compile the module for the first time and save it.
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
kwargs = {
    "use_python": False,
    "enabled_precisions": {torch.float32},
    "make_refitable": True,
}

torch.manual_seed(0)
inputs = [torch.rand((1, 3, 224, 224)).to("cuda")]

compile_spec = {
    "use_python": False,
    "enabled_precisions": {torch.float32},
    "make_refitable": True,
}

model = models.resnet18(pretrained=False).eval().to("cuda")
mutable_module = torch_trt.MutableTorchTensorRTModule(model, **compile_spec)
mutable_module(*inputs)

mutable_module.conv1.weight = nn.Parameter(mutable_module.original_model.conv1.weight)
mutable_module.update_refit_condition()

print("Refit successfully!")


# class net(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv1 = nn.Conv2d(3, 12, 3, padding=1)
#         self.bn = nn.BatchNorm2d(12)
#         self.conv2 = nn.Conv2d(12, 12, 3, padding=1)
#         self.fc1 = nn.Linear(12 * 56 * 56, 10)

#     def forward(self, x, b=5, c=None, d=None):
#         x = self.conv1(x)
#         x = F.relu(x)
#         x = self.bn(x)
#         x = F.max_pool2d(x, (2, 2))
#         x = self.conv2(x)
#         x = F.relu(x)
#         x = F.max_pool2d(x, (2, 2))
#         x = torch.flatten(x, 1)
#         x = x + b
#         if c is not None:
#             x = x * c
#         if d is not None:
#             x = x - d["value"]
#         return self.fc1(x)

# torch.manual_seed(0)
# model = net().eval().to("cuda")
# args = [torch.rand((1, 3, 224, 224)).to("cuda")]
# kwargs = {
#     # "d": {"value": torch.tensor(8).to("cuda")},
#     # "b": torch.tensor(6).to("cuda"),
# }

# compile_spec = {
#     "enabled_precisions": {torch.float},
#     "pass_through_build_failures": True,
#     "optimization_level": 1,
#     "min_block_size": 1,
#     "ir": "dynamo",
#     "make_refitable": True
# }

# mutable_module = torch_trt.MutableTorchTensorRTModule(model, **compile_spec)
# mutable_module(*args, **kwargs)
# torch.manual_seed(2)
# model2 = net().eval().to("cuda")
# mutable_module.load_state_dict(model2.state_dict())

# # Check the output
# expected_outputs, refitted_outputs = model2(*args, **kwargs), mutable_module(*args, **kwargs)
# for expected_output, refitted_output in zip(expected_outputs, refitted_outputs):

#     print((expected_output- refitted_output).std())
#     print()

# print("Refit successfully!")
# torch_trt.MutableTorchTensorRTModule.save(mutable_module, "mutable_module.pkl")
# reload = torch_trt.MutableTorchTensorRTModule.load("mutable_module.pkl")

# expected_outputs, refitted_outputs = reload(*args, **kwargs), mutable_module(*args, **kwargs)
# for expected_output, refitted_output in zip(expected_outputs, refitted_outputs):

#     print((expected_output- refitted_output).std())


# # Clean up model env
# torch._dynamo.reset()
