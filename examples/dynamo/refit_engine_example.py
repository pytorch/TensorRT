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
import torch_tensorrt as trt
import torchvision.models as models
from torch_tensorrt.dynamo import refit_module_weights

np.random.seed(0)
torch.manual_seed(0)
inputs = [torch.rand((1, 3, 224, 224)).to("cuda")]


# %%
# Compile the module for the first time and save it.
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

model = models.resnet18(pretrained=False).eval().to("cuda")
exp_program = torch.export.export(model, tuple(inputs))
enabled_precisions = {torch.float}
debug = False
workspace_size = 20 << 30
min_block_size = 0
use_python_runtime = False
torch_executed_ops = {}
trt_gm = trt.dynamo.compile(
    exp_program,
    tuple(inputs),
    use_python_runtime=use_python_runtime,
    enabled_precisions=enabled_precisions,
    debug=debug,
    min_block_size=min_block_size,
    torch_executed_ops=torch_executed_ops,
    refit=True,
)  # Output is a torch.fx.GraphModule

# Save the graph module as an exported program
# This is only supported when use_python_runtime = False
trt.save(trt_gm, "./compiled.ep", inputs=inputs)


# %%
# Refit the module with update model weights
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# Create and compile the updated model
model2 = models.resnet18(pretrained=True).eval().to("cuda")
exp_program2 = torch.export.export(model2, tuple(inputs))


compiled_trt_gm = trt.load("./compiled.ep")

# This returns a new module with updated weights
new_trt_gm = refit_module_weights(
    compiled_module=compiled_trt_gm,
    new_weight_module=exp_program2,
    inputs=inputs,
)

# Check the output
expected_outputs, refitted_outputs = exp_program2.module()(*inputs), new_trt_gm(*inputs)
for expected_output, refitted_output in zip(expected_outputs, refitted_outputs):
    assert torch.allclose(
        expected_output, refitted_output, 1e-2, 1e-2
    ), "Refit Result is not correct. Refit failed"

print("Refit successfully!")

# %%
# Alterative Workflow using Python Runtime
# -----------------------------

# Currently python runtime does not support engine serialization. So the refitting will be done in the same runtime.
# This usecase is more useful when you need to switch different weights in the same runtime, such as using Stable Diffusion.
