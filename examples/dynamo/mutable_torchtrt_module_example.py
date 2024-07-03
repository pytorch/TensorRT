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
import torch_tensorrt as torch_trt
import torchvision.models as models

np.random.seed(0)
torch.manual_seed(0)
inputs = [torch.rand((1, 3, 224, 224)).to("cuda")]


# %%
# Compile the module for the first time and save it.
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
kwargs = {
    'use_python': False,
    'enabled_precisions': {torch.float16, torch.float32},
    'mutable': True
}

model = models.resnet18(pretrained=False).eval().to("cuda")
model2 = models.resnet18(pretrained=True).eval().to("cuda")
mutable_module = torch_trt.compile(inputs=inputs, 
                                   module=model, 
                                   **kwargs)

# Save the graph module as an exported program
# This is only supported when use_python_runtime = False
mutable_module.load_state_dict(model2.state_dict())


# Check the output
expected_outputs, refitted_outputs = model2(*inputs), mutable_module(*inputs)
for expected_output, refitted_output in zip(expected_outputs, refitted_outputs):
    assert torch.allclose(
        expected_output, refitted_output, 1e-2, 1e-2
    ), "Refit Result is not correct. Refit failed"

print("Refit successfully!")
