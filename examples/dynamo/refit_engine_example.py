"""
.. _refit_engine_example:

Refitting Torch-TensorRT Programs with New Weights
===================================================================

Compilation is an expensive operation as it involves many graph transformations, translations
and optimizations applied on the model. In cases were the weights of a model might be updated
occasionally (e.g. inserting LoRA adapters), the large cost of recompilation can make it infeasible
to use TensorRT if the compiled program needed to be built from scratch each time. Torch-TensorRT
provides a PyTorch native mechanism to update the weights of a compiled TensorRT program without
recompiling from scratch through weight refitting.

In this tutorial, we are going to walk through

    1. Compiling a PyTorch model to a TensorRT Graph Module
    2. Save and load a graph module
    3. Refit the graph module

This tutorial focuses mostly on the AOT workflow where it is most likely that a user might need to
manually refit a module. In the JIT workflow, weight changes trigger recompilation. As the engine
has previously been built, with an engine cache enabled, Torch-TensorRT can automatically recognize
a previously built engine, trigger refit and short cut recompilation on behalf of the user (see: :ref:`engine_caching_example`).
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
from torch_tensorrt.dynamo import refit_module_weights

np.random.seed(0)
torch.manual_seed(0)
inputs = [torch.rand((1, 3, 224, 224)).to("cuda")]


# %%
# Make a refittable Compilation Program
# ---------------------------------------
#
# The inital step is to compile a module and save it as with a normal. Note that there is an
# additional parameter `immutable_weights` that is set to `False`. This parameter is used to
# indicate that the engine being built should support weight refitting later. Engines built without
# these setttings will not be able to be refit.
#
# In this case we are going to compile a ResNet18 model with randomly initialized weights and save it.

model = models.resnet18(pretrained=False).eval().to("cuda")
exp_program = torch.export.export(model, tuple(inputs))
enabled_precisions = {torch.float}
debug = False
workspace_size = 20 << 30
min_block_size = 0
use_python_runtime = False
torch_executed_ops = {}
trt_gm = torch_trt.dynamo.compile(
    exp_program,
    tuple(inputs),
    use_python_runtime=use_python_runtime,
    enabled_precisions=enabled_precisions,
    debug=debug,
    min_block_size=min_block_size,
    torch_executed_ops=torch_executed_ops,
    immutable_weights=False,
    reuse_cached_engines=False,
)  # Output is a torch.fx.GraphModule

# Save the graph module as an exported program
torch_trt.save(trt_gm, "./compiled.ep", inputs=inputs)


# %%
# Refit the Program with Pretrained Weights
# ------------------------------------------
#
# Random weights are not useful for inference. But now instead of recompiling the model, we can
# refit the model with the pretrained weights. This is done by setting up another PyTorch module
# with the target weights and exporting it as an ExportedProgram. Then the ``refit_module_weights``
# function is used to update the weights of the compiled module with the new weights.

# Create and compile the updated model
model2 = models.resnet18(pretrained=True).eval().to("cuda")
exp_program2 = torch.export.export(model2, tuple(inputs))


compiled_trt_ep = torch_trt.load("./compiled.ep")

# This returns a new module with updated weights
new_trt_gm = refit_module_weights(
    compiled_module=compiled_trt_ep,
    new_weight_module=exp_program2,
    arg_inputs=inputs,
)

# Check the output
model2.to("cuda")
expected_outputs, refitted_outputs = exp_program2.module()(*inputs), new_trt_gm(*inputs)
for expected_output, refitted_output in zip(expected_outputs, refitted_outputs):
    assert torch.allclose(
        expected_output, refitted_output, 1e-2, 1e-2
    ), "Refit Result is not correct. Refit failed"

print("Refit successfully!")

# %%
#
# Advanced Usage
# -----------------------------
#
# There are a number of settings you can use to control the refit process
#
# Weight Map Cache
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Weight refitting works by matching the weights of the compiled module with the new weights from
# the user supplied ExportedProgram. Since 1:1 name matching from PyTorch to TensorRT is hard to accomplish,
# the only gaurenteed way to match weights at *refit-time* is to pass the new ExportedProgram through the
# early phases of the compilation process to generate near identical weight names. This can be expensive
# and is not always necessary.
#
# To avoid this, **At initial compile**, Torch-TensorRt will attempt to cache a direct mapping from PyTorch
# weights to TensorRT weights. This cache is stored in the compiled module as metadata and can be used
# to speed up refit. If the cache is not present, the refit system will fallback to rebuilding the mapping at
# refit-time. Use of this cache is controlled by the ``use_weight_map_cache`` parameter.
#
# Since the cache uses a heuristic based system for matching PyTorch and TensorRT weights, you may want to verify the refitting. This can be done by setting
# ``verify_output`` to True and providing sample ``arg_inputs`` and ``kwarg_inputs``. When this is done, the refit
# system will run the refitted module and the user supplied module on the same inputs and compare the outputs.
#
# In-Place Refit
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# ``in_place`` allows the user to refit the module in place. This is useful when the user wants to update the weights
# of the compiled module without creating a new module.
