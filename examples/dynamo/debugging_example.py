"""
.. _debugging_torchtrt:

Using the Torch-TensorRT Debugger
==========================================================

This tutorial demonstrates how the use the Torch-TensorRT debugger to inspect different components of the
compiler.
"""

# %%
# Imports and Model Definition
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

import torch
import torch_tensorrt
import torchvision.models as models

# %%

model = models.resnet18(pretrained=True).half().eval().to("cuda")
inputs = [torch.randn((1, 3, 224, 224)).to("cuda").half()]
enabled_precisions = {torch.half}
workspace_size = 20 << 30
min_block_size = 7

# %%
# Compilation with `torch_tensorrt.compile`
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# Compile within the debugging context to control different aspects of the compilation pipeline
with torch_tensorrt.dynamo._Debugger.Debugger(engine_builder_monitor=True, break_in_remove_assert_nodes=True):
    optimized_model = torch_tensorrt.compile(
        model,
        #ir="torch_compile",
        inputs=inputs,
        enabled_precisions=enabled_precisions,
        workspace_size=workspace_size,
        min_block_size=min_block_size,
    )

    # Does not cause recompilation (same batch size as input)
    new_inputs = [torch.randn((1, 3, 224, 224)).half().to("cuda")]
    new_outputs = optimized_model(*new_inputs)
    print(new_outputs)

# Does not cause recompilation (same batch size as input)
new_inputs = [torch.randn((1, 3, 224, 224)).half().to("cuda")]
new_outputs = optimized_model(*new_inputs)
print(new_outputs)


optimized_model = torch_tensorrt.compile(
    model,
    #ir="torch_compile",
    inputs=inputs,
    enabled_precisions=enabled_precisions,
    workspace_size=workspace_size,
    min_block_size=min_block_size,
)

# Does not cause recompilation (same batch size as input)
new_inputs = [torch.randn((1, 3, 224, 224)).half().to("cuda")]
new_outputs = optimized_model(*new_inputs)
print(new_outputs)
