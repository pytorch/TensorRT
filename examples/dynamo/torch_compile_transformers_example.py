"""
.. _torch_compile_transformer:

Compiling BERT using the `torch.compile` backend
==============================================================

This interactive script is intended as a sample of the Torch-TensorRT workflow with `torch.compile` on a BERT model.
"""

# %%
# Imports and Model Definition
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

import torch
import torch_tensorrt
from transformers import BertModel

# %%

# Initialize model with float precision and sample inputs
model = BertModel.from_pretrained("bert-base-uncased").to("cuda").eval()
inputs = [
    torch.randint(0, 2, (1, 14), dtype=torch.int32).to("cuda"),
    torch.randint(0, 2, (1, 14), dtype=torch.int32).to("cuda"),
]


# %%
# Optional Input Arguments to `torch_tensorrt.compile`
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# Enabled precision for TensorRT optimization
enabled_precisions = {torch.float}

# Workspace size for TensorRT
workspace_size = 20 << 30

# Maximum number of TRT Engines
# (Lower value allows more graph segmentation)
min_block_size = 7

# Operations to Run in Torch, regardless of converter support
torch_executed_ops = {}

# %%
# Compilation with `torch.compile`
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# Define backend compilation keyword arguments
compilation_kwargs = {
    "enabled_precisions": enabled_precisions,
    "workspace_size": workspace_size,
    "min_block_size": min_block_size,
    "torch_executed_ops": torch_executed_ops,
}

# Build and compile the model with torch.compile, using Torch-TensorRT backend
optimized_model = torch.compile(
    model,
    backend="torch_tensorrt",
    dynamic=False,
    options=compilation_kwargs,
)
optimized_model(*inputs)

# %%
# Equivalently, we could have run the above via the convenience frontend, as so:
# `torch_tensorrt.compile(model, ir="torch_compile", inputs=inputs, **compilation_kwargs)`

# %%
# Inference
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# Does not cause recompilation (same batch size as input)
new_inputs = [
    torch.randint(0, 2, (1, 14), dtype=torch.int32).to("cuda"),
    torch.randint(0, 2, (1, 14), dtype=torch.int32).to("cuda"),
]
new_outputs = optimized_model(*new_inputs)

# %%

# Does cause recompilation (new batch size)
new_inputs = [
    torch.randint(0, 2, (4, 14), dtype=torch.int32).to("cuda"),
    torch.randint(0, 2, (4, 14), dtype=torch.int32).to("cuda"),
]
new_outputs = optimized_model(*new_inputs)

# %%
# Cleanup
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# Finally, we use Torch utilities to clean up the workspace
torch._dynamo.reset()

# %%
# Cuda Driver Error Note
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Occasionally, upon exiting the Python runtime after Dynamo compilation with `torch_tensorrt`,
# one may encounter a Cuda Driver Error. This issue is related to https://github.com/NVIDIA/TensorRT/issues/2052
# and can be resolved by wrapping the compilation/inference in a function and using a scoped call, as in::
#
#       if __name__ == '__main__':
#           compile_engine_and_infer()
