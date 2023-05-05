"""
Dynamo Compile Transformers Example
=========================

This interactive script is intended as a sample of the `torch_tensorrt.dynamo.compile` workflow on a transformer-based model."""

# %%
# Imports and Model Definition
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

import torch
import torch_tensorrt
from transformers import BertModel

# %%

# Initialize model with float precision and sample inputs
model = BertModel.from_pretrained("bert-base-uncased").eval().to("cuda")
inputs = [
    torch.randint(0, 2, (1, 14), dtype=torch.int32).to("cuda"),
    torch.randint(0, 2, (1, 14), dtype=torch.int32).to("cuda"),
]


# %%
# Optional Input Arguments to `torch_tensorrt.dynamo.compile`
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# Enabled precision for TensorRT optimization
enabled_precisions = {torch.float}

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

# Build and compile the model with torch.compile, using tensorrt backend
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

with torch.no_grad():
    torch.cuda.empty_cache()
