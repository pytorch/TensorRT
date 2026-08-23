"""
.. _compile_with_dynamic_inputs:

Compiling Models with Dynamic Input Shapes
==========================================================

Dynamic shapes are essential when your model
needs to handle varying batch sizes or sequence lengths at inference time without recompilation.

The example uses a Vision Transformer-style model with expand and reshape operations,
which are common patterns that benefit from dynamic shape handling.
"""

# %%
# Imports and Model Definition
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

import logging

import torch
import torch.nn as nn
import torch_tensorrt

logging.basicConfig(level=logging.DEBUG)

torch.manual_seed(0)

# %%


# Define a model with expand and reshape operations
# This is a simplified Vision Transformer pattern with:
# - A learnable class token that needs to expand to match batch size
# - A QKV projection followed by reshaping for multi-head attention
class ExpandReshapeModel(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.embed_dim = embed_dim
        self.qkv_proj = nn.Linear(self.embed_dim, self.embed_dim * 3)

    def forward(self, x: torch.Tensor):
        batch_size = x.shape[0]
        cls_token = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        x = self.qkv_proj(x)
        reshaped_qkv = x.reshape(batch_size, x.size(1), 3, 12, -1)
        return reshaped_qkv


model = ExpandReshapeModel(embed_dim=768).cuda().eval()
x = torch.randn(4, 196, 768).cuda()

# %%
# Approach 1: JIT Compilation with `torch.compile`
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# The first approach uses PyTorch's `torch.compile` with the TensorRT backend.
# This is a Just-In-Time (JIT) compilation method where the model is compiled
# during the first inference call.
#
# Key points:
#
# - Use `torch._dynamo.mark_dynamic()` to specify which dimensions are dynamic
# - The `index` parameter indicates which dimension (0 = batch dimension)
# - Provide `min` and `max` bounds for the dynamic dimension
# - The model will work for any batch size within the specified range

x1 = x.clone()
torch._dynamo.mark_dynamic(x1, index=0, min=2, max=32)
trt_module = torch.compile(model, backend="tensorrt")
out1 = trt_module(x1)

# %%
# Approach 2: AOT Compilation with `torch_tensorrt.compile`
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# The second approach uses Ahead-Of-Time (AOT) compilation with `torch_tensorrt.compile`.
# This compiles the model upfront before inference.
#
# Key points:
#
# - Use `torch_tensorrt.Input()` to specify dynamic shape ranges
# - Provide `min_shape`, `opt_shape`, and `max_shape` for each input
# - The `opt_shape` is used for optimization and should represent typical input sizes
# - Set `ir="dynamo"` to use the Dynamo frontend

x2 = x.clone()
example_input = torch_tensorrt.Input(
    min_shape=[1, 196, 768],
    opt_shape=[4, 196, 768],
    max_shape=[32, 196, 768],
    dtype=torch.float32,
)
trt_module = torch_tensorrt.compile(model, ir="dynamo", inputs=example_input)
out2 = trt_module(x2)

# %%
# Approach 3: AOT with `torch.export` + Dynamo Compile
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# The third approach uses PyTorch 2.0's `torch.export` API combined with
# Torch-TensorRT's Dynamo compiler. This provides the most explicit control
# over dynamic shapes.
#
# Key points:
#
# - Use `torch.export.Dim()` to define symbolic dimensions with constraints
# - Create a `dynamic_shapes` dictionary mapping inputs to their dynamic dimensions
# - Export the model to an `ExportedProgram` with these constraints
# - Compile the exported program with `torch_tensorrt.dynamo.compile`

x3 = x.clone()
bs = torch.export.Dim("bs", min=1, max=32)
dynamic_shapes = {"x": {0: bs}}
exp_program = torch.export.export(model, (x3,), dynamic_shapes=dynamic_shapes)
trt_module = torch_tensorrt.dynamo.compile(exp_program, (x3,))
out3 = trt_module(x3)

# %%
# Verify All Approaches Produce Identical Results
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# All three approaches should produce the same numerical results.
# This verification ensures that dynamic shape handling works correctly
# across different compilation methods.

assert torch.allclose(out1, out2)
assert torch.allclose(out1, out3)
assert torch.allclose(out2, out3)

print("All three approaches produced identical results!")
