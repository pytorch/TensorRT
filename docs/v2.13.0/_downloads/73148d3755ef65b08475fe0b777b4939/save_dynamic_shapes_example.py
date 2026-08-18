"""
.. _save_dynamic_shapes:

Saving and Loading Models with Dynamic Shapes
==============================================

This example demonstrates how to save and load Torch-TensorRT compiled models
with dynamic input shapes. When you compile a model with dynamic shapes,
you need to preserve the dynamic shape specifications when saving the model
to ensure it can handle variable input sizes after deserialization.

The API is designed to feel similar to torch.export's handling of dynamic shapes
for consistency and ease of use.
"""

# %%
# Imports and Model Definition
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

import tempfile

import torch
import torch.nn as nn
import torch_tensorrt


# %%
# Define a simple model that we'll compile with dynamic batch size
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(16 * 224 * 224, 10)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = x.flatten(1)
        x = self.linear(x)
        return x


# %%
# Compile with Dynamic Shapes
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# First, we compile the model with dynamic batch dimension

model = MyModel().eval().cuda()

# Define example input with batch size 2
example_input = torch.randn(2, 3, 224, 224).cuda()

# Define dynamic batch dimension using torch.export.Dim
# This allows batch sizes from 1 to 32
dyn_batch = torch.export.Dim("batch", min=1, max=32)

# Specify which dimensions are dynamic
dynamic_shapes = {"x": {0: dyn_batch}}

# Export the model with dynamic shapes
exp_program = torch.export.export(
    model, (example_input,), dynamic_shapes=dynamic_shapes, strict=False
)

# Compile with Torch-TensorRT
compile_spec = {
    "inputs": [
        torch_tensorrt.Input(
            min_shape=(1, 3, 224, 224),
            opt_shape=(8, 3, 224, 224),
            max_shape=(32, 3, 224, 224),
            dtype=torch.float32,
        )
    ],
    "enabled_precisions": {torch.float32},
    "min_block_size": 1,
}

trt_gm = torch_tensorrt.dynamo.compile(exp_program, **compile_spec)

# %%
# Test Compiled Model with Different Batch Sizes
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# Test with batch size 4
input_bs4 = torch.randn(4, 3, 224, 224).cuda()
output_bs4 = trt_gm(input_bs4)

# Test with batch size 16
input_bs16 = torch.randn(16, 3, 224, 224).cuda()
output_bs16 = trt_gm(input_bs16)

# %%
# Save the Model with Dynamic Shapes
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# The key is to pass the same dynamic_shapes specification to save()

with tempfile.TemporaryDirectory() as tmpdir:
    save_path = f"{tmpdir}/dynamic_model.ep"

    # Save with dynamic_shapes parameter - this is crucial for preserving dynamic behavior
    torch_tensorrt.save(
        trt_gm,
        save_path,
        output_format="exported_program",
        arg_inputs=[example_input],
        dynamic_shapes=dynamic_shapes,  # Same as used during export
    )

    # %%
    # Load and Test the Saved Model
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    # Load the saved model
    loaded_model = torch_tensorrt.load(save_path).module()

    # Test with the same batch sizes to verify dynamic shapes are preserved
    output_loaded_bs4 = loaded_model(input_bs4)

    output_loaded_bs16 = loaded_model(input_bs16)

    assert torch.allclose(output_bs4, output_loaded_bs4, rtol=1e-3, atol=1e-3)
    assert torch.allclose(output_bs16, output_loaded_bs16, rtol=1e-3, atol=1e-3)

# %%
# Example with Multiple Dynamic Dimensions
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


class MultiDimModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, stride=1, padding=1)

    def forward(self, x):
        return self.conv(x)


model2 = MultiDimModel().eval().cuda()
example_input2 = torch.randn(2, 3, 128, 128).cuda()

# Define dynamic dimensions for batch and spatial dimensions
dyn_batch2 = torch.export.Dim("batch", min=1, max=16)
dyn_height = torch.export.Dim("height", min=64, max=512)
dyn_width = torch.export.Dim("width", min=64, max=512)

dynamic_shapes2 = {"x": {0: dyn_batch2, 2: dyn_height, 3: dyn_width}}

exp_program2 = torch.export.export(
    model2, (example_input2,), dynamic_shapes=dynamic_shapes2, strict=False
)

compile_spec2 = {
    "inputs": [
        torch_tensorrt.Input(
            min_shape=(1, 3, 64, 64),
            opt_shape=(8, 3, 256, 256),
            max_shape=(16, 3, 512, 512),
            dtype=torch.float32,
        )
    ],
    "enabled_precisions": {torch.float32},
}

trt_gm2 = torch_tensorrt.dynamo.compile(exp_program2, **compile_spec2)

with tempfile.TemporaryDirectory() as tmpdir:
    save_path2 = f"{tmpdir}/multi_dim_model.ep"

    torch_tensorrt.save(
        trt_gm2,
        save_path2,
        output_format="exported_program",
        arg_inputs=[example_input2],
        dynamic_shapes=dynamic_shapes2,
    )

    loaded_model2 = torch_tensorrt.load(save_path2).module()

    # Test with different input shapes
    test_input = torch.randn(4, 3, 256, 256).cuda()
    output = loaded_model2(test_input)
