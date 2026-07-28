"""
.. _save_dynamic_shapes_both_methods:

Saving Models with Dynamic Shapes - Both Methods
=================================================

This example demonstrates BOTH methods for saving Torch-TensorRT compiled models
with dynamic input shapes:

1. **Method 1**: Using torch.export.Dim (explicit dynamic_shapes parameter)
2. **Method 2**: Using torch_tensorrt.Input with min/opt/max (automatic inference)

"""

# %%
# Imports and Model Definition
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

import tempfile

import torch
import torch.nn as nn
import torch_tensorrt


# %%
# Define a simple model
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)

    def forward(self, x):
        return self.linear(x)


model = SimpleModel().eval().cuda()

# %%
# Method 1: Explicit dynamic_shapes with torch.export.Dim
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# This follows torch.export's API pattern

example_input = torch.randn(4, 10).cuda()

# Define dynamic dimension explicitly
dyn_batch = torch.export.Dim("batch", min=1, max=32)
dynamic_shapes = {"x": {0: dyn_batch}}

# Export with dynamic shapes
exp_program = torch.export.export(
    model, (example_input,), dynamic_shapes=dynamic_shapes, strict=False
)

# Compile with TensorRT
trt_module_method1 = torch_tensorrt.dynamo.compile(
    exp_program,
    inputs=[
        torch_tensorrt.Input(
            min_shape=(1, 10),
            opt_shape=(8, 10),
            max_shape=(32, 10),
            dtype=torch.float32,
        )
    ],
    enabled_precisions={torch.float32},
    min_block_size=1,
)

with tempfile.TemporaryDirectory() as tmpdir:
    save_path = f"{tmpdir}/model_method1.ep"

    # Save with explicit dynamic_shapes parameter
    torch_tensorrt.save(
        trt_module_method1,
        save_path,
        output_format="exported_program",
        arg_inputs=[example_input],
        dynamic_shapes=dynamic_shapes,  # Explicit!
        retrace=True,
    )

    # Load and test
    loaded_model = torch_tensorrt.load(save_path).module()
    output_bs4 = loaded_model(torch.randn(4, 10).cuda())
    output_bs16 = loaded_model(torch.randn(16, 10).cuda())


# %%
# Method 2: Automatic inference from torch_tensorrt.Input
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# Redefine model for fresh compile
model2 = SimpleModel().eval().cuda()

inputs = [
    torch_tensorrt.Input(
        min_shape=(1, 10),
        opt_shape=(8, 10),
        max_shape=(32, 10),
        dtype=torch.float32,
        name="x",
    )
]

# Compile directly with torch_tensorrt.compile
trt_module_method2 = torch_tensorrt.compile(model2, ir="dynamo", inputs=inputs)


with tempfile.TemporaryDirectory() as tmpdir:
    save_path = f"{tmpdir}/model_method2.ep"

    # Save with Input objects - dynamic_shapes inferred automatically!
    # No need to specify dynamic_shapes explicitly
    torch_tensorrt.save(
        trt_module_method2,
        save_path,
        output_format="exported_program",
        arg_inputs=inputs,  # Pass the same Input objects used for compile
        retrace=True,
    )

    # Load and test
    loaded_model = torch_tensorrt.load(save_path).module()
    output_bs4 = loaded_model(torch.randn(4, 10).cuda())
    output_bs16 = loaded_model(torch.randn(16, 10).cuda())


# %%
# Multiple Dynamic Dimensions Example
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


class ConvModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, padding=1)

    def forward(self, x):
        return self.conv(x)


model3 = ConvModel().eval().cuda()

# Multiple dynamic dimensions: batch, height, width
inputs_multi = [
    torch_tensorrt.Input(
        min_shape=(1, 3, 64, 64),
        opt_shape=(8, 3, 256, 256),
        max_shape=(16, 3, 512, 512),
        dtype=torch.float32,
        name="image",
    )
]

trt_module_multi = torch_tensorrt.compile(model3, ir="dynamo", inputs=inputs_multi)

with tempfile.TemporaryDirectory() as tmpdir:
    save_path = f"{tmpdir}/model_multi_dim.ep"

    torch_tensorrt.save(
        trt_module_multi,
        save_path,
        arg_inputs=inputs_multi,  # Automatically infers all 3 dynamic dims!
        retrace=True,
    )

    loaded_model = torch_tensorrt.load(save_path).module()

    # Test with different shapes
    out1 = loaded_model(torch.randn(4, 3, 128, 128).cuda())
    out2 = loaded_model(torch.randn(12, 3, 384, 384).cuda())
