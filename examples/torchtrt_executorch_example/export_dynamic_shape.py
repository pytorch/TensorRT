"""
.. _executorch_export_dynamic:

Saving a Torch-TensorRT Model with Dynamic Shapes in ExecuTorch Format (.pte)
==============================================================================

This example demonstrates how to compile a model with Torch-TensorRT using
dynamic (range-based) input shapes and save it as an ExecuTorch ``.pte`` file.

The TRT engine is built with a shape profile: batch size can vary between 1
and 8, and the spatial dimensions between 2 and 8, while the channel dimension
is fixed at 3.  The ExecuTorch runtime will select the correct binding sizes
at execute() time based on the actual input shapes.

Prerequisites
-------------
Install Torch-TensorRT with the ExecuTorch extra before running this example::

    pip install -e ".[executorch]"

See https://pytorch.org/executorch/stable/getting-started-setup.html for details.
"""

# %%
# Imports and Model Definition
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

import argparse

import torch
import torch_tensorrt

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_path", default="model_dynamic.pte", help="Path to save the .pte file"
)
args = parser.parse_args()


class MyModel(torch.nn.Module):
    def forward(self, x):
        return x + 1


# %%
# Compile with Torch-TensorRT using dynamic shapes
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# torch_tensorrt.Input with min/opt/max shapes builds a TRT optimization
# profile that covers the full range.  The exported program is traced at the
# opt shape; min/max define the runtime range accepted by the engine.

with torch.no_grad():
    model = MyModel().eval().cuda()
    # Trace at the opt shape
    opt_input = (torch.randn((4, 3, 4, 4)).cuda(),)

    # Mark variable dimensions so the ExecuTorch .pte allocates them as
    # resizable tensors.  Without dynamic_shapes the portable runtime
    # pre-allocates fixed-size tensors at the export shape and rejects
    # inputs of any other size at execute() time.
    batch = torch.export.Dim("batch", min=1, max=8)
    spatial = torch.export.Dim("spatial", min=2, max=8)
    dynamic_shapes = {"x": {0: batch, 2: spatial, 3: spatial}}

    exported_program = torch.export.export(
        model, opt_input, dynamic_shapes=dynamic_shapes
    )
    compile_settings = {
        "arg_inputs": [
            torch_tensorrt.Input(
                min_shape=(1, 3, 2, 2),
                opt_shape=(4, 3, 4, 4),
                max_shape=(8, 3, 8, 8),
                dtype=torch.float32,
            ),
        ],
        "min_block_size": 1,
    }
    trt_gm = torch_tensorrt.dynamo.compile(exported_program, **compile_settings)

    # %%
    # Save as ExecuTorch .pte format
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    torch_tensorrt.save(
        trt_gm,
        args.model_path,
        output_format="executorch",
        arg_inputs=opt_input,
        retrace=False,
    )

    print(f"Saved {args.model_path} successfully.")
