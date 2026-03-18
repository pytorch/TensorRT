"""
.. _executorch_export:

Saving a Torch-TensorRT Model in ExecuTorch Format (.pte)
=========================================================

This example demonstrates how to compile a model with Torch-TensorRT and save it
as an ExecuTorch ``.pte`` file, which can be loaded by the ExecuTorch runtime
(e.g., on embedded or mobile devices with a TensorRT-capable backend).

Prerequisites
-------------
Install ExecuTorch before running this example::

    pip install executorch

See https://pytorch.org/executorch/stable/getting-started-setup.html for details.
"""

# %%
# Imports and Model Definition
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

import torch
import torch_tensorrt


class MyModel(torch.nn.Module):
    def forward(self, x):
        return x + 1


# %%
# Compile with Torch-TensorRT
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Export the model, compile it with TensorRT, then save as .pte

with torch.no_grad():
    model = MyModel().eval().cuda()
    example_input = (torch.randn((2, 3, 4, 4)).cuda(),)

    exported_program = torch.export.export(model, example_input)
    compile_settings = {
        "arg_inputs": [
            torch_tensorrt.Input(shape=(2, 3, 4, 4), dtype=torch.float32),
        ],
        "min_block_size": 1,
    }
    trt_gm = torch_tensorrt.dynamo.compile(exported_program, **compile_settings)

    # %%
    # Save as ExecuTorch .pte format (loadable by the ExecuTorch runtime)
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    # The TensorRT engine is serialized inside the .pte using the same blob format
    # as the Torch-TensorRT runtime (vector of strings), so one engine format for
    # both ExecuTorch and non-ExecuTorch deployment.
    torch_tensorrt.save(
        trt_gm, "model.pte", output_format="executorch", arg_inputs=example_input
    )

    print("Saved model.pte successfully.")
