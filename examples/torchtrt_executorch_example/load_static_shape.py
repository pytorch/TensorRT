"""
.. _executorch_load:

Loading a Torch-TensorRT Model from ExecuTorch Format (.pte)
=============================================================

This example demonstrates how to load a ``.pte`` file produced by
``export_static_shape.py`` and run inference with the ExecuTorch runtime.

Prerequisites
-------------
- ExecuTorch installed with a runtime that includes the TensorRT backend.
- Run ``export_static_shape.py`` first to produce ``model.pte``.
"""

# %%
# Imports
# ^^^^^^^

import torch
import torch_tensorrt  # noqa: F401  -- loads libtorchtrt_runtime.so, which registers TensorRTBackend with the ExecuTorch runtime via its static initializer
from executorch.extension.pybindings import portable_lib as runtime

# %%
# Load the .pte file
# ^^^^^^^^^^^^^^^^^^
# _load_for_executorch returns an ExecuTorchModule whose methods mirror the
# original model's exported methods (e.g. "forward").

executorch_module = runtime._load_for_executorch("model.pte")

# %%
# Run inference
# ^^^^^^^^^^^^^
# Inputs must be passed as a list of tensors matching the static shapes used
# at export time: (2, 3, 4, 4) float32 on CUDA.

example_input = torch.randn((2, 3, 4, 4)).cuda()
outputs = executorch_module.forward([example_input])

print("Output shape:", outputs[0].shape)
print("Output dtype:", outputs[0].dtype)

# %%
# Verify against eager mode
# ^^^^^^^^^^^^^^^^^^^^^^^^^


class MyModel(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + 1


model = MyModel().eval().cuda()
with torch.no_grad():
    expected = model(example_input)

torch.testing.assert_close(outputs[0], expected, rtol=1e-3, atol=1e-3)
print("Output matches eager mode.")
