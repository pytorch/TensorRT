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

import ctypes
import os

import torch
import torch_tensorrt  # noqa: F401  -- loads libtorchtrt.so / libtorchtrt_runtime.so

# libqnn_executorch_backend.so carries the ExecuTorch runtime (including
# executorch::runtime::internal::vlogf and register_backend). It must be
# loaded with RTLD_GLOBAL so its symbols are visible to subsequently
# dlopen'd libraries (libtrt_executorch_backend.so and portable_lib).
_executorch_path = os.environ.get("EXECUTORCH_PATH", "/home/lanl/git/executorch")
ctypes.CDLL(
    os.path.join(_executorch_path, "backends/qualcomm/libqnn_executorch_backend.so"),
    mode=ctypes.RTLD_GLOBAL,
)

# Load the TensorRT ExecuTorch backend shared library, which runs a static
# initializer that calls executorch::runtime::register_backend("TensorRTBackend").
# Resolves runtime symbols from libqnn_executorch_backend.so loaded above.
_lib_dir = os.path.join(os.path.dirname(torch_tensorrt.__file__), "lib")
ctypes.CDLL(os.path.join(_lib_dir, "libtrt_executorch_backend.so"))

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

example_input = torch.randn(
    (2, 3, 4, 4)
)  # CPU tensor; execute() stages it to CUDA internally
outputs = executorch_module.forward([example_input])

print("Output shape:", outputs[0].shape)
print("Output dtype:", outputs[0].dtype)

# %%
# Verify against eager mode
# ^^^^^^^^^^^^^^^^^^^^^^^^^


class MyModel(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + 1


model = MyModel().eval()
with torch.no_grad():
    expected = model(example_input)

torch.testing.assert_close(outputs[0], expected, rtol=1e-3, atol=1e-3)
print("Output matches eager mode.")
