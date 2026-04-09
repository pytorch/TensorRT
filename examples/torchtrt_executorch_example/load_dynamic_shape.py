"""
.. _executorch_load_dynamic:

Loading a Torch-TensorRT Dynamic-Shape Model from ExecuTorch Format (.pte)
===========================================================================

This example demonstrates how to load a ``.pte`` file produced by
``export_dynamic_shape.py`` and run inference at several different input
shapes within the compiled min/max range.

Prerequisites
-------------
- ExecuTorch installed with a runtime that includes the TensorRT backend.
- Run ``export_dynamic_shape.py`` first to produce ``model_dynamic.pte``.
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

executorch_module = runtime._load_for_executorch("model_dynamic.pte")

# %%
# Run inference at multiple shapes within the compiled profile
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Inputs must be CPU tensors; execute() stages them to CUDA internally.
# All shapes must lie within the min/max range used at export time:
#   batch: [1, 8]  channels: 3 (fixed)  spatial: [2, 8]


class MyModel(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + 1


ref_model = MyModel().eval()

test_shapes = [
    (1, 3, 2, 2),  # minimum shape
    (4, 3, 4, 4),  # opt shape (used at export time)
    (8, 3, 8, 8),  # maximum shape
    (2, 3, 6, 6),  # arbitrary shape within range
]

for shape in test_shapes:
    example_input = torch.randn(shape)  # CPU tensor
    outputs = executorch_module.forward([example_input])
    with torch.no_grad():
        expected = ref_model(example_input)

    torch.testing.assert_close(outputs[0], expected, rtol=1e-3, atol=1e-3)
    print(f"shape={shape}  output={outputs[0].shape}  dtype={outputs[0].dtype}  OK")

print("All dynamic-shape inference runs passed.")
