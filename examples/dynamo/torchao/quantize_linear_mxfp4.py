"""
.. _quantize_linear_mxfp4:

TorchAO MXFP4 Dynamic Activation + MX Weight (Linear)
=====================================================

This example quantizes a toy nn.Linear with TorchAO
MXDynamicActivationMXWeightConfig-shaped MXFP4 storage (packed FP4 E2M1
weights, E8M0 block scales, block size 32), keeps an explicit
torchao_trt.dequantize_mxfp4 in the exported graph, and compiles with the
Torch-TensorRT Dynamo backend.

There is no TorchAO MXFP4 weight-only config. Native MXFP4xMXFP4 kernels
need B200/B300. This path uses emulated storage and a weight DQ prologue
into a high-precision GEMM — activations stay BF16.

MXFP4 requires the last two weight dims to be divisible by 32. TorchAO's
default MXTensor Linear path does not emit a DQ op TensorRT can map, so
weights are promoted to MXTensorNonDecomposed before export.

Requirements:

* NVIDIA GPU with TensorRT FP4 and E8M0 constants (TRT ≥ 10.8)
* torchao (prototype.mx_formats)
* torch-tensorrt with the torchao_trt.dequantize_mxfp4 converter

"""

# %%
# Imports
# ^^^^^^^
# This example lives in examples/dynamo/torchao/. Move that directory off
# the front of sys.path so import torchao resolves the PyPI package
# instead of this folder.

import sys
from pathlib import Path

_EXAMPLE_DIR = str(Path(__file__).resolve().parent)
if sys.path and Path(sys.path[0]).resolve() == Path(_EXAMPLE_DIR):
    sys.path.pop(0)

import torch
import torch_tensorrt as torchtrt

sys.path.insert(0, _EXAMPLE_DIR)
from mxfp4_utils import pre_process_model_for_export, quantize_linear_mxfp4

# %%
# Define a linear model and quantize weights to MXFP4
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


class LinearModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # K and N must be divisible by MX block_size=32.
        self.linear = torch.nn.Linear(3072, 4096)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


model = LinearModel().eval().to(dtype=torch.bfloat16, device="cuda")
example_input = torch.randn(32, 3072, dtype=torch.bfloat16, device="cuda")

quantize_linear_mxfp4(model)
processed_model = pre_process_model_for_export(model)

# %%
# Export and compile
# ^^^^^^^^^^^^^^^^^^
# Torch-TensorRT marks dequantize_mxfp4 impure in constant folding so the
# packed FP4 weight is not folded into a dense BF16 constant.

exp_program = torch.export.export(processed_model, (example_input,), strict=True)

trt_model = torchtrt.dynamo.compile(
    exp_program,
    inputs=[example_input],
    min_block_size=1,
    use_explicit_typing=True,
    require_full_compilation=True,
    immutable_weights=True,
)

output = trt_model(example_input)
print(output)
