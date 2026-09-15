"""
.. _quantize_linear_int4_woq:

TorchAO INT4 Weight-Only Quantization (Linear)
==============================================

This example quantizes a toy nn.Linear to **symmetric** group-wise INT4
with TorchAO, keeps an explicit dequantize_affine in the exported graph,
and compiles with the Torch-TensorRT Dynamo backend.

The intended engine keeps an INT4 weight constant plus a DQ prologue into GEMM.
TorchAO's default INT4 config is asymmetric (nonzero zero-point); TensorRT's
IDequantizeLayer rejects that, so the helper uses TorchAO's symmetric
quantizer. Parent Int4Tensor Linear uses fused mslk kernels and never calls
dequantize(), so weights are promoted to Int4TensorNonDecomposed before
export.

This is 4-bit **integer** weight-only quantization (storage + DQ + high-precision
GEMM), not NVFP4 Tensor Core MMA.

Requirements:

* NVIDIA GPU
* torchao
* torch-tensorrt with the INT4-aware torchao.dequantize_affine converter

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
from int4_utils import pre_process_model_for_export, quantize_linear_int4_symmetric
from utils import exclude_dq_from_constant_folding

# %%
# Define a linear model and quantize weights to symmetric INT4
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


class LinearModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(3072, 4096)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


model = LinearModel().eval().to(dtype=torch.bfloat16, device="cuda")
example_input = torch.randn(32, 3072, dtype=torch.bfloat16, device="cuda")

quantize_linear_int4_symmetric(model, group_size=128)
processed_model = pre_process_model_for_export(model)

# %%
# Export and compile
# ^^^^^^^^^^^^^^^^^^
# Wrap export in exclude_dq_from_constant_folding so inductor does not fold
# dequantize_affine before Torch-TensorRT lowering. Use
# immutable_weights=True: the converter re-packs int8 qdata into an INT4
# constant at build time, and engine refit would push unpacked int8 at an INT4
# prototype.

with exclude_dq_from_constant_folding():
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
