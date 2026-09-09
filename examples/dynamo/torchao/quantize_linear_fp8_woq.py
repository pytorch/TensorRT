"""
.. _quantize_linear_fp8_woq:

TorchAO FP8 Weight-Only Quantization (Linear)
=============================================

This example quantizes a toy ``nn.Linear`` with TorchAO
``Float8WeightOnlyConfig``, keeps an explicit ``dequantize_affine`` in the
exported graph, and compiles with the Torch-TensorRT Dynamo backend.

The intended engine keeps an FP8 weight constant plus a DQ prologue into GEMM.
On Blackwell, Myelin can fuse that prologue into the matmul. On other GPUs the
DQ + GEMM may run as two kernels — that is still correct as long as the FP8
weight is **not** constant-folded into a dense FP16/BF16 weight.

Requirements:

* NVIDIA GPU with FP8 support (Hopper or newer). Blackwell for DQ fusion.
* ``torchao``
* ``torch-tensorrt`` with the ``torchao.dequantize_affine`` converter

"""

# %%
# Imports
# ^^^^^^^
# This example lives in ``examples/dynamo/torchao/``. Move that directory off
# the front of ``sys.path`` so ``import torchao`` resolves the PyPI package
# instead of this folder.

import sys
from pathlib import Path

_EXAMPLE_DIR = str(Path(__file__).resolve().parent)
if sys.path and Path(sys.path[0]).resolve() == Path(_EXAMPLE_DIR):
    sys.path.pop(0)

import torch
import torch_tensorrt as torchtrt
from torchao.quantization import Float8WeightOnlyConfig, quantize_

sys.path.insert(0, _EXAMPLE_DIR)
from utils import exclude_dq_from_constant_folding, pre_process_model_for_export

# %%
# Define a linear model and quantize weights to FP8
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


class LinearModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(3072, 4096)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


model = LinearModel().eval().to(dtype=torch.bfloat16, device="cuda")
example_input = torch.randn(32, 3072, dtype=torch.bfloat16, device="cuda")

quantize_(model, Float8WeightOnlyConfig())
processed_model = pre_process_model_for_export(model)

# %%
# Export and compile
# ^^^^^^^^^^^^^^^^^^
# Wrap export in ``exclude_dq_from_constant_folding`` so inductor does not fold
# ``dequantize_affine`` before Torch-TensorRT lowering. Torch-TensorRT also
# marks that op impure in its own constant-folding pass.

with exclude_dq_from_constant_folding():
    exp_program = torch.export.export(processed_model, (example_input,), strict=True)

trt_model = torchtrt.dynamo.compile(
    exp_program,
    inputs=[example_input],
    min_block_size=1,
    use_explicit_typing=True,
    require_full_compilation=True,
)

output = trt_model(example_input)
print(output)
