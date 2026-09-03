"""
.. _quantize_linear_fp8_static:

TorchAO Static FP8 Quantization (Linear)
========================================

This example calibrates a two-layer Linear model with TorchAO observers, then
rewrites those layers so **activations and weights** are quantized to FP8
(e4m3). Export keeps explicit ``quantize_affine_float8_non_decomposed`` /
``dequantize_affine_float8_non_decomposed`` nodes, which Torch-TensorRT maps
to ``IQuantizeLayer`` / ``IDequantizeLayer``.

Contrast with :ref:`quantize_linear_fp8_woq`, which quantizes **weights only**
and needs no calibration. Static FP8 can run the GEMM itself in FP8 (Tensor
Cores) after Q/DQ fusion.

Graph after export (one Linear)::

    BF16 act ──► Q ──► FP8 act ──► DQ ──┐
                                        ▼
                                     aten.linear
                                        ▲
    FP8 weight ──► DQ ──────────────────┘

Requirements:

* NVIDIA GPU with FP8 support (Hopper or newer)
* ``torchao``
* ``torch-tensorrt`` with the TorchAO float8_non_decomposed converters

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

sys.path.insert(0, _EXAMPLE_DIR)
from static_fp8_utils import quantize_static_fp8


def sqnr(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Signal-to-quantization-noise ratio in dB (higher is closer)."""
    a = a.float().flatten()
    b = b.float().flatten()
    signal = torch.norm(a)
    noise = torch.norm(a - b)
    return 20 * torch.log10(signal / noise.clamp_min(1e-12))


# %%
# Define a small two-layer Linear model
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


class LinearModel(torch.nn.Module):
    def __init__(self, in_features=256, hidden=512, out_features=128):
        super().__init__()
        self.linear1 = torch.nn.Linear(in_features, hidden, bias=False)
        self.linear2 = torch.nn.Linear(hidden, out_features, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.linear1(x))


model = LinearModel().eval().to(dtype=torch.bfloat16, device="cuda")
example_input = torch.randn(32, 256, dtype=torch.bfloat16, device="cuda")

# %%
# Calibrate, then insert static FP8 Q/DQ
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Observers record per-tensor activation min/max and per-channel weight min/max
# over a few forward passes. Those scales are baked into ``QuantizedLinearQDQ``.

with torch.no_grad():
    fp_out = model(example_input)

quantize_static_fp8(model, (example_input,), calibration_steps=10)

with torch.no_grad():
    quant_out = model(example_input)
print(f"eager SQNR after static FP8: {sqnr(fp_out, quant_out):.2f} dB")

# %%
# Export and compile
# ^^^^^^^^^^^^^^^^^^
# The exported graph should contain
# ``quantize_affine_float8_non_decomposed`` (activations) and
# ``dequantize_affine_float8_non_decomposed`` (activations and weights).

exp_program = torch.export.export(model, (example_input,), strict=True)
exp_program.graph_module.print_readable()

trt_model = torchtrt.dynamo.compile(
    exp_program,
    inputs=[example_input],
    enabled_precisions={torch.float8_e4m3fn},
    min_block_size=1,
    require_full_compilation=True,
)

with torch.no_grad():
    trt_out = trt_model(example_input)
    if isinstance(trt_out, (list, tuple)):
        trt_out = trt_out[0]
print(f"TRT SQNR vs quantized eager: {sqnr(quant_out, trt_out):.2f} dB")
print(trt_out)
