# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

"""
.. _fused_kernel_example:

Replace annotated operations with a custom kernel inside TensorRT
=================================================================

Run with CUDA, Triton >=3.5, and TensorRT QDP support::

    python examples/dynamo/fused_kernel.py

``fused_kernel`` captures add -> ReLU -> multiply and registers the supplied
Triton kernel as their replacement automatically. Supply the kernel and launch
configuration; Torch-TensorRT handles operator and plugin registration.
The original model still executes ordinary PyTorch operations eagerly.
The zero-argument kernel factory keeps the Triton object outside model tracing.

The compiled model contains one TensorRT engine: native TensorRT arithmetic
before and after a plugin layer running the fused Triton kernel. Fusion applies
to the operations inside the annotated scope; the surrounding TensorRT layers
remain separate. The example checks plugin ownership and prints the engine's
layer information to show the generated plugin explicitly.

The supplied kernel must implement the entire annotated computation. This
example uses equal-shaped contiguous CUDA float32 inputs and checks numerical
agreement, including NaN and infinity; automatic registration does not prove
kernel correctness or speed.
"""

import json

import tensorrt.plugin as trtp
import torch
import triton
import triton.language as tl

import torch_tensorrt
from torch_tensorrt.dynamo.region_compilation import get_region_records
from torch_tensorrt.dynamo.runtime import TorchTensorRTModule

BLOCK = 256


@triton.jit
def bias_relu_scale_kernel(
    x_ptr, bias_ptr, scale_ptr, n_elements, output_ptr, BLOCK: tl.constexpr
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    bias = tl.load(bias_ptr + offsets, mask=mask, other=0.0)
    scale = tl.load(scale_ptr + offsets, mask=mask, other=0.0)
    shifted = x + bias
    activated = tl.where(shifted < 0.0, 0.0, shifted)  # Preserve NaNs like ReLU.
    tl.store(output_ptr + offsets, activated * scale, mask=mask)


def launch_grid(inputs, outputs):
    return (trtp.cdiv(inputs[0].shape_expr.numel(), BLOCK),)


def kernel_extras(inputs, outputs):
    # n_elements follows the input pointers and precedes the output pointer.
    return [trtp.SymInt32(inputs[0].shape_expr.numel())]


class Model(torch.nn.Module):
    def forward(self, x, bias, scale):
        x = x + 1.0
        with torch_tensorrt.region.fused_kernel(
            kernel=lambda: bias_relu_scale_kernel,
            constexprs={"BLOCK": BLOCK},
            grid=launch_grid,
            extra_args_fn=kernel_extras,
        ):
            x = x + bias
            x = torch.relu(x)
            x = x * scale
        return x - 0.5


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("This example requires a CUDA GPU")
    torch.manual_seed(0)
    # 771 elements exercise the mask on the final, partially filled block.
    inputs = tuple(torch.randn(3, 257, device="cuda") for _ in range(3))
    model = Model().eval().cuda()
    compiled = torch_tensorrt.compile(
        model,
        inputs=list(inputs),
        ir="dynamo",
        strict=True,
        offload_module_to_cpu=False,
        min_block_size=1,
        require_full_compilation=True,
    )
    engines = [m for m in compiled.modules() if isinstance(m, TorchTensorRTModule)]
    assert len(engines) == 1, f"Expected one TensorRT engine, got {len(engines)}"
    records = [r for r in get_region_records(compiled) if r.route == "plugin"]
    assert len(records) == 1, f"Expected one plugin region, got {len(records)}"
    record = records[0]
    assert record.owner is not None, "The plugin region has no owning engine"
    engine = compiled.get_submodule(record.owner)
    assert engine is engines[0], "The plugin must belong to the single TensorRT engine"
    # The outer FX graph only shows an engine call. Inspect the built engine
    # itself to verify that it contains the generated plugin operator.
    layer_info = engine.get_layer_info()
    assert str(record.plugin_op) in layer_info, "Plugin missing from the built engine"

    with torch.inference_mode():
        torch.testing.assert_close(compiled(*inputs), model(*inputs))
        other = tuple(torch.randn_like(value) for value in inputs)
        other[0].view(-1)[:3] = other[0].new_tensor([torch.nan, torch.inf, -torch.inf])
        torch.testing.assert_close(compiled(*other), model(*other), equal_nan=True)
    print("Compiled graph (one TensorRT engine call):")
    print(compiled.graph)
    print(f"\nExecution inside TensorRT engine {record.owner}:")
    print("  TensorRT: x + 1.0")
    print("  -> TensorRT plugin: bias_relu_scale_kernel (add bias -> ReLU -> scale)")
    print("  -> TensorRT: x - 0.5")
    print(f"Plugin operator: {record.plugin_op}")
    print("Engine layers (from TensorRT's inspector):")
    print(json.dumps(json.loads(layer_info)["Layers"], indent=2))
    print(
        "One TensorRT engine includes the fused kernel; both input cases match PyTorch."
    )


if __name__ == "__main__":
    main()
