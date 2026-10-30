# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
.. _cutedsl_op:

Register a CuTe DSL kernel as an AOT TensorRT plugin
===================================================

``torch_tensorrt.kernels.cutedsl_op`` compiles a raw ``@cute.kernel`` to PTX,
registers a PyTorch custom operator, and lowers it to a TensorRT Ahead-of-Time
Quick Deployable Plugin. The PTX is embedded in the engine.

Install ``nvidia-cutlass-dsl>=4.8.1,<5`` and use a CUDA GPU with a compatible
driver, plus TensorRT with QDP support. From a source checkout, the optional
frontend dependencies can be installed with ``pip install --group kernels``.
Run this example with ``python examples/dynamo/cutedsl_op.py``.

This example adds one to float32 tensors with a dynamic element count. The
optional eager implementation is a PyTorch reference; the compiled TensorRT
engine executes the CuTe DSL kernel.
"""

import cutlass
import cutlass.cute as cute
import tensorrt.plugin as trtp
import torch

import torch_tensorrt
import torch_tensorrt.kernels as ttk

# %%
# Define a raw CuTe DSL kernel. Runtime arguments must follow the QDP order:
# input pointers, int32 launch metadata, then output pointers.

BLOCK_SIZE = 256


@cute.kernel
def add_one_kernel(
    x: cute.Pointer,
    n_elements: cutlass.Int32,
    out: cute.Pointer,
    block_size: cutlass.Constexpr,
):
    tx, _, _ = cute.arch.thread_idx()
    bx, _, _ = cute.arch.block_idx()
    index = bx * block_size + tx
    if index < n_elements:
        out[index] = x[index] + cutlass.Float32(1.0)


# %%
# Register the operator. Shapes in the launch callbacks stay symbolic so one
# engine handles every element count allowed by the input profile.


def add_one_meta(x: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(x, memory_format=torch.contiguous_format)


def add_one_eager(x: torch.Tensor) -> torch.Tensor:
    return (x + 1).contiguous()


ttk.cutedsl_op(
    "torch_tensorrt_example::cutedsl_add_one",
    kernel=add_one_kernel,
    signature={"x": "*fp32", "n_elements": "i32", "out": "*fp32"},
    meta_fn=add_one_meta,
    grid=lambda inputs, outputs: (trtp.cdiv(inputs[0].shape_expr.numel(), BLOCK_SIZE),),
    block_size=BLOCK_SIZE,
    constants={"block_size": BLOCK_SIZE},
    extra_args_fn=lambda inputs, outputs: [trtp.SymInt32(inputs[0].shape_expr.numel())],
    eager_fn=add_one_eager,
)


# %%
# Compile a dynamic-shape engine, check that the custom operator was converted,
# and compare several shapes, including partial blocks, against the reference.


class AddOne(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ops.torch_tensorrt_example.cutedsl_add_one(x)


if __name__ == "__main__":
    model = AddOne().cuda().eval()
    trt_model = torch_tensorrt.compile(
        model,
        ir="dynamo",
        inputs=[
            torch_tensorrt.Input(
                min_shape=(127,),
                opt_shape=(1024,),
                max_shape=(4097,),
                dtype=torch.float32,
            )
        ],
        min_block_size=1,
        require_full_compilation=True,
        pass_through_build_failures=True,
    )

    for size in (127, 1024, 4097):
        x = torch.randn(size, device="cuda", dtype=torch.float32)
        torch.testing.assert_close(trt_model(x), model(x), atol=1e-5, rtol=1e-5)
        print(f"CuTe DSL AOT plugin matched the reference for {size} elements")
