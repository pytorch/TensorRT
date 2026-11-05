# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import platform
import unittest
from typing import Tuple

import torch
import torch.nn as nn
import torch_tensorrt
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests

# Triton is Linux-only, skip on Windows
if platform.system() != "Windows":
    import triton
    import triton.language as tl
else:
    triton = None
    tl = None

from ..conversion.harness import DispatchTestCase

if triton is not None:

    @triton.jit
    def elementwise_mul_kernel(X, Y, Z, BLOCK_SIZE: tl.constexpr):
        # Program ID determines the block of data each thread will process
        pid = tl.program_id(0)
        # Compute the range of elements that this thread block will work on
        block_start = pid * BLOCK_SIZE
        # Range of indices this thread will handle
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        # Load elements from the X and Y tensors
        x_vals = tl.load(X + offsets)
        y_vals = tl.load(Y + offsets)
        # Perform the element-wise multiplication
        z_vals = x_vals * y_vals
        # Store the result in Z
        tl.store(Z + offsets, z_vals)

else:
    # Dummy function for Windows
    def elementwise_mul_kernel(*args, **kwargs):
        pass


@torch.library.custom_op("torchtrt_ex::elementwise_mul", mutates_args=())  # type: ignore[misc]
def elementwise_mul(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    # Ensure the tensors are on the GPU
    assert X.is_cuda and Y.is_cuda, "Tensors must be on CUDA device."
    assert X.shape == Y.shape, "Tensors must have the same shape."

    # Create output tensor
    Z = torch.empty_like(X)

    # Define block size
    BLOCK_SIZE = 64

    # Grid of programs
    grid = lambda meta: (X.numel() // meta["BLOCK_SIZE"],)

    # Launch the kernel
    elementwise_mul_kernel[grid](X, Y, Z, BLOCK_SIZE=BLOCK_SIZE)

    return Z


@torch.library.register_fake("torchtrt_ex::elementwise_mul")
def elementwise_mul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return x


if torch_tensorrt.ENABLED_FEATURES.qdp_plugin:
    torch_tensorrt.dynamo.conversion.plugins.custom_op(
        "torchtrt_ex::elementwise_mul", supports_dynamic_shapes=True
    )


@torch.library.custom_op("torchtrt_ex::int8_weight_mm", mutates_args=())  # type: ignore[misc]
def int8_weight_mm(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    # torch._int_mm takes int8 operands only.
    return torch._int_mm(x.to(torch.int8), w).to(torch.float32)


@torch.library.register_fake("torchtrt_ex::int8_weight_mm")
def int8_weight_mm(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return x.new_empty((x.shape[0], w.shape[1]))


if torch_tensorrt.ENABLED_FEATURES.qdp_plugin:
    torch_tensorrt.dynamo.conversion.plugins.custom_op(
        "torchtrt_ex::int8_weight_mm", supports_dynamic_shapes=False
    )


@unittest.skipIf(
    platform.system() == "Windows",
    "Triton is not supported on Windows",
)
@unittest.skipIf(
    not torch_tensorrt.ENABLED_FEATURES.qdp_plugin,
    "QDP Plugin is not available",
)
class TestAutomaticPlugin(DispatchTestCase):

    @parameterized.expand(
        [
            ((64, 64), torch.float),
            ((256, 256), torch.int),
        ]
    )
    def test_mul_plugin_float(self, input_shape, dtype):
        class elementwise_mul(nn.Module):
            def forward(self, lhs, rhs):
                return torch.ops.torchtrt_ex.elementwise_mul.default(lhs, rhs)

        inputs = [
            torch.randint(0, 5, input_shape, device="cuda", dtype=dtype),
            torch.randint(0, 5, input_shape, device="cuda", dtype=dtype),
        ]

        self.run_test(elementwise_mul(), inputs)

    def test_int8_weight_plugin(self):
        # TensorRT takes an INT8 constant in front of a plugin, so the weight
        # must reach the plugin as int8.
        class Int8WeightMM(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer(
                    "w",
                    torch.randint(-128, 128, (64, 32), dtype=torch.int8, device="cuda"),
                )

            def forward(self, x):
                return torch.ops.torchtrt_ex.int8_weight_mm.default(x, self.w)

        inputs = [torch.randint(-8, 8, (32, 64), device="cuda", dtype=torch.float)]

        self.run_test(Int8WeightMM(), inputs)


if __name__ == "__main__":
    run_tests()
