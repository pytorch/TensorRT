from typing import Tuple

import torch
import torch.nn as nn
import torch_tensorrt
import triton
import triton.language as tl
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests

from ..conversion.harness import DispatchTestCase


@triton.jit
def elementwise_scale_mul_kernel(X, Y, Z, a, b, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    # Compute the range of elements that this thread block will work on
    block_start = pid * BLOCK_SIZE
    # Range of indices this thread will handle
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Load elements from the X and Y tensors
    x_vals = tl.load(X + offsets)
    y_vals = tl.load(Y + offsets)
    # Perform the element-wise multiplication
    z_vals = x_vals * y_vals * a + b
    # Store the result in Z
    tl.store(Z + offsets, z_vals)


@torch.library.custom_op("torchtrt_ex::elementwise_scale_mul", mutates_args=())  # type: ignore[misc]
def elementwise_scale_mul(
    X: torch.Tensor, Y: torch.Tensor, b: float = 0.2, a: int = 2
) -> torch.Tensor:
    # Ensure the tensors are on the GPU
    assert X.is_cuda and Y.is_cuda, "Tensors must be on CUDA device."
    assert X.shape == Y.shape, "Tensors must have the same shape."

    # Create output tensor
    Z = torch.empty_like(X)

    # Define block size
    BLOCK_SIZE = 1024

    # Grid of programs
    grid = lambda meta: (X.numel() // meta["BLOCK_SIZE"],)

    # Launch the kernel with parameters a and b
    elementwise_scale_mul_kernel[grid](X, Y, Z, a, b, BLOCK_SIZE=BLOCK_SIZE)

    return Z


@torch.library.register_fake("torchtrt_ex::elementwise_scale_mul")
def _(x: torch.Tensor, y: torch.Tensor, b: float = 0.2, a: int = 2) -> torch.Tensor:
    return x


torch_tensorrt.dynamo.conversion.plugins.custom_op(
    "torchtrt_ex::elementwise_scale_mul", supports_dynamic_shapes=True
)


class TestAutomaticPlugin(DispatchTestCase):
    @parameterized.expand(
        [
            ((64, 64), torch.float),
            ((256, 256), torch.int),
        ]
    )
    def test_scale_mul_plugin_float(self, input_shape, dtype):
        class elementwise_scale_mul(nn.Module):
            def forward(self, lhs, rhs):
                return torch.ops.torchtrt_ex.elementwise_scale_mul.default(
                    lhs, rhs, b=1, a=0
                )

        inputs = [
            torch.randint(0, 5, input_shape, device="cuda", dtype=dtype),
            torch.randint(0, 5, input_shape, device="cuda", dtype=dtype),
        ]

        self.run_test(elementwise_scale_mul(), inputs)


if __name__ == "__main__":
    run_tests()
