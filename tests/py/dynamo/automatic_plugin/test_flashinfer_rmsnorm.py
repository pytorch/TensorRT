import pytest

flashinfer = pytest.importorskip("flashinfer")
import unittest

import torch
import torch.nn as nn
import torch_tensorrt
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt._enums import dtype

from ..conversion.harness import DispatchTestCase


@torch.library.custom_op("flashinfer::rmsnorm", mutates_args=())  # type: ignore[misc]
def flashinfer_rmsnorm(
    input: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    return flashinfer.norm.rmsnorm(input, weight)


@torch.library.register_fake("flashinfer::rmsnorm")
def _(input: torch.Tensor, weight: torch.Tensor, b: float = 1e-6) -> torch.Tensor:
    return input


torch_tensorrt.dynamo.conversion.plugins.custom_op(
    "flashinfer::rmsnorm", supports_dynamic_shapes=True
)


@unittest.skip("Not Available")
class TestAutomaticPlugin(DispatchTestCase):
    @parameterized.expand(
        [
            ((64, 64), (64,), torch.float16),
            ((256, 256), (256,), torch.float16),
        ]
    )
    def test_rmsnorm_float(self, input_shape, weight_shape, data_type):
        class rmsnorm(nn.Module):
            def forward(self, input, weight):
                return torch.ops.flashinfer.rmsnorm.default(input, weight)

        inputs = [
            torch.randn(input_shape, device="cuda", dtype=data_type),
            torch.randn(weight_shape, device="cuda", dtype=data_type),
        ]

        self.run_test(rmsnorm(), inputs, precision=dtype.f16)


if __name__ == "__main__":
    run_tests()
