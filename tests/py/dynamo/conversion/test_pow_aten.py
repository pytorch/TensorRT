import torch
import torch.nn as nn
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt import Input

from .harness import DispatchTestCase


class TestPowConverter(DispatchTestCase):
    @parameterized.expand(
        [
            ("2d", (2, 1)),
            ("3d", (2, 1, 2)),
        ]
    )
    def test_pow_tensor_tensor(self, _, shape):
        class pow(nn.Module):
            def forward(self, lhs_val, rhs_val):
                return torch.ops.aten.pow.Tensor_Tensor(lhs_val, rhs_val)

        inputs = [torch.randn(shape), torch.randn(shape)]
        self.run_test(
            pow(),
            inputs,
        )

    @parameterized.expand(
        [
            ("2d", (2, 1), 1),
            ("3d", (2, 1, 2), 2.0),
        ]
    )
    def test_pow_scalar(self, _, shape, scalar):
        class pow(nn.Module):
            def forward(self, rhs_val):
                return torch.ops.aten.pow.Scalar(scalar, rhs_val)

        inputs = [torch.randn(shape)]
        self.run_test(
            pow(),
            inputs,
        )

    @parameterized.expand(
        [
            ("2d", (2, 1), 1),
            ("3d", (2, 1, 2), 2.0),
        ]
    )
    def test_pow_tensor_scalar(self, _, shape, scalar):
        class pow(nn.Module):
            def forward(self, lhs_val):
                return torch.ops.aten.pow.Tensor_Scalar(lhs_val, scalar)

        inputs = [torch.randn(shape)]
        self.run_test(
            pow(),
            inputs,
        )


if __name__ == "__main__":
    run_tests()
