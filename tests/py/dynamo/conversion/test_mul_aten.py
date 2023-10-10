import torch
import torch.nn as nn
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt import Input

from .harness import DispatchTestCase


class TestMulConverter(DispatchTestCase):
    @parameterized.expand(
        [
            ("2d", (2, 1)),
            ("3d", (2, 1, 2)),
        ]
    )
    def test_mul_tensor(self, _, shape):
        class mul(nn.Module):
            def forward(self, lhs_val, rhs_val):
                return torch.ops.aten.mul.Tensor(lhs_val, rhs_val)

        inputs = [torch.randn(shape), torch.randn(shape)]
        self.run_test(
            mul(),
            inputs,
        )

    @parameterized.expand(
        [
            ("2d_int", (2, 1), 1),
            ("3d_int", (2, 1, 2), 2),
            ("2d_float", (2, 1), 1.0),
            ("3d_float", (2, 1, 2), 2.0),
        ]
    )
    def test_mul_scalar(self, _, shape, scalar):
        class mul(nn.Module):
            def forward(self, lhs_val):
                return torch.ops.aten.mul.Tensor(lhs_val, scalar)

        inputs = [torch.randn(shape)]
        self.run_test(
            mul(),
            inputs,
        )


if __name__ == "__main__":
    run_tests()
