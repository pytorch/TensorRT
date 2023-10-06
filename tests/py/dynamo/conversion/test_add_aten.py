import torch
import torch.nn as nn
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt import Input

from .harness import DispatchTestCase


class TestAddConverter(DispatchTestCase):
    @parameterized.expand(
        [
            ("2d", (2, 1)),
            ("3d", (2, 1, 2)),
        ]
    )
    def test_add_tensor(self, _, shape):
        class add(nn.Module):
            def forward(self, lhs_val, rhs_val):
                return torch.ops.aten.add.Tensor(lhs_val, rhs_val)

        inputs = [torch.randn(shape), torch.randn(shape)]
        self.run_test(
            add(),
            inputs,
        )

    @parameterized.expand(
        [
            ("2d", (2, 1), 1),
            ("3d", (2, 1, 2), 2.0),
        ]
    )
    def test_add_tensor_alpha(self, _, shape, alpha):
        class add(nn.Module):
            def forward(self, lhs_val, rhs_val):
                return torch.ops.aten.add.Tensor(lhs_val, rhs_val, alpha=alpha)

        inputs = [torch.randn(shape), torch.randn(shape)]
        self.run_test(
            add(),
            inputs,
        )

    @parameterized.expand(
        [
            ("2d", (2, 1), 1.0),
            ("3d", (2, 1, 2), 2),
        ]
    )
    def test_add_scalar(self, _, shape, scalar):
        class add(nn.Module):
            def forward(self, lhs_val):
                return torch.ops.aten.add.Tensor(lhs_val, scalar)

        inputs = [torch.randn(shape)]
        self.run_test(
            add(),
            inputs,
        )

    @parameterized.expand(
        [
            ("2d", (2, 1), 1.0, 1.0),
            ("3d", (2, 1, 2), 2, 2),
        ]
    )
    def test_add_scalar_alpha(self, _, shape, scalar, alpha):
        class add(nn.Module):
            def forward(self, lhs_val):
                return torch.ops.aten.add.Tensor(lhs_val, scalar, alpha=alpha)

        inputs = [torch.randn(shape)]
        self.run_test(
            add(),
            inputs,
        )


if __name__ == "__main__":
    run_tests()
