import torch
import torch.nn as nn
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests

from .harness import DispatchTestCase


class TestGtConverter(DispatchTestCase):
    @parameterized.expand(
        [
            ("2d", (5, 3)),
            ("3d", (5, 3, 2)),
        ]
    )
    def test_gt_tensor(self, _, shape):
        class gt(nn.Module):
            def forward(self, lhs_val, rhs_val):
                return torch.ops.aten.gt.Tensor(lhs_val, rhs_val)

        inputs = [torch.randn(shape), torch.randn(shape)]
        self.run_test(
            gt(),
            inputs,
        )

    @parameterized.expand(
        [
            ("2d", (5, 3), 1),
            ("3d", (5, 3, 2), 2.0),
        ]
    )
    def test_gt_tensor_scalar(self, _, shape, scalar):
        class gt(nn.Module):
            def forward(self, lhs_val):
                return torch.ops.aten.gt.Tensor(lhs_val, torch.tensor(scalar))

        inputs = [torch.randn(shape)]
        self.run_test(
            gt(),
            inputs,
        )

    @parameterized.expand(
        [
            ("2d", (5, 3), 1),
            ("3d", (5, 3, 2), 2.0),
        ]
    )
    def test_gt_scalar(self, _, shape, scalar):
        class gt(nn.Module):
            def forward(self, lhs_val):
                return torch.ops.aten.gt.Scalar(lhs_val, scalar)

        inputs = [torch.randn(shape)]
        self.run_test(
            gt(),
            inputs,
        )


if __name__ == "__main__":
    run_tests()
