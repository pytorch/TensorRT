import torch
import torch.nn as nn
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt import Input

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

    @parameterized.expand(
        [
            ("2d_2d", (5, 3), (5, 1)),
            ("3d_2d", (5, 3, 2), (3, 1)),
            ("4d_3d", (5, 3, 4, 1), (3, 1, 1)),
        ]
    )
    def test_gt_tensor_broadcast(self, _, lshape, rshape):
        class gt(nn.Module):
            def forward(self, lhs_val, rhs_val):
                return torch.ops.aten.gt.Tensor(lhs_val, rhs_val)

        inputs = [
            torch.randint(0, 3, lshape, dtype=torch.int32),
            torch.randint(0, 3, rshape, dtype=torch.int32),
        ]
        self.run_test(
            gt(),
            inputs,
        )

    @parameterized.expand(
        [
            ("2d_2d", (2, 3), (4, 3), (5, 3), (2, 3), (4, 3), (5, 3)),
            ("3d_2d", (2, 2, 2), (2, 3, 2), (2, 4, 2), (2, 1), (3, 1), (4, 1)),
        ]
    )
    def test_gt_dynamic_tensor(self, *args):
        class gt(nn.Module):
            def forward(self, lhs_val, rhs_val):
                return torch.ops.aten.gt.Tensor(lhs_val, rhs_val)

        input_specs = [
            Input(
                min_shape=args[1],
                opt_shape=args[2],
                max_shape=args[3],
            ),
            Input(
                min_shape=args[4],
                opt_shape=args[5],
                max_shape=args[6],
            ),
        ]

        self.run_test_with_dynamic_shape(
            gt(),
            input_specs,
        )


if __name__ == "__main__":
    run_tests()
