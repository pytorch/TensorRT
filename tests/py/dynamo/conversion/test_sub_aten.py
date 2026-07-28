import torch
import torch.nn as nn
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt import Input

from .harness import DispatchTestCase


class TestSubConverter(DispatchTestCase):
    @parameterized.expand(
        [
            ("2d", (2, 1)),
            ("3d", (2, 1, 2)),
        ]
    )
    def test_sub_tensor(self, _, shape):
        class sub(nn.Module):
            def forward(self, lhs_val, rhs_val):
                return torch.ops.aten.sub.Tensor(lhs_val, rhs_val)

        inputs = [torch.randn(shape), torch.randn(shape)]
        self.run_test(
            sub(),
            inputs,
        )

    @parameterized.expand(
        [
            ("2d", (2, 1), 1),
            ("3d", (2, 1, 2), 2.0),
        ]
    )
    def test_sub_tensor_alpha(self, _, shape, alpha):
        class sub(nn.Module):
            def forward(self, lhs_val, rhs_val):
                return torch.ops.aten.sub.Tensor(lhs_val, rhs_val, alpha=alpha)

        inputs = [torch.randn(shape), torch.randn(shape)]
        self.run_test(
            sub(),
            inputs,
        )

    @parameterized.expand(
        [
            ("2d", (2, 1), 1.0),
            ("3d", (2, 1, 2), 2),
        ]
    )
    def test_sub_scalar(self, _, shape, scalar):
        class sub(nn.Module):
            def forward(self, lhs_val):
                return torch.ops.aten.sub.Tensor(lhs_val, scalar)

        inputs = [torch.randn(shape)]
        self.run_test(
            sub(),
            inputs,
        )

    @parameterized.expand(
        [
            ("2d", (2, 1), 1.0, 1.0),
            ("3d", (2, 1, 2), 2, 2),
        ]
    )
    def test_sub_scalar_alpha(self, _, shape, scalar, alpha):
        class sub(nn.Module):
            def forward(self, lhs_val):
                return torch.ops.aten.sub.Tensor(lhs_val, scalar, alpha=alpha)

        inputs = [torch.randn(shape)]
        self.run_test(
            sub(),
            inputs,
        )

    @parameterized.expand(
        [
            (
                "3d_2d_alpha_float32",
                torch.float32,
                (1, 1, 1),
                (3, 2, 2),
                (3, 2, 4),
                (1, 1),
                (2, 2),
                (2, 4),
                1.5,
            ),
            (
                "2d_2d_alpha_int32",
                torch.int32,
                (3, 2),
                (3, 2),
                (3, 3),
                (3, 2),
                (3, 2),
                (3, 3),
                2,
            ),
        ]
    )
    def test_dynamic_shape_sub(self, *args):
        class sub(nn.Module):
            def forward(self, lhs_val, rhs_val):
                return torch.ops.aten.sub.Tensor(lhs_val, rhs_val, alpha=args[8])

        input_specs = [
            Input(
                min_shape=args[2],
                opt_shape=args[3],
                max_shape=args[4],
                dtype=args[1],
            ),
            Input(
                min_shape=args[5],
                opt_shape=args[6],
                max_shape=args[7],
                dtype=args[1],
            ),
        ]

        self.run_test_with_dynamic_shape(sub(), input_specs)

    @parameterized.expand(
        [
            (
                "3d_scalar_float32",
                torch.float32,
                (1, 1, 1),
                (3, 2, 2),
                (3, 2, 4),
                0.3,
            )
        ]
    )
    def test_dynamic_shape_sub_scalar(self, *args):
        class sub(nn.Module):
            def forward(self, lhs_val):
                return torch.ops.aten.sub.Tensor(lhs_val, args[5])

        input_specs = [
            Input(
                min_shape=args[2],
                opt_shape=args[3],
                max_shape=args[4],
                dtype=args[1],
            ),
        ]

        self.run_test_with_dynamic_shape(sub(), input_specs)

    @parameterized.expand(
        [("scalar_2d_alpha_float32", torch.float32, (1, 1), (2, 2), (3, 4), 0.3, 1.5)]
    )
    def test_dynamic_shape_sub_scalar_alpha(self, *args):
        class sub(nn.Module):
            def forward(self, rhs_val):
                return torch.ops.aten.sub.Tensor(args[5], rhs_val, alpha=args[6])

        input_specs = [
            Input(
                min_shape=args[2],
                opt_shape=args[3],
                max_shape=args[4],
                dtype=args[1],
            ),
        ]

        self.run_test_with_dynamic_shape(sub(), input_specs)


if __name__ == "__main__":
    run_tests()
