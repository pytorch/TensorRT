import torch
import torch.nn as nn
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt import Input

from .harness import DispatchTestCase


class TestRemainderConverter(DispatchTestCase):
    @parameterized.expand(
        [
            ("1d", (5,), 3),
            ("2d", (2, 1), 1.0),
            ("3d", (2, 1, 2), 2),
        ]
    )
    def test_remainder_scalar(self, _, shape, scalar):
        class Remainder(nn.Module):
            def forward(self, lhs_val):
                return torch.ops.aten.remainder.Scalar(lhs_val, scalar)

        inputs = [torch.randn(shape)]
        self.run_test(
            Remainder(),
            inputs,
        )

    def test_remainder_scalar_int(self, scalar=3):
        class Remainder(nn.Module):
            def forward(self, lhs_val):
                return torch.ops.aten.remainder.Scalar(lhs_val, scalar)

        inputs = [torch.tensor([0, 1, 2, 3, 4, -1, -2, -3, -4], dtype=torch.float32)]
        self.run_test(
            Remainder(),
            inputs,
        )

    @parameterized.expand(
        [
            ("1d", (5,)),
            ("2d", (2, 1)),
            ("3d", (2, 1, 2)),
        ]
    )
    def test_remainder_tensor(self, _, shape):
        class Remainder(nn.Module):
            def forward(self, lhs_val, rhs_val):
                return torch.ops.aten.remainder.Tensor(lhs_val, rhs_val)

        inputs = [torch.randn(shape), torch.randn(shape)]
        self.run_test(
            Remainder(),
            inputs,
        )

    @parameterized.expand(
        [
            (
                "2d_dim_dtype_half",
                (1, 1),
                (2, 2),
                (4, 4),
                torch.half,
                torch.half,
            ),
            (
                "3d_dim_dtype_float",
                (1, 1, 1),
                (1, 2, 3),
                (3, 3, 3),
                torch.float,
                torch.float,
            ),
        ]
    )
    def test_remainder_dynamic_shape(
        self, _, min_shape, opt_shape, max_shape, type, output_type
    ):
        class remainder(nn.Module):
            def forward(self, lhs_val, rhs_val):
                return torch.ops.aten.remainder.Tensor(lhs_val, rhs_val)

        class remainder_scalar(nn.Module):
            def forward(self, lhs_val, rhs_val):
                return torch.ops.aten.remainder.Scalar(lhs_val, 2)

        class mod_operator(nn.Module):
            def forward(self, lhs_val, rhs_val):
                return lhs_val % rhs_val

        input_specs = [
            Input(
                min_shape=min_shape,
                opt_shape=opt_shape,
                max_shape=max_shape,
                dtype=type,
            ),
            Input(
                min_shape=min_shape,
                opt_shape=opt_shape,
                max_shape=max_shape,
                dtype=type,
            ),
        ]
        self.run_test_with_dynamic_shape(
            remainder(), input_specs, output_dtypes=[output_type]
        )
        self.run_test_with_dynamic_shape(
            remainder_scalar(), input_specs, output_dtypes=[output_type]
        )
        self.run_test_with_dynamic_shape(
            mod_operator(), input_specs, output_dtypes=[output_type]
        )


if __name__ == "__main__":
    run_tests()
