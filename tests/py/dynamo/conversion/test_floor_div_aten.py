import torch
import torch.nn as nn
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt import Input

from .harness import DispatchTestCase


class TestFloorDivConverter(DispatchTestCase):
    @parameterized.expand(
        [
            ("2d", (2, 1)),
            ("3d", (2, 1, 2)),
        ]
    )
    def test_floor_div_default(self, _, shape):
        class floor_div(nn.Module):
            def forward(self, lhs_val, rhs_val):
                return torch.ops.aten.floor_divide.default(lhs_val, rhs_val)

        inputs = [torch.randn(shape), torch.randn(shape)]
        self.run_test(
            floor_div(),
            inputs,
        )

    @parameterized.expand(
        [
            ("2d", (2, 1), 1),
            ("3d", (2, 1, 2), 2.0),
        ]
    )
    def test_floor_div_tensor_scalar(self, _, shape, scalar):
        class floor_div(nn.Module):
            def forward(self, lhs_val):
                return torch.ops.aten.floor_divide.default(
                    lhs_val, torch.tensor(scalar)
                )

        inputs = [torch.randn(shape)]
        self.run_test(
            floor_div(),
            inputs,
        )

    @parameterized.expand(
        [
            ("2d", (2, 1), 1),
            ("3d", (2, 1, 2), 2.0),
        ]
    )
    def test_floor_div_scalar(self, _, shape, scalar):
        class floor_div(nn.Module):
            def forward(self, lhs_val):
                return torch.ops.aten.floor_divide.default(lhs_val, scalar)

        inputs = [torch.randn(shape)]
        self.run_test(
            floor_div(),
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
    def test_floor_div_dynamic_shape(
        self, _, min_shape, opt_shape, max_shape, type, output_type
    ):
        class floor_div(nn.Module):
            def forward(self, lhs_val, rhs_val):
                return torch.ops.aten.floor_divide.default(lhs_val, rhs_val)

        class floor_div_operator(nn.Module):
            def forward(self, lhs_val, rhs_val):
                return lhs_val // rhs_val

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
            floor_div(), input_specs, output_dtypes=[output_type]
        )
        self.run_test_with_dynamic_shape(
            floor_div_operator(), input_specs, output_dtypes=[output_type]
        )


if __name__ == "__main__":
    run_tests()
