import torch
import torch.nn as nn
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt import Input

from .harness import DispatchTestCase


class TestMinimumConverter(DispatchTestCase):
    @parameterized.expand(
        [
            ("2d", (2, 1)),
            ("3d", (2, 1, 2)),
        ]
    )
    def test_minimum(self, _, shape):
        class Minimum(nn.Module):
            def forward(self, lhs_val, rhs_val):
                return torch.ops.aten.minimum.default(lhs_val, rhs_val)

        inputs = [torch.randn(shape), torch.randn(shape)]
        self.run_test(
            Minimum(),
            inputs,
            use_dynamo_tracer=True,
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
    def test_minimum_dynamic_shape(
        self, _, min_shape, opt_shape, max_shape, type, output_type
    ):
        class Minimum(nn.Module):
            def forward(self, lhs_val, rhs_val):
                return torch.ops.aten.minimum.default(lhs_val, rhs_val)

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
            Minimum(), input_specs, output_dtypes=[output_type]
        )


if __name__ == "__main__":
    run_tests()
