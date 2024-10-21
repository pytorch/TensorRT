import torch
import torch.nn as nn
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt import Input

from .harness import DispatchTestCase


class TestLogicalAndConverter(DispatchTestCase):
    @parameterized.expand(
        [
            ("2d", (2, 1)),
            ("3d", (2, 1, 2)),
        ]
    )
    def test_logical_and(self, _, shape):
        class logical_and(nn.Module):
            def forward(self, lhs_val, rhs_val):
                return torch.ops.aten.logical_and.default(lhs_val, rhs_val)

        inputs = [torch.randn(shape), torch.randn(shape)]
        self.run_test(
            logical_and(),
            inputs,
        )

    @parameterized.expand(
        [
            (
                "2d_dim_dtype_float",
                (1, 1),
                (2, 2),
                (4, 4),
                torch.float,
            ),
            (
                "3d_dim_dtype_bool",
                (1, 1, 1),
                (1, 2, 3),
                (3, 3, 3),
                torch.bool,
            ),
        ]
    )
    def test_logical_and_dynamic_shape(self, _, min_shape, opt_shape, max_shape, type):
        class logical_and(nn.Module):
            def forward(self, lhs_val, rhs_val):
                return torch.ops.aten.logical_and.default(lhs_val, rhs_val)

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
        self.run_test_with_dynamic_shape(logical_and(), input_specs)


if __name__ == "__main__":
    run_tests()
