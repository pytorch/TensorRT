import torch
import torch.nn as nn
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt import Input

from .harness import DispatchTestCase


class TestNegConverter(DispatchTestCase):
    @parameterized.expand(
        [
            ("2d_dim_dtype_float", (2, 2), torch.float),
            ("3d_dim_dtype_float", (2, 2, 2), torch.float),
            ("2d_dim_dtype_half", (2, 2), torch.half),
            ("3d_dim_dtype_half", (2, 2, 2), torch.half),
        ]
    )
    def test_neg_float(self, _, x, type):
        class neg(nn.Module):
            def forward(self, input):
                return torch.ops.aten.neg.default(input)

        inputs = [torch.randn(x, dtype=type)]
        self.run_test(neg(), inputs, precision=type)

    @parameterized.expand(
        [
            ("2d_dim_dtype_int32", (2, 2), torch.int32, 0, 5),
            ("3d_dim_dtype_int32", (2, 2, 2), torch.int32, 0, 5),
        ]
    )
    def test_neg_int(self, _, x, type, min, max):
        class neg(nn.Module):
            def forward(self, input):
                return torch.ops.aten.neg.default(input)

        inputs = [torch.randint(min, max, x, dtype=type)]
        self.run_test(
            neg(),
            inputs,
            check_dtype=False,
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
    def test_dynamic_shape_neg(
        self, _, min_shape, opt_shape, max_shape, type, output_type
    ):
        class neg(nn.Module):
            def forward(self, input):
                return torch.ops.aten.neg.default(input)

        input_specs = [
            Input(
                min_shape=min_shape,
                opt_shape=opt_shape,
                max_shape=max_shape,
                dtype=type,
            ),
        ]

        self.run_test_with_dynamic_shape(
            neg(), input_specs, output_dtypes=[output_type]
        )


if __name__ == "__main__":
    run_tests()
