import torch
import torch.nn as nn
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt import Input

from .harness import DispatchTestCase


class TestAtanhConverter(DispatchTestCase):
    @parameterized.expand(
        [
            ((10,), torch.float),
            ((1, 20), torch.float),
            ((2, 3, 4), torch.float),
            ((2, 3, 4, 5), torch.float),
        ]
    )
    def test_atanh_float(self, input_shape, dtype):
        class atanh(nn.Module):
            def forward(self, input):
                return torch.ops.aten.atanh.default(input)

        inputs = [torch.randn(input_shape, dtype=dtype)]
        self.run_test(
            atanh(),
            inputs,
        )

    @parameterized.expand(
        [
            ((10,), torch.int, 0, 5),
            ((1, 20), torch.int32, -10, 10),
            ((2, 3, 4), torch.int, -5, 5),
        ]
    )
    def test_atanh_int(self, input_shape, dtype, low, high):
        class atanh(nn.Module):
            def forward(self, input):
                return torch.ops.aten.atanh.default(input)

        inputs = [torch.randint(low, high, input_shape, dtype=dtype)]
        self.run_test(
            atanh(),
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
                (2, 2, 3),
                (3, 3, 3),
                torch.float,
                torch.float,
            ),
            (
                "3d_dim_dtype_int32",
                (1, 1, 1),
                (2, 2, 4),
                (2, 3, 5),
                torch.int32,
                torch.float,
            ),
        ]
    )
    def test_dynamic_shape_atanh(
        self, _, min_shape, opt_shape, max_shape, type, output_type
    ):
        class atanh(nn.Module):
            def forward(self, input):
                return torch.ops.aten.atanh.default(input)

        input_specs = [
            Input(
                min_shape=min_shape,
                opt_shape=opt_shape,
                max_shape=max_shape,
                dtype=type,
            ),
        ]
        self.run_test_with_dynamic_shape(
            atanh(), input_specs, output_dtypes=[output_type]
        )


if __name__ == "__main__":
    run_tests()
