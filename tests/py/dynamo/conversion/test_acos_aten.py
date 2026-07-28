import torch
import torch.nn as nn
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt import Input

from .harness import DispatchTestCase


class TestAcosConverter(DispatchTestCase):
    @parameterized.expand(
        [
            ((10,), torch.float),
            ((1, 20), torch.float),
            ((2, 3, 4), torch.float),
            ((2, 3, 4, 5), torch.float),
        ]
    )
    def test_acos_float(self, input_shape, dtype):
        class acos(nn.Module):
            def forward(self, input):
                return torch.ops.aten.acos.default(input)

        inputs = [torch.randn(input_shape, dtype=dtype)]
        self.run_test(
            acos(),
            inputs,
        )

    @parameterized.expand(
        [
            ((10,), torch.int, 0, 5),
            ((1, 20), torch.int32, -10, 10),
            ((2, 3, 4), torch.int, -5, 5),
        ]
    )
    def test_acos_int(self, input_shape, dtype, low, high):
        class acos(nn.Module):
            def forward(self, input):
                return torch.ops.aten.acos.default(input)

        inputs = [torch.randint(low, high, input_shape, dtype=dtype)]
        self.run_test(
            acos(),
            inputs,
        )

    @parameterized.expand(
        [
            (
                "3d_dim_dtype_int32",
                (3, 2, 1),
                (3, 2, 3),
                (3, 3, 4),
                torch.int32,
                torch.float32,
            ),
            (
                "2d_dim_dtype_float16",
                (1, 1),
                (2, 2),
                (4, 4),
                torch.float16,
                torch.float16,
            ),
            (
                "3d_dim_dtype_float",
                (1, 1, 1),
                (2, 2, 3),
                (3, 3, 3),
                torch.float,
                torch.float,
            ),
        ]
    )
    def test_acos_dynamic_shape(
        self, _, min_shape, opt_shape, max_shape, type, output_type
    ):
        class acos(nn.Module):
            def forward(self, input):
                return torch.ops.aten.acos.default(input)

        input_specs = [
            Input(
                min_shape=min_shape,
                opt_shape=opt_shape,
                max_shape=max_shape,
                dtype=type,
            ),
        ]

        self.run_test_with_dynamic_shape(
            acos(),
            input_specs,
            output_dtypes=[output_type],
        )


if __name__ == "__main__":
    run_tests()
