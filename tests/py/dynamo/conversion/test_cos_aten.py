import torch
import torch.nn as nn
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt import Input

from .harness import DispatchTestCase


class TestCosConverter(DispatchTestCase):
    @parameterized.expand(
        [
            ((10,), torch.float),
            ((1, 20), torch.float),
            ((2, 3, 4), torch.float),
            ((2, 3, 4, 5), torch.float),
        ]
    )
    def test_cos_float(self, input_shape, dtype):
        class cos(nn.Module):
            def forward(self, input):
                return torch.ops.aten.cos.default(input)

        inputs = [torch.randn(input_shape, dtype=dtype)]
        self.run_test(
            cos(),
            inputs,
        )

    @parameterized.expand(
        [
            ((10,), torch.int, 0, 5),
            ((1, 20), torch.int32, -10, 10),
            ((2, 3, 4), torch.int, -5, 5),
        ]
    )
    def test_cos_int(self, input_shape, dtype, low, high):
        class cos(nn.Module):
            def forward(self, input):
                return torch.ops.aten.cos.default(input)

        inputs = [torch.randint(low, high, input_shape, dtype=dtype)]
        self.run_test(
            cos(),
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
                torch.randint(-128, 127, (3, 3, 4), dtype=torch.int32),
            ),
            (
                "2d_dim_dtype_float16",
                (1, 1),
                (2, 2),
                (4, 4),
                torch.float16,
                torch.randn((4, 4), dtype=torch.float16),
            ),
            (
                "3d_dim_dtype_float",
                (1, 1, 1),
                (1, 2, 3),
                (3, 3, 3),
                torch.float,
                torch.randn((3, 3, 3), dtype=torch.float32),
            ),
        ]
    )
    def test_dynamic_shape_cos(self, _, min_shape, opt_shape, max_shape, type, tensor):
        class cos(nn.Module):
            def forward(self, input):
                return torch.ops.aten.cos.default(input)

        input_specs = [
            Input(
                min_shape=min_shape,
                opt_shape=opt_shape,
                max_shape=max_shape,
                dtype=type,
                torch_tensor=tensor,
            ),
        ]

        self.run_test_with_dynamic_shape(
            cos(),
            input_specs,
            use_example_tensors=False,
        )


if __name__ == "__main__":
    run_tests()
