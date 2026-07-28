import torch
import torch.nn as nn
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt import Input

from .harness import DispatchTestCase


class TestErfConverter(DispatchTestCase):
    @parameterized.expand(
        [
            ("2d_dim_dtype_float", (2, 2), torch.float),
            ("3d_dim_dtype_float", (2, 2, 2), torch.float),
            ("2d_dim_dtype_half", (2, 2), torch.half),
            ("3d_dim_dtype_half", (2, 2, 2), torch.half),
        ]
    )
    def test_erf_float(self, _, x, type):
        class erf(nn.Module):
            def forward(self, input):
                return torch.ops.aten.erf.default(input)

        inputs = [torch.randn(x, dtype=type)]
        self.run_test(erf(), inputs, precision=type)

    @parameterized.expand(
        [
            ("2d_dim_dtype_int32", (2, 2), torch.int32, 0, 5),
            ("3d_dim_dtype_int32", (2, 2, 2), torch.int32, 0, 5),
        ]
    )
    def test_erf_int(self, _, x, type, min, max):
        class erf(nn.Module):
            def forward(self, input):
                return torch.ops.aten.erf.default(input)

        inputs = [torch.randint(min, max, x, dtype=type)]
        self.run_test(
            erf(),
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
    def test_dynamic_shape_erf(
        self, _, min_shape, opt_shape, max_shape, type, output_type
    ):
        class erf(nn.Module):
            def forward(self, input):
                return torch.ops.aten.erf.default(input)

        input_specs = [
            Input(
                min_shape=min_shape,
                opt_shape=opt_shape,
                max_shape=max_shape,
                dtype=type,
            ),
        ]

        self.run_test_with_dynamic_shape(
            erf(), input_specs, output_dtypes=[output_type]
        )


if __name__ == "__main__":
    run_tests()
