import unittest

import torch
import torch.nn as nn
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt import Input

from .harness import DispatchTestCase


class TestAtan2Converter(DispatchTestCase):
    @parameterized.expand(
        [
            ((10,), torch.float),
            ((1, 20), torch.float),
            ((2, 3, 4), torch.float),
            ((2, 3, 4, 5), torch.float),
        ]
    )
    def test_atan2_lhs_const(self, input_shape, dtype):
        class atan2(nn.Module):
            def forward(self, lhs_val, rhs_val):
                return torch.ops.aten.atan2.default(lhs_val, rhs_val)

        inputs = [
            torch.randn(input_shape, dtype=dtype),
            torch.rand(1),
        ]

        self.run_test(
            atan2(),
            inputs,
        )

    @parameterized.expand(
        [
            ((10,), torch.float),
            ((1, 20), torch.float),
            ((2, 3, 4), torch.float),
            ((2, 3, 4, 5), torch.float),
        ]
    )
    def test_atan2_rhs_const(self, input_shape, dtype):
        class atan2(nn.Module):
            def forward(self, lhs_val, rhs_val):
                return torch.ops.aten.atan2.default(lhs_val, rhs_val)

        inputs = [
            torch.rand(1),
            torch.randn(input_shape, dtype=dtype),
        ]

        self.run_test(
            atan2(),
            inputs,
        )

    @parameterized.expand(
        [
            ((10,), torch.float),
            ((1, 20), torch.float),
            ((2, 3, 4), torch.float),
            ((2, 3, 4, 5), torch.float),
        ]
    )
    def test_atan2_float(self, input_shape, dtype):
        class atan2(nn.Module):
            def forward(self, lhs_val, rhs_val):
                return torch.ops.aten.atan2.default(lhs_val, rhs_val)

        inputs = [
            torch.randn(input_shape, dtype=dtype),
            torch.randn(input_shape, dtype=dtype),
        ]

        self.run_test(
            atan2(),
            inputs,
        )

    @parameterized.expand(
        [
            ((50,), torch.int, -5, 5),
            ((1, 20), torch.int32, -5, 5),
            ((2, 3, 4), torch.int, -5, 5),
        ]
    )
    def test_atan2_int(self, input_shape, dtype, low, high):
        class atan2(nn.Module):
            def forward(self, lhs_val, rhs_val):
                return torch.ops.aten.atan2.default(lhs_val, rhs_val)

        inputs = [
            torch.randint(low, high, input_shape, dtype=dtype),
            torch.randint(low, high, input_shape, dtype=dtype),
        ]
        self.run_test(
            atan2(),
            inputs,
        )

    @parameterized.expand(
        [
            (torch.float, 0.0, 0.0),
            (torch.float, 0.0, torch.rand(1)),
            (torch.float, torch.rand(1), 0.0),
            (torch.int, 0, 0),
            (torch.int, 0, torch.randint(-5, 5, (1,))),
            (torch.int, torch.randint(1, 10, (1,)), 0),
        ]
    )
    def test_atan2_zero(self, dtype, x_val, y_val):
        class atan2(nn.Module):
            def forward(self, lhs_val, rhs_val):
                return torch.ops.aten.atan2.default(lhs_val, rhs_val)

        if isinstance(x_val, torch.Tensor):
            x_val = x_val.item()
        if isinstance(y_val, torch.Tensor):
            y_val = y_val.item()

        inputs = [
            torch.tensor([x_val], dtype=dtype),
            torch.tensor([y_val], dtype=dtype),
        ]

        self.run_test(
            atan2(),
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
    def test_dynamic_shape_atan2(
        self, _, min_shape, opt_shape, max_shape, type, output_type
    ):
        class atan2(nn.Module):
            def forward(self, lhs_val, rhs_val):
                return torch.ops.aten.atan2.default(lhs_val, rhs_val)

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
            atan2(), input_specs, output_dtypes=[output_type]
        )


# torch.ops.aten.atan2.out will be decomposed/partitioned into core aten ops which torch_tensorrt supported in run_on_acc and
# non supported ops in run_on_gpu in dynamo tracer, it works via torch_tensorrt.dynamo.compile workflow
# but it won't be valid for our converter test framework, so skip it here.
@unittest.skip("skip torch.ops.aten.atan2.out converter test")
class TestAtan2OutConverter(DispatchTestCase):
    @parameterized.expand(
        [
            # dynamo trace does not allow output to be in a different shape
            # raise Unsupported(msg, case_name=case_name)
            # torch._dynamo.exc.Unsupported: out variants with resizing on graph inputs
            ((5,), (5,), torch.float),
            ((10,), (10,), torch.float),
        ]
    )
    def test_atan2_float(self, input_shape, out_shape, dtype):
        class atan2_out(nn.Module):
            def forward(self, lhs_val, rhs_val, out):
                return torch.ops.aten.atan2.out(lhs_val, rhs_val, out=out)

        out = torch.empty(out_shape)

        inputs = [
            torch.randn(input_shape, dtype=dtype),
            torch.randn(input_shape, dtype=dtype),
            out,
        ]

        self.run_test(
            atan2_out(),
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
        ]
    )
    def test_dynamic_shape_atan2_out(
        self, _, min_shape, opt_shape, max_shape, type, output_type
    ):
        class atan2(nn.Module):
            def forward(self, lhs_val, rhs_val, out):
                return torch.ops.aten.atan2.out(lhs_val, rhs_val, out=out)

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
            Input(
                min_shape=min_shape,
                opt_shape=opt_shape,
                max_shape=max_shape,
                dtype=type,
            ),
        ]
        self.run_test_with_dynamic_shape(
            atan2(),
            input_specs,
            output_dtypes=[output_type],
            use_dynamo_tracer=False,
        )


if __name__ == "__main__":
    run_tests()
