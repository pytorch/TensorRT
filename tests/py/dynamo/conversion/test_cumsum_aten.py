import torch
import torch.nn as nn
import torch_tensorrt
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt._utils import is_tensorrt_rtx_version_supported
from torch_tensorrt.dynamo.conversion import UnsupportedOperatorException

from .harness import DispatchTestCase


class TestCumsumConverter(DispatchTestCase):
    @parameterized.expand(
        [
            ((1,), 0),
            ((2,), 0),
            ((3,), -1),
        ]
    )
    def test_cumsum_1D(self, shape, dim):
        class Cumsum(nn.Module):
            def forward(self, x):
                return torch.ops.aten.cumsum.default(x, dim)

        inputs = [torch.randn(shape)]
        if not is_tensorrt_rtx_version_supported("1.7"):
            with self.assertRaises(UnsupportedOperatorException):
                self.run_test(
                    Cumsum(),
                    inputs,
                    immutable_weights=False,
                    use_dynamo_tracer=True,
                )
            return

        self.run_test(Cumsum(), inputs, immutable_weights=False, use_dynamo_tracer=True)

    @parameterized.expand(
        [
            ((3, 1), 0),
            ((3, 1), 1),
            ((2, 3), -1),
            ((2, 3), -2),
        ]
    )
    def test_cumsum_2D(self, shape, dims):
        class Cumsum(nn.Module):
            def forward(self, x):
                return torch.ops.aten.cumsum.default(x, dims)

        inputs = [torch.randn(shape)]
        self.run_test(
            Cumsum(),
            inputs,
            immutable_weights=True,
        )

    @parameterized.expand(
        [
            ((2, 3, 3), 0),
            ((4, 2, 3), 1),
            ((1, 2, 3), 2),
            ((1, 2, 3), -1),
            ((1, 2, 3), -2),
        ]
    )
    def test_cumsum_3D(self, shape, dims):
        class Cumsum(nn.Module):
            def forward(self, x):
                return torch.ops.aten.cumsum.default(x, dims)

        inputs = [torch.randn(shape)]
        self.run_test(
            Cumsum(),
            inputs,
            immutable_weights=True,
        )

    @parameterized.expand(
        [
            ((1,), (2,), (3,), 0),
            ((1,), (2,), (3,), -1),
            ((2, 3), (2, 4), (2, 5), 0),
            ((2, 3), (3, 4), (4, 5), -1),
            ((1, 2, 2), (2, 2, 3), (3, 3, 3), 0),
            ((1, 2, 2), (2, 2, 3), (3, 2, 3), -2),
            ((1, 2, 2, 3), (2, 2, 3, 4), (3, 3, 4, 5), -3),
            ((1, 2, 2, 3), (2, 2, 3, 4), (3, 3, 4, 5), -2),
        ]
    )
    def test_cumsum_dynamic_shape(self, min_shape, opt_shape, max_shape, dims):
        class Cumsum(nn.Module):
            def forward(self, x):
                return torch.ops.aten.cumsum.default(x, dims)

        inputs = [
            torch_tensorrt.Input(
                min_shape=min_shape,
                opt_shape=opt_shape,
                max_shape=max_shape,
            ),
        ]
        positive_dim = dims if dims >= 0 else len(min_shape) + dims
        has_static_trip_count = (
            min_shape[positive_dim]
            == opt_shape[positive_dim]
            == max_shape[positive_dim]
        )
        if has_static_trip_count and not is_tensorrt_rtx_version_supported("1.7"):
            with self.assertRaises(UnsupportedOperatorException):
                self.run_test_with_dynamic_shape(
                    Cumsum(),
                    inputs,
                    immutable_weights=False,
                    use_dynamo_tracer=True,
                )
            return

        self.run_test_with_dynamic_shape(
            Cumsum(),
            inputs,
            immutable_weights=False,
            use_dynamo_tracer=True,
        )

    @parameterized.expand(
        [
            ("int32", torch.int32, 2**24, 6),
            ("int64", torch.int64, 10**15, 4),
        ]
    )
    def test_cumsum_integer_stays_exact(self, _, dtype, first, count):
        """The running total has to be accumulated in an integer type.

        This asserts equality rather than going through run_test, whose tolerance is
        relative: accumulating in float32 puts a total of 1e15 out by about 13 million,
        which is well inside that tolerance and so invisible to an approximate comparison.
        """

        class Cumsum(nn.Module):
            def forward(self, x):
                return torch.ops.aten.cumsum.default(x, 1)

        values = torch.tensor([[first] + [1] * (count - 1)], dtype=dtype, device="cuda")
        module = Cumsum().eval().cuda()
        expected = module(values)

        compiled = torch_tensorrt.dynamo.compile(
            torch.export.export(module, (values,)),
            inputs=[values],
            min_block_size=1,
            enabled_precisions={torch.float32},
            truncate_double=True,
        )
        self.assertTrue(
            torch.equal(compiled(values).cpu(), expected.cpu()),
            f"expected {expected.flatten().tolist()}, "
            f"got {compiled(values).flatten().tolist()}",
        )

    @parameterized.expand(
        [
            ("float16", torch.float16),
            ("bfloat16", torch.bfloat16),
        ]
    )
    def test_cumsum_reduced_precision_float_stays_accurate(self, _, dtype):
        """A long running sum in a narrow float drifts, because every partial total is
        rounded to the operand type. eager accumulates in float32, so the engine has to as
        well and cast the result back, or a thousand-long sum is off by many units.
        """

        class Cumsum(nn.Module):
            def forward(self, x):
                return torch.ops.aten.cumsum.default(x, 1)

        values = torch.ones((2, 1000), dtype=dtype, device="cuda")
        module = Cumsum().eval().cuda()
        expected = module(values)

        compiled = torch_tensorrt.dynamo.compile(
            torch.export.export(module, (values,)),
            inputs=[values],
            min_block_size=1,
            enabled_precisions={torch.float32},
            truncate_double=True,
        )
        result = compiled(values)
        self.assertEqual(result.dtype, expected.dtype)
        max_abs_diff = (result.float() - expected.float()).abs().max().item()
        self.assertLess(
            max_abs_diff,
            1.0,
            f"cumsum drifted by {max_abs_diff}, so it accumulated in {dtype} not float32",
        )

    @parameterized.expand(
        [
            ("bool", torch.bool),
            ("int32", torch.int32),
        ]
    )
    def test_cumsum_integer_accumulates_in_int64(self, _, dtype):
        """A bool input must not accumulate in bool, or every non-zero running total comes
        back as True."""

        class Cumsum(nn.Module):
            def forward(self, x):
                return torch.ops.aten.cumsum.default(x, 1)

        if dtype is torch.bool:
            inputs = [torch.tensor([[True, False, True, True, False, True]])]
        else:
            inputs = [torch.tensor([[1, 0, 1, 1, 0, 1]], dtype=dtype)]
        self.run_test(
            Cumsum(),
            inputs,
            use_dynamo_tracer=True,
        )

    def test_cumsum_honors_dtype_argument(self):
        """The keyword-only dtype argument selects the accumulator type."""

        class Cumsum(nn.Module):
            def forward(self, x):
                return torch.ops.aten.cumsum.default(x, 1, dtype=torch.float32)

        inputs = [torch.tensor([[1, 0, 1, 1]], dtype=torch.int32)]
        self.run_test(
            Cumsum(),
            inputs,
            use_dynamo_tracer=True,
        )

    def test_cumsum_rank4(self):
        """Rank 4 static, which the existing cases cover only with dynamic shapes."""

        class Cumsum(nn.Module):
            def forward(self, x):
                return torch.ops.aten.cumsum.default(x, 2)

        inputs = [torch.randn(1, 2, 5, 3)]
        self.run_test(
            Cumsum(),
            inputs,
            use_dynamo_tracer=True,
        )


if __name__ == "__main__":
    run_tests()
