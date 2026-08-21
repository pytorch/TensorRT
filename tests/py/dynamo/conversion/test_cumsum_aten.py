import torch
import torch.nn as nn
import torch_tensorrt
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests

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
        self.run_test(
            Cumsum(),
            inputs,
            immutable_weights=False,
        )

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
        self.run_test_with_dynamic_shape(
            Cumsum(),
            inputs,
            immutable_weights=False,
        )

    @parameterized.expand(
        [
            (torch.int32, torch.int32),  # explicit dtype keeps int32
            (torch.int32, None),  # integral promotes to int64
            (torch.int64, None),
            (torch.float16, None),
            (torch.bfloat16, None),
            (torch.float32, torch.float16),
            (torch.float32, torch.bfloat16),
        ]
    )
    def test_cumsum_dtype(self, input_dtype, out_dtype):
        class Cumsum(nn.Module):
            def forward(self, x):
                if out_dtype is None:
                    return torch.ops.aten.cumsum.default(x, 0)
                return torch.ops.aten.cumsum.default(x, 0, dtype=out_dtype)

        # 1,2,3,4 accumulate exactly in every dtype under test
        inputs = [torch.tensor([1, 2, 3, 4], dtype=input_dtype)]
        self.run_test(
            Cumsum(),
            inputs,
            use_dynamo_tracer=True,
            immutable_weights=False,
        )

    @parameterized.expand(
        [
            (torch.int32, None),
            (torch.int64, None),
            (torch.float32, torch.int64),
        ]
    )
    def test_cumsum_accumulator_is_exact(self, input_dtype, out_dtype):
        class Cumsum(nn.Module):
            def forward(self, x):
                if out_dtype is None:
                    return torch.ops.aten.cumsum.default(x, 0)
                return torch.ops.aten.cumsum.default(x, 0, dtype=out_dtype)

        # 2**24+1 is unrepresentable in float32, so a float accumulator stalls
        # here while an integral one keeps counting; the sums must be exact
        inputs = [torch.tensor([2**24, 1, 1, 1], dtype=input_dtype)]
        self.run_test(
            Cumsum(),
            inputs,
            rtol=0,
            atol=0,
            use_dynamo_tracer=True,
            immutable_weights=False,
        )

    @parameterized.expand([(torch.float16,), (torch.bfloat16,)])
    def test_cumsum_dynamic_shape_dtype(self, input_dtype):
        class Cumsum(nn.Module):
            def forward(self, x):
                return torch.ops.aten.cumsum.default(x, 0)

        # a dynamic non-cumsum dim sends the seed down full's shape-tensor path
        inputs = [
            torch_tensorrt.Input(
                min_shape=(1, 2),
                opt_shape=(2, 3),
                max_shape=(3, 4),
                dtype=input_dtype,
            ),
        ]
        self.run_test_with_dynamic_shape(
            Cumsum(),
            inputs,
            immutable_weights=False,
        )


if __name__ == "__main__":
    run_tests()
