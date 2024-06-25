import torch
import torch.nn as nn
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt import Input

from .harness import DispatchTestCase


class TestSumConverter(DispatchTestCase):
    @parameterized.expand(
        [
            ((1, 2),),
            ((3, 2, 4),),
            ((2, 3, 4, 5),),
            ((6, 7, 5, 4, 5),),
        ]
    )
    def test_sum_dim_int_default(self, input_shape):
        class Sum(nn.Module):
            def forward(self, x):
                return torch.ops.aten.sum.default(x)

        inputs = [torch.randn(*input_shape)]
        self.run_test(
            Sum(),
            inputs,
        )

    @parameterized.expand(
        [
            ((3, 2, 4), 1, True),
            ((2, 3, 4, 5), 3, True),
            ((2, 3, 4, 5), None, False),
            ((6, 7, 5, 4, 5), 4, False),
            ((1, 5, 2, 1), -3, False),
            ((1, 5, 2, 3), -2, True),
        ]
    )
    def test_sum_dim_int(self, input_shape, dim, keep_dims):
        class Sum(nn.Module):
            def forward(self, x):
                return torch.ops.aten.sum.dim_IntList(x, dim, keep_dims)

        inputs = [torch.randn(*input_shape)]
        self.run_test(
            Sum(),
            inputs,
        )

    @parameterized.expand(
        [
            ((1, 2, 4), [], True),
            ((3, 2, 4), [1], True),
            ((2, 1, 4, 5), None, True),
            ((2, 3, 4, 5), [0, 1, 2, 3], False),
            ((6, 7, 5, 4, 5), [1, 3, 4], False),
            ((6, 7, 5, 4, 5), [-5, -4, -2], False),
        ]
    )
    def test_sum_dim_tuple(self, input_shape, dim, keep_dims):
        class Sum(nn.Module):
            def forward(self, x):
                return torch.ops.aten.sum.dim_IntList(x, dim, keep_dims)

        inputs = [torch.randn(*input_shape)]
        self.run_test(
            Sum(),
            inputs,
        )

    @parameterized.expand(
        [
            ((3, 2, 4), 1, True, torch.int32, 0, 5),
            ((2, 3, 4, 5), None, True, torch.int32, -10, 10),
            ((2, 3, 4, 5), 2, False, torch.int32, -5, 0),
            ((6, 7, 5, 4, 5), 4, False, torch.int32, -5, 5),
        ]
    )
    def test_sum_dim_int_int(self, input_shape, dim, keep_dims, dtype, low, high):
        class Sum(nn.Module):
            def forward(self, x):
                return torch.ops.aten.sum.dim_IntList(x, dim, keep_dims)

        inputs = [torch.randint(low, high, input_shape, dtype=dtype)]
        self.run_test(
            Sum(),
            inputs,
        )

    @parameterized.expand(
        [
            ((1, 2, 4), [], True, torch.int32, 0, 5),
            ((3, 2, 4), [1], True, torch.int32, 0, 5),
            ((2, 1, 4, 5), [0, 3], True, torch.int32, -10, 10),
            ((2, 3, 4, 5), None, False, torch.int32, -5, 0),
            ((6, 7, 5, 4, 5), [1, 3, 4], False, torch.int32, -5, 5),
            ((6, 7, 5, 4, 5), [1, 3, 4], False, torch.bool, 0, 2),
            ((4, 7, 1, 5), None, True, torch.bool, 0, 2),
        ]
    )
    def test_sum_dim_tuple_int(self, input_shape, dim, keep_dims, dtype, low, high):
        class Sum(nn.Module):
            def forward(self, x):
                return torch.ops.aten.sum.dim_IntList(x, dim, keep_dims)

        inputs = [torch.randint(low, high, input_shape, dtype=dtype)]
        self.run_test(
            Sum(),
            inputs,
        )


class TestPrimsSumConverter(DispatchTestCase):
    @parameterized.expand(
        [
            ((3, 2, 4), [1]),
            ((2, 1, 4, 5), [1, 2]),
            ((2, 3, 4, 5), [0, 1, 2, 3]),
            ((6, 7, 5, 4, 5), [1, 3, 4]),
        ]
    )
    def test_sum_dim_sequence(self, input_shape, dim):
        class Sum(nn.Module):
            def forward(self, x):
                return torch.ops.prims.sum.default(x, dim)

        inputs = [torch.randn(*input_shape)]
        self.run_test(
            Sum(),
            inputs,
        )

    @parameterized.expand(
        [
            ([0], (2, 3), (2, 4), (3, 5)),
            ([1], (2, 3), (2, 4), (3, 5)),
            (
                [
                    2,
                ],
                (2, 2, 4),
                (2, 3, 4),
                (3, 4, 5),
            ),
            ([0, 1], (2, 2, 4), (2, 3, 4), (3, 4, 5)),
        ]
    )
    def test_sum_dynamic_shape(self, dim, min_shape, opt_shape, max_shape):
        class Sum(nn.Module):
            def forward(self, x):
                return torch.ops.prims.sum.default(x, dim)

        input_specs = [
            Input(
                dtype=torch.float32,
                min_shape=min_shape,
                opt_shape=opt_shape,
                max_shape=max_shape,
            ),
        ]
        self.run_test_with_dynamic_shape(
            Sum(),
            input_specs,
        )


if __name__ == "__main__":
    run_tests()
