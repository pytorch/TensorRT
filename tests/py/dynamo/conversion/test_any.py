import torch
import torch.nn as nn
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests

from .harness import DispatchTestCase


class TestAnyConverter(DispatchTestCase):
    @parameterized.expand(
        [
            (
                "3d",
                (3, 2, 4),
            ),
            (
                "4d",
                (2, 3, 4, 5),
            ),
            ("5d", (6, 7, 5, 4, 5)),
        ]
    )
    def test_any_default_float_dtype(self, _, input_shape):
        class Any(nn.Module):
            def forward(self, x):
                return torch.ops.aten.any.default(x)

        inputs = [torch.randn(*input_shape)]
        self.run_test(Any(), inputs)

    @parameterized.expand(
        [
            ((3, 2, 4), 1, True),
            ((2, 3, 4, 5), 3, True),
            ((2, 3, 4, 5), 2, False),
            ((6, 7, 5, 4, 5), 4, False),
            ((1, 5, 2, 1), -1, True),
        ]
    )
    def test_any_dim_float_dtype(self, input_shape, dim, keep_dims):
        class AnyDim(nn.Module):
            def forward(self, x):
                return torch.ops.aten.any.dim(x, dim, keep_dims)

        inputs = [torch.randn(*input_shape)]
        self.run_test(AnyDim(), inputs)

    @parameterized.expand(
        [
            ((3, 2, 4), [1], True),
            ((2, 1, 4, 5), [0, 3], True),
            ((2, 3, 4, 5), [0, 1, 2, 3], False),
            ((6, 7, 5, 4, 5), [1, 3, 4], False),
        ]
    )
    def test_any_dims_tuple_float_dtype(self, input_shape, dims, keep_dims):
        class AnyDims(nn.Module):
            def forward(self, x):
                return torch.ops.aten.any.dims(x, dims, keep_dims)

        inputs = [torch.randn(*input_shape)]
        self.run_test(AnyDims(), inputs)

    @parameterized.expand(
        [
            ((3, 2, 4), torch.int, 0, 5),
            ((2, 3, 4, 5), torch.int, -10, 10),
            ((2, 3, 4, 5), torch.int32, -5, 0),
            ((6, 7, 5, 4, 5), torch.int32, -5, 5),
            ((1, 5, 2, 1), torch.int32, -5, 5),
        ]
    )
    def test_any_default_int_dtype(self, input_shape, dtype, low, high):
        class Any(nn.Module):
            def forward(self, x):
                return torch.ops.aten.any.default(x)

        inputs = [torch.randint(low, high, input_shape, dtype=dtype)]
        self.run_test(
            Any(),
            inputs,
        )

    @parameterized.expand(
        [
            ((3, 2, 4), 1, True, torch.int, 0, 5),
            ((2, 3, 4, 5), 3, True, torch.int, -10, 10),
            ((2, 3, 4, 5), 2, False, torch.int32, -5, 0),
            ((6, 7, 5, 4, 5), 4, False, torch.int32, -5, 5),
            ((1, 5, 2, 1), -4, False, torch.int32, -5, 5),
        ]
    )
    def test_any_dim_int_dtype(self, input_shape, dim, keep_dims, dtype, low, high):
        class AnyDim(nn.Module):
            def forward(self, x):
                return torch.ops.aten.any.dim(x, dim, keep_dims)

        inputs = [torch.randint(low, high, input_shape, dtype=dtype)]
        self.run_test(
            AnyDim(),
            inputs,
        )

    @parameterized.expand(
        [
            ((3, 2, 4), [1], True, torch.int, 0, 5),
            ((2, 1, 4, 5), [0, 3], True, torch.int, -10, 10),
            ((2, 3, 4, 5), [0, 1, 2, 3], False, torch.int32, -5, 0),
            ((6, 7, 5, 4, 5), [1, 3, 4], False, torch.int32, -5, 5),
            ((1, 5, 2, 1), [-3, -1], False, torch.int32, -5, 5),
        ]
    )
    def test_any_dims_tuple_int_dtype(
        self, input_shape, dims, keep_dims, dtype, low, high
    ):
        class AnyDims(nn.Module):
            def forward(self, x):
                return torch.ops.aten.any.dims(x, dims, keep_dims)

        inputs = [torch.randint(low, high, input_shape, dtype=dtype)]
        self.run_test(
            AnyDims(),
            inputs,
        )

    @parameterized.expand(
        [
            ((2, 3, 4), torch.int, -5, 0),
            ((6, 7, 5, 4, 5), torch.int, -5, 5),
            ((1, 5, 2, 1), torch.int, -5, 5),
        ]
    )
    def test_any_default_bool_dtype(self, input_shape, dtype, low, high):
        class Any(nn.Module):
            def forward(self, x):
                return torch.ops.aten.any.default(x)

        inputs = [torch.randint(low, high, input_shape, dtype=dtype).bool()]
        self.run_test(
            Any(),
            inputs,
        )

    @parameterized.expand(
        [
            ((3, 2, 4), 1, True, torch.int, 0, 5),
            ((2, 3, 4, 5), 3, True, torch.int, -10, 10),
            ((2, 3, 4, 5), 2, False, torch.int32, -5, 0),
            ((6, 7, 5, 4, 5), 4, False, torch.int32, -5, 5),
            ((1, 5, 2, 1), -4, False, torch.int32, -5, 5),
        ]
    )
    def test_any_dim_bool_dtype(self, input_shape, dim, keep_dims, dtype, low, high):
        class AnyDim(nn.Module):
            def forward(self, x):
                return torch.ops.aten.any.dim(x, dim, keep_dims)

        inputs = [torch.randint(low, high, input_shape, dtype=dtype).bool()]
        self.run_test(
            AnyDim(),
            inputs,
        )

    @parameterized.expand(
        [
            ((3, 2, 4), [1], True, torch.int, 0, 5),
            ((2, 1, 4, 5), [0, 3], True, torch.int, -10, 10),
            ((2, 3, 4, 5), [0, 1, 2, 3], False, torch.int32, -5, 0),
            ((6, 7, 5, 4, 5), [1, 3, 4], False, torch.int32, -5, 5),
            ((1, 5, 2, 1), [-3, -1], False, torch.int32, -5, 5),
        ]
    )
    def test_any_dims_tuple_bool_dtype(
        self, input_shape, dims, keep_dims, dtype, low, high
    ):
        class AnyDims(nn.Module):
            def forward(self, x):
                return torch.ops.aten.any.dims(x, dims, keep_dims)

        inputs = [torch.randint(low, high, input_shape, dtype=dtype).bool()]
        self.run_test(
            AnyDims(),
            inputs,
        )


if __name__ == "__main__":
    run_tests()
