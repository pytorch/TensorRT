import torch
import torch.nn as nn
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt import Input

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


class TestAnyConverterDynamic(DispatchTestCase):
    @parameterized.expand(
        [
            (
                "3d_dynamic_float",
                (2, 1, 1),
                (2, 2, 2),
                (3, 2, 4),
                torch.float,
            ),
            (
                "2d_dynamic_int32",
                (2, 2),
                (2, 2),
                (3, 2),
                torch.int32,
            ),
            (
                "4d_dynamic_bool",
                (1, 2, 1, 1),
                (2, 2, 2, 2),
                (2, 2, 4, 3),
                torch.bool,
            ),
        ]
    )
    def test_any_dynamic(self, _, min_shape, opt_shape, max_shape, type):
        class Any(nn.Module):
            def forward(self, x):
                return torch.ops.aten.any.default(x)

        input_specs = [
            Input(
                min_shape=min_shape,
                opt_shape=opt_shape,
                max_shape=max_shape,
                dtype=type,
            ),
        ]
        self.run_test_with_dynamic_shape(
            Any(),
            input_specs,
        )

    @parameterized.expand(
        [
            (
                "3d_dynamic_dim_float",
                (2, 1, 1),
                (2, 2, 2),
                (3, 2, 4),
                torch.float,
                2,
                True,
            ),
            (
                "4d_dynamic_dim_int32",
                (1, 1, 4, 1),
                (2, 2, 4, 2),
                (2, 4, 4, 3),
                torch.int32,
                -2,
                False,
            ),
            (
                "3d_dynamic_dim_bool",
                (2, 1, 1),
                (2, 2, 2),
                (3, 2, 4),
                torch.bool,
                0,
                True,
            ),
        ]
    )
    def test_any_dynamic_dim(
        self, _, min_shape, opt_shape, max_shape, type, dim, keep_dims
    ):
        class AnyDim(nn.Module):
            def forward(self, x):
                return torch.ops.aten.any.dim(x, dim, keep_dims)

        input_specs = [
            Input(
                min_shape=min_shape,
                opt_shape=opt_shape,
                max_shape=max_shape,
                dtype=type,
            ),
        ]
        self.run_test_with_dynamic_shape(
            AnyDim(),
            input_specs,
        )

    @parameterized.expand(
        [
            (
                "3d_dynamic_dims_float",
                (2, 1, 1),
                (2, 2, 2),
                (3, 2, 4),
                torch.float,
                [1, 2],
                True,
            ),
            (
                "4d_dynamic_dims_int32",
                (1, 1, 4, 1),
                (2, 2, 4, 2),
                (2, 4, 4, 3),
                torch.int32,
                [2, -1],
                False,
            ),
            (
                "3d_dynamic_dims_bool",
                (1, 4, 1),
                (2, 4, 2),
                (4, 4, 3),
                torch.bool,
                [0, 1, 2],
                False,
            ),
        ]
    )
    def test_any_dynamic_dims(
        self, _, min_shape, opt_shape, max_shape, type, dims, keep_dims
    ):
        class AnyDims(nn.Module):
            def forward(self, x):
                return torch.ops.aten.any.dims(x, dims, keep_dims)

        input_specs = [
            Input(
                min_shape=min_shape,
                opt_shape=opt_shape,
                max_shape=max_shape,
                dtype=type,
            ),
        ]
        self.run_test_with_dynamic_shape(
            AnyDims(),
            input_specs,
        )


if __name__ == "__main__":
    run_tests()
