import torch
import torch.nn as nn
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt import Input

from .harness import DispatchTestCase


class TestAminConverter(DispatchTestCase):
    @parameterized.expand(
        [
            ((3, 2, 4), 1, True),
            ((2, 3, 4, 5), 3, True),
            ((2, 3, 4, 5), 2, False),
            ((6, 7, 5, 4, 5), 4, False),
            ((1, 5, 2, 1), -1, True),
        ]
    )
    def test_amin_dim_int_default(self, input_shape, dim, keep_dims):
        class Amin(nn.Module):
            def forward(self, x):
                return torch.ops.aten.amin.default(x, dim, keep_dims)

        inputs = [torch.randn(*input_shape)]
        self.run_test(
            Amin(),
            inputs,
        )

    @parameterized.expand(
        [
            ((1, 2, 4), [], True),
            ((3, 2, 4), [1], True),
            ((2, 1, 4, 5), [0, 3], True),
            ((2, 3, 4, 5), [0, 1, 2, 3], False),
            ((6, 7, 5, 4, 5), [1, 3, 4], False),
        ]
    )
    def test_amin_dim_tuple_default(self, input_shape, dim, keep_dims):
        class Amin(nn.Module):
            def forward(self, x):
                return torch.ops.aten.amin.default(x, dim, keep_dims)

        inputs = [torch.randn(*input_shape)]
        self.run_test(
            Amin(),
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
    def test_amin_dim_int_int(self, input_shape, dim, keep_dims, dtype, low, high):
        class Amin(nn.Module):
            def forward(self, x):
                return torch.ops.aten.amin.default(x, dim, keep_dims)

        inputs = [torch.randint(low, high, input_shape, dtype=dtype)]
        self.run_test(
            Amin(),
            inputs,
            check_dtype=False,
        )

    @parameterized.expand(
        [
            ((1, 2, 4), [], True, torch.int, 0, 5),
            ((3, 2, 4), [1], True, torch.int, 0, 5),
            ((2, 1, 4, 5), [0, 3], True, torch.int, -10, 10),
            ((2, 3, 4, 5), [0, 1, 2, 3], False, torch.int32, -5, 0),
            ((6, 7, 5, 4, 5), [1, 3, 4], False, torch.int32, -5, 5),
            ((1, 5, 2, 1), [-3, -1], False, torch.int32, -5, 5),
        ]
    )
    def test_amin_dim_tuple_int(self, input_shape, dim, keep_dims, dtype, low, high):
        class Amin(nn.Module):
            def forward(self, x):
                return torch.ops.aten.amin.default(x, dim, keep_dims)

        inputs = [torch.randint(low, high, input_shape, dtype=dtype)]
        self.run_test(
            Amin(),
            inputs,
            check_dtype=False,
        )

    @parameterized.expand(
        [
            ((0, 1), True, (2, 3, 5), (3, 4, 6), (4, 5, 7)),
            ((0,), False, (2, 3, 5), (3, 4, 6), (4, 5, 7)),
            (1, True, (2, 3, 5), (3, 4, 6), (4, 5, 7)),
            (2, False, (2, 3, 5), (3, 4, 6), (4, 5, 7)),
            (-1, True, (2, 3, 5), (3, 4, 6), (4, 5, 7)),
            ((-1, 0), True, (2, 3, 5), (3, 4, 6), (4, 5, 7)),
        ]
    )
    def test_amin_dynamic_shape(self, dim, keep_dim, min_shape, opt_shape, max_shape):
        class Amin(nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.dim = dim

            def forward(self, x):
                return torch.ops.aten.amin.default(x, dim, keep_dim)

        input_specs = [
            Input(
                dtype=torch.float32,
                min_shape=min_shape,
                opt_shape=opt_shape,
                max_shape=max_shape,
            ),
        ]
        self.run_test_with_dynamic_shape(
            Amin(dim),
            input_specs,
        )


if __name__ == "__main__":
    run_tests()
