import torch
import torch.nn as nn
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests

from .harness import DispatchTestCase


class TestAmaxConverter(DispatchTestCase):
    @parameterized.expand(
        [
            ((3, 2, 4), 1, True),
            ((2, 3, 4, 5), 3, True),
            ((2, 3, 4, 5), 2, False),
            ((6, 7, 5, 4, 5), 4, False),
            ((1, 5, 2, 1), -1, True),
        ]
    )
    def test_amax_dim_int_default(self, input_shape, dim, keep_dims):
        class Amax(nn.Module):
            def forward(self, x):
                return torch.ops.aten.amax.default(x, dim, keep_dims)

        inputs = [torch.randn(*input_shape)]
        self.run_test(
            Amax(),
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
    def test_amax_dim_tuple_default(self, input_shape, dim, keep_dims):
        class Amax(nn.Module):
            def forward(self, x):
                return torch.ops.aten.amax.default(x, dim, keep_dims)

        inputs = [torch.randn(*input_shape)]
        self.run_test(
            Amax(),
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
    def test_amax_dim_int_int(self, input_shape, dim, keep_dims, dtype, low, high):
        class Amax(nn.Module):
            def forward(self, x):
                return torch.ops.aten.amax.default(x, dim, keep_dims)

        inputs = [torch.randint(low, high, input_shape, dtype=dtype)]
        self.run_test(
            Amax(),
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
    def test_amax_dim_tuple_int(self, input_shape, dim, keep_dims, dtype, low, high):
        class Amax(nn.Module):
            def forward(self, x):
                return torch.ops.aten.amax.default(x, dim, keep_dims)

        inputs = [torch.randint(low, high, input_shape, dtype=dtype)]
        self.run_test(
            Amax(),
            inputs,
            check_dtype=False,
        )


if __name__ == "__main__":
    run_tests()
