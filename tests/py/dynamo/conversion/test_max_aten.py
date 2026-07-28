import torch
import torch.nn as nn
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt import Input

from .harness import DispatchTestCase


class TestMaxConverter(DispatchTestCase):
    @parameterized.expand(
        [
            ((1, 2),),
            ((3, 2, 4),),
            ((2, 3, 4, 5),),
            ((6, 7, 5, 4, 5),),
        ]
    )
    def test_max_dim_int_default(self, input_shape):
        class Max(nn.Module):
            def forward(self, x):
                return torch.ops.aten.max.default(x)

        inputs = [torch.randn(*input_shape)]
        self.run_test(
            Max(),
            inputs,
        )

    @parameterized.expand(
        [
            ((3, 2, 4), 1, True),
            ((2, 3, 4, 5), 3, True),
            ((6, 7, 5, 4, 5), 4, False),
            ((1, 5, 2, 1), -3, False),
            ((1, 5, 2, 3), -2, True),
        ]
    )
    def test_max_dim_int(self, input_shape, dim, keep_dims):
        class Max(nn.Module):
            def forward(self, x):
                return torch.ops.aten.max.dim(x, dim, keep_dims)[0]

        inputs = [torch.randn(*input_shape)]
        self.run_test(
            Max(),
            inputs,
        )

    @parameterized.expand(
        [
            ((3, 2, 4), 1, True, torch.int, 0, 5),
            ((2, 3, 4, 5), 2, False, torch.int32, -5, 0),
            ((6, 7, 5, 4, 5), 4, False, torch.int32, -5, 5),
        ]
    )
    def test_max_dim_int_int(self, input_shape, dim, keep_dims, dtype, low, high):
        class Max(nn.Module):
            def forward(self, x):
                return torch.ops.aten.max.dim(x, dim, keep_dims)[0]

        inputs = [torch.randint(low, high, input_shape, dtype=dtype)]
        self.run_test(
            Max(),
            inputs,
            check_dtype=False,
        )

    @parameterized.expand(
        [
            (1, True, (2, 2, 3), (2, 3, 3), (3, 3, 4)),
            (2, False, (2, 3, 5), (3, 4, 6), (4, 5, 7)),
            (-1, True, (2, 3, 5), (3, 4, 6), (4, 5, 7)),
        ]
    )
    def test_max_dim_dynamic_shape(
        self, dim, keep_dim, min_shape, opt_shape, max_shape
    ):
        class Max(nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.dim = dim

            def forward(self, x):
                return torch.ops.aten.max.dim(x, dim, keep_dim)[0]

        input_specs = [
            Input(
                dtype=torch.float32,
                min_shape=min_shape,
                opt_shape=opt_shape,
                max_shape=max_shape,
            ),
        ]
        self.run_test_with_dynamic_shape(
            Max(dim),
            input_specs,
        )

    @parameterized.expand(
        [
            ((2, 2, 3), (2, 3, 3), (3, 3, 4)),
            ((2, 3, 5), (3, 4, 6), (4, 5, 7)),
            ((2, 3, 5), (3, 4, 6), (4, 5, 7)),
        ]
    )
    def test_max_default_dynamic_shape(self, min_shape, opt_shape, max_shape):
        class Max(nn.Module):
            def forward(self, x):
                return torch.ops.aten.max.default(x)

        input_specs = [
            Input(
                dtype=torch.float32,
                min_shape=min_shape,
                opt_shape=opt_shape,
                max_shape=max_shape,
            ),
        ]
        self.run_test_with_dynamic_shape(
            Max(),
            input_specs,
        )


if __name__ == "__main__":
    run_tests()
