import torch
import torch.nn as nn
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt import Input

from .harness import DispatchTestCase


class TestProdConverter(DispatchTestCase):
    @parameterized.expand(
        [
            ((1, 2),),
            ((3, 2, 4),),
            ((2, 3, 4, 5),),
            ((6, 7, 5, 4, 5),),
        ]
    )
    def test_prod_dim_int_default(self, input_shape):
        class Prod(nn.Module):
            def forward(self, x):
                return torch.prod(x)

        inputs = [torch.randn(input_shape)]
        self.run_test(
            Prod(),
            inputs,
            use_dynamo_tracer=True,
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
    def test_prod_dim_int(self, input_shape, dim, keep_dims):
        class Prod(nn.Module):
            def forward(self, x):
                return torch.prod(x, dim, keep_dims)

        inputs = [torch.randn(input_shape)]
        self.run_test(
            Prod(),
            inputs,
            use_dynamo_tracer=True,
        )

    @parameterized.expand(
        [
            ((3, 2, 4), 1, True, torch.int, 0, 5),
            ((2, 3, 4, 5), 2, False, torch.int32, -5, 0),
            ((6, 7, 5, 4, 5), 4, False, torch.int32, -5, 5),
        ]
    )
    def test_prod_dim_int_int(self, input_shape, dim, keep_dims, dtype, low, high):
        class Prod(nn.Module):
            def forward(self, x):
                return torch.prod(x, dim, keep_dims)

        inputs = [torch.randint(low, high, input_shape, dtype=dtype)]
        self.run_test(
            Prod(),
            inputs,
            check_dtype=False,
            use_dynamo_tracer=True,
        )

    @parameterized.expand(
        [
            (0, (2, 3), (2, 4), (3, 5)),
            (1, (2, 3), (2, 4), (3, 5)),
            (2, (2, 2, 4), (2, 3, 4), (3, 4, 5)),
            (-1, (2, 2, 4), (2, 3, 4), (3, 4, 5)),
        ]
    )
    def test_prod_dynamic_shape(self, dim, min_shape, opt_shape, max_shape):
        class Prod(nn.Module):
            def forward(self, x):
                return torch.prod(x, dim)

        input_specs = [
            Input(
                dtype=torch.float32,
                min_shape=min_shape,
                opt_shape=opt_shape,
                max_shape=max_shape,
            ),
        ]
        self.run_test_with_dynamic_shape(
            Prod(),
            input_specs,
            use_dynamo_tracer=True,
        )


if __name__ == "__main__":
    run_tests()
