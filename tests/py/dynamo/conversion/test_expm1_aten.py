from math import exp

import torch
import torch.nn as nn
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt import Input

from .harness import DispatchTestCase


class TestExpConverter(DispatchTestCase):
    @parameterized.expand(
        [
            ((10,), torch.float),
            ((1, 20), torch.float),
            ((2, 3, 4), torch.float),
            ((2, 3, 4, 5), torch.float),
        ]
    )
    def test_expm1_float(self, input_shape, dtype):
        class expm1(nn.Module):
            def forward(self, input):
                return torch.ops.aten.expm1.default(input)

        inputs = [torch.randn(input_shape, dtype=dtype)]
        self.run_test(
            expm1(),
            inputs,
        )

    @parameterized.expand(
        [
            ((1,), (3,), (5,)),
            ((1, 20), (2, 20), (3, 20)),
            ((2, 3, 4), (3, 4, 5), (4, 5, 6)),
            ((2, 3, 4, 5), (3, 5, 5, 6), (4, 5, 6, 7)),
        ]
    )
    def test_expm1_float_dynamic_shape(self, min_shape, opt_shape, max_shape):
        class expm1(nn.Module):
            def forward(self, input):
                return torch.ops.aten.expm1.default(input)

        input_specs = [
            Input(
                dtype=torch.float32,
                min_shape=min_shape,
                opt_shape=opt_shape,
                max_shape=max_shape,
            ),
        ]
        self.run_test_with_dynamic_shape(
            expm1(),
            input_specs,
        )

    @parameterized.expand(
        [
            (torch.full((1, 20), exp(1), dtype=torch.float),),
            (torch.full((2, 3, 4), exp(2), dtype=torch.float),),
            (torch.full((2, 3, 4, 5), exp(3), dtype=torch.float),),
        ]
    )
    def test_expm1_exp_const_float(self, data):
        class expm1(nn.Module):
            def forward(self, input):
                return torch.ops.aten.expm1.default(input)

        inputs = [data]
        self.run_test(
            expm1(),
            inputs,
        )

    @parameterized.expand(
        [
            ((10,), torch.int, 0, 5),
            ((1, 20), torch.int32, -10, 10),
            ((2, 3, 4), torch.int, -5, 5),
        ]
    )
    def test_exp_int(self, input_shape, dtype, low, high):
        class expm1(nn.Module):
            def forward(self, input):
                return torch.ops.aten.expm1.default(input)

        inputs = [torch.randint(low, high, input_shape, dtype=dtype)]
        self.run_test(
            expm1(),
            inputs,
        )


if __name__ == "__main__":
    run_tests()
