import torch
import torch.nn as nn
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt import Input

from .harness import DispatchTestCase


class TestNonZeroConverter(DispatchTestCase):
    @parameterized.expand(
        [
            ((10,), torch.int),
            ((1, 20), torch.int32),
            ((2, 3), torch.int64),
            ((2, 3, 4), torch.float),
            ((2, 3, 4, 5), torch.float),
        ]
    )
    def test_non_zero(self, input_shape, dtype):
        class NonZero(nn.Module):
            def forward(self, input):
                return torch.ops.aten.nonzero.default(input)

        inputs = [torch.randint(low=0, high=3, size=input_shape, dtype=dtype)]
        self.run_test(
            NonZero(),
            inputs,
        )

    @parameterized.expand(
        [
            (
                "1d",
                (1,),
                (10,),
                (100,),
                torch.int32,
            ),
            (
                "2d",
                (1, 2),
                (5, 10),
                (20, 40),
                torch.float16,
            ),
            (
                "3d",
                (1, 2, 3),
                (5, 10, 20),
                (30, 40, 50),
                torch.float,
            ),
        ]
    )
    def test_nonzero_dynamic_shape(self, _, min_shape, opt_shape, max_shape, dtype):
        class NonZero(nn.Module):
            def forward(self, input):
                return torch.ops.aten.nonzero.default(input)

        input_specs = [
            Input(
                min_shape=min_shape,
                opt_shape=opt_shape,
                max_shape=max_shape,
                dtype=dtype,
            ),
        ]

        self.run_test_with_dynamic_shape(NonZero(), input_specs)


if __name__ == "__main__":
    run_tests()
