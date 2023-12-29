import torch
import torch.nn as nn
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests

from .harness import DispatchTestCase


class TestTruncConverter(DispatchTestCase):
    @parameterized.expand(
        [
            ((10,),),
            ((1, 20),),
            ((2, 3, 4),),
            ((2, 3, 4, 5),),
        ]
    )
    def test_trunc_float(self, shape):
        class Trunc(nn.Module):
            def forward(self, input):
                return torch.ops.aten.trunc.default(input)

        inputs = [torch.randn(shape)]
        self.run_test(
            Trunc(),
            inputs,
            enable_passes=True,
        )

    @parameterized.expand(
        [
            ((10,),),
            ((1, 20),),
            ((2, 3, 4),),
            ((2, 3, 4, 5),),
        ]
    )
    def test_trunc_int(self, shape):
        class Trunc(nn.Module):
            def forward(self, input):
                return torch.ops.aten.trunc.default(input)

        inputs = [torch.randint(-10, 10, shape, dtype=torch.int32)]
        self.run_test(
            Trunc(),
            inputs,
            enable_passes=True,
        )


if __name__ == "__main__":
    run_tests()
