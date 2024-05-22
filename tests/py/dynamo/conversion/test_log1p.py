import torch
import torch.nn as nn
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt import Input

from .harness import DispatchTestCase


class TestLog1pConverter(DispatchTestCase):
    @parameterized.expand(
        [
            ((10,), torch.float),
            ((1, 20), torch.float),
            ((2, 3, 4), torch.float),
            ((2, 3, 4, 5), torch.float),
        ]
    )
    def test_log1p_float(self, input_shape, dtype):
        class Log1p(nn.Module):
            def forward(self, input):
                return torch.ops.aten.log1p.default(input)

        inputs = [
            torch.randn(input_shape, dtype=dtype).abs() + 0.001
        ]  # ensure positive input
        self.run_test(
            Log1p(),
            inputs,
        )

    @parameterized.expand(
        [
            ((10,), torch.int, 0, 5),
            ((1, 20), torch.int, 0, 10),
            ((2, 3, 4), torch.int, 0, 5),
            ((2, 3, 4, 5), torch.int, 0, 5),
        ]
    )
    def test_log1p_int(self, input_shape, dtype, low, high):
        class Log1p(nn.Module):
            def forward(self, input):
                return torch.ops.aten.log1p.default(input)

        inputs = [
            torch.randint(low, high, input_shape, dtype=dtype).abs() + 0.001
        ]  # ensure positive input
        self.run_test(
            Log1p(),
            inputs,
        )

    @parameterized.expand(
        [
            (torch.full((1, 20), 2, dtype=torch.float),),
            (torch.full((2, 3, 4), 3, dtype=torch.float),),
            (torch.full((2, 3, 4, 5), 4, dtype=torch.float),),
        ]
    )
    def test_log1p_const_float(self, data):
        class Log1p(nn.Module):
            def forward(self, input):
                return torch.ops.aten.log1p.default(input)

        inputs = [data]
        self.run_test(
            Log1p(),
            inputs,
        )


if __name__ == "__main__":
    run_tests()
