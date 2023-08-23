import torch
import torch.nn as nn
from harness import DispatchTestCase
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt import Input


class TestCeilConverter(DispatchTestCase):
    @parameterized.expand(
        [
            ((10,), torch.float),
            ((1, 20), torch.float),
            ((2, 3, 4), torch.float),
            ((2, 3, 4, 5), torch.float),
        ]
    )
    def test_ceil_float(self, input_shape, dtype):
        class ceil(nn.Module):
            def forward(self, input):
                return torch.ceil(input)

        inputs = [torch.randn(input_shape, dtype=dtype)]
        self.run_test(
            ceil(),
            inputs,
            expected_ops={torch.ops.aten.ceil.default},
        )

    @parameterized.expand(
        [
            ((10,), torch.int, 0, 5),
            ((1, 20), torch.int32, -10, 10),
            ((2, 3, 4), torch.int, -5, 5),
        ]
    )
    def test_ceil_int(self, input_shape, dtype, low, high):
        class ceil(nn.Module):
            def forward(self, input):
                return torch.ceil(input)

        inputs = [torch.randint(low, high, input_shape, dtype=dtype)]
        self.run_test(
            ceil(),
            inputs,
            expected_ops={torch.ops.aten.ceil.default},
            check_dtype=False,
        )


if __name__ == "__main__":
    run_tests()
