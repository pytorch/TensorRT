import torch
import torch.nn as nn
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests

from .harness import DispatchTestCase


class TestLogicalNotConverter(DispatchTestCase):
    @parameterized.expand(
        [
            (torch.tensor([True, False, False, True]),),
            (torch.tensor([[True, False, True], [True, False, False]]),),
        ]
    )
    def test_logical_not_bool(self, data):
        class logical_not(nn.Module):
            def forward(self, input):
                return torch.logical_not(input)

        inputs = [data]
        self.run_test(
            logical_not(),
            inputs,
            expected_ops={torch.ops.aten.logical_not.default},
            output_dtypes=[torch.bool],
        )

    @parameterized.expand(
        [
            ((10,), torch.int, 0, 3),
            ((1, 20), torch.int32, -2, 2),
            ((2, 3, 4), torch.int, -5, 5),
        ]
    )
    def test_logical_not_int(self, input_shape, dtype, low, high):
        class logical_not(nn.Module):
            def forward(self, input):
                return torch.logical_not(input)

        inputs = [torch.randint(low, high, input_shape, dtype=dtype)]
        self.run_test(
            logical_not(),
            inputs,
            expected_ops={torch.ops.aten.logical_not.default},
            output_dtypes=[torch.bool],
        )

    @parameterized.expand(
        [
            ((10,), torch.float),
            ((1, 5), torch.float),
            ((2, 3, 4), torch.float),
        ]
    )
    def test_logical_not_float(self, input_shape, dtype):
        class logical_not(nn.Module):
            def forward(self, input):
                return torch.logical_not(input)

        inputs = [torch.randn(input_shape, dtype=dtype)]
        self.run_test(
            logical_not(),
            inputs,
            expected_ops={torch.ops.aten.logical_not.default},
            output_dtypes=[torch.bool],
        )


if __name__ == "__main__":
    run_tests()
