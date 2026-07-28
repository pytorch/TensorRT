import torch
import torch.nn as nn
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt import Input

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
                return torch.ops.aten.logical_not.default(input)

        inputs = [data]
        self.run_test(
            logical_not(),
            inputs,
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
                return torch.ops.aten.logical_not.default(input)

        inputs = [torch.randint(low, high, input_shape, dtype=dtype)]
        self.run_test(
            logical_not(),
            inputs,
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
                return torch.ops.aten.logical_not.default(input)

        inputs = [torch.randn(input_shape, dtype=dtype)]
        self.run_test(
            logical_not(),
            inputs,
        )

    @parameterized.expand(
        [
            ((10,), (11,), (13,)),
            ((1, 5), (2, 5), (3, 5)),
            ((2, 3, 4), (2, 3, 5), (3, 4, 6)),
        ]
    )
    def test_logical_not_dynamic_shape(self, min_shape, opt_shape, max_shape):
        class logical_not(nn.Module):
            def forward(self, input):
                return torch.ops.aten.logical_not.default(input)

        input_specs = [
            Input(
                dtype=torch.float32,
                min_shape=min_shape,
                opt_shape=opt_shape,
                max_shape=max_shape,
            ),
        ]
        self.run_test_with_dynamic_shape(
            logical_not(),
            input_specs,
        )


if __name__ == "__main__":
    run_tests()
