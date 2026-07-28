import torch
import torch.nn as nn
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt import Input

from .harness import DispatchTestCase


class TestLogConverter(DispatchTestCase):
    @parameterized.expand(
        [
            ((10,), torch.float),
            ((1, 20), torch.float),
            ((2, 3, 4), torch.float),
            ((2, 3, 4, 5), torch.float),
        ]
    )
    def test_log_float(self, input_shape, dtype):
        class log2(nn.Module):
            def forward(self, input):
                return torch.ops.aten.log2.default(input)

        inputs = [torch.randn(input_shape, dtype=dtype)]
        self.run_test(
            log2(),
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
    def test_log_float_dynamic_shape(self, min_shape, opt_shape, max_shape):
        class log2(nn.Module):
            def forward(self, input):
                return torch.ops.aten.log2.default(input)

        input_specs = [
            Input(
                dtype=torch.float32,
                min_shape=min_shape,
                opt_shape=opt_shape,
                max_shape=max_shape,
            ),
        ]
        self.run_test_with_dynamic_shape(
            log2(),
            input_specs,
        )

    @parameterized.expand(
        [
            ((10,), torch.int, 0, 5),
            ((1, 20), torch.int, -10, 10),
            ((2, 3, 4), torch.int, -5, 5),
        ]
    )
    def test_log_int(self, input_shape, dtype, low, high):
        class log2(nn.Module):
            def forward(self, input):
                return torch.ops.aten.log2.default(input)

        inputs = [torch.randint(low, high, input_shape, dtype=dtype)]
        self.run_test(
            log2(),
            inputs,
        )


if __name__ == "__main__":
    run_tests()
