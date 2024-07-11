import torch
import torch.nn as nn
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt import Input

from .harness import DispatchTestCase


class TestSignConverter(DispatchTestCase):
    @parameterized.expand(
        [
            ((10,), torch.float),
            ((1, 20), torch.float),
            ((2, 3, 4), torch.float),
            ((2, 3, 4, 5), torch.float),
        ]
    )
    def test_sign_float(self, input_shape, dtype):
        class sign(nn.Module):
            def forward(self, input):
                return torch.ops.aten.sign.default(input)

        inputs = [torch.randn(input_shape, dtype=dtype)]
        self.run_test(
            sign(),
            inputs,
        )

    @parameterized.expand(
        [
            ((10,), (11,), (12,)),
            ((1, 3, 4), (2, 3, 5), (3, 4, 6)),
            ((2, 3, 4, 5), (3, 5, 4, 5), (4, 6, 4, 5)),
        ]
    )
    def test_sign_dynamic_shape(self, min_shape, opt_shape, max_shape):
        class sign(nn.Module):
            def forward(self, input):
                return torch.ops.aten.sign.default(input)

        input_specs = [
            Input(
                dtype=torch.float32,
                min_shape=min_shape,
                opt_shape=opt_shape,
                max_shape=max_shape,
            ),
        ]
        self.run_test_with_dynamic_shape(
            sign(),
            input_specs,
        )

    @parameterized.expand(
        [
            ((10,), torch.int, -2, 2),
            ((1, 20), torch.int32, -10, 10),
            ((2, 3, 4), torch.int, -100, 100),
        ]
    )
    def test_sign_int(self, input_shape, dtype, low, high):
        class sign(nn.Module):
            def forward(self, input):
                return torch.ops.aten.sign.default(input)

        inputs = [torch.randint(low, high, input_shape, dtype=dtype)]
        self.run_test(
            sign(),
            inputs,
            check_dtype=False,
        )


if __name__ == "__main__":
    run_tests()
