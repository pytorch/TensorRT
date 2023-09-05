import torch
import torch.nn as nn
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests

from .harness import DispatchTestCase


class TestIsInfConverter(DispatchTestCase):
    @parameterized.expand(
        [
            (
                torch.tensor(
                    [
                        1.23,
                        -4.56,
                        float("inf"),
                        float("-inf"),
                        -100.0,
                        float("nan"),
                        0.13,
                        -0.13,
                        3.14159265,
                    ]
                ),
            ),
        ]
    )
    def test_isinf_float(self, data):
        class isinf(nn.Module):
            def forward(self, input):
                return torch.isinf(input)

        inputs = [data]
        self.run_test(
            isinf(),
            inputs,
            expected_ops={torch.ops.aten.isinf.default},
            output_dtypes=[torch.bool],
        )

    @parameterized.expand(
        [
            ((10,), torch.int, 0, 5),
            ((1, 20), torch.int32, -10, 10),
            ((2, 3, 4), torch.int, -5, 5),
        ]
    )
    def test_isinf_int(self, input_shape, dtype, low, high):
        class isinf(nn.Module):
            def forward(self, input):
                return torch.isinf(input)

        inputs = [torch.randint(low, high, input_shape, dtype=dtype)]
        self.run_test(
            isinf(),
            inputs,
            expected_ops={torch.ops.aten.isinf.default},
            output_dtypes=[torch.bool],
        )


if __name__ == "__main__":
    run_tests()
