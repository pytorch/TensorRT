import torch
import torch.nn as nn
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests

from .harness import DispatchTestCase


class TestBmmConverter(DispatchTestCase):
    @parameterized.expand(
        [
            ("10_3_5", (10, 3, 4), (10, 4, 5)),
            ("1_10_1", (1, 10, 1), (1, 1, 1)),
            ("1_1_1", (1, 1, 1), (1, 1, 1)),
        ]
    )
    def test_bmm(self, _, input_shape, mat2_shape):
        class BMM(nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, input, mat2):
                return torch.bmm(input, mat2)

        inputs = [torch.randn(*input_shape), torch.randn(*mat2_shape)]

        self.run_test(
            BMM(),
            inputs,
            disable_passes=True,
            expected_ops={torch.ops.aten.bmm.default},
        )


if __name__ == "__main__":
    run_tests()
