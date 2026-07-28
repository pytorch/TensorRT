import torch
import torch.nn as nn
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests

from .harness import DispatchTestCase


class TestAddmmConverter(DispatchTestCase):
    @parameterized.expand(
        [
            ((2, 2), (2, 3), (3, 2)),
            ((4, 6), (4, 5), (5, 6)),
            ((2, 1), (2, 3), (3, 1)),
            ((4, 1), (4, 1), (1, 1)),
            ((1, 2), (1, 3), (3, 2)),
        ]
    )
    def test_addmm(self, input_shape, mat1_shape, mat2_shape):
        class Addmm(nn.Module):
            def forward(self, input, mat1, mat2):
                return torch.ops.aten.addmm.default(input, mat1, mat2)

        inputs = [
            torch.randn(input_shape),
            torch.randn(mat1_shape),
            torch.randn(mat2_shape),
        ]

        self.run_test(
            Addmm(),
            inputs,
        )

    @parameterized.expand(
        [
            ((2, 2), (2, 3), (3, 2), 1.0, 1.0),
            ((4, 6), (4, 5), (5, 6), 1.2, 0.8),
            ((2, 1), (2, 3), (3, 1), 3, 2),
            ((4, 1), (4, 1), (1, 1), 1, 1),
            ((1, 2), (1, 3), (3, 2), 2, 1.0),
            ((1, 2), (1, 3), (3, 2), 1, 2.0),
        ]
    )
    def test_addmm_scale(self, input_shape, mat1_shape, mat2_shape, beta, alpha):
        class Addmm(nn.Module):
            def forward(self, input, mat1, mat2):
                return torch.ops.aten.addmm.default(
                    input, mat1, mat2, beta=beta, alpha=alpha
                )

        inputs = [
            torch.randn(input_shape),
            torch.randn(mat1_shape),
            torch.randn(mat2_shape),
        ]

        self.run_test(
            Addmm(),
            inputs,
        )


if __name__ == "__main__":
    run_tests()
