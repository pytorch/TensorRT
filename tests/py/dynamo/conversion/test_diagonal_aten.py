import torch
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt import Input

from .harness import DispatchTestCase


class TestAsStridedConverter(DispatchTestCase):
    @parameterized.expand(
        [
            (
                (3, 3),
                1,
                0,
                1,
            ),
            (
                (3, 3),
                1,
                0,
                -1,
            ),
            (
                (3, 4),
                1,
                0,
                1,
            ),
            (
                (5, 4, 2),
                -1,
                1,
                2,
            ),
            (
                (5, 4, 2),
                1,
                2,
                0,
            ),
            (
                (6, 5, 4),
                1,
                0,
                1,
            ),
            (
                (2, 5, 4, 2),
                0,
                0,
                1,
            ),
            (
                (2, 5, 4, 2),
                1,
                1,
                2,
            ),
            (
                (2, 5, 4, 2),
                1,
                -1,
                2,
            ),
            (
                (2, 5, 4, 2),
                1,
                1,
                -2,
            ),
            (
                (2, 5, 4, 2),
                1,
                -1,
                -2,
            ),
            (
                (2, 5, 4, 2),
                0,
                0,
                2,
            ),
            (
                (2, 5, 4, 2),
                -1,
                1,
                2,
            ),
            (
                (2, 5, 4, 2, 6),
                1,
                1,
                2,
            ),
            (
                (2, 5, 4, 2, 5, 6),
                1,
                1,
                2,
            ),
        ]
    )
    def test_diagonal(
        self,
        input_shape,
        offset,
        dim1,
        dim2,
    ):
        class TestModule(torch.nn.Module):
            def forward(self, x):
                return torch.ops.aten.diagonal.default(x, offset, dim1, dim2)

        inputs = [torch.randn(input_shape)]
        self.run_test(
            TestModule(),
            inputs,
            enable_passes=True,
        )


if __name__ == "__main__":
    run_tests()
