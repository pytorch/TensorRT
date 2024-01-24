import torch
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests

from .harness import DispatchTestCase


class TestPixelShuffleConverter(DispatchTestCase):
    @parameterized.expand(
        [
            ((1, 1, 1), 1),
            ((12, 3, 4), 2),
            ((1, 9, 4, 4), 3),
            ((2, 32, 2, 3), 4),
            ((1, 10, 36, 2, 4), 6),
        ]
    )
    def test_pixel_shuffle(self, shape, upscale_factor):
        class PixelShuffle(torch.nn.Module):
            def forward(self, x):
                return torch.ops.aten.pixel_shuffle.default(x, upscale_factor)

        inputs = [torch.randn(shape)]
        self.run_test(
            PixelShuffle(),
            inputs,
        )


if __name__ == "__main__":
    run_tests()
