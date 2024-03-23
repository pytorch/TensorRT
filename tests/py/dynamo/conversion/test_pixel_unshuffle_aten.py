import torch
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests

from .harness import DispatchTestCase


class TestPixelUnshuffleConverter(DispatchTestCase):
    @parameterized.expand(
        [
            ((1, 1, 1), 1),
            ((1, 1, 12, 12), 3),
            ((2, 3, 4, 25, 30), 5),
        ]
    )
    def test_pixel_unshuffle(self, shape, downscale_factor):
        class PixelUnshuffle(torch.nn.Module):
            def forward(self, x):
                return torch.ops.aten.pixel_unshuffle.default(x, downscale_factor)

        inputs = [torch.randn(shape)]
        self.run_test(
            PixelUnshuffle(),
            inputs,
        )


if __name__ == "__main__":
    run_tests()
