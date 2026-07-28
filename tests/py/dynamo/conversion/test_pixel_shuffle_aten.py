import torch
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt import Input

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

    @parameterized.expand(
        [
            (
                (1, 1, 1),
                (2, 2, 2),
                (3, 3, 3),
                torch.float,
                1,
            ),
        ]
    )
    def test_dynamic_shape_pixel_shuffle(
        self, min_shape, opt_shape, max_shape, type, upscale_factor
    ):
        class PixelShuffle(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return torch.ops.aten.pixel_shuffle.default(x, upscale_factor)

        input_specs = [
            Input(
                min_shape=min_shape,
                opt_shape=opt_shape,
                max_shape=max_shape,
                dtype=type,
            ),
        ]

        self.run_test_with_dynamic_shape(PixelShuffle(), input_specs)


if __name__ == "__main__":
    run_tests()
