import torch
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests

from .harness import DispatchTestCase


class TestUpsampleConverter(DispatchTestCase):
    # test case for nearest upsample, using output_size, scale_factors is disabled here
    @parameterized.expand(
        [
            ("upsample_nearest2d.vec_outshape_0", (2, 2), (4, 4)),
            ("upsample_nearest2d.vec_outshape_1", (2, 2), (5, 5)),
        ]
    )
    def test_upsample_nearest_output_shape(self, _, input_shape, output_shape):
        class Upsample(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, input):
                return torch.ops.aten.upsample_nearest2d.vec(input, output_shape, None)

        input = [torch.randn([1, 1] + list(input_shape))]
        self.run_test(Upsample(), input)

    # test case for nearest upsample, using scale_factors, output_size is disabled here
    @parameterized.expand(
        [
            ("upsample_nearest2d.vec_scale_0", (2, 2), (2, 2)),
            ("upsample_nearest2d.vec_scale_1", (2, 2), (1.5, 1.5)),
        ]
    )
    def test_upsample_nearest_scale_factor(self, _, input_shape, scale_factor):
        class Upsample(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, input):
                return torch.ops.aten.upsample_nearest2d.vec(input, None, scale_factor)

        input = [torch.randn([1, 1] + list(input_shape))]
        self.run_test(Upsample(), input)

    # test case for bilinear upsample, using output_size, scale_factors is disabled here
    @parameterized.expand(
        [
            ("upsample_bilinear2d.vec_outshape_0", (2, 2), (4, 4), True),
            ("upsample_bilinear2d.vec_outshape_1", (2, 2), (5, 5), True),
        ]
    )
    def test_upsample_bilinear_output_shape(
        self, _, input_shape, output_shape, align_corners
    ):
        class Upsample(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, input):
                return torch.ops.aten.upsample_bilinear2d.vec(
                    input,
                    output_shape,
                    align_corners,
                    None,
                )

        input = [torch.randn([1, 1] + list(input_shape))]
        self.run_test(Upsample(), input)

    # test case for bilinear upsample, using scale_factors, output_shape is disabled here
    @parameterized.expand(
        [
            ("upsample_bilinear2d.vec_scale_0", (2, 2), (2, 2), True),
            ("upsample_bilinear2d.vec_scale_1", (2, 2), (1.5, 1.5), True),
        ]
    )
    def test_upsample_bilinear_scale_factors(
        self, _, input_shape, scale_factors, align_corners
    ):
        class Upsample(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, input):
                return torch.ops.aten.upsample_bilinear2d.vec(
                    input,
                    None,
                    align_corners,
                    scale_factors,
                )

        input = [torch.randn([1, 1] + list(input_shape))]
        self.run_test(Upsample(), input)


if __name__ == "__main__":
    run_tests()
