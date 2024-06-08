import torch
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests

from .harness import DispatchTestCase


class TestUpsampleConverter(DispatchTestCase):
    @parameterized.expand(
        [
            ((2,), (4,)),
            ((2,), (5,)),
        ]
    )
    def test_nearest1d_default(self, input_shape, output_size):
        class Upsample(torch.nn.Module):
            def forward(self, input):
                return torch.ops.aten.upsample_nearest1d.default(input, output_size)

        input = [torch.randn([1, 1] + list(input_shape))]
        self.run_test(Upsample(), input)

    @parameterized.expand(
        [
            ((2,), (4,), None),
            ((2,), (5,), None),
            ((2,), None, (2,)),
            ((2,), None, (1.5,)),
        ]
    )
    def test_nearest1d_vec(self, input_shape, output_size, scale_factors):
        class Upsample(torch.nn.Module):
            def forward(self, input):
                return torch.ops.aten.upsample_nearest1d.vec(
                    input, output_size, scale_factors
                )

        input = [torch.randn([1, 1] + list(input_shape))]
        self.run_test(Upsample(), input)

    @parameterized.expand(
        [
            ((2, 2), (4, 4)),
            ((2, 2), (5, 5)),
        ]
    )
    def test_nearest2d_default(self, input_shape, output_size):
        class Upsample(torch.nn.Module):
            def forward(self, input):
                return torch.ops.aten.upsample_nearest2d.default(input, output_size)

        input = [torch.randn([1, 1] + list(input_shape))]
        self.run_test(Upsample(), input)

    @parameterized.expand(
        [
            ((2, 2), (4, 4), None),
            ((2, 2), (5, 5), None),
            ((2, 2), None, (2, 2)),
            ((2, 2), None, (1.5, 1.5)),
        ]
    )
    def test_nearest2d_vec(self, input_shape, output_size, scale_factors):
        class Upsample(torch.nn.Module):
            def forward(self, input):
                return torch.ops.aten.upsample_nearest2d.vec(
                    input, output_size, scale_factors
                )

        input = [torch.randn([1, 1] + list(input_shape))]
        self.run_test(Upsample(), input)

    @parameterized.expand(
        [
            ((2, 2, 2), (4, 4, 4)),
            ((2, 2, 2), (5, 5, 5)),
        ]
    )
    def test_nearest3d_default(self, input_shape, output_size):
        class Upsample(torch.nn.Module):
            def forward(self, input):
                return torch.ops.aten.upsample_nearest3d.default(input, output_size)

        input = [torch.randn([1, 1] + list(input_shape))]
        self.run_test(Upsample(), input)

    @parameterized.expand(
        [
            ((2, 2, 2), (4, 4, 4), None),
            ((2, 2, 2), (5, 5, 5), None),
            ((2, 2, 2), None, (2, 2, 2)),
            ((2, 2, 2), None, (1.5, 1.5, 1.5)),
        ]
    )
    def test_nearest3d_vec(self, input_shape, output_size, scale_factors):
        class Upsample(torch.nn.Module):
            def forward(self, input):
                return torch.ops.aten.upsample_nearest3d.vec(
                    input, output_size, scale_factors
                )

        input = [torch.randn([1, 1] + list(input_shape))]
        self.run_test(Upsample(), input)

    @parameterized.expand(
        [
            ((2,), (4,), True),
            ((2,), (4,), False),
            ((2,), (5,), True),
            ((2,), (5,), False),
        ]
    )
    def test_linear1d_default(self, input_shape, output_size, align_corners):
        class Upsample(torch.nn.Module):
            def forward(self, input):
                return torch.ops.aten.upsample_linear1d.default(
                    input, output_size, align_corners
                )

        input = [torch.randn([1, 1] + list(input_shape))]
        self.run_test(Upsample(), input)

    @parameterized.expand(
        [
            ((2,), (4,), True, None),
            ((2,), (4,), False, None),
            ((2,), (5,), True, None),
            ((2,), (5,), False, None),
            ((2,), None, True, (2,)),
            ((2,), None, False, (2,)),
            ((2,), None, True, (1.5,)),
            ((2,), None, False, (1.5,)),
        ]
    )
    def test_linear1d_vec(self, input_shape, output_size, align_corners, scale_factors):
        class Upsample(torch.nn.Module):
            def forward(self, input):
                return torch.ops.aten.upsample_linear1d.vec(
                    input, output_size, align_corners, scale_factors
                )

        input = [torch.randn([1, 1] + list(input_shape))]
        self.run_test(Upsample(), input)

    @parameterized.expand(
        [
            ((2, 2), (4, 4), True),
            ((2, 2), (4, 4), False),
            ((2, 2), (5, 5), True),
            ((2, 2), (5, 5), False),
        ]
    )
    def test_bilinear2d_default(self, input_shape, output_size, align_corners):
        class Upsample(torch.nn.Module):
            def forward(self, input):
                return torch.ops.aten.upsample_bilinear2d.default(
                    input, output_size, align_corners
                )

        input = [torch.randn([1, 1] + list(input_shape))]
        self.run_test(Upsample(), input)

    @parameterized.expand(
        [
            ((2, 2), (4, 4), True, None),
            ((2, 2), (4, 4), False, None),
            ((2, 2), (5, 5), True, None),
            ((2, 2), (5, 5), False, None),
            ((2, 2), None, True, (2, 2)),
            ((2, 2), None, False, (2, 2)),
            ((2, 2), None, True, (1.5, 1.5)),
            ((2, 2), None, False, (1.5, 1.5)),
        ]
    )
    def test_bilinear2d_vec(
        self, input_shape, output_size, align_corners, scale_factors
    ):
        class Upsample(torch.nn.Module):
            def forward(self, input):
                return torch.ops.aten.upsample_bilinear2d.vec(
                    input, output_size, align_corners, scale_factors
                )

        input = [torch.randn([1, 1] + list(input_shape))]
        self.run_test(Upsample(), input)

    @parameterized.expand(
        [
            ((2, 2, 2), (4, 4, 4), True),
            ((2, 2, 2), (4, 4, 4), False),
            ((2, 2, 2), (5, 5, 5), True),
            ((2, 2, 2), (5, 5, 5), False),
        ]
    )
    def test_trilinear3d_default(self, input_shape, output_size, align_corners):
        class Upsample(torch.nn.Module):
            def forward(self, input):
                return torch.ops.aten.upsample_trilinear3d.default(
                    input, output_size, align_corners
                )

        input = [torch.randn([1, 1] + list(input_shape))]
        self.run_test(Upsample(), input)

    @parameterized.expand(
        [
            ((2, 2, 2), (4, 4, 4), True, None),
            ((2, 2, 2), (4, 4, 4), False, None),
            ((2, 2, 2), (5, 5, 5), True, None),
            ((2, 2, 2), (5, 5, 5), False, None),
            ((2, 2, 2), None, True, (2, 2, 2)),
            ((2, 2, 2), None, False, (2, 2, 2)),
            ((2, 2, 2), None, True, (1.5, 1.5, 1.5)),
            ((2, 2, 2), None, False, (1.5, 1.5, 1.5)),
        ]
    )
    def test_trilinear3d_vec(
        self, input_shape, output_size, align_corners, scale_factors
    ):
        class Upsample(torch.nn.Module):
            def forward(self, input):
                return torch.ops.aten.upsample_trilinear3d.vec(
                    input, output_size, align_corners, scale_factors
                )

        input = [torch.randn([1, 1] + list(input_shape))]
        self.run_test(Upsample(), input)

    @parameterized.expand(
        [
            ((2, 2), (4, 4), True),
            ((2, 2), (4, 4), False),
            ((2, 2), (5, 5), True),
            ((2, 2), (5, 5), False),
        ]
    )
    def test_bicubic2d_default(self, input_shape, output_size, align_corners):
        class Upsample(torch.nn.Module):
            def forward(self, input):
                return torch.ops.aten.upsample_bicubic2d.default(
                    input, output_size, align_corners
                )

        input = [torch.randn([1, 1] + list(input_shape))]
        self.run_test(Upsample(), input)

    @parameterized.expand(
        [
            ((2, 2), (4, 4), True, None),
            ((2, 2), (4, 4), False, None),
            ((2, 2), (5, 5), True, None),
            ((2, 2), (5, 5), False, None),
            ((2, 2), None, True, (2, 2)),
            ((2, 2), None, False, (2, 2)),
            ((2, 2), None, True, (1.5, 1.5)),
            ((2, 2), None, False, (1.5, 1.5)),
        ]
    )
    def test_bicubic2d_vec(
        self, input_shape, output_size, align_corners, scale_factors
    ):
        class Upsample(torch.nn.Module):
            def forward(self, input):
                return torch.ops.aten.upsample_bicubic2d.vec(
                    input, output_size, align_corners, scale_factors
                )

        input = [torch.randn([1, 1] + list(input_shape))]
        self.run_test(Upsample(), input)


if __name__ == "__main__":
    run_tests()
