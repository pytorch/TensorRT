import torch
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt import Input

from .harness import DispatchTestCase


class TestUpsampleConverter(DispatchTestCase):
    @parameterized.expand(
        [
            ([7], [3], None),
            ([7], None, [1.5]),
        ]
    )
    def test_nearest1d(self, input_size, output_size, scale_factors):
        class TestModule(torch.nn.Module):
            def forward(self, x):
                return torch.ops.aten.upsample_nearest1d.vec(
                    x, output_size, scale_factors
                )

        inputs = [torch.randn([1, 1] + input_size)]
        self.run_test(TestModule(), inputs)

    @parameterized.expand(
        [
            ([3], None),
            (None, [1.5]),
        ]
    )
    def test_nearest1d_dynamic_shape(self, output_size, scale_factors):
        class TestModule(torch.nn.Module):
            def forward(self, x):
                return torch.ops.aten.upsample_nearest1d.vec(
                    x, output_size, scale_factors
                )

        input_specs = [
            Input(
                min_shape=(1, 1, 1),
                opt_shape=(5, 5, 5),
                max_shape=(9, 9, 9),
                dtype=torch.float32,
            )
        ]
        self.run_test_with_dynamic_shape(TestModule(), input_specs)

    @parameterized.expand(
        [
            ([7, 7], [3, 3], None),
            ([7, 7], None, [0.5, 1.5]),
        ]
    )
    def test_nearest2d(self, input_size, output_size, scale_factors):
        class TestModule(torch.nn.Module):
            def forward(self, x):
                return torch.ops.aten.upsample_nearest2d.vec(
                    x, output_size, scale_factors
                )

        inputs = [torch.randn([1, 1] + input_size)]
        self.run_test(TestModule(), inputs)

    @parameterized.expand(
        [
            ([3, 3], None),
            (None, [0.5, 1.5]),
        ]
    )
    def test_nearest2d_dynamic_shape(self, output_size, scale_factors):
        class TestModule(torch.nn.Module):
            def forward(self, x):
                return torch.ops.aten.upsample_nearest2d.vec(
                    x, output_size, scale_factors
                )

        input_specs = [
            Input(
                min_shape=(1, 1, 1, 1),
                opt_shape=(5, 5, 5, 5),
                max_shape=(9, 9, 9, 9),
                dtype=torch.float32,
            )
        ]
        self.run_test_with_dynamic_shape(TestModule(), input_specs)

    @parameterized.expand(
        [
            ([7, 7, 7], [3, 3, 3], None),
            ([7, 7, 7], None, [0.5, 1.0, 1.5]),
        ]
    )
    def test_nearest3d(self, input_size, output_size, scale_factors):
        class TestModule(torch.nn.Module):
            def forward(self, x):
                return torch.ops.aten.upsample_nearest3d.vec(
                    x, output_size, scale_factors
                )

        inputs = [torch.randn([1, 1] + input_size)]
        self.run_test(TestModule(), inputs)

    @parameterized.expand(
        [
            ([3, 3, 3], None),
            (None, [0.5, 1.0, 1.5]),
        ]
    )
    def test_nearest3d_dynamic_shape(self, output_size, scale_factors):
        class TestModule(torch.nn.Module):
            def forward(self, x):
                return torch.ops.aten.upsample_nearest3d.vec(
                    x, output_size, scale_factors
                )

        input_specs = [
            Input(
                min_shape=(1, 1, 1, 1, 1),
                opt_shape=(5, 5, 5, 5, 5),
                max_shape=(9, 9, 9, 9, 9),
                dtype=torch.float32,
            )
        ]
        self.run_test_with_dynamic_shape(TestModule(), input_specs)

    @parameterized.expand(
        [
            ([7], [3], True, None),
            ([7], [3], False, None),
            ([7], None, True, [1.5]),
            ([7], None, False, [1.5]),
        ]
    )
    def test_linear1d(self, input_size, output_size, align_corners, scale_factors):
        class TestModule(torch.nn.Module):
            def forward(self, x):
                return torch.ops.aten.upsample_linear1d.vec(
                    x, output_size, align_corners, scale_factors
                )

        inputs = [torch.randn([1, 1] + input_size)]
        self.run_test(TestModule(), inputs)

    @parameterized.expand(
        [
            ([3], True, None),
            ([3], False, None),
            (None, True, [1.5]),
            (None, False, [1.5]),
        ]
    )
    def test_linear1d_dynamic_shape(self, output_size, align_corners, scale_factors):
        class TestModule(torch.nn.Module):
            def forward(self, x):
                return torch.ops.aten.upsample_linear1d.vec(
                    x, output_size, align_corners, scale_factors
                )

        input_specs = [
            Input(
                min_shape=(1, 1, 1),
                opt_shape=(5, 5, 5),
                max_shape=(9, 9, 9),
                dtype=torch.float32,
            )
        ]
        self.run_test_with_dynamic_shape(TestModule(), input_specs)

    @parameterized.expand(
        [
            ([7, 7], [3, 3], True, None),
            ([7, 7], [3, 3], False, None),
            ([7, 7], None, True, [0.5, 1.5]),
            ([7, 7], None, False, [0.5, 1.5]),
        ]
    )
    def test_bilinear2d(self, input_size, output_size, align_corners, scale_factors):
        class TestModule(torch.nn.Module):
            def forward(self, x):
                return torch.ops.aten.upsample_bilinear2d.vec(
                    x, output_size, align_corners, scale_factors
                )

        inputs = [torch.randn([1, 1] + input_size)]
        self.run_test(TestModule(), inputs)

    @parameterized.expand(
        [
            ([3, 3], True, None),
            ([3, 3], False, None),
            (None, True, [0.5, 1.5]),
            (None, False, [0.5, 1.5]),
        ]
    )
    def test_bilinear2d_dynamic_shape(self, output_size, align_corners, scale_factors):
        class TestModule(torch.nn.Module):
            def forward(self, x):
                return torch.ops.aten.upsample_bilinear2d.vec(
                    x, output_size, align_corners, scale_factors
                )

        input_specs = [
            Input(
                min_shape=(1, 1, 1, 1),
                opt_shape=(5, 5, 5, 5),
                max_shape=(9, 9, 9, 9),
                dtype=torch.float32,
            )
        ]
        self.run_test_with_dynamic_shape(TestModule(), input_specs)

    @parameterized.expand(
        [
            ([7, 7, 7], [3, 3, 3], True, None),
            ([7, 7, 7], [3, 3, 3], False, None),
            ([7, 7, 7], None, True, [0.5, 1.0, 1.5]),
            ([7, 7, 7], None, False, [0.5, 1.0, 1.5]),
        ]
    )
    def test_trilinear3d(self, input_size, output_size, align_corners, scale_factors):
        class TestModule(torch.nn.Module):
            def forward(self, x):
                return torch.ops.aten.upsample_trilinear3d.vec(
                    x, output_size, align_corners, scale_factors
                )

        inputs = [torch.randn([1, 1] + input_size)]
        self.run_test(TestModule(), inputs)

    @parameterized.expand(
        [
            ([3, 3, 3], True, None),
            ([3, 3, 3], False, None),
            (None, True, [0.5, 1.0, 1.5]),
            (None, False, [0.5, 1.0, 1.5]),
        ]
    )
    def test_trilinear3d_dynamic_shape(self, output_size, align_corners, scale_factors):
        class TestModule(torch.nn.Module):
            def forward(self, x):
                return torch.ops.aten.upsample_trilinear3d.vec(
                    x, output_size, align_corners, scale_factors
                )

        input_specs = [
            Input(
                min_shape=(1, 1, 1, 1, 1),
                opt_shape=(5, 5, 5, 5, 5),
                max_shape=(9, 9, 9, 9, 9),
                dtype=torch.float32,
            )
        ]
        self.run_test_with_dynamic_shape(TestModule(), input_specs)

    @parameterized.expand(
        [
            ([7, 7], [3, 3], True, None),
            ([7, 7], [3, 3], False, None),
            ([7, 7], None, True, [0.5, 1.5]),
            ([7, 7], None, False, [0.5, 1.5]),
        ]
    )
    def test_bicubic2d(self, input_size, output_size, align_corners, scale_factors):
        class TestModule(torch.nn.Module):
            def forward(self, x):
                return torch.ops.aten.upsample_bicubic2d.vec(
                    x, output_size, align_corners, scale_factors
                )

        inputs = [torch.randn([1, 1] + input_size)]
        self.run_test(TestModule(), inputs)

    @parameterized.expand(
        [
            ([3, 3], True, None),
            ([3, 3], False, None),
            (None, True, [0.5, 1.5]),
            (None, False, [0.5, 1.5]),
        ]
    )
    def test_bicubic2d_dynamic_shape(self, output_size, align_corners, scale_factors):
        class TestModule(torch.nn.Module):
            def forward(self, x):
                return torch.ops.aten.upsample_bicubic2d.vec(
                    x, output_size, align_corners, scale_factors
                )

        input_specs = [
            Input(
                min_shape=(1, 1, 1, 1),
                opt_shape=(5, 5, 5, 5),
                max_shape=(9, 9, 9, 9),
                dtype=torch.float32,
            )
        ]
        self.run_test_with_dynamic_shape(TestModule(), input_specs)

    @parameterized.expand(
        [
            ([torch.tensor(3), 3], None),
            (None, [torch.tensor(0.5), 1.5]),
        ]
    )
    def test_nearest2d_mixed_dynamic_shape(self, output_size, scale_factors):
        class TestModule(torch.nn.Module):
            def forward(self, x):
                out_size = output_size
                scale = scale_factors

                return torch.ops.aten.upsample_nearest2d.vec(x, out_size, scale)

        input_specs = [
            Input(
                min_shape=(1, 1, 1, 1),
                opt_shape=(5, 5, 5, 5),
                max_shape=(9, 9, 9, 9),
                dtype=torch.float32,
            )
        ]
        self.run_test_with_dynamic_shape(TestModule(), input_specs)

    @parameterized.expand(
        [
            # Mix of Tensor and int in output_size
            ([torch.tensor(3), 3], None),
            # Mix of Tensor and float in scale_factors
            (None, [torch.tensor(0.5), 1.5]),
        ]
    )
    def test_nearest2d_mixed_static_input(self, output_size, scale_factors):
        class TestModule(torch.nn.Module):
            def forward(self, x):
                out_size = output_size
                scale = scale_factors
                return torch.ops.aten.upsample_nearest2d.vec(x, out_size, scale)

        input_size = [7, 7]  # H, W
        inputs = [torch.randn([1, 1] + input_size)]  # shape [1, 1, 7, 7]

        self.run_test(TestModule(), inputs)


if __name__ == "__main__":
    run_tests()
