import torch
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt import Input

from .harness import DispatchTestCase


class TestUpsampleConverter(DispatchTestCase):
    @parameterized.expand(
        [
            ([7], [3], None),
            ([7], [10], 1.5),
        ]
    )
    def test_nearest1d(self, input_size, output_size, scales):
        class TestModule(torch.nn.Module):
            def forward(self, x):
                return torch.ops.aten.upsample_nearest1d.default(x, output_size, scales)

        inputs = [torch.randn([1, 1] + input_size)]
        self.run_test(TestModule(), inputs)

    @parameterized.expand(
        [
            ([3], None),
            ([13], 1.5),
        ]
    )
    def test_nearest1d_dynamic_shape(self, output_size, scales):
        class TestModule(torch.nn.Module):
            def forward(self, x):
                return torch.ops.aten.upsample_nearest1d.default(x, output_size, scales)

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
            ([7, 7], [3, 3], None, None),
            ([7, 7], [3, 10], 0.5, 1.5),
        ]
    )
    def test_nearest2d(self, input_size, output_size, scales_h, scales_w):
        class TestModule(torch.nn.Module):
            def forward(self, x):
                return torch.ops.aten.upsample_nearest2d.default(
                    x, output_size, scales_h, scales_w
                )

        inputs = [torch.randn([1, 1] + input_size)]
        self.run_test(TestModule(), inputs)

    @parameterized.expand(
        [
            ([3, 3], None, None),
            ([4, 13], 0.5, 1.5),
        ]
    )
    def test_nearest2d_dynamic_shape(self, output_size, scales_h, scales_w):
        class TestModule(torch.nn.Module):
            def forward(self, x):
                return torch.ops.aten.upsample_nearest2d.default(
                    x, output_size, scales_h, scales_w
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
            ([7, 7, 7], [3, 3, 3], None, None, None),
            ([7, 7, 7], [3, 7, 10], 0.5, 1.0, 1.5),
        ]
    )
    def test_nearest3d(self, input_size, output_size, scales_d, scales_h, scales_w):
        class TestModule(torch.nn.Module):
            def forward(self, x):
                return torch.ops.aten.upsample_nearest3d.default(
                    x, output_size, scales_d, scales_h, scales_w
                )

        inputs = [torch.randn([1, 1] + input_size)]
        self.run_test(TestModule(), inputs)

    @parameterized.expand(
        [
            ([3, 3, 3], None, None, None),
            ([4, 9, 13], 0.5, 1.0, 1.5),
        ]
    )
    def test_nearest3d_dynamic_shape(self, output_size, scales_d, scales_h, scales_w):
        class TestModule(torch.nn.Module):
            def forward(self, x):
                return torch.ops.aten.upsample_nearest3d.default(
                    x, output_size, scales_d, scales_h, scales_w
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
            ([7], [10], True, 1.5),
            ([7], [10], False, 1.5),
        ]
    )
    def test_linear1d(self, input_size, output_size, align_corners, scales):
        class TestModule(torch.nn.Module):
            def forward(self, x):
                return torch.ops.aten.upsample_linear1d.default(
                    x, output_size, align_corners, scales
                )

        inputs = [torch.randn([1, 1] + input_size)]
        self.run_test(TestModule(), inputs)

    @parameterized.expand(
        [
            ([3], True, None),
            ([3], False, None),
            ([13], True, 1.5),
            ([13], False, 1.5),
        ]
    )
    def test_linear1d_dynamic_shape(self, output_size, align_corners, scales):
        class TestModule(torch.nn.Module):
            def forward(self, x):
                return torch.ops.aten.upsample_linear1d.default(
                    x, output_size, align_corners, scales
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
            ([7, 7], [3, 3], True, None, None),
            ([7, 7], [3, 3], False, None, None),
            ([7, 7], [3, 10], True, 0.5, 1.5),
            ([7, 7], [3, 10], False, 0.5, 1.5),
        ]
    )
    def test_bilinear2d(
        self, input_size, output_size, align_corners, scales_h, scales_w
    ):
        class TestModule(torch.nn.Module):
            def forward(self, x):
                return torch.ops.aten.upsample_bilinear2d.default(
                    x, output_size, align_corners, scales_h, scales_w
                )

        inputs = [torch.randn([1, 1] + input_size)]
        self.run_test(TestModule(), inputs)

    @parameterized.expand(
        [
            ([3, 3], True, None, None),
            ([3, 3], False, None, None),
            ([4, 13], True, 0.5, 1.5),
            ([4, 13], False, 0.5, 1.5),
        ]
    )
    def test_bilinear2d_dynamic_shape(
        self, output_size, align_corners, scales_h, scales_w
    ):
        class TestModule(torch.nn.Module):
            def forward(self, x):
                return torch.ops.aten.upsample_bilinear2d.default(
                    x, output_size, align_corners, scales_h, scales_w
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
            ([7, 7, 7], [3, 3, 3], True, None, None, None),
            ([7, 7, 7], [3, 3, 3], False, None, None, None),
            ([7, 7, 7], [3, 7, 10], True, 0.5, 1.0, 1.5),
            ([7, 7, 7], [3, 7, 10], False, 0.5, 1.0, 1.5),
        ]
    )
    def test_trilinear3d(
        self, input_size, output_size, align_corners, scales_d, scales_h, scales_w
    ):
        class TestModule(torch.nn.Module):
            def forward(self, x):
                return torch.ops.aten.upsample_trilinear3d.default(
                    x, output_size, align_corners, scales_d, scales_h, scales_w
                )

        inputs = [torch.randn([1, 1] + input_size)]
        self.run_test(TestModule(), inputs)

    @parameterized.expand(
        [
            ([3, 3, 3], True, None, None, None),
            ([3, 3, 3], False, None, None, None),
            ([4, 9, 13], True, 0.5, 1.0, 1.5),
            ([4, 9, 13], False, 0.5, 1.0, 1.5),
        ]
    )
    def test_trilinear3d_dynamic_shape(
        self, output_size, align_corners, scales_d, scales_h, scales_w
    ):
        class TestModule(torch.nn.Module):
            def forward(self, x):
                return torch.ops.aten.upsample_trilinear3d.default(
                    x, output_size, align_corners, scales_d, scales_h, scales_w
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
            ([7, 7], [3, 3], True, None, None),
            ([7, 7], [3, 3], False, None, None),
            ([7, 7], [3, 10], True, 0.5, 1.5),
            ([7, 7], [3, 10], False, 0.5, 1.5),
        ]
    )
    def test_bicubic2d(
        self, input_size, output_size, align_corners, scales_h, scales_w
    ):
        class TestModule(torch.nn.Module):
            def forward(self, x):
                return torch.ops.aten.upsample_bicubic2d.default(
                    x, output_size, align_corners, scales_h, scales_w
                )

        inputs = [torch.randn([1, 1] + input_size)]
        self.run_test(TestModule(), inputs)

    @parameterized.expand(
        [
            ([3, 3], True, None, None),
            ([3, 3], False, None, None),
            ([4, 13], True, 0.5, 1.5),
            ([4, 13], False, 0.5, 1.5),
        ]
    )
    def test_bicubic2d_dynamic_shape(
        self, output_size, align_corners, scales_h, scales_w
    ):
        class TestModule(torch.nn.Module):
            def forward(self, x):
                return torch.ops.aten.upsample_bicubic2d.default(
                    x, output_size, align_corners, scales_h, scales_w
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


if __name__ == "__main__":
    run_tests()
