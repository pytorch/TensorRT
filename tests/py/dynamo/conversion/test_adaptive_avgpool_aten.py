import torch
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt import Input

from .harness import DispatchTestCase


class TestAdaptiveAvgPoolConverter(DispatchTestCase):
    @parameterized.expand(
        [
            (
                (2, 3),
                2,
            ),
            (
                (2, 8),
                8,
            ),
            (
                (1, 2, 3),
                2,
            ),
            (
                (2, 2, 8),
                16,
            ),
            (
                (2, 3),
                (1,),
            ),
            (
                (2, 3),
                (2,),
            ),
            (
                (2, 8),
                (4,),
            ),
            (
                (2, 8),
                (16,),
            ),
            (
                (2, 3, 1),
                (1,),
            ),
            (
                (2, 3, 2),
                (2,),
            ),
            (
                (2, 3, 4),
                (4,),
            ),
            (
                (2, 2, 32),
                (31,),
            ),
            (
                (2, 2, 32),
                (64,),
            ),
        ]
    )
    def test_adaptive_avg_pool1d(
        self,
        input_shape,
        output_size,
    ):
        class TestModule(torch.nn.Module):
            def forward(self, x):
                return torch.ops.aten.adaptive_avg_pool1d.default(x, output_size)

        inputs = [torch.randn(input_shape)]
        self.run_test(
            TestModule(),
            inputs,
            enable_passes=True,
        )

    @parameterized.expand(
        [
            (
                (1, 3, 3),
                (2, 3, 3),
                (3, 3, 3),
                torch.float,
                (2,),
            ),
        ]
    )
    def test_dynamic_shape_adaptive_pool1d(
        self,
        min_shape,
        opt_shape,
        max_shape,
        type,
        output_size,
    ):
        class adaptive_pool1d(torch.nn.Module):
            def forward(self, x):
                return torch.ops.aten.adaptive_avg_pool1d.default(x, output_size)

        input_specs = [
            Input(
                min_shape=min_shape,
                opt_shape=opt_shape,
                max_shape=max_shape,
                dtype=type,
            ),
        ]

        self.run_test_with_dynamic_shape(adaptive_pool1d(), input_specs)

    @parameterized.expand(
        [
            # 3d input
            (
                (1, 2, 3),
                (1, 2),
            ),
            (
                (1, 2, 3),
                (2, 3),
            ),
            (
                (1, 2, 8),
                (4, 4),
            ),
            (
                (2, 3, 2),
                (5, 3),
            ),
            (
                (2, 8, 16),
                (4, 8),
            ),
            (
                (2, 8, 16),
                (8, 8),
            ),
            # 4d input
            (
                (1, 1, 4, 3),
                (4, 8),
            ),
            (
                (3, 2, 3, 2),
                (1, 5),
            ),
            (
                (4, 2, 2, 8),
                (5, 2),
            ),
            (
                (3, 2, 3, 3),
                (6, 4),
            ),
            (
                (1, 2, 3, 2),
                (2, 2),
            ),
            (
                (2, 2, 32, 16),
                (8, 8),
            ),
            (
                (2, 2, 32, 32),
                (31, 16),
            ),
            (
                (1, 1, 64, 64),
                (64, 16),
            ),
        ]
    )
    def test_adaptive_avg_pool2d(
        self,
        input_shape,
        output_size,
    ):
        class TestModule(torch.nn.Module):
            def forward(self, x):
                return torch.ops.aten.adaptive_avg_pool2d.default(x, output_size)

        inputs = [torch.randn(input_shape)]
        self.run_test(
            TestModule(),
            inputs,
            enable_passes=True,
        )

    @parameterized.expand(
        [
            (
                (1, 1, 3, 3),
                (2, 2, 3, 3),
                (3, 3, 3, 3),
                torch.float,
                (2, 2),
            ),
        ]
    )
    def test_dynamic_shape_adaptive_pool2d(
        self,
        min_shape,
        opt_shape,
        max_shape,
        type,
        output_size,
    ):
        class adaptive_pool2d(torch.nn.Module):
            def forward(self, x):
                return torch.ops.aten.adaptive_avg_pool2d.default(x, output_size)

        input_specs = [
            Input(
                min_shape=min_shape,
                opt_shape=opt_shape,
                max_shape=max_shape,
                dtype=type,
            ),
        ]

        self.run_test_with_dynamic_shape(adaptive_pool2d(), input_specs)

    @parameterized.expand(
        [
            # 4d input
            (
                (1, 1, 4, 3),
                (4, 8, 2),
            ),
            (
                (1, 2, 3, 1),
                (1, 5, 2),
            ),
            (
                (1, 2, 3, 2),
                (1, 5, 3),
            ),
            (
                (4, 2, 2, 8),
                (8, 5, 2),
            ),
            (
                (3, 2, 3, 3),
                (6, 4, 1),
            ),
            (
                (1, 2, 3, 2),
                (2, 2, 2),
            ),
            (
                (2, 2, 32, 16),
                (8, 8, 8),
            ),
            (
                (2, 2, 32, 32),
                (31, 16, 64),
            ),
            (
                (1, 1, 64, 64),
                (64, 16, 1),
            ),
            # 5d input
            (
                (1, 1, 1, 4, 3),
                (4, 8, 2),
            ),
            (
                (4, 3, 1, 2, 3),
                (2, 4, 6),
            ),
            (
                (1, 4, 2, 2, 2),
                (5, 2, 4),
            ),
            (
                (3, 2, 3, 3, 2),
                (6, 4, 1),
            ),
            (
                (2, 2, 32, 16, 8),
                (8, 8, 8),
            ),
            (
                (2, 2, 32, 32, 32),
                (31, 16, 64),
            ),
            (
                (1, 1, 64, 64, 64),
                (64, 16, 1),
            ),
        ]
    )
    def test_adaptive_avgpool3d(
        self,
        input_shape,
        output_size,
    ):
        class TestModule(torch.nn.Module):
            def forward(self, x):
                return torch.ops.aten.adaptive_avg_pool3d.default(x, output_size)

        inputs = [torch.randn(input_shape)]
        self.run_test(
            TestModule(),
            inputs,
            enable_passes=True,
        )

    @parameterized.expand(
        [
            (
                (1, 1, 3, 3, 3),
                (2, 2, 3, 3, 3),
                (3, 3, 3, 3, 3),
                torch.float,
                (2, 2, 2),
            ),
        ]
    )
    def test_dynamic_shape_adaptive_pool3d(
        self,
        min_shape,
        opt_shape,
        max_shape,
        type,
        output_size,
    ):
        class adaptive_pool3d(torch.nn.Module):
            def forward(self, x):
                return torch.ops.aten.adaptive_avg_pool3d.default(x, output_size)

        input_specs = [
            Input(
                min_shape=min_shape,
                opt_shape=opt_shape,
                max_shape=max_shape,
                dtype=type,
            ),
        ]

        self.run_test_with_dynamic_shape(adaptive_pool3d(), input_specs)


if __name__ == "__main__":
    run_tests()
