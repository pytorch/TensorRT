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
            # use_dynamo_tracer=True,
            enable_passes=True,
        )

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
            # use_dynamo_tracer=True,
            enable_passes=True,
        )

    @parameterized.expand(
        [
            ((1, 2),),
        ]
    )
    def test_adaptive_avg_pool2d_dynamic(self, output_size):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                out = torch.ops.aten.adaptive_avg_pool2d.default(x, output_size)
                return out

        input_specs = [
            Input(
                shape=(-1, 2, 3, 2),
                dtype=torch.float32,
                shape_ranges=[((1, 2, 3, 2), (3, 2, 3, 2), (10, 2, 3, 2))],
            ),
        ]
        self.run_test_with_dynamic_shape(
            TestModule(),
            input_specs,
        )

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
            # use_dynamo_tracer=True,
            enable_passes=True,
        )

    @parameterized.expand(
        [
            ((1, 2, 3),),
        ]
    )
    def test_adaptive_avg_pool3d_dynamic(self, output_size):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                out = torch.ops.aten.adaptive_avg_pool3d.default(x, output_size)
                return out

        input_specs = [
            Input(
                shape=(-1, 2, 3, 1, 4),
                dtype=torch.float32,
                shape_ranges=[((1, 2, 3, 1, 4), (3, 2, 3, 1, 4), (10, 2, 3, 1, 4))],
            ),
        ]
        self.run_test_with_dynamic_shape(
            TestModule(),
            input_specs,
        )


if __name__ == "__main__":
    run_tests()
