import torch
import torch.nn as nn
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt import Input

from .harness import DispatchTestCase


class TestGELUConverter(DispatchTestCase):
    @parameterized.expand(
        [
            ("none",),
            ("tanh",),
        ]
    )
    def test_gelu(self, approximate):
        class TestModule(nn.Module):
            def forward(self, x):
                return torch.ops.aten.gelu.default(x, approximate=approximate)

        inputs = [torch.randn(1, 10)]
        self.run_test(TestModule(), inputs)

    @parameterized.expand(
        [
            ("none",),
            ("tanh",),
        ]
    )
    def test_gelu_with_dynamic_shape(self, approximate):
        class TestModule(nn.Module):
            def forward(self, x):
                return torch.ops.aten.gelu.default(x, approximate=approximate)

        input_specs = [
            Input(
                min_shape=(1, 1, 1),
                opt_shape=(1, 2, 3),
                max_shape=(3, 3, 3),
                dtype=torch.float32,
            ),
        ]
        self.run_test_with_dynamic_shape(TestModule(), input_specs)

    @parameterized.expand(
        [
            ("none",),
            ("tanh",),
        ]
    )
    def test_gelu_with_dynamic_shape_four_dimensions(self, approximate):
        class TestModule(nn.Module):
            def forward(self, x):
                return torch.ops.aten.gelu.default(x, approximate=approximate)

        input_specs = [
            Input(
                min_shape=(1, 1, 1, 5),
                opt_shape=(1, 2, 3, 5),
                max_shape=(3, 3, 3, 5),
                dtype=torch.float32,
            ),
        ]
        self.run_test_with_dynamic_shape(TestModule(), input_specs)


if __name__ == "__main__":
    run_tests()
