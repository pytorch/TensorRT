import torch
import torch.nn as nn
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt import Input

from .harness import DispatchTestCase


class TestCatConverter(DispatchTestCase):
    @parameterized.expand(
        [
            ("pos", 1),
            ("neg", -2),
        ]
    )
    def test_cat(self, _, dim):
        class Cat(nn.Module):
            def forward(self, x, y, z):
                return torch.ops.aten.cat.default((x, y, z), dim)

        inputs = [torch.randn(1, 2, 3), torch.randn(1, 1, 3), torch.randn(1, 3, 3)]
        self.run_test(
            Cat(),
            inputs,
        )

    @parameterized.expand(
        [
            ("pos", 1),
            ("neg", -2),
        ]
    )
    def test_cat_dynamic_shape(self, _, dim):
        class Cat(nn.Module):
            def forward(self, x, y):
                return torch.ops.aten.cat.default((x, y), dim)

        input_specs = [
            Input(
                shape=(16, -1, 3),
                dtype=torch.float32,
                shape_ranges=[((16, 2, 3), (16, 3, 3), (16, 32, 3))],
            ),
            Input(
                shape=(16, -1, 3),
                dtype=torch.float32,
                shape_ranges=[((16, 2, 3), (16, 16, 3), (16, 32, 3))],
            ),
        ]
        self.run_test_with_dynamic_shape(
            Cat(),
            input_specs,
        )

    def test_cat_no_dim(self):
        class Cat(nn.Module):
            def forward(self, x, y, z):
                return torch.ops.aten.cat.default((x, y, z))

        inputs = [torch.randn(2, 1, 3), torch.randn(1, 1, 3), torch.randn(3, 1, 3)]
        self.run_test(
            Cat(),
            inputs,
        )

    def test_cat_dynamic_shape_no_dim(self):
        class Cat(nn.Module):
            def forward(self, x, y):
                return torch.ops.aten.cat.default((x, y))

        input_specs = [
            Input(
                shape=(-1, 16, 3),
                dtype=torch.float32,
                shape_ranges=[((2, 16, 3), (3, 16, 3), (32, 16, 3))],
            ),
            Input(
                shape=(-1, 16, 3),
                dtype=torch.float32,
                shape_ranges=[((2, 16, 3), (3, 16, 3), (32, 16, 3))],
            ),
        ]
        self.run_test_with_dynamic_shape(
            Cat(),
            input_specs,
        )


if __name__ == "__main__":
    run_tests()
