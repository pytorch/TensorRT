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
    def test_cat_dim_in_kwargs(self, _, dim):
        class Cat(nn.Module):
            def forward(self, x, y, z):
                return torch.ops.aten.cat.default((x, y, z), dim=dim)

        inputs = [torch.randn(1, 2, 3), torch.randn(1, 1, 3), torch.randn(1, 3, 3)]
        self.run_test(
            Cat(),
            inputs,
        )

    @parameterized.expand(
        [
            ("pos", 0),
            ("neg", -3),
        ]
    )
    def test_cat_with_scalar_inputs(self, _, dim):
        # Ensure scalar tensor wrap works
        class Cat(nn.Module):
            def forward(self, x, y):
                # y is a scalar, x is a tensor
                return torch.ops.aten.cat.default((x, y), dim)

        x = torch.randn(1, 2, 3, device="cuda")
        y = torch.ones_like(x) * 5.0  # simulate scalar broadcast
        inputs = [x, y]
        self.run_test(Cat(), inputs)

    @parameterized.expand(
        [
            ("pos", 0),
            ("neg", -3),
        ]
    )
    def test_cat_with_empty_tensor(self, _, dim):
        # Handle empty tensor in concat
        class Cat(nn.Module):
            def forward(self, x):
                y = torch.empty(0, 2, 3, device="cuda")
                return torch.ops.aten.cat.default((x, y), dim)

        inputs = [
            torch.randn(1, 2, 3, device="cuda"),
        ]
        self.run_test(Cat(), inputs)

    @parameterized.expand(
        [
            ("pos", 2),
            ("neg", -1),
        ]
    )
    def test_cat_with_different_dtypes(self, _, dim):
        # check dtype promotion path in concat
        class Cat(nn.Module):
            def forward(self, x, y):
                return torch.ops.aten.cat.default((x, y), dim)

        inputs = [
            torch.ones(1, 2, 3, dtype=torch.float32, device="cuda"),
            torch.ones(1, 2, 3, dtype=torch.float16, device="cuda"),
        ]
        self.run_test(Cat(), inputs)

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
                dtype=torch.float32,
                min_shape=(16, 2, 3),
                opt_shape=(16, 3, 3),
                max_shape=(16, 32, 3),
            ),
            Input(
                dtype=torch.float32,
                min_shape=(16, 2, 3),
                opt_shape=(16, 16, 3),
                max_shape=(16, 32, 3),
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
                dtype=torch.float32,
                min_shape=(2, 16, 3),
                opt_shape=(3, 16, 3),
                max_shape=(32, 16, 3),
            ),
            Input(
                dtype=torch.float32,
                min_shape=(2, 16, 3),
                opt_shape=(3, 16, 3),
                max_shape=(32, 16, 3),
            ),
        ]
        self.run_test_with_dynamic_shape(
            Cat(),
            input_specs,
        )

    @parameterized.expand(
        [
            ("pos", 1),
            ("neg", -2),
        ]
    )
    def test_cat_dynamic_shape_dim(self, _, dim):
        class Cat(nn.Module):
            def forward(self, x, y):
                return torch.ops.aten.cat.default((x, y), dim)

        input_specs = [
            Input(
                dtype=torch.float32,
                min_shape=(2, 1, 1),
                opt_shape=(3, 1, 2),
                max_shape=(4, 1, 3),
            ),
            Input(
                dtype=torch.float32,
                min_shape=(2, 2, 1),
                opt_shape=(3, 3, 2),
                max_shape=(4, 4, 3),
            ),
        ]
        self.run_test_with_dynamic_shape(
            Cat(),
            input_specs,
        )


if __name__ == "__main__":
    run_tests()
