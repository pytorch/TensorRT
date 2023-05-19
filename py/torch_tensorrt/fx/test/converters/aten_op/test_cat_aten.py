import torch
import torch.nn as nn
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt.fx.tools.common_fx2trt import DispatchTestCase, InputTensorSpec


class TestCatConverter(DispatchTestCase):
    @parameterized.expand(
        [
            ("pos", 1),
            # ("neg", -2), #dim can not have dynamic input
        ]
    )
    def test_cat(self, _, dim):
        class Cat(nn.Module):
            def forward(self, x, y, z):
                return torch.cat((x, y, z), dim)

        inputs = [torch.randn(1, 2, 3), torch.randn(1, 1, 3), torch.randn(1, 3, 3)]
        self.run_test(
            Cat(),
            inputs,
            expected_ops={torch.ops.aten.cat.default},
        )

    @parameterized.expand(
        [
            ("pos", 1),
            # ("neg", -2), #dim can not have dynamic input
        ]
    )
    def test_cat_dynamic_shape(self, _, dim):
        class Cat(nn.Module):
            def forward(self, x, y):
                return torch.cat((x, y), dim)

        input_specs = [
            InputTensorSpec(
                shape=(16, -1, 3),
                dtype=torch.float32,
                shape_ranges=[((16, 2, 3), (16, 3, 3), (16, 32, 3))],
            ),
            InputTensorSpec(
                shape=(16, -1, 3),
                dtype=torch.float32,
                shape_ranges=[((16, 2, 3), (16, 16, 3), (16, 32, 3))],
            ),
        ]
        self.run_test_with_dynamic_shape(
            Cat(),
            input_specs,
            expected_ops={torch.ops.aten.cat.default},
        )


if __name__ == "__main__":
    run_tests()
