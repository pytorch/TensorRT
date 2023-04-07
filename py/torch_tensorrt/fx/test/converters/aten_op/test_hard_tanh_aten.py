import torch
import torch.nn as nn
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt.fx.tools.common_fx2trt import DispatchTestCase, InputTensorSpec


class TestHardTanHConverter(DispatchTestCase):
    def test_hardtanh(self):
        class TestModule(nn.Module):
            def forward(self, x):
                return nn.functional.hardtanh(x)

        inputs = [torch.randn(1, 10)]
        self.run_test(
            TestModule(), inputs, expected_ops={torch.ops.aten.hardtanh.default}
        )

    def test_hardtanh_with_dynamic_shape(self):
        class TestModule(nn.Module):
            def forward(self, x):
                return nn.functional.hardtanh(x)

        input_specs = [
            InputTensorSpec(
                shape=(-1, -1, -1),
                dtype=torch.float32,
                shape_ranges=[((1, 1, 1), (1, 2, 3), (3, 3, 3))],
            ),
        ]
        self.run_test_with_dynamic_shape(
            TestModule(), input_specs, expected_ops={torch.ops.aten.hardtanh.default}
        )

    def test_hardtanh_with_dynamic_shape_four_dimensions(self):
        class TestModule(nn.Module):
            def forward(self, x):
                return nn.functional.hardtanh(x)

        input_specs = [
            InputTensorSpec(
                shape=(-1, -1, -1, -1),
                dtype=torch.float32,
                shape_ranges=[((1, 1, 1, 5), (1, 2, 3, 5), (3, 3, 3, 5))],
            ),
        ]

        self.run_test_with_dynamic_shape(
            TestModule(), input_specs, expected_ops={torch.ops.aten.hardtanh.default}
        )


if __name__ == "__main__":
    run_tests()
