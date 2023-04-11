import torch
import torch.nn as nn
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt.dynamo.fx_ts_compat.tools.common_fx2trt import DispatchTestCase, InputTensorSpec


class TestReLUConverter(DispatchTestCase):
    def test_relu(self):
        class TestModule(nn.Module):
            def forward(self, x):
                return nn.functional.relu(x)

        inputs = [torch.randn(1, 10)]
        self.run_test(TestModule(), inputs, expected_ops={torch.ops.aten.relu.default})

    def test_relu_with_dynamic_shape(self):
        class TestModule(nn.Module):
            def forward(self, x):
                return nn.functional.relu(x)

        input_specs = [
            InputTensorSpec(
                shape=(-1, -1, -1),
                dtype=torch.float32,
                shape_ranges=[((1, 1, 1), (1, 2, 3), (3, 3, 3))],
            ),
        ]
        self.run_test_with_dynamic_shape(
            TestModule(), input_specs, expected_ops={torch.ops.aten.relu.default}
        )

    def test_relu_with_dynamic_shape_four_dimensions(self):
        class TestModule(nn.Module):
            def forward(self, x):
                return nn.functional.relu(x)

        input_specs = [
            InputTensorSpec(
                shape=(-1, -1, -1, -1),
                dtype=torch.float32,
                shape_ranges=[((1, 1, 1, 5), (1, 2, 3, 5), (3, 3, 3, 5))],
            ),
        ]

        self.run_test_with_dynamic_shape(
            TestModule(), input_specs, expected_ops={torch.ops.aten.relu.default}
        )


if __name__ == "__main__":
    run_tests()
