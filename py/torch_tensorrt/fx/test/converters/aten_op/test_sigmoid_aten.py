import torch
import torch.nn as nn
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt.fx.utils import LowerPrecision
from torch_tensorrt.fx.tools.common_fx2trt import DispatchTestCase, InputTensorSpec


class TestSigmoidConverter(DispatchTestCase):
    def test_sigmoid(self):
        class TestModule(nn.Module):
            def forward(self, x):
                return nn.functional.sigmoid(x)

        inputs = [torch.randn(1, 10)]
        self.run_test(
            TestModule(), inputs, expected_ops={torch.ops.aten.sigmoid.default}
        )

    def test_sigmoid_with_dynamic_shape(self):
        class TestModule(nn.Module):
            def forward(self, x):
                return nn.functional.sigmoid(x)

        input_specs = [
            InputTensorSpec(
                shape=(-1, -1, -1),
                dtype=torch.float32,
                shape_ranges=[((1, 1, 1), (1, 2, 3), (3, 3, 3))],
            ),
        ]
        self.run_test_with_dynamic_shape(
            TestModule(), input_specs, expected_ops={torch.ops.aten.sigmoid.default}
        )

    def test_sigmoid_with_dynamic_shape_four_dimensions(self):
        class TestModule(nn.Module):
            def forward(self, x):
                return nn.functional.sigmoid(x)

        input_specs = [
            InputTensorSpec(
                shape=(-1, -1, -1, -1),
                dtype=torch.float32,
                shape_ranges=[((1, 1, 1, 5), (1, 2, 3, 5), (3, 3, 3, 5))],
            ),
        ]

        self.run_test_with_dynamic_shape(
            TestModule(), input_specs, expected_ops={torch.ops.aten.sigmoid.default}
        )

    def test_sigmoid_fp16(self):
        class TestModule(nn.Module):
            def forward(self, x):
                return nn.functional.sigmoid(x)

        inputs = [torch.randn(1, 10)]
        self.run_test(
            TestModule(),
            inputs,
            expected_ops={torch.ops.aten.sigmoid.default},
            precision=torch.half,
        )


if __name__ == "__main__":
    run_tests()
