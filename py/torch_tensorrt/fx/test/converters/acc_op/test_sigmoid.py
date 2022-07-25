import torch
import torch.nn as nn
import torch_tensorrt.fx.tracer.acc_tracer.acc_ops as acc_ops
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt.fx.tools.common_fx2trt import AccTestCase, InputTensorSpec


class TestSigmoid(AccTestCase):
    def test_sigmoid(self):
        class Sigmoid(nn.Module):
            def forward(self, x):
                return torch.sigmoid(x)

        inputs = [torch.randn(1, 2, 3)]
        self.run_test(Sigmoid(), inputs, expected_ops={acc_ops.sigmoid})

    def test_sigmoid_with_dynamic_shape_four_dimensions(self):
        class Sigmoid(nn.Module):
            def forward(self, x):
                return torch.sigmoid(x)

        input_specs = [
            InputTensorSpec(
                shape=(-1, -1, -1, -1),
                dtype=torch.float32,
                shape_ranges=[((1, 1, 1, 1), (1, 2, 3, 3), (3, 3, 3, 3))],
            ),
        ]
        self.run_test_with_dynamic_shape(
            Sigmoid(), input_specs, expected_ops={acc_ops.sigmoid}
        )


if __name__ == "__main__":
    run_tests()
