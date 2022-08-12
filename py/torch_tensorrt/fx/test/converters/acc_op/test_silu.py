import torch
from torch import nn
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt.fx.tools.common_fx2trt import AccTestCase, InputTensorSpec
from torch_tensorrt.fx.tracer.acc_tracer import acc_ops


class TestSilu(AccTestCase):
    def test_silu(self):
        class Silu(nn.Module):
            def forward(self, x):
                return torch.nn.functional.silu(x)

        inputs = [torch.randn(1, 2, 3)]
        self.run_test(Silu(), inputs, expected_ops={acc_ops.sigmoid, acc_ops.mul})

    def test_silu_with_dynamic_shape(self):
        class Silu(nn.Module):
            def forward(self, x):
                return torch.nn.functional.silu(x)

        input_specs = [
            InputTensorSpec(
                shape=(-1, -1, -1),
                dtype=torch.float32,
                shape_ranges=[((1, 1, 1), (1, 2, 3), (3, 3, 3))],
            ),
        ]
        self.run_test_with_dynamic_shape(
            Silu(), input_specs, expected_ops={acc_ops.sigmoid, acc_ops.mul}
        )

    def test_silu_with_dynamic_shape_four_dimensions(self):
        class Silu(nn.Module):
            def forward(self, x):
                return torch.nn.functional.silu(x)

        input_specs = [
            InputTensorSpec(
                shape=(-1, -1, -1, -1),
                dtype=torch.float32,
                shape_ranges=[((1, 1, 1, 1), (1, 2, 3, 3), (3, 3, 3, 3))],
            ),
        ]

        self.run_test_with_dynamic_shape(
            Silu(), input_specs, expected_ops={acc_ops.sigmoid, acc_ops.mul}
        )


if __name__ == "__main__":
    run_tests()
