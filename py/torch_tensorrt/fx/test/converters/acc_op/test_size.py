import torch
import torch.nn as nn
import torch_tensorrt.fx.tracer.acc_tracer.acc_ops as acc_ops
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt.fx.tools.common_fx2trt import AccTestCase, InputTensorSpec


class TestSizeConverter(AccTestCase):
    def test_size(self):
        class Size(nn.Module):
            def forward(self, x):
                bs = x.size(0)
                return x.view(bs, -1)

        inputs = [torch.randn(1, 2, 3, 4)]
        self.run_test(Size(), inputs, expected_ops={acc_ops.size})

    def test_size_param(self):
        class Size(nn.Module):
            def __init__(self, x):
                super().__init__()
                self.param = torch.nn.Parameter(x)

            def forward(self, y):
                bs = self.param.size(0)
                return y.view(bs, -1)

        self.run_test(
            Size(torch.randn(1, 2, 3, 4)),
            [torch.randn(1, 2, 3, 4)],
            expected_ops={acc_ops.size},
        )

    def test_size_dynamic_shape(self):
        class Size(nn.Module):
            def forward(self, x):
                bs = x.size(0)
                return x.view(bs, -1)

        input_specs = [
            InputTensorSpec(
                shape=(-1, 12, 32),
                dtype=torch.float32,
                shape_ranges=[((1, 12, 32), (3, 12, 32), (100, 12, 32))],
            ),
        ]
        self.run_test_with_dynamic_shape(
            Size(), input_specs, expected_ops={acc_ops.size}
        )

    def test_size_dynamic_shape_four_dimensions(self):
        class Size(nn.Module):
            def forward(self, x):
                bs = x.size(0)
                return x.view(bs, -1)

        input_specs = [
            InputTensorSpec(
                shape=(-1, -1, -1, -1),
                dtype=torch.float32,
                shape_ranges=[((1, 12, 32, 3), (3, 12, 32, 3), (100, 12, 32, 3))],
            ),
        ]

        self.run_test_with_dynamic_shape(
            Size(), input_specs, expected_ops={acc_ops.size}
        )


if __name__ == "__main__":
    run_tests()
