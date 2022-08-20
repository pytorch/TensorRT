import torch
import torch_tensorrt.fx.tracer.acc_tracer.acc_ops as acc_ops
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt.fx.tools.common_fx2trt import AccTestCase, InputTensorSpec


class TestMaximumConverter(AccTestCase):
    def test_maximum(self):
        class Maximum(torch.nn.Module):
            def forward(self, x, y):
                return torch.maximum(x, y)

        inputs = [
            torch.randn(3, 4),
            torch.randn(3, 4),
        ]
        self.run_test(Maximum(), inputs, expected_ops={acc_ops.maximum})


class TestMaximumConverterWithDynamicShape(AccTestCase):
    def test_maximum(self):
        class Maximum(torch.nn.Module):
            def forward(self, x, y):
                return torch.maximum(x, y)

        input_specs = [
            InputTensorSpec(
                shape=(-1, -1, -1, -1),
                dtype=torch.float32,
                shape_ranges=[((1, 1, 1, 1), (2, 3, 4, 5), (2, 3, 10, 10))],
            ),
            InputTensorSpec(
                shape=(-1, -1, -1, -1),
                dtype=torch.float32,
                shape_ranges=[((1, 1, 1, 1), (2, 3, 4, 5), (2, 3, 10, 10))],
            ),
        ]

        self.run_test_with_dynamic_shape(
            Maximum(), input_specs, expected_ops={acc_ops.maximum}
        )


class TestMaximumMethodConverter(AccTestCase):
    def test_maximum(self):
        class Maximum(torch.nn.Module):
            def forward(self, x, y):
                return x.maximum(y)

        inputs = [
            torch.randn(3, 4),
            torch.randn(3, 4),
        ]
        self.run_test(Maximum(), inputs, expected_ops={acc_ops.maximum})


class TestMaximumMethodConverterWithDynamicShape(AccTestCase):
    def test_maximum(self):
        class Maximum(torch.nn.Module):
            def forward(self, x, y):
                return x.maximum(y)

        input_specs = [
            InputTensorSpec(
                shape=(-1, -1, -1, -1),
                dtype=torch.float32,
                shape_ranges=[((1, 1, 1, 1), (2, 3, 4, 5), (2, 3, 10, 10))],
            ),
            InputTensorSpec(
                shape=(-1, -1, -1, -1),
                dtype=torch.float32,
                shape_ranges=[((1, 1, 1, 1), (2, 3, 4, 5), (2, 3, 10, 10))],
            ),
        ]

        self.run_test_with_dynamic_shape(
            Maximum(), input_specs, expected_ops={acc_ops.maximum}
        )


if __name__ == "__main__":
    run_tests()
