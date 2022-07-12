import torch
import torch_tensorrt.fx.tracer.acc_tracer.acc_ops as acc_ops
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt.fx.tools.common_fx2trt import AccTestCase, InputTensorSpec


class TestMinimumConverter(AccTestCase):
    def test_minimum(self):
        class Minimum(torch.nn.Module):
            def forward(self, x, y):
                return torch.minimum(x, y)

        inputs = [
            torch.randn(3, 4),
            torch.randn(3, 4),
        ]
        self.run_test(Minimum(), inputs, expected_ops={acc_ops.minimum})


class TestMinimumMethodConverter(AccTestCase):
    def test_minimum(self):
        class Minimum(torch.nn.Module):
            def forward(self, x, y):
                return x.minimum(y)

        inputs = [
            torch.randn(3, 4),
            torch.randn(3, 4),
        ]
        self.run_test(Minimum(), inputs, expected_ops={acc_ops.minimum})


class TestMinimumConverterWithDynamicShape(AccTestCase):
    def test_minimum(self):
        class Minimum(torch.nn.Module):
            def forward(self, x, y):
                return torch.minimum(x, y)

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
            Minimum(), input_specs, expected_ops={acc_ops.minimum}
        )


class TestMinimumMethodConverterWithDynamicShape(AccTestCase):
    def test_minimum(self):
        class Minimum(torch.nn.Module):
            def forward(self, x, y):
                return x.minimum(y)

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
            Minimum(), input_specs, expected_ops={acc_ops.minimum}
        )


if __name__ == "__main__":
    run_tests()
