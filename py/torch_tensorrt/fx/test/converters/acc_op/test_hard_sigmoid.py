import torch
from parameterized import parameterized
from torch import nn
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt.fx.tools.common_fx2trt import AccTestCase, InputTensorSpec
from torch_tensorrt.fx.tracer.acc_tracer import acc_ops


class TestHardSigmoid(AccTestCase):
    @parameterized.expand(
        [
            ("3", 3),
            ("0", 0),
            ("-3", -4),
        ]
    )
    def test_hardsigmoid(self, _, pad):
        class Hardsigmoid(nn.Module):
            def forward(self, x):
                return torch.nn.functional.hardsigmoid(x)

        inputs = [torch.randn(1, 2, 3) + pad]
        self.run_test(Hardsigmoid(), inputs, expected_ops={acc_ops.hardsigmoid})

    def test_hardsigmoid_with_dynamic_shape(self):
        class Hardsigmoid(nn.Module):
            def forward(self, x):
                return torch.nn.functional.hardsigmoid(x)

        input_specs = [
            InputTensorSpec(
                shape=(-1, -1, -1),
                dtype=torch.float32,
                shape_ranges=[((1, 1, 1), (1, 2, 3), (3, 3, 3))],
            ),
        ]
        self.run_test_with_dynamic_shape(
            Hardsigmoid(), input_specs, expected_ops={acc_ops.hardsigmoid}
        )

    def test_hardsigmoid_with_dynamic_shape_four_dimensions(self):
        class Hardsigmoid(nn.Module):
            def forward(self, x):
                return torch.nn.functional.hardsigmoid(x)

        input_specs = [
            InputTensorSpec(
                shape=(-1, -1, -1, -1),
                dtype=torch.float32,
                shape_ranges=[((1, 1, 1, 1), (1, 2, 3, 3), (3, 3, 3, 3))],
            ),
        ]
        self.run_test_with_dynamic_shape(
            Hardsigmoid(), input_specs, expected_ops={acc_ops.hardsigmoid}
        )


if __name__ == "__main__":
    run_tests()
