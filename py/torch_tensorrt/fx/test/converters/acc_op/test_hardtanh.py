import torch
import torch.nn as nn
import torch_tensorrt.fx.tracer.acc_tracer.acc_ops as acc_ops
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt.fx.tools.common_fx2trt import AccTestCase, InputTensorSpec


class TestHardtanhConverter(AccTestCase):
    @parameterized.expand(
        [
            (-2.0, 6),
            (0, 1),
            (0.5, 7),
        ]
    )
    def test_hardtanh(self, test_min_value, test_max_value):
        class Hardtanh(nn.Module):
            def forward(self, x):
                return nn.functional.hardtanh(
                    x, min_val=test_min_value, max_val=test_max_value
                )

        inputs = [torch.randn(2, 10, 10, 10)]
        self.run_test(Hardtanh(), inputs, expected_ops={acc_ops.hardtanh})


class TestHardtanhConverterWithDynamicShape(AccTestCase):
    @parameterized.expand(
        [
            (-2.0, 6),
            (0, 1),
            (0.5, 7),
        ]
    )
    def test_hardtanh(self, test_min_value, test_max_value):
        class Hardtanh(nn.Module):
            def forward(self, x):
                return nn.functional.hardtanh(
                    x, min_val=test_min_value, max_val=test_max_value
                )

        input_specs = [
            InputTensorSpec(
                shape=(-1, -1, -1, -1),
                dtype=torch.float32,
                shape_ranges=[((1, 1, 1, 1), (1, 2, 3, 3), (3, 3, 3, 3))],
            ),
        ]

        self.run_test_with_dynamic_shape(
            Hardtanh(), input_specs, expected_ops={acc_ops.hardtanh}
        )


if __name__ == "__main__":
    run_tests()
