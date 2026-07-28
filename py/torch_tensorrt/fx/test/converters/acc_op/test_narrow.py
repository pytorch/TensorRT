import torch
import torch.nn as nn
import torch_tensorrt.fx.tracer.acc_tracer.acc_ops as acc_ops
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt.fx.tools.common_fx2trt import AccTestCase, InputTensorSpec


class TestNarrowConverterWithDynamicShape(AccTestCase):
    @parameterized.expand(
        [
            ("positive_dim", 1, 0, 1),
        ]
    )
    def test_narrow(self, _, dim, start, length):
        class Narrow(nn.Module):
            def forward(self, x):
                return x.narrow(dim, start, length)

        input_specs = [
            InputTensorSpec(
                shape=(-1, 3, -1, -1),
                dtype=torch.float32,
                shape_ranges=[((1, 3, 5, 5), (2, 3, 5, 5), (2, 3, 5, 5))],
            ),
        ]

        self.run_test_with_dynamic_shape(
            Narrow(), input_specs, expected_ops={acc_ops.slice_tensor}
        )


class TestNarrowConverter(AccTestCase):
    @parameterized.expand(
        [
            ("positive_dim", 1, 0, 1),
            ("negative_dim", -1, 1, 2),
        ]
    )
    def test_narrow(self, _, dim, start, length):
        class Narrow(nn.Module):
            def forward(self, x):
                return x.narrow(dim, start, length)

        inputs = [torch.randn(1, 2, 3, 4)]
        self.run_test(
            Narrow(),
            inputs,
            expected_ops={acc_ops.slice_tensor},
            test_explicit_batch_dim=False,
        )


if __name__ == "__main__":
    run_tests()
