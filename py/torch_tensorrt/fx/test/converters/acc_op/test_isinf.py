import unittest

import torch

import torch_tensorrt.fx.tracer.acc_tracer.acc_ops as acc_ops
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt.fx.tools.common_fx2trt import AccTestCase, InputTensorSpec


@unittest.skip("Implementation is commented out due to accuracy issue T113156424")
class TestInfConverter(AccTestCase):
    def test_isinf(self):
        class Test(torch.nn.Module):
            def forward(self, x):
                return torch.isinf(x)

        input = torch.randn(2, 3)
        input[0][0] = float("inf")
        input[0][1] = float("-inf")
        input.cuda()
        inputs = [
            input,
        ]
        self.run_test(
            Test(), inputs, expected_ops={acc_ops.isinf}, test_implicit_batch_dim=False
        )

    def test_isinf_large(self):
        class Test(torch.nn.Module):
            def forward(self, x):
                return torch.isinf(x)

        input = torch.randn(2, 3, 4, 5)
        input[0][0][0][:] = float("inf")
        input[0][0][1][:] = float("-inf")
        input.cuda()
        inputs = [
            input,
        ]
        self.run_test(
            Test(), inputs, expected_ops={acc_ops.isinf}, test_implicit_batch_dim=False
        )

    def test_isinf_large_with_dynamic_shape_four_dimensions(self):
        class Test(torch.nn.Module):
            def forward(self, x):
                return torch.isinf(x)

        input_specs = [
            InputTensorSpec(
                shape=(-1, -1, -1, -1),
                dtype=torch.float32,
                shape_ranges=[((1, 1, 1, 1), (1, 2, 3, 3), (3, 3, 3, 3))],
            ),
        ]

        self.run_test_with_dynamic_shape(
            Test(), input_specs, expected_ops={acc_ops.isinf}
        )


if __name__ == "__main__":
    run_tests()
