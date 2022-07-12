import torch
import torch.nn as nn
import torch_tensorrt.fx.tracer.acc_tracer.acc_ops as acc_ops
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt.fx.tools.common_fx2trt import AccTestCase

# from torch_tensorrt.fx.tools.common_fx2trt import InputTensorSpec


class TestAnyConverters(AccTestCase):
    @parameterized.expand(
        [
            ("bool", torch.bool),
            ("int", torch.int),
            ("float", torch.float),
        ]
    )
    def test_ops(self, _, input_dtype):
        class TestModule(nn.Module):
            def forward(self, x):
                return torch.any(x)

        inputs = [torch.randn(2, 3).to(input_dtype)]
        self.run_test(
            TestModule(),
            inputs,
            expected_ops={acc_ops.any},
            test_implicit_batch_dim=False,
        )

    @parameterized.expand(
        [
            ("bool", torch.bool, 0),
            ("int", torch.int, 1),
            ("float", torch.float, 0),
        ]
    )
    def test_ops_dim(self, _, input_dtype, dim):
        class TestModule(nn.Module):
            def forward(self, x):
                return torch.any(x, dim, keepdim=True)

        inputs = [torch.randn(2, 3).to(input_dtype)]
        self.run_test(
            TestModule(), inputs, expected_ops={}, test_implicit_batch_dim=False
        )

    @parameterized.expand(
        [
            ("bool", torch.bool),
            ("int", torch.int),
            ("float", torch.float),
        ]
    )
    def test_ops_method(self, _, input_dtype):
        class TestModule(nn.Module):
            def forward(self, x):
                return x.any()

        inputs = [torch.randn(2, 3).to(input_dtype)]
        self.run_test(
            TestModule(),
            inputs,
            expected_ops={acc_ops.any},
            test_implicit_batch_dim=False,
        )

    # Testing with shape (-1, -1, -1, -1) results into error: torch.zeros(tuple([*input_t.shape])). Trying to create tensor with negative dimension -1: [-1, -1, -1, -1]
    """
    def test_ops_with_dynamic_shape_four_dimensions(self):
        class TestModule(nn.Module):
            def forward(self, x):
                return torch.any(x)

        input_specs = [
            InputTensorSpec(
                shape=(-1, -1, -1, -1),
                dtype=torch.float32,
                shape_ranges=[((1, 1, 256, 256), (3, 3, 256, 256), (5, 5, 256, 256))],
            ),
        ]

        self.run_test_with_dynamic_shape(
            TestModule(), input_specs, expected_ops={acc_ops.any}
        )
    """


if __name__ == "__main__":
    run_tests()
