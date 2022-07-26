import torch
import torch.nn as nn
import torch_tensorrt.fx.tracer.acc_tracer.acc_ops as acc_ops
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt.fx.tools.common_fx2trt import AccTestCase, InputTensorSpec


class TestNewOnesConverter(AccTestCase):
    def test_newone(self):
        class TestModule(nn.Module):
            def forward(self, x):
                return x.new_ones((3, 5), dtype=torch.float16)

        inputs = [torch.randn(1, 10)]
        self.run_test(
            TestModule(),
            inputs,
            expected_ops={acc_ops.new_ones},
            test_implicit_batch_dim=False,
        )

    def test_newone_no_dtype(self):
        class TestModule(nn.Module):
            def forward(self, x):
                return x.new_ones((3, 5))

        inputs = [torch.randn(1, 10)]
        self.run_test(
            TestModule(),
            inputs,
            expected_ops={acc_ops.new_ones},
            test_implicit_batch_dim=False,
        )

    def test_newone_device(self):
        class TestModule(nn.Module):
            def forward(self, x):
                return x.new_ones((3, 5), device="cuda")

        inputs = [torch.randn(1, 10)]
        self.run_test(
            TestModule(),
            inputs,
            expected_ops={acc_ops.new_ones},
            test_implicit_batch_dim=False,
        )


class TestNewOnesConverterWithDynamicShape(AccTestCase):
    def test_newone(self):
        class TestModule(nn.Module):
            def forward(self, x):
                return x.new_ones((3, 5), dtype=torch.float16)

        input_specs = [
            InputTensorSpec(
                shape=(-1, -1, -1, -1),
                dtype=torch.float32,
                shape_ranges=[((1, 1, 1, 1), (2, 3, 4, 5), (2, 3, 10, 10))],
            ),
        ]

        self.run_test_with_dynamic_shape(
            TestModule(), input_specs, expected_ops={acc_ops.new_ones}
        )

    def test_newone_no_dtype(self):
        class TestModule(nn.Module):
            def forward(self, x):
                return x.new_ones((3, 5))

        input_specs = [
            InputTensorSpec(
                shape=(-1, -1, -1, -1),
                dtype=torch.float32,
                shape_ranges=[((1, 1, 1, 1), (2, 3, 4, 5), (2, 3, 10, 10))],
            ),
        ]

        self.run_test_with_dynamic_shape(
            TestModule(), input_specs, expected_ops={acc_ops.new_ones}
        )

    def test_newone_device(self):
        class TestModule(nn.Module):
            def forward(self, x):
                return x.new_ones((3, 5), device="cuda")

        input_specs = [
            InputTensorSpec(
                shape=(-1, -1, -1, -1),
                dtype=torch.float32,
                shape_ranges=[((1, 1, 1, 1), (2, 3, 4, 5), (2, 3, 10, 10))],
            ),
        ]

        self.run_test_with_dynamic_shape(
            TestModule(), input_specs, expected_ops={acc_ops.new_ones}
        )


if __name__ == "__main__":
    run_tests()
