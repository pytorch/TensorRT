import torch
import torch.nn as nn
import torch_tensorrt.fx.tracer.acc_tracer.acc_ops as acc_ops
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt.fx.tools.common_fx2trt import AccTestCase, InputTensorSpec


class TestSplitConverter(AccTestCase):
    @parameterized.expand(
        [
            ("split_size", 3, 1),
            ("sections", [5, 2, 3], 1),
        ]
    )
    def test_split(self, _, split_size_or_sections, dim):
        class Split(nn.Module):
            def forward(self, x):
                return x.split(split_size_or_sections, dim)[0]

        inputs = [torch.randn(1, 10)]
        self.run_test(
            Split(),
            inputs,
            expected_ops={
                acc_ops.split
                if isinstance(split_size_or_sections, int)
                else acc_ops.slice_tensor
            },
            test_explicit_batch_dim=False,
        )

    @parameterized.expand(
        [
            ("split_with_size", [2, 3, 5], 1),
        ]
    )
    def test_split_with_size(self, _, split_size, dim):
        class Split(nn.Module):
            def forward(self, x):
                return x.split_with_sizes(split_size, dim)

        inputs = [torch.randn(1, 10)]
        self.run_test(
            Split(),
            inputs,
            expected_ops={acc_ops.slice_tensor},
            test_explicit_batch_dim=False,
        )

    @parameterized.expand(
        [
            ("split_size", 3, 1),
            ("sections", [5, 2, 3], 1),
        ]
    )
    def test_split_with_dynamic_shape(self, _, split_size_or_sections, dim):
        class Split(nn.Module):
            def forward(self, x):
                return x.split(split_size_or_sections, dim)[0]

        input_specs = [
            InputTensorSpec(
                shape=(-1, 10, -1),
                dtype=torch.float32,
                shape_ranges=[((1, 10, 10), (5, 10, 15), (10, 10, 20))],
            ),
        ]
        self.run_test_with_dynamic_shape(
            Split(),
            input_specs,
            expected_ops={
                acc_ops.split
                if isinstance(split_size_or_sections, int)
                else acc_ops.slice_tensor
            },
        )

    # Testing with (-1, -1, -1) results into following error:
    # AssertionError: Can't chunk on dynamic shape dimension!

    @parameterized.expand(
        [
            ("split_with_size", [2, 3, 5], 1),
        ]
    )
    def test_split_with_size_dynamic_shape(self, _, split_size, dim):
        class Split(nn.Module):
            def forward(self, x):
                return x.split_with_sizes(split_size, dim)

        input_specs = [
            InputTensorSpec(
                shape=(-1, 10, -1),
                dtype=torch.float32,
                shape_ranges=[((1, 10, 20), (5, 10, 20), (10, 10, 20))],
            ),
        ]
        self.run_test_with_dynamic_shape(
            Split(),
            input_specs,
            expected_ops={acc_ops.slice_tensor},
        )


if __name__ == "__main__":
    run_tests()
