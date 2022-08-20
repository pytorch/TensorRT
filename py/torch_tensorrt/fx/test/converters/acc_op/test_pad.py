import unittest

import tensorrt as trt
import torch
import torch.nn as nn

import torch_tensorrt.fx.tracer.acc_tracer.acc_ops as acc_ops
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt.fx.tools.common_fx2trt import AccTestCase

# from torch_tensorrt.fx.tools.common_fx2trt import AccTestCase, InputTensorSpec


class TestPadConverter(AccTestCase):
    @parameterized.expand(
        [
            ("1d", (1, 2), 9),
            ("2d", (2, 0, 0, 1), 10),
        ]
    )
    def test_pad_value(self, _, pad, value):
        class Pad(nn.Module):
            def forward(self, x):
                return torch.nn.functional.pad(x, pad, value=value)

        inputs = [torch.randn(1, 2, 3, 4)]
        self.run_test(
            Pad(),
            inputs,
            expected_ops={acc_ops.pad},
            # enable value will not work with implicit batch
            test_implicit_batch_dim=False,
        )

    @parameterized.expand(
        [
            ("1d", (1, 2)),
            ("2d", (2, 0, 0, 1)),
        ]
    )
    def test_pad(self, _, pad):
        class Pad(nn.Module):
            def forward(self, x):
                return torch.nn.functional.pad(x, pad)

        inputs = [torch.randn(1, 2, 3, 4)]
        self.run_test(
            Pad(),
            inputs,
            expected_ops={acc_ops.pad},
            # enable value will not work with implicit batch
            test_implicit_batch_dim=False,
        )

    # Testing with (-1, 3, 3, 3) results into following error:
    # test_pad_with_dynamic_shape_four_dimensions_0_2d (deeplearning.trt.torch_tensorrt.py.torch_tensorrt.fx.test.converters.acc_op.test_pad.TestPadConverter) ... [07/15/2022-09:23:18] [TRT] [E] 2: [intInterval.cpp::max::26] Error Code 2: Internal Error (Assertion !empty() failed. )
    # Segmentation fault (core dumped)

    """
    def test_pad_with_dynamic_shape_four_dimensions(self):
        class Pad(nn.Module):
            def forward(self, x):
                return torch.nn.functional.pad(x, (1, 1))

        input_specs = [
            InputTensorSpec(
                shape=(-1, 3, 3, 3),
                dtype=torch.float32,
                shape_ranges=[((1, 3, 3, 3), (2, 3, 3, 3), (2, 3, 3, 3))],
            ),
        ]

        self.run_test_with_dynamic_shape(Pad(), input_specs, expected_ops={acc_ops.pad})
    """

    @parameterized.expand(
        [
            ("3d", (2, 2, 3, 1, 2, 2)),
        ]
    )
    @unittest.skipIf(
        trt.__version__ < "8.2",
        "Padding 3d only supported in TensorRT 8.2 and later",
    )
    def test_pad_3d(self, _, pad):
        class Pad(nn.Module):
            def forward(self, x):
                return torch.nn.functional.pad(x, pad)

        inputs = [torch.randn(1, 2, 3, 4)]
        self.run_test(
            Pad(),
            inputs,
            expected_ops={acc_ops.pad},
            # enable value will not work with implicit batch
            test_implicit_batch_dim=False,
        )


if __name__ == "__main__":
    run_tests()
