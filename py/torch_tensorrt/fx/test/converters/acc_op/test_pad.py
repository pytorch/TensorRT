import unittest

import fx2trt_oss.tracer.acc_tracer.acc_ops as acc_ops
import tensorrt as trt
import torch
import torch.nn as nn
from parameterized import param, parameterized
from torch.testing._internal.common_fx2trt import AccTestCase
from torch.testing._internal.common_utils import run_tests


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
