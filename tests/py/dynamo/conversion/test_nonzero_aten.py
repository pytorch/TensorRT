import unittest

import torch
import torch.nn as nn
import torch_tensorrt
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt import Input
from torch_tensorrt.dynamo.conversion.aten_ops_converters import nonzero_validator

from .harness import DispatchTestCase


@unittest.skipIf(
    torch_tensorrt.ENABLED_FEATURES.tensorrt_rtx,
    "nonzero is not supported for tensorrt_rtx",
)
class TestNonZeroConverter(DispatchTestCase):
    @parameterized.expand(
        [
            ((10,), torch.int),
            ((1, 20), torch.int32),
            ((2, 3), torch.int64),
            ((2, 3, 4), torch.float),
            ((2, 3, 4, 5), torch.float),
        ]
    )
    def test_nonzero_dds(self, input_shape, dtype):
        class NonZero(nn.Module):
            # This is a DDS network
            def forward(self, input):
                out = torch.ops.aten.nonzero.default(input)
                return out

        inputs = [torch.randint(low=0, high=3, size=input_shape, dtype=dtype)]
        self.run_test(
            NonZero(),
            inputs,
        )

    @parameterized.expand(
        [
            ((10,), torch.int),
            ((1, 20), torch.int32),
            ((2, 3), torch.int64),
            ((2, 3, 4), torch.float),
            ((2, 3, 4, 5), torch.float),
        ]
    )
    def test_nonzero_non_dds(self, input_shape, dtype):
        class NonZero(nn.Module):
            # This is a static network
            def forward(self, input):
                out = torch.ops.aten.nonzero.default(input)
                out = torch.ops.aten.sum.dim_IntList(out, 0)
                return out

        inputs = [torch.randint(low=0, high=3, size=input_shape, dtype=dtype)]
        self.run_test(
            NonZero(),
            inputs,
        )

    @parameterized.expand(
        [
            (
                "1d",
                (1,),
                (10,),
                (100,),
                torch.int32,
            ),
            (
                "2d",
                (1, 2),
                (5, 10),
                (20, 40),
                torch.float16,
            ),
            (
                "3d",
                (1, 2, 3),
                (5, 10, 20),
                (30, 40, 50),
                torch.float,
            ),
        ]
    )
    def test_nonzero_dynamic_shape_dds(self, _, min_shape, opt_shape, max_shape, dtype):
        class NonZero(nn.Module):
            def forward(self, input):
                return torch.ops.aten.nonzero.default(input)

        input_specs = [
            Input(
                min_shape=min_shape,
                opt_shape=opt_shape,
                max_shape=max_shape,
                dtype=dtype,
            ),
        ]

        self.run_test_with_dynamic_shape(NonZero(), input_specs)

    @parameterized.expand(
        [
            (
                "1d",
                (1,),
                (10,),
                (100,),
                torch.int32,
            ),
            (
                "2d",
                (1, 2),
                (5, 10),
                (20, 40),
                torch.float16,
            ),
            (
                "3d",
                (1, 2, 3),
                (5, 10, 20),
                (30, 40, 50),
                torch.float,
            ),
        ]
    )
    def test_nonzero_dynamic_shape_non_dds(
        self, _, min_shape, opt_shape, max_shape, dtype
    ):
        class NonZero(nn.Module):
            def forward(self, input):
                out = torch.ops.aten.nonzero.default(input)
                out = torch.ops.aten.sum.dim_IntList(out, 0)
                return out

        input_specs = [
            Input(
                min_shape=min_shape,
                opt_shape=opt_shape,
                max_shape=max_shape,
                dtype=dtype,
            ),
        ]

        self.run_test_with_dynamic_shape(NonZero(), input_specs)


class TestNonZeroValidator(unittest.TestCase):
    """nonzero_validator decides, at partition time, whether nonzero reaches TensorRT.

    It has to reject on TensorRT-RTX so the partitioner leaves the node in a
    PyTorch block; a converter-side raise would fail the build instead.
    """

    @staticmethod
    def _nonzero_node() -> torch.fx.Node:
        class NonZero(nn.Module):
            def forward(self, x):
                return torch.nonzero(x)

        gm = torch.export.export(NonZero(), (torch.tensor([0, 3, 0, 5]),)).module()
        return next(
            n
            for n in gm.graph.nodes
            if n.op == "call_function" and n.target == torch.ops.aten.nonzero.default
        )

    @unittest.skipUnless(
        torch_tensorrt.ENABLED_FEATURES.tensorrt_rtx,
        "nonzero_validator only rejects on tensorrt_rtx",
    )
    def test_nonzero_validator_false_on_rtx(self):
        self.assertFalse(nonzero_validator(self._nonzero_node()))

    @unittest.skipIf(
        torch_tensorrt.ENABLED_FEATURES.tensorrt_rtx,
        "On non-RTX, nonzero_validator always passes",
    )
    def test_nonzero_validator_true_on_non_rtx(self):
        self.assertTrue(nonzero_validator(self._nonzero_node()))


if __name__ == "__main__":
    run_tests()
