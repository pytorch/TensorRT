import torch
import torch.nn as nn
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt import Input

from .harness import DispatchTestCase


class TestCopyConverter(DispatchTestCase):
    @parameterized.expand(
        [
            ((3,), (3,), False),
            ((1, 10), (1, 10), False),
            ((2, 3, 4), (2, 3, 4), True),
            ((2, 3, 4, 5), (2, 3, 4, 5), True),
        ]
    )
    def test_copy_float(self, input_shape, src_shape, non_blocking):
        class Copy(nn.Module):
            def forward(self, input, src):
                return torch.ops.aten.copy.default(input, src, non_blocking)

        inputs = [torch.randn(input_shape), torch.randn(src_shape)]
        self.run_test(
            Copy(),
            inputs,
        )

    @parameterized.expand(
        [
            (
                "1d_float32",
                torch.float32,
                (1,),
                (2,),
                (3,),
                False,
            ),
            (
                "2d_float32",
                torch.float32,
                (1, 1),
                (2, 2),
                (3, 3),
                False,
            ),
            (
                "3d_float32",
                torch.float32,
                (1, 1, 1),
                (2, 2, 2),
                (3, 3, 3),
                True,
            ),
            (
                "4d_float32",
                torch.float32,
                (1, 1, 1, 1),
                (2, 2, 2, 2),
                (3, 3, 3, 3),
                True,
            ),
        ]
    )
    def test_dynamic_shape_copy_float(self, *args):
        class Copy(nn.Module):
            def forward(self, input, src):
                return torch.ops.aten.copy.default(input, src, args[5])

        input_specs = [
            Input(
                min_shape=args[2],
                opt_shape=args[3],
                max_shape=args[4],
                dtype=args[1],
            ),
            Input(
                min_shape=args[2],
                opt_shape=args[3],
                max_shape=args[4],
                dtype=args[1],
            ),
        ]

        self.run_test_with_dynamic_shape(
            Copy(),
            input_specs,
        )


if __name__ == "__main__":
    run_tests()
