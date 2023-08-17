import torch
import torch.nn as nn
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests

from .harness import DispatchTestCase


class TestMatMulConverter(DispatchTestCase):
    @parameterized.expand(
        [
            ("2_2", (2, 3), (3, 2)),
            ("2_2", (2, 3), (3, 1)),
            # FIXME torch.ops.aten.mv.default for (2,3), (3,1) - should mv be lowered to mm?
            # (2,3), (3,) torch.ops.aten.mv.default
            # Following cases use torch.ops.aten.bmm.defauly
            # ("4_3", (3,1,3,2), (2,2,3)),
            # ("3_4", (3,1,3,2), (2,2,3)),
            # ("3_4", (2, 2, 3), (3, 1, 3, 3)),
            # ("4_2", (1, 2, 2, 3), (3, 2)),
        ]
    )
    def test_matmul_other_constant(self, _, input_shape, other_shape):
        class MatMul(nn.Module):
            def __init__(self):
                super().__init__()
                self.other = nn.Parameter(torch.randn(*other_shape))

            def forward(self, input):
                return torch.matmul(input, self.other)

        inputs = [torch.randn(*input_shape)]

        self.run_test(
            MatMul(),
            inputs,
            expected_ops={torch.ops.aten.mm.default},
        )

    @parameterized.expand(
        [
            ("2_2", (2, 3), (3, 2)),
            ("1_2", (1, 3), (3, 2)),
            # FIXME torch.ops.aten.mv.default for (2,3), (3,1) - should mv be lowered to mm?
            # (2,3), (3,) torch.ops.aten.mv.default
            # Following cases use torch.ops.aten.bmm.defauly
            # ("4_3", (3,1,3,2), (2,2,3)),
            # ("3_4", (3,1,3,2), (2,2,3)),
            # ("3_4", (2, 2, 3), (3, 1, 3, 3)),
            # ("4_2", (1, 2, 2, 3), (3, 2)),
        ]
    )
    def test_matmul_input_constant(self, _, input_shape, other_shape):
        class MatMul(nn.Module):
            def __init__(self):
                super().__init__()
                self.input = nn.Parameter(torch.randn(*input_shape))

            def forward(self, other):
                return torch.matmul(self.input, other)

        inputs = [torch.randn(*other_shape)]

        self.run_test(
            MatMul(),
            inputs,
            expected_ops={torch.ops.aten.mm.default},
        )

    @parameterized.expand(
        [
            ("2_2", (2, 3), (3, 2)),
            # ("2_3", (2, 3), (2, 3, 4)),
            # ("4_4", (2, 2, 2, 3), (2, 1, 3, 2)),
            # ("4_2", (2, 1, 2, 3), (3, 2)),
            # ("2_1", (2, 3), (3,)),
            # ("1_2", (3,), (3, 2)),
            # ("1_1", (3,), (3,)),
        ]
    )
    def test_matmul(self, _, input_shape, other_shape):
        class MatMul(nn.Module):
            def forward(self, input, other):
                return torch.matmul(input, other)

        inputs = [torch.randn(*input_shape), torch.randn(*other_shape)]

        self.run_test(
            MatMul(),
            inputs,
            expected_ops={torch.ops.aten.mm.default},
        )

    # FIXME: dynamic shape is giving bmm


if __name__ == "__main__":
    run_tests()
