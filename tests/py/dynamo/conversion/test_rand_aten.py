import torch
import torch.nn as nn
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests

from .harness import DispatchTestCase

rand_ops = [
    (
        "rand_one_dimension",
        (lambda shape: torch.ops.aten.rand(shape)),
        [1],
    ),
    (
        "rand_two_dimension",
        (lambda shape: torch.ops.aten.rand(shape)),
        [2,3],
    ),
    (
        "rand_three_dimension",
        (lambda shape: torch.ops.aten.rand(shape)),
        [2,3,4],
    ),
    (
        "randn_one_dimension",
        (lambda shape: torch.ops.aten.randn(shape)),
        [1],
    ),
    (
        "randn_two_dimension",
        (lambda shape: torch.ops.aten.randn(shape)),
        [2,3],
    ),
    (
        "randn_three_dimension",
        (lambda shape: torch.ops.aten.randn(shape)),
        [2,3,4],
    ),
    (
        "randperm_one_case",
        (lambda x: torch.ops.aten.randperm(x)),
        [1],
    ),
    (
        "randperm_two_case",
        (lambda x: torch.ops.aten.randperm(x)),
        [150],
    ),
    (
        "randperm_three_case",
        (lambda x: torch.ops.aten.randperm(x)),
        [1500],
    ),

]

class TestRandConverter(DispatchTestCase):
    @parameterized.expand(
        [
            (
                rand_op[0],
                rand_op[1],
            )
            for rand_op in rand_ops
        ]
    )
    def test_rand(self, _, op, shape_or_input):
        class TestModule(nn.Module):
            def __init__(self, rand_op):
                super().__init__()
                self.rand_op = rand_op

            def forward(self, x):
                return self.rand_op(x)

        inputs = [shape_or_input]
        grid_model = TestModule(op)
        self.run_test(grid_model, inputs)


if __name__ == "__main__":
    run_tests()