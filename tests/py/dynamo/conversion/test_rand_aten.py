import torch
import torch.nn as nn
import torch_tensorrt
from parameterized import parameterized
from torch.testing._internal.common_utils import TestCase, run_tests

rand_ops = [
    (
        "rand_one_dimension",
        (lambda shape: torch.ops.aten.rand(shape)),
        [1],
    ),
    (
        "rand_two_dimension",
        (lambda shape: torch.ops.aten.rand(shape)),
        [2, 3],
    ),
    (
        "rand_three_dimension",
        (lambda shape: torch.ops.aten.rand(shape)),
        [2, 3, 4],
    ),
    (
        "randn_one_dimension",
        (lambda shape: torch.ops.aten.randn(shape)),
        [1],
    ),
    (
        "randn_two_dimension",
        (lambda shape: torch.ops.aten.randn(shape)),
        [2, 3],
    ),
    (
        "randn_three_dimension",
        (lambda shape: torch.ops.aten.randn(shape)),
        [2, 3, 4],
    ),
    (
        "randperm_one_case",
        (lambda x: torch.ops.aten.randperm(x)),
        1,
    ),
    (
        "randperm_two_case",
        (lambda x: torch.ops.aten.randperm(x)),
        150,
    ),
    (
        "randperm_three_case",
        (lambda x: torch.ops.aten.randperm(x)),
        1500,
    ),
]


class TestRandConverter(TestCase):
    @parameterized.expand(
        [
            (
                rand_op[0],
                rand_op[1],
                rand_op[2],
            )
            for rand_op in rand_ops
        ]
    )
    def test_rand(self, _, op, shape_or_input):
        class TestModule(nn.Module):
            def __init__(self, rand_op, size):
                super().__init__()
                self.rand_op = rand_op
                self.size = size

            def forward(self):
                return self.rand_op(self.size)

        grid_model = TestModule(op, shape_or_input)
        # cannot use self.run_test() since it expects input in form of tensor

        # self.run_test(grid_model, None)
        fx_graph = torch.fx.symbolic_trace(grid_model)
        torch._dynamo.reset()

        optimized_model = torch_tensorrt.compile(
            fx_graph,
            "torch_compile",
            None,
            min_block_size=1,
            pass_through_build_failures=True,
            truncate_long_and_double=True,
            debug=True,
        )
        optimized_model_results = optimized_model().detach().cpu()
        torch_model_results = fx_graph().detach().cpu()
        max_diff = float(
            torch.max(torch.abs(optimized_model_results - torch_model_results))
        )
        self.assertAlmostEqual(
            max_diff,
            0,
            4,
            f"TRT outputs don't match with the original model.",
        )


if __name__ == "__main__":
    run_tests()
