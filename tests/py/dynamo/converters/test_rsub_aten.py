import torch
import torch.nn as nn
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests, TestCase
from torch_tensorrt.dynamo import compile
from torch_tensorrt.dynamo.test_utils import DispatchTestCase


class TestRSubConverter(DispatchTestCase):
    @parameterized.expand(
        [
            ("2d_dim_alpha", (2, 1), 2),
            ("3d_dim_alpha", (2, 1, 2), 2),
        ]
    )
    def test_rsub_same(self, _, x, alpha):
        class rsub(nn.Module):
            def forward(self, input):
                return torch.rsub(input, input, alpha=alpha)

        inputs = [torch.randn(x)]
        self.run_test(
            rsub(),
            inputs,
            expected_ops={torch.ops.aten.rsub.Tensor},
        )

    # @parameterized.expand(
    #     [
    #         ("2d_dim_alpha", (2, 1), 2),
    #         ("3d_dim_alpha", (2, 1, 2), 2),
    #     ]
    # )
    # def test_rsub_diff(self, _, x, alpha):
    #     class rsub(nn.Module):
    #         def forward(self, inputOne, inputTwo):
    #             return torch.rsub(inputOne, inputTwo, alpha=alpha)

    #     inputOne = [torch.randn(x)]
    #     inputTwo = [torch.randn(x)]
    #     inputs = (inputOne, inputTwo)
    #     self.run_test(
    #         rsub(),
    #         inputs,
    #         expected_ops={torch.ops.aten.rsub.Tensor},
    #     )


class TestRSubDiff(TestCase):
    def test_rsub_diff(self):
        class rsub_diff(nn.Module):
            def forward(self, inputOne, inputTwo):
                return torch.rsub(inputOne, inputTwo, alpha=2)

        inputOne = torch.randn(2, 1).cuda()
        inputTwo = torch.randn(2, 1).cuda()
        alpha = 2
        inputs = [inputOne, inputTwo]
        fx_graph = torch.fx.symbolic_trace(rsub_diff())
        torch._dynamo.reset()
        optimized_model = compile(
            fx_graph, inputs, min_block_size=1, pass_through_build_failures=True
        )
        optimized_model_results = optimized_model(*inputs).detach().cpu()
        torch_model_results = fx_graph(*inputs).detach().cpu()
        max_diff = float(
            torch.max(torch.abs(optimized_model_results - torch_model_results))
        )
        self.assertAlmostEqual(
            max_diff,
            0,
            5,
            f"Reciprocal TRT outputs don't match with the original model.",
        )


if __name__ == "__main__":
    run_tests()
    # Test two
