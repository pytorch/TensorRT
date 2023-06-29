import torch
from utils import lower_graph_testing
from torch.testing._internal.common_utils import run_tests, TestCase
from torch_tensorrt.dynamo import compile


class TestMaxPool1D(TestCase):
    def test_pre_aot_lowering_maxpool1d(self):
        class MaxPool1D(torch.nn.Module):
            def __init__(self, *args, **kwargs) -> None:
                super().__init__(*args, **kwargs)
                self.maxpool = torch.nn.MaxPool1d(2)

            def forward(self, x):
                return self.maxpool(x)

        # Operations expected to be included in the traced graph after decompositions
        expected_ops = {torch.ops.tensorrt.maxpool1d.default}

        inputs = [
            torch.rand(
                9,
                16,
                2,
            ).cuda(),
        ]

        fx_graph = torch.fx.symbolic_trace(MaxPool1D())
        _, expected_ops_unseen = lower_graph_testing(
            fx_graph, inputs, expected_ops=expected_ops, min_block_size=1
        )

        self.assertEquals(
            len(expected_ops_unseen),
            0,
            f"The following expected ops were not encountered: {expected_ops_unseen}",
        )

        torch._dynamo.reset()

        # Validate that the results between Torch and Torch-TRT are similar
        optimized_model = compile(
            fx_graph, inputs, min_block_size=1, pass_through_build_failures=True
        )
        optimized_model_results = optimized_model(*inputs).detach().cpu()
        torch_model_results = fx_graph(*inputs).detach().cpu()

        max_diff = torch.max(torch.abs(optimized_model_results - torch_model_results))
        self.assertAlmostEqual(
            max_diff, 0, f"Maxpool1d TRT outputs don't match with the original model."
        )


class TestEinsum(TestCase):
    def test_pre_aot_lowering_einsum(self):
        class Einsum(torch.nn.Module):
            def forward(self, x, y):
                return torch.einsum("ij,ji->ij", x, y)

        # Operations expected to be included in the traced graph after decompositions
        expected_ops = {torch.ops.tensorrt.einsum.default}

        inputs = [
            torch.rand(
                16,
                16,
            ).cuda(),
            torch.rand(
                16,
                16,
            ).cuda(),
        ]

        fx_graph = torch.fx.symbolic_trace(Einsum())
        _, expected_ops_unseen = lower_graph_testing(
            fx_graph, inputs, expected_ops=expected_ops, min_block_size=1
        )

        self.assertEquals(
            len(expected_ops_unseen),
            0,
            f"The following expected ops were not encountered: {expected_ops_unseen}",
        )

        torch._dynamo.reset()

        # Validate that the results between Torch and Torch-TRT are similar
        optimized_model = compile(
            fx_graph, inputs, min_block_size=1, pass_through_build_failures=True
        )
        optimized_model_results = optimized_model(*inputs).detach().cpu()
        torch_model_results = fx_graph(*inputs).detach().cpu()

        max_diff = torch.max(torch.abs(optimized_model_results - torch_model_results))
        self.assertAlmostEqual(
            max_diff, 0, f"Einsum TRT outputs don't match with the original model."
        )


if __name__ == "__main__":
    run_tests()
