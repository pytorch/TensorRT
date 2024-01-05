import torch
import torch_tensorrt
from torch.testing._internal.common_utils import TestCase, run_tests

from ..testing_utilities import DECIMALS_OF_AGREEMENT, lower_graph_testing


class TestLowering(TestCase):
    def test_lowering_inplace_op(self):
        class InPlace(torch.nn.Module):
            def __init__(self, *args, **kwargs) -> None:
                super().__init__(*args, **kwargs)

            def forward(self, x, y):
                x += 1
                x = torch.ops.aten.add_.Tensor(x, y)
                x = torch.ops.aten.relu_.default(x)
                return x

        # Operations expected to be included in the traced graph after decompositions
        expected_ops = {torch.ops.aten.add.Tensor, torch.ops.aten.relu.default}

        inputs = [
            torch.rand(
                5,
            ),
            torch.rand(
                5,
            ),
        ]

        fx_graph = torch.fx.symbolic_trace(InPlace())
        _, expected_ops_unseen = lower_graph_testing(
            fx_graph, inputs, expected_ops=expected_ops, min_block_size=2
        )

        self.assertEquals(
            len(expected_ops_unseen),
            0,
            f"The following expected ops were not encountered: {expected_ops_unseen}",
        )

    def test_lowering_alias_replacement(self):
        class Alias(torch.nn.Module):
            def __init__(self, *args, **kwargs) -> None:
                super().__init__(*args, **kwargs)

            def forward(self, x):
                y = torch.ops.aten.alias.default(x) + 1
                return y

        # Operations expected to be removed in the traced graph after decompositions
        unexpected_ops = {torch.ops.aten.alias.default}

        inputs = [
            torch.rand(
                5,
            ),
        ]

        fx_graph = torch.fx.symbolic_trace(Alias())
        unexpected_ops_seen, _ = lower_graph_testing(
            fx_graph, inputs, unexpected_ops=unexpected_ops, min_block_size=1
        )

        self.assertEquals(
            len(unexpected_ops_seen),
            0,
            f"The following unexpected ops were encountered: {unexpected_ops_seen}",
        )

    def test_lowering_rsqrt(self):
        class Rsqrt(torch.nn.Module):
            def __init__(self, *args, **kwargs) -> None:
                super().__init__(*args, **kwargs)

            def forward(self, x):
                y = torch.ops.aten.rsqrt.default(x)
                return y

        # Operations expected to be removed in the traced graph after decompositions
        expected_ops = {torch.ops.aten.sqrt.default, torch.ops.aten.div.Tensor}
        unexpected_ops = {
            torch.ops.aten.rsqrt.default,
            torch.ops.aten.reciprocal.default,
        }

        inputs = [
            torch.randint(1, 10, (5,), dtype=torch.int32),
        ]

        fx_graph = torch.fx.symbolic_trace(Rsqrt())
        unexpected_ops_seen, expected_ops_unseen = lower_graph_testing(
            fx_graph,
            inputs,
            expected_ops=expected_ops,
            unexpected_ops=unexpected_ops,
            min_block_size=1,
        )

        self.assertEquals(
            len(unexpected_ops_seen),
            0,
            f"The following unexpected ops were encountered: {unexpected_ops_seen}",
        )

        self.assertEquals(
            len(expected_ops_unseen),
            0,
            f"The following expected ops were not encountered: {expected_ops_unseen}",
        )

    def test_lowering_reciprocal(self):
        class Reciprocal(torch.nn.Module):
            def __init__(self, *args, **kwargs) -> None:
                super().__init__(*args, **kwargs)

            def forward(self, x):
                y = torch.ops.aten.reciprocal.default(x)
                return y

        # Operations expected to be removed in the traced graph after decompositions
        expected_ops = {torch.ops.aten.div.Tensor}
        unexpected_ops = {torch.ops.aten.reciprocal.default}

        inputs = [
            torch.randn(
                5,
            ).cuda()
        ]

        fx_graph = torch.fx.symbolic_trace(Reciprocal())
        unexpected_ops_seen, expected_ops_unseen = lower_graph_testing(
            fx_graph,
            inputs,
            expected_ops=expected_ops,
            unexpected_ops=unexpected_ops,
            min_block_size=1,
        )

        self.assertEquals(
            len(unexpected_ops_seen),
            0,
            f"The following unexpected ops were encountered: {unexpected_ops_seen}",
        )

        self.assertEquals(
            len(expected_ops_unseen),
            0,
            f"The following expected ops were not encountered: {expected_ops_unseen}",
        )

        torch._dynamo.reset()

        # Validate that the results between Torch and Torch-TRT are similar
        optimized_model = torch_tensorrt.compile(
            fx_graph,
            "torch_compile",
            inputs,
            min_block_size=1,
            pass_through_build_failures=True,
        )
        optimized_model_results = optimized_model(*inputs).detach().cpu()
        torch_model_results = fx_graph(*inputs).detach().cpu()

        max_diff = float(
            torch.max(torch.abs(optimized_model_results - torch_model_results))
        )
        self.assertAlmostEqual(
            max_diff,
            0,
            DECIMALS_OF_AGREEMENT,
            f"Reciprocal TRT outputs don't match with the original model.",
        )

    def test_lowering_prims_var(self):
        class Var(torch.nn.Module):
            def forward(self, x):
                y = torch.var(x)
                return y

        # Operations expected to be removed in the traced graph after decompositions
        expected_ops = {
            torch.ops.aten.mean.dim,
            torch.ops.aten.sub.Tensor,
            torch.ops.aten.mul.Tensor,
            torch.ops.aten.sum.dim_IntList,
            torch.ops.aten.div.Tensor,
        }
        unexpected_ops = {torch.ops.aten.var.default, torch.ops.prims.div.default}

        inputs = [
            torch.randn(
                5,
                10,
                1,
            ).cuda()
        ]

        fx_graph = torch.fx.symbolic_trace(Var())
        unexpected_ops_seen, expected_ops_unseen = lower_graph_testing(
            fx_graph,
            inputs,
            expected_ops=expected_ops,
            unexpected_ops=unexpected_ops,
            min_block_size=1,
        )

        self.assertEquals(
            len(unexpected_ops_seen),
            0,
            f"The following unexpected ops were encountered: {unexpected_ops_seen}",
        )

        self.assertEquals(
            len(expected_ops_unseen),
            0,
            f"The following expected ops were not encountered: {expected_ops_unseen}",
        )

        torch._dynamo.reset()

        # Validate that the results between Torch and Torch-TRT are similar
        optimized_model = torch_tensorrt.compile(
            fx_graph,
            "torch_compile",
            inputs,
            min_block_size=1,
            pass_through_build_failures=True,
        )
        optimized_model_results = optimized_model(*inputs).detach().cpu()
        torch_model_results = fx_graph(*inputs).detach().cpu()

        max_diff = float(
            torch.max(torch.abs(optimized_model_results - torch_model_results))
        )
        self.assertAlmostEqual(
            max_diff,
            0,
            DECIMALS_OF_AGREEMENT,
            f"Var TRT outputs don't match with the original model.",
        )

    def test_lowering_maxpool1d_functional(self):
        class MaxPool1d(torch.nn.Module):
            def forward(self, x):
                y = torch.nn.functional.max_pool1d(x, 3)
                return y

        # Operations expected to be removed in the traced graph after decompositions
        expected_ops = {torch.ops.aten.max_pool2d.default}
        unexpected_ops = {
            torch.ops.aten.max_pool1d_with_indices.default,
            torch.ops.aten.max_pool2d_with_indices.default,
        }

        inputs = [torch.randn(4, 8, 27).cuda()]

        fx_graph = torch.fx.symbolic_trace(MaxPool1d())
        unexpected_ops_seen, expected_ops_unseen = lower_graph_testing(
            fx_graph,
            inputs,
            expected_ops=expected_ops,
            unexpected_ops=unexpected_ops,
            min_block_size=1,
        )

        self.assertEquals(
            len(unexpected_ops_seen),
            0,
            f"The following unexpected ops were encountered: {unexpected_ops_seen}",
        )

        self.assertEquals(
            len(expected_ops_unseen),
            0,
            f"The following expected ops were not encountered: {expected_ops_unseen}",
        )

        torch._dynamo.reset()

        # Validate that the results between Torch and Torch-TRT are similar
        optimized_model = torch_tensorrt.compile(
            fx_graph,
            "torch_compile",
            inputs,
            min_block_size=1,
            pass_through_build_failures=True,
        )
        optimized_model_results = optimized_model(*inputs).detach().cpu()
        torch_model_results = fx_graph(*inputs).detach().cpu()

        max_diff = float(
            torch.max(torch.abs(optimized_model_results - torch_model_results))
        )
        self.assertAlmostEqual(
            max_diff,
            0,
            DECIMALS_OF_AGREEMENT,
            f"MaxPool1d TRT outputs don't match with the original model.",
        )

    def test_lowering_maxpool_2d_module(self):
        class MaxPool2d(torch.nn.Module):
            def __init__(self, *args, **kwargs) -> None:
                super().__init__(*args, **kwargs)
                self.maxpool = torch.nn.MaxPool2d((5, 3), stride=(2, 1))

            def forward(self, x):
                y = self.maxpool(x)
                return y

        # Operations expected to be removed in the traced graph after decompositions
        expected_ops = {torch.ops.aten.max_pool2d.default}
        unexpected_ops = {torch.ops.aten.max_pool2d_with_indices.default}

        inputs = [torch.randn(1, 3, 25, 30).cuda()]

        fx_graph = torch.fx.symbolic_trace(MaxPool2d())
        unexpected_ops_seen, expected_ops_unseen = lower_graph_testing(
            fx_graph,
            inputs,
            expected_ops=expected_ops,
            unexpected_ops=unexpected_ops,
            min_block_size=1,
        )

        self.assertEquals(
            len(unexpected_ops_seen),
            0,
            f"The following unexpected ops were encountered: {unexpected_ops_seen}",
        )

        self.assertEquals(
            len(expected_ops_unseen),
            0,
            f"The following expected ops were not encountered: {expected_ops_unseen}",
        )

        torch._dynamo.reset()

        # Validate that the results between Torch and Torch-TRT are similar
        optimized_model = torch_tensorrt.compile(
            fx_graph,
            "torch_compile",
            inputs,
            min_block_size=1,
            pass_through_build_failures=True,
        )
        optimized_model_results = optimized_model(*inputs).detach().cpu()
        torch_model_results = fx_graph(*inputs).detach().cpu()

        max_diff = float(
            torch.max(torch.abs(optimized_model_results - torch_model_results))
        )
        self.assertAlmostEqual(
            max_diff,
            0,
            DECIMALS_OF_AGREEMENT,
            f"MaxPool2d TRT outputs don't match with the original model.",
        )

    def test_lowering_maxpool_3d_module(self):
        class MaxPool3d(torch.nn.Module):
            def __init__(self, *args, **kwargs) -> None:
                super().__init__(*args, **kwargs)
                self.maxpool = torch.nn.MaxPool3d(3)

            def forward(self, x):
                y = self.maxpool(x)
                return y

        # Operations expected to be removed in the traced graph after decompositions
        expected_ops = {torch.ops.aten.max_pool3d.default}
        unexpected_ops = {torch.ops.aten.max_pool3d_with_indices.default}

        inputs = [torch.randn(4, 8, 27, 72, 96).cuda()]

        fx_graph = torch.fx.symbolic_trace(MaxPool3d())
        unexpected_ops_seen, expected_ops_unseen = lower_graph_testing(
            fx_graph,
            inputs,
            expected_ops=expected_ops,
            unexpected_ops=unexpected_ops,
            min_block_size=1,
        )

        self.assertEquals(
            len(unexpected_ops_seen),
            0,
            f"The following unexpected ops were encountered: {unexpected_ops_seen}",
        )

        self.assertEquals(
            len(expected_ops_unseen),
            0,
            f"The following expected ops were not encountered: {expected_ops_unseen}",
        )

        torch._dynamo.reset()

        # Validate that the results between Torch and Torch-TRT are similar
        optimized_model = torch_tensorrt.compile(
            fx_graph,
            "torch_compile",
            inputs,
            min_block_size=1,
            pass_through_build_failures=True,
        )
        optimized_model_results = optimized_model(*inputs).detach().cpu()
        torch_model_results = fx_graph(*inputs).detach().cpu()

        max_diff = float(
            torch.max(torch.abs(optimized_model_results - torch_model_results))
        )
        self.assertAlmostEqual(
            max_diff,
            0,
            DECIMALS_OF_AGREEMENT,
            f"MaxPool3d TRT outputs don't match with the original model.",
        )

    def test_lowering_select_scatter_dimZero_module(self):
        class selectScatter(torch.nn.Module):
            def __init__(self, *args, **kwargs) -> None:
                super().__init__(*args, **kwargs)

            def forward(self, x, src, dim, index):
                y = torch.ops.aten.select_scatter.default(x, src, dim, index)
                return y

        # Operations expected to be removed in the traced graph after decompositions
        expected_ops = {
            torch.ops.aten.slice.Tensor,
            torch.ops.aten.squeeze.dim,
            torch.ops.aten.cat.default,
        }
        unexpected_ops = {torch.ops.aten.select_scatter.default}

        inputs = [torch.zeros(2, 2).cuda(), torch.ones(2).cuda(), 0, 0]

        fx_graph = torch.fx.symbolic_trace(selectScatter())
        unexpected_ops_seen, expected_ops_unseen = lower_graph_testing(
            fx_graph,
            inputs,
            expected_ops=expected_ops,
            unexpected_ops=unexpected_ops,
            min_block_size=1,
        )

        self.assertEquals(
            len(unexpected_ops_seen),
            0,
            f"The following unexpected ops were encountered: {unexpected_ops_seen}",
        )

        self.assertEquals(
            len(expected_ops_unseen),
            0,
            f"The following expected ops were not encountered: {expected_ops_unseen}",
        )

        torch._dynamo.reset()

        # Validate that the results between Torch and Torch-TRT are similar
        optimized_model = torch_tensorrt.compile(
            fx_graph,
            "torch_compile",
            inputs,
            min_block_size=1,
            pass_through_build_failures=True,
        )
        optimized_model_results = optimized_model(*inputs).detach().cpu()
        torch_model_results = fx_graph(*inputs).detach().cpu()

        max_diff = float(
            torch.max(torch.abs(optimized_model_results - torch_model_results))
        )
        self.assertAlmostEqual(
            max_diff,
            0,
            DECIMALS_OF_AGREEMENT,
            f"Select_scatter TRT outputs don't match with the original model.",
        )

    def test_lowering_select_scatter_dimOne_module(self):
        class selectScatter(torch.nn.Module):
            def __init__(self, *args, **kwargs) -> None:
                super().__init__(*args, **kwargs)

            def forward(self, x, src, dim, index):
                y = torch.ops.aten.select_scatter.default(x, src, dim, index)
                return y

        # Operations expected to be removed in the traced graph after decompositions
        expected_ops = {
            torch.ops.aten.slice.Tensor,
            torch.ops.aten.squeeze.dim,
            torch.ops.aten.cat.default,
        }
        unexpected_ops = {torch.ops.aten.select_scatter.default}

        inputs = [torch.zeros(2, 2).cuda(), torch.ones(2).cuda(), 1, 0]

        fx_graph = torch.fx.symbolic_trace(selectScatter())
        unexpected_ops_seen, expected_ops_unseen = lower_graph_testing(
            fx_graph,
            inputs,
            expected_ops=expected_ops,
            unexpected_ops=unexpected_ops,
            min_block_size=1,
        )

        self.assertEquals(
            len(unexpected_ops_seen),
            0,
            f"The following unexpected ops were encountered: {unexpected_ops_seen}",
        )

        self.assertEquals(
            len(expected_ops_unseen),
            0,
            f"The following expected ops were not encountered: {expected_ops_unseen}",
        )

        torch._dynamo.reset()

        # Validate that the results between Torch and Torch-TRT are similar
        optimized_model = torch_tensorrt.compile(
            fx_graph,
            "torch_compile",
            inputs,
            min_block_size=1,
            pass_through_build_failures=True,
        )
        optimized_model_results = optimized_model(*inputs).detach().cpu()
        torch_model_results = fx_graph(*inputs).detach().cpu()

        max_diff = float(
            torch.max(torch.abs(optimized_model_results - torch_model_results))
        )
        self.assertAlmostEqual(
            max_diff,
            0,
            DECIMALS_OF_AGREEMENT,
            f"Select_scatter TRT outputs don't match with the original model.",
        )


if __name__ == "__main__":
    run_tests()
