import unittest

import torch
import torch_tensorrt
from parameterized import parameterized
from torch.testing._internal.common_cuda import (
    PLATFORM_SUPPORTS_CUDNN_ATTENTION,
    PLATFORM_SUPPORTS_FLASH_ATTENTION,
)
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

        self.assertEqual(
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

        self.assertEqual(
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

        self.assertEqual(
            len(unexpected_ops_seen),
            0,
            f"The following unexpected ops were encountered: {unexpected_ops_seen}",
        )

        self.assertEqual(
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

        self.assertEqual(
            len(unexpected_ops_seen),
            0,
            f"The following unexpected ops were encountered: {unexpected_ops_seen}",
        )

        self.assertEqual(
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

        self.assertEqual(
            len(unexpected_ops_seen),
            0,
            f"The following unexpected ops were encountered: {unexpected_ops_seen}",
        )

        self.assertEqual(
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

        self.assertEqual(
            len(unexpected_ops_seen),
            0,
            f"The following unexpected ops were encountered: {unexpected_ops_seen}",
        )

        self.assertEqual(
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

        self.assertEqual(
            len(unexpected_ops_seen),
            0,
            f"The following unexpected ops were encountered: {unexpected_ops_seen}",
        )

        self.assertEqual(
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

        self.assertEqual(
            len(unexpected_ops_seen),
            0,
            f"The following unexpected ops were encountered: {unexpected_ops_seen}",
        )

        self.assertEqual(
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

    def test_lowering_full_like_module(self):
        class FullLike(torch.nn.Module):
            def __init__(self, *args, **kwargs) -> None:
                super().__init__(*args, **kwargs)

            def forward(self, x):
                y = torch.full_like(x, 2.0)
                return y

        # Operations expected to be removed in the traced graph after decompositions
        expected_ops = {torch.ops.aten.full.default}
        unexpected_ops = {torch.ops.aten.full_like.default}

        inputs = [torch.randn(3, 3, dtype=torch.float32).cuda()]

        fx_graph = torch.fx.symbolic_trace(FullLike())
        unexpected_ops_seen, expected_ops_unseen = lower_graph_testing(
            fx_graph,
            inputs,
            expected_ops=expected_ops,
            unexpected_ops=unexpected_ops,
            min_block_size=1,
        )

        self.assertEqual(
            len(unexpected_ops_seen),
            0,
            f"The following unexpected ops were encountered: {unexpected_ops_seen}",
        )

        self.assertEqual(
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
            truncate_double=True,
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
            f"FullLike TRT outputs don't match with the original model.",
        )

    def test_lowering_empty_like_module(self):
        class emptyLike(torch.nn.Module):
            def __init__(self, *args, **kwargs) -> None:
                super().__init__(*args, **kwargs)

            def forward(self, x):
                c = torch.ops.aten.add(x, x)
                y = torch.ops.aten.empty_like.default(c)
                d = y + c
                return d

        # Operations expected to be removed in the traced graph after decompositions
        expected_ops = {torch.ops.aten.add.Tensor}
        unexpected_ops = {
            torch.ops.aten.empty_like.default,
            torch.ops.aten.empty_permuted.default,
        }

        inputs = [torch.zeros(3, 2).cuda()]

        fx_graph = torch.fx.symbolic_trace(emptyLike())
        unexpected_ops_seen, expected_ops_unseen = lower_graph_testing(
            fx_graph,
            inputs,
            expected_ops=expected_ops,
            unexpected_ops=unexpected_ops,
            min_block_size=1,
        )

        self.assertEqual(
            len(unexpected_ops_seen),
            0,
            f"The following unexpected ops were encountered: {unexpected_ops_seen}",
        )

        self.assertEqual(
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
            truncate_double=True,
            pass_through_build_failures=True,
        )
        optimized_model_results = optimized_model(*inputs).detach().cpu()
        torch_model_results = fx_graph(*inputs).detach().cpu()

        optimized_model_results_shape = optimized_model_results.size()
        torch_model_results_shape = torch_model_results.size()

        self.assertEqual(
            optimized_model_results_shape,
            torch_model_results_shape,
            f"The optimized model results shape and torch model results shape should be equal in empty_like",
        )

    def test_lowering_slice_scatter_dimOne_module(self):
        class sliceScatter(torch.nn.Module):
            def __init__(self, *args, **kwargs) -> None:
                super().__init__(*args, **kwargs)

            def forward(self, x, src, dim, start=None, end=None, step=1):
                y = torch.ops.aten.slice_scatter(x, src, dim, start, end, step)
                return y

        # Operations expected to be removed in the traced graph after decompositions
        expected_ops = {
            torch.ops.aten.scatter.src,
        }
        unexpected_ops = {torch.ops.aten.select_scatter}

        inputs = [torch.zeros(8, 8).cuda(), torch.ones(8, 2).cuda(), 1, 6, None, 1]

        fx_graph = torch.fx.symbolic_trace(sliceScatter())
        unexpected_ops_seen, expected_ops_unseen = lower_graph_testing(
            fx_graph,
            inputs,
            expected_ops=expected_ops,
            unexpected_ops=unexpected_ops,
            min_block_size=1,
        )

        self.assertEqual(
            len(unexpected_ops_seen),
            0,
            f"The following unexpected ops were encountered: {unexpected_ops_seen}",
        )

        self.assertEqual(
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
            truncate_double=True,
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
            f"Slice_scatter TRT outputs don't match with the original model.",
        )

    def test_lowering_slice_scatter_dimZero_StepTwo_module(self):
        class sliceScatter(torch.nn.Module):
            def __init__(self, *args, **kwargs) -> None:
                super().__init__(*args, **kwargs)

            def forward(self, x, src, dim, start, end, step):
                y = torch.ops.aten.slice_scatter.default(x, src, dim, start, end, step)
                return y

        # Operations expected to be removed in the traced graph after decompositions
        expected_ops = {
            torch.ops.aten.scatter.src,
        }
        unexpected_ops = {torch.ops.aten.slice_scatter}

        inputs = [torch.zeros(8, 8).cuda(), torch.ones(2, 8).cuda(), 0, 2, 6, 2]

        fx_graph = torch.fx.symbolic_trace(sliceScatter())

        unexpected_ops_seen, expected_ops_unseen = lower_graph_testing(
            fx_graph,
            inputs,
            expected_ops=expected_ops,
            unexpected_ops=unexpected_ops,
            min_block_size=1,
        )

        self.assertEqual(
            len(unexpected_ops_seen),
            0,
            f"The following unexpected ops were encountered: {unexpected_ops_seen}",
        )

        self.assertEqual(
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
            truncate_double=True,
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
            f"Slice_scatter TRT outputs don't match with the original model.",
        )

    def test_lowering_slice_scatter_dimOne_3d_module(self):
        class sliceScatter(torch.nn.Module):
            def __init__(self, *args, **kwargs) -> None:
                super().__init__(*args, **kwargs)

            def forward(self, x, src, dim, start, end, step):
                y = torch.ops.aten.slice_scatter.default(x, src, dim, start, end, step)
                return y

        # Operations expected to be removed in the traced graph after decompositions
        expected_ops = {
            torch.ops.aten.scatter.src,
        }
        unexpected_ops = {torch.ops.aten.slice_scatter}

        inputs = [
            torch.zeros(8, 8, 8).cuda(),
            torch.ones(8, 2, 8).cuda(),
            1,
            6,
            None,
            1,
        ]

        fx_graph = torch.fx.symbolic_trace(sliceScatter())

        unexpected_ops_seen, expected_ops_unseen = lower_graph_testing(
            fx_graph,
            inputs,
            expected_ops=expected_ops,
            unexpected_ops=unexpected_ops,
            min_block_size=1,
        )

        self.assertEqual(
            len(unexpected_ops_seen),
            0,
            f"The following unexpected ops were encountered: {unexpected_ops_seen}",
        )

        self.assertEqual(
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
            truncate_double=True,
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
            f"Slice_scatter TRT outputs don't match with the original model.",
        )

    def test_lowering_select_scatter_dimZero_module(self):
        class selectScatter(torch.nn.Module):
            def __init__(self, *args, **kwargs) -> None:
                super().__init__(*args, **kwargs)

            def forward(self, x, src, dim, index):
                y = torch.ops.aten.select_scatter.default(x, src, dim, index)
                return y

        # Operations expected to be removed in the traced graph after decompositions
        expected_ops = {torch.ops.aten.scatter.src, torch.ops.aten.unsqueeze.default}
        unexpected_ops = {
            torch.ops.aten.select_scatter.default,
            torch.ops.aten.slice_scatter.default,
        }

        inputs = [torch.zeros(2, 2).cuda(), torch.ones(2).cuda(), 0, 0]

        fx_graph = torch.fx.symbolic_trace(selectScatter())
        unexpected_ops_seen, expected_ops_unseen = lower_graph_testing(
            fx_graph,
            inputs,
            expected_ops=expected_ops,
            unexpected_ops=unexpected_ops,
            min_block_size=1,
        )

        self.assertEqual(
            len(unexpected_ops_seen),
            0,
            f"The following unexpected ops were encountered: {unexpected_ops_seen}",
        )

        self.assertEqual(
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
            truncate_and_double=True,
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
        expected_ops = {torch.ops.aten.scatter.src, torch.ops.aten.unsqueeze.default}
        unexpected_ops = {
            torch.ops.aten.select_scatter.default,
            torch.ops.aten.slice_scatter.default,
        }

        inputs = [torch.zeros(2, 2).cuda(), torch.ones(2).cuda(), 1, 0]

        fx_graph = torch.fx.symbolic_trace(selectScatter())
        unexpected_ops_seen, expected_ops_unseen = lower_graph_testing(
            fx_graph,
            inputs,
            expected_ops=expected_ops,
            unexpected_ops=unexpected_ops,
            min_block_size=1,
        )

        self.assertEqual(
            len(unexpected_ops_seen),
            0,
            f"The following unexpected ops were encountered: {unexpected_ops_seen}",
        )

        self.assertEqual(
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
            truncate_double=True,
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

    def test_lowering_select_scatter_multidimension_module(self):
        class selectScatter(torch.nn.Module):
            def __init__(self, *args, **kwargs) -> None:
                super().__init__(*args, **kwargs)

            def forward(self, x, src, dim, index):
                y = torch.ops.aten.select_scatter.default(x, src, dim, index)
                return y

        # Operations expected to be removed in the traced graph after decompositions
        expected_ops = {torch.ops.aten.scatter.src, torch.ops.aten.unsqueeze.default}
        unexpected_ops = {
            torch.ops.aten.select_scatter.default,
            torch.ops.aten.slice_scatter.default,
        }

        inputs = [torch.zeros(2, 3, 4).cuda(), torch.ones(2, 4).cuda(), 1, 0]

        fx_graph = torch.fx.symbolic_trace(selectScatter())
        unexpected_ops_seen, expected_ops_unseen = lower_graph_testing(
            fx_graph,
            inputs,
            expected_ops=expected_ops,
            unexpected_ops=unexpected_ops,
            min_block_size=1,
        )

        self.assertEqual(
            len(unexpected_ops_seen),
            0,
            f"The following unexpected ops were encountered: {unexpected_ops_seen}",
        )

        self.assertEqual(
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
            truncate_double=True,
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

    empty_ops = [
        (
            "empty_stride_one_dimension_firstcase",
            [5, 5],
            [1, 2],
            None,
        ),
        (
            "empty_stride_two_dimension_secondcase",
            [5, 5],
            [2, 2],
            None,
        ),
        (
            "empty_three_dimension",
            [8, 8, 8],
            [1, 2, 3],
            torch.int32,
        ),
    ]

    @parameterized.expand(
        [(empty_op[0], empty_op[1], empty_op[2], empty_op[3]) for empty_op in empty_ops]
    )
    def test_empty_stride(self, _, shape_or_input, stride, data_type):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, input):
                # The add operation is added otherwise it returns an empty graph post lowering passes
                add_tensor = torch.ops.aten.add(input[0], input[0])
                shape_or_input[0] = input[0].shape[0]
                empty_strided = torch.ops.aten.empty_strided.default(
                    shape_or_input, stride, dtype=data_type
                )
                add_tensor = empty_strided.cuda() + add_tensor
                return add_tensor

        # Operations expected to be included in the traced graph after decompositions
        unexpected_ops = {
            torch.ops.aten.empty_strided.default,
            torch.ops.aten.empty_permuted.default,
        }
        expected_ops = {torch.ops.aten.add.Tensor}

        input = [torch.randint(1, 3, shape_or_input, dtype=torch.int32).cuda()]
        inputs = [input]

        fx_graph = torch.fx.symbolic_trace(TestModule())

        unexpected_ops_seen, expected_ops_unseen = lower_graph_testing(
            fx_graph,
            inputs,
            expected_ops=expected_ops,
            unexpected_ops=unexpected_ops,
            min_block_size=2,
        )

        torch._dynamo.reset()

        self.assertEqual(
            len(unexpected_ops_seen),
            0,
            f"The following unexpected ops were encountered: {unexpected_ops_seen}",
        )

        self.assertEqual(
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
            truncate_double=True,
            pass_through_build_failures=True,
        )
        optimized_model_results = optimized_model(*inputs).detach().cpu()
        torch_model_results = fx_graph(*inputs).detach().cpu()

        self.assertEqual(
            optimized_model_results.shape,
            torch_model_results.shape,
            f"The optimized model results shape and torch model results shape should be equal in empty_stride",
        )

    @parameterized.expand(
        [
            (
                "scatter_add_zero_dim_indexOne_constant",
                0,
                torch.tensor([[0, 1, 2, 0]]).cuda(),
                torch.tensor([[1, 2, 3, 4]], dtype=torch.int32).cuda(),
                {torch.ops.aten.add.Tensor},
            ),
            (
                "scatter_add_zero_dim_indexTwo_constant",
                0,
                torch.tensor([[0, 1, 2, 0], [1, 2, 1, 1]]).cuda(),
                torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.int32).cuda(),
                {torch.ops.aten.add.Tensor},
            ),
            (
                "scatter_add_one_dim_indexOne_constant",
                1,
                torch.tensor([[0, 1, 2, 0]]).cuda(),
                torch.tensor([[1, 2, 3, 1]], dtype=torch.int32).cuda(),
                {
                    torch.ops.aten.add.Tensor,
                },
            ),
            (
                "scatter_add_one_dim_indexTwo_constant",
                1,
                torch.tensor([[0, 1, 2, 0], [1, 2, 1, 1]]).cuda(),
                torch.tensor([[1, 2, 3, 1], [5, 6, 5, 5]], dtype=torch.int32).cuda(),
                {
                    torch.ops.aten.add.Tensor,
                },
            ),
            (
                "scatter_add_one_dim_indexTwo_constant",
                1,
                torch.tensor([[0, 1, 2, 0], [1, 2, 1, 1], [3, 2, 1, 2]]).cuda(),
                torch.tensor(
                    [[1, 2, 3, 1], [5, 6, 5, 5], [2, 4, 3, 2]], dtype=torch.int32
                ).cuda(),
                {
                    torch.ops.aten.add.Tensor,
                },
            ),
        ]
    )
    def test_scatter_add(self, _, dim, index, src, expected_ops_param):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, input):
                return torch.ops.aten.scatter_add.default(input, dim, index, src)

        # Operations expected to be included in the traced graph after decompositions
        expected_ops = expected_ops_param
        unexpected_ops = {torch.ops.aten.scatter_add.default}

        input = torch.zeros(3, 5, dtype=torch.int32).cuda()
        inputs = [input]

        fx_graph = torch.fx.symbolic_trace(TestModule())
        unexpected_ops_seen, expected_ops_unseen = lower_graph_testing(
            fx_graph,
            inputs,
            expected_ops=expected_ops,
            unexpected_ops=unexpected_ops,
            min_block_size=2,
        )

        self.assertEqual(
            len(expected_ops_unseen),
            0,
            f"The following expected ops were not encountered: {expected_ops_unseen}",
        )

        self.assertEqual(
            len(unexpected_ops_seen),
            0,
            f"The following expected ops were not encountered: {unexpected_ops_seen}",
        )

        torch._dynamo.reset()

        # Validate that the results between Torch and Torch-TRT are similar
        optimized_model = torch_tensorrt.compile(
            fx_graph,
            "torch_compile",
            inputs,
            min_block_size=1,
            truncate_double=True,
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
            f"Scatter_add TRT outputs don't match with the original model.",
        )

    @parameterized.expand(
        [
            ############################sum###########################
            (
                "scatter_reduce_add_zero_dim_indexOne_constant",
                0,
                torch.tensor([[0, 1, 2, 0]]).cuda(),
                torch.tensor([[1, 2, 3, 4]], dtype=torch.int32).cuda(),
                {torch.ops.aten.add.Tensor},
                torch.zeros(3, 5, dtype=torch.int32).cuda(),
                "sum",
            ),
            (
                "scatter_reduce_add_zero_dim_indexTwo_constant",
                0,
                torch.tensor([[0, 1, 2, 0], [1, 2, 1, 1]]).cuda(),
                torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.int32).cuda(),
                {torch.ops.aten.add.Tensor, torch.ops.aten.scatter.src},
                torch.zeros(3, 5, dtype=torch.int32).cuda(),
                "sum",
            ),
            (
                "scatter_reduce_add_one_dim_indexOne_constant",
                1,
                torch.tensor([[0, 1, 2, 0]]).cuda(),
                torch.tensor([[1, 2, 3, 1]], dtype=torch.int32).cuda(),
                {
                    torch.ops.aten.add.Tensor,
                    torch.ops.aten.scatter.src,
                },
                torch.zeros(3, 5, dtype=torch.int32).cuda(),
                "sum",
            ),
            (
                "scatter_reduce_add_one_dim_indexTwo_constant",
                1,
                torch.tensor([[0, 1, 2, 0], [1, 2, 1, 1]]).cuda(),
                torch.tensor([[1, 2, 3, 1], [5, 6, 5, 5]], dtype=torch.int32).cuda(),
                {
                    torch.ops.aten.add.Tensor,
                    torch.ops.aten.scatter.src,
                },
                torch.zeros(3, 5, dtype=torch.int32).cuda(),
                "sum",
            ),
            (
                "scatter_reduce_add_one_dim_indexOne_constant_3D",
                1,
                torch.tensor(
                    [[[0, 1, 2, 0], [1, 2, 1, 1]], [[3, 2, 1, 2], [0, 1, 2, 0]]]
                ).cuda(),
                torch.tensor(
                    [[[1, 2, 3, 1], [5, 6, 5, 5]], [[2, 4, 3, 2], [1, 2, 3, 1]]],
                    dtype=torch.int32,
                ).cuda(),
                {
                    torch.ops.aten.add.Tensor,
                    torch.ops.aten.scatter.src,
                },
                torch.zeros(3, 5, 6, dtype=torch.int32).cuda(),
                "sum",
            ),
            ###########################prod###########################
            (
                "scatter_reduce_prod_zero_dim_indexOne_constant",
                0,
                torch.tensor([[0, 1, 2, 0]]).cuda(),
                torch.tensor([[1, 2, 3, 4]], dtype=torch.int32).cuda(),
                {
                    torch.ops.aten.mul.Tensor,
                    torch.ops.aten.scatter.src,
                },
                torch.ones(3, 5, dtype=torch.int32).cuda(),
                "prod",
            ),
            (
                "scatter_reduce_prod_zero_dim_indexTwo_constant",
                0,
                torch.tensor([[0, 1, 2, 0], [1, 2, 1, 1]]).cuda(),
                torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.int32).cuda(),
                {
                    torch.ops.aten.mul.Tensor,
                    torch.ops.aten.scatter.src,
                },
                torch.ones(3, 5, dtype=torch.int32).cuda(),
                "prod",
            ),
            (
                "scatter_reduce_prod_one_dim_indexOne_constant",
                1,
                torch.tensor([[0, 1, 2, 0]]).cuda(),
                torch.tensor([[1, 2, 3, 1]], dtype=torch.int32).cuda(),
                {
                    torch.ops.aten.mul.Tensor,
                    torch.ops.aten.scatter.src,
                },
                torch.ones(3, 5, dtype=torch.int32).cuda(),
                "prod",
            ),
            (
                "scatter_reduce_prod_one_dim_indexTwo_constant",
                1,
                torch.tensor([[0, 1, 2, 0], [1, 2, 1, 1]]).cuda(),
                torch.tensor([[1, 2, 3, 1], [5, 6, 5, 5]], dtype=torch.int32).cuda(),
                {
                    torch.ops.aten.mul.Tensor,
                    torch.ops.aten.scatter.src,
                },
                torch.ones(3, 5, dtype=torch.int32).cuda(),
                "prod",
            ),
            (
                "scatter_reduce_prod_one_dim_indexTwo_constant_3D",
                1,
                torch.tensor(
                    [[[0, 1, 2, 0], [1, 2, 1, 1]], [[3, 2, 1, 2], [0, 1, 2, 0]]]
                ).cuda(),
                torch.tensor(
                    [[[1, 2, 3, 1], [5, 6, 5, 5]], [[2, 4, 3, 2], [1, 2, 3, 1]]],
                    dtype=torch.int32,
                ).cuda(),
                {
                    torch.ops.aten.mul.Tensor,
                    torch.ops.aten.scatter.src,
                },
                torch.ones(3, 5, 6, dtype=torch.int32).cuda(),
                "prod",
            ),
            # #############################mean###########################
            (
                "scatter_reduce_mean_zero_dim_indexOne_constant",
                0,
                torch.tensor([[0, 1, 2, 0]]).cuda(),
                torch.tensor([[1, 2, 3, 4]], dtype=torch.int32).cuda(),
                {
                    torch.ops.aten.add.Tensor,
                    torch.ops.aten.div.Tensor_mode,
                },
                torch.zeros(3, 5, dtype=torch.int32).cuda(),
                "mean",
            ),
            (
                "scatter_reduce_mean_zero_dim_indexTwo_constant",
                0,
                torch.tensor([[0, 1, 2, 0], [1, 2, 1, 1]]).cuda(),
                torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.int32).cuda(),
                {
                    torch.ops.aten.add.Tensor,
                    torch.ops.aten.div.Tensor_mode,
                    torch.ops.aten.scatter.src,
                },
                torch.zeros(3, 5, dtype=torch.int32).cuda(),
                "mean",
            ),
            (
                "scatter_reduce_mean_one_dim_indexOne_constant",
                1,
                torch.tensor([[0, 1, 2, 0]]).cuda(),
                torch.tensor([[1, 2, 3, 1]], dtype=torch.int32).cuda(),
                {
                    torch.ops.aten.add.Tensor,
                    torch.ops.aten.div.Tensor_mode,
                    torch.ops.aten.scatter.src,
                },
                torch.zeros(3, 5, dtype=torch.int32).cuda(),
                "mean",
            ),
            (
                "scatter_reduce_mean_one_dim_indexTwo_constant",
                1,
                torch.tensor([[0, 1, 2, 0], [1, 2, 1, 1]]).cuda(),
                torch.tensor([[1, 2, 3, 1], [5, 6, 5, 5]], dtype=torch.int32).cuda(),
                {
                    torch.ops.aten.add.Tensor,
                    torch.ops.aten.div.Tensor_mode,
                    torch.ops.aten.scatter.src,
                },
                torch.zeros(3, 5, dtype=torch.int32).cuda(),
                "mean",
            ),
            (
                "scatter_reduce_mean_one_dim_indexTwo_constant_3D",
                1,
                torch.tensor(
                    [[[0, 1, 2, 0], [1, 2, 1, 1]], [[3, 2, 1, 2], [0, 1, 2, 0]]]
                ).cuda(),
                torch.tensor(
                    [[[1, 2, 3, 1], [5, 6, 5, 5]], [[2, 4, 3, 2], [1, 2, 3, 1]]],
                    dtype=torch.int32,
                ).cuda(),
                {
                    torch.ops.aten.add.Tensor,
                    torch.ops.aten.div.Tensor_mode,
                    torch.ops.aten.scatter.src,
                },
                torch.zeros(3, 5, 6, dtype=torch.int32).cuda(),
                "mean",
            ),
            # #############################amax###########################
            (
                "scatter_reduce_amax_zero_dim_indexOne_constant",
                0,
                torch.tensor([[0, 1, 2, 0]]).cuda(),
                torch.tensor([[1, 2, 3, 4]], dtype=torch.int32).cuda(),
                {
                    torch.ops.aten.maximum.default,
                    torch.ops.aten.scatter.src,
                },
                torch.zeros(3, 5, dtype=torch.int32).cuda(),
                "amax",
            ),
            (
                "scatter_reduce_amax_zero_dim_indexTwo_constant",
                0,
                torch.tensor([[0, 1, 2, 0], [1, 2, 1, 1]]).cuda(),
                torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.int32).cuda(),
                {
                    torch.ops.aten.maximum.default,
                    torch.ops.aten.scatter.src,
                },
                torch.zeros(3, 5, dtype=torch.int32).cuda(),
                "amax",
            ),
            (
                "scatter_reduce_amax_one_dim_indexOne_constant",
                1,
                torch.tensor([[0, 1, 2, 0]]).cuda(),
                torch.tensor([[1, 2, 3, 1]], dtype=torch.int32).cuda(),
                {
                    torch.ops.aten.maximum.default,
                    torch.ops.aten.scatter.src,
                },
                torch.zeros(3, 5, dtype=torch.int32).cuda(),
                "amax",
            ),
            (
                "scatter_reduce_amax_one_dim_indexTwo_constant",
                1,
                torch.tensor([[0, 1, 2, 0], [1, 2, 1, 1]]).cuda(),
                torch.tensor([[1, 2, 3, 1], [5, 6, 5, 5]], dtype=torch.int32).cuda(),
                {
                    torch.ops.aten.maximum.default,
                    torch.ops.aten.scatter.src,
                },
                torch.zeros(3, 5, dtype=torch.int32).cuda(),
                "amax",
            ),
            (
                "scatter_reduce_amax_one_dim_indexTwo_constant_3D",
                1,
                torch.tensor(
                    [[[0, 1, 2, 0], [1, 2, 1, 1]], [[3, 2, 1, 2], [0, 1, 2, 0]]]
                ).cuda(),
                torch.tensor(
                    [[[1, 2, 3, 1], [5, 6, 5, 5]], [[2, 4, 3, 2], [1, 2, 3, 1]]],
                    dtype=torch.int32,
                ).cuda(),
                {
                    torch.ops.aten.maximum.default,
                    torch.ops.aten.scatter.src,
                },
                torch.zeros(3, 5, 6, dtype=torch.int32).cuda(),
                "amax",
            ),
            # #############################amin###########################
            (
                "scatter_reduce_amin_zero_dim_indexOne_constant",
                0,
                torch.tensor([[0, 1, 2, 0]]).cuda(),
                torch.tensor([[1, 2, 3, 4]], dtype=torch.int32).cuda(),
                {
                    torch.ops.aten.minimum.default,
                    torch.ops.aten.scatter.src,
                },
                torch.zeros(3, 5, dtype=torch.int32).cuda(),
                "amin",
            ),
            (
                "scatter_reduce_amin_zero_dim_indexTwo_constant",
                0,
                torch.tensor([[0, 1, 2, 0], [1, 2, 1, 1]]).cuda(),
                torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.int32).cuda(),
                {
                    torch.ops.aten.minimum.default,
                    torch.ops.aten.scatter.src,
                },
                torch.zeros(3, 5, dtype=torch.int32).cuda(),
                "amin",
            ),
            (
                "scatter_reduce_amin_one_dim_indexOne_constant",
                1,
                torch.tensor([[0, 1, 2, 0]]).cuda(),
                torch.tensor([[1, 2, 3, 1]], dtype=torch.int32).cuda(),
                {
                    torch.ops.aten.minimum.default,
                    torch.ops.aten.scatter.src,
                },
                torch.zeros(3, 5, dtype=torch.int32).cuda(),
                "amin",
            ),
            (
                "scatter_reduce_amin_one_dim_indexTwo_constant",
                1,
                torch.tensor([[0, 1, 2, 0], [1, 2, 1, 1]]).cuda(),
                torch.tensor([[1, 2, 3, 1], [5, 6, 5, 5]], dtype=torch.int32).cuda(),
                {
                    torch.ops.aten.minimum.default,
                    torch.ops.aten.scatter.src,
                },
                torch.zeros(3, 5, dtype=torch.int32).cuda(),
                "amin",
            ),
            (
                "scatter_reduce_amin_one_dim_indexTwo_constant_3D",
                1,
                torch.tensor(
                    [[[0, 1, 2, 0], [1, 2, 1, 1]], [[3, 2, 1, 2], [0, 1, 2, 0]]]
                ).cuda(),
                torch.tensor(
                    [[[1, 2, 3, 1], [5, 6, 5, 5]], [[2, 4, 3, 2], [1, 2, 3, 1]]],
                    dtype=torch.int32,
                ).cuda(),
                {
                    torch.ops.aten.minimum.default,
                    torch.ops.aten.scatter.src,
                },
                torch.zeros(3, 5, 6, dtype=torch.int32).cuda(),
                "amin",
            ),
        ]
    )
    def test_scatter_reduce(
        self, _, dim, index, src, expected_ops_param, input_reduce_op, reduce_op_str
    ):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, input):

                return torch.ops.aten.scatter_reduce_.two(
                    input, dim, index, src, reduce=reduce_op_str
                )

        # Operations expected to be included in the traced graph after decompositions
        expected_ops = expected_ops_param
        unexpected_ops = {torch.ops.aten.scatter_reduce_.two}

        input = torch.zeros(3, 5, dtype=torch.int32).cuda()
        inputs = [input_reduce_op]

        fx_graph = torch.fx.symbolic_trace(TestModule())
        unexpected_ops_seen, expected_ops_unseen = lower_graph_testing(
            fx_graph,
            inputs,
            expected_ops=expected_ops,
            unexpected_ops=unexpected_ops,
        )

        self.assertEqual(
            len(expected_ops_unseen),
            0,
            f"The following expected ops were not encountered: {expected_ops_unseen}",
        )

        self.assertEqual(
            len(unexpected_ops_seen),
            0,
            f"The following expected ops were not encountered: {unexpected_ops_seen}",
        )

        torch._dynamo.reset()

        # Validate that the results between Torch and Torch-TRT are similar
        optimized_model = torch_tensorrt.compile(
            fx_graph,
            "torch_compile",
            inputs,
            min_block_size=3,
            truncate_double=True,
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
            f"Scatter_reduce TRT outputs don't match with the original model.",
        )

    @parameterized.expand(
        [
            (torch.float, False),
            (torch.half, False),
            (torch.half, True),
        ]
    )
    def test_lowering_log_softmax(self, dtype, half_to_float):
        class TestModule(torch.nn.Module):
            def forward(self, x):
                return torch.ops.aten._log_softmax.default(x, 1, half_to_float)

        # Operations expected to be removed in the traced graph after decompositions
        expected_ops = {torch.ops.aten._softmax.default, torch.ops.aten.log.default}
        unexpected_ops = {torch.ops.aten._log_softmax.default}

        inputs = [torch.randn(1, 3, 5, 7, dtype=dtype, device="cuda")]

        fx_graph = torch.fx.symbolic_trace(TestModule())
        unexpected_ops_seen, expected_ops_unseen = lower_graph_testing(
            fx_graph,
            inputs,
            expected_ops=expected_ops,
            unexpected_ops=unexpected_ops,
            min_block_size=1,
        )

        self.assertEqual(
            len(unexpected_ops_seen),
            0,
            f"The following unexpected ops were encountered: {unexpected_ops_seen}",
        )

        self.assertEqual(
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
            f"Log_softmax TRT outputs don't match with the original model.",
        )

    @parameterized.expand(
        [
            ((1, 3, 5), True),
            ((1, 3, 5), False),
            ((2, 4, 6, 8), True),
            ((2, 4, 6, 8), False),
            ((3, 6, 9, 12, 15), True),
            ((3, 6, 9, 12, 15), False),
        ]
    )
    def test_lowering_instance_norm(self, shape, use_input_stats):
        class TestModule(torch.nn.Module):
            def forward(self, input, weight, bias, running_mean=None, running_var=None):
                return torch.ops.aten.instance_norm.default(
                    input,
                    weight,
                    bias,
                    running_mean,
                    running_var,
                    use_input_stats,
                    0.1,
                    1e-05,
                    True,
                )

        # Operations expected to be removed in the traced graph after decompositions
        unexpected_ops = {torch.ops.aten.instance_norm.default}

        inputs = [
            torch.randn(shape, device="cuda"),
            torch.randn(shape[1], device="cuda"),
            torch.randn(shape[1], device="cuda"),
        ]
        if not use_input_stats:
            inputs += [
                torch.randn(shape[1], device="cuda"),
                torch.rand(shape[1], device="cuda"),
            ]

        fx_graph = torch.fx.symbolic_trace(TestModule())
        unexpected_ops_seen, _ = lower_graph_testing(
            fx_graph, inputs, unexpected_ops=unexpected_ops, min_block_size=1
        )

        self.assertEqual(
            len(unexpected_ops_seen),
            0,
            f"The following unexpected ops were encountered: {unexpected_ops_seen}",
        )

        torch._dynamo.reset()

        # Validate that the results between Torch and Torch-TRT are similar
        optimized_model = torch_tensorrt.compile(
            fx_graph, "dynamo", inputs, min_block_size=1
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
            "Instance_norm TRT outputs don't match with the original model.",
        )

    @parameterized.expand(
        [
            (True, False, None, False),
            (False, True, 0.123, True),
        ]
    )
    def test_lowering_scaled_dot_product_attention(
        self, attn, is_causal, scale, enable_gqa
    ):
        class TestModule(torch.nn.Module):
            def forward(self, query, key, value, attn_mask=None):
                return torch.ops.aten.scaled_dot_product_attention.default(
                    query,
                    key,
                    value,
                    attn_mask,
                    0.0,
                    is_causal,
                    scale=scale,
                    enable_gqa=enable_gqa,
                )

        # Operations expected to be removed in the traced graph after decompositions
        unexpected_ops = {torch.ops.aten.scaled_dot_product_attention.default}

        inputs = [
            torch.rand(2, 4, 8, 16, dtype=torch.half, device="cuda"),
            torch.rand(2, 4, 8, 16, dtype=torch.half, device="cuda"),
            torch.rand(2, 4, 8, 16, dtype=torch.half, device="cuda"),
        ]
        if attn:
            inputs += [torch.rand(2, 4, 8, 8, dtype=torch.half, device="cuda")]

        fx_graph = torch.fx.symbolic_trace(TestModule())
        unexpected_ops_seen, _ = lower_graph_testing(
            fx_graph, inputs, unexpected_ops=unexpected_ops, min_block_size=1
        )

        self.assertEqual(
            len(unexpected_ops_seen),
            0,
            f"The following unexpected ops were encountered: {unexpected_ops_seen}",
        )

        torch._dynamo.reset()

        # Validate that the results between Torch and Torch-TRT are similar
        optimized_model = torch_tensorrt.compile(
            fx_graph,
            "dynamo",
            inputs,
            enabled_precisions={torch.half},
            min_block_size=1,
        )
        optimized_model_results = optimized_model(*inputs).detach().cpu()
        torch_model_results = fx_graph(*inputs).detach().cpu()

        max_diff = float(
            torch.max(torch.abs(optimized_model_results - torch_model_results))
        )
        self.assertAlmostEqual(
            max_diff,
            0,
            DECIMALS_OF_AGREEMENT - 1,
            "Scaled_dot_product_attention TRT outputs don't match with the original model.",
        )

    @parameterized.expand(
        [
            (False, None),
            (True, 0.123),
        ]
    )
    @unittest.skipUnless(
        PLATFORM_SUPPORTS_FLASH_ATTENTION, "Platform doesn't support Flash attention"
    )
    def test_lowering_scaled_dot_product_flash_attention(self, is_causal, scale):
        class TestModule(torch.nn.Module):
            def forward(self, query, key, value):
                return torch.ops.aten._scaled_dot_product_flash_attention.default(
                    query,
                    key,
                    value,
                    0.0,
                    is_causal,
                    False,
                    scale=scale,
                )[0]

        # Operations expected to be removed in the traced graph after decompositions
        unexpected_ops = {torch.ops.aten._scaled_dot_product_flash_attention.default}

        inputs = [
            torch.rand(2, 4, 8, 16, dtype=torch.half, device="cuda"),
            torch.rand(2, 4, 8, 16, dtype=torch.half, device="cuda"),
            torch.rand(2, 4, 8, 16, dtype=torch.half, device="cuda"),
        ]

        fx_graph = torch.fx.symbolic_trace(TestModule())
        unexpected_ops_seen, _ = lower_graph_testing(
            fx_graph, inputs, unexpected_ops=unexpected_ops, min_block_size=1
        )

        self.assertEqual(
            len(unexpected_ops_seen),
            0,
            f"The following unexpected ops were encountered: {unexpected_ops_seen}",
        )

        torch._dynamo.reset()

        # Validate that the results between Torch and Torch-TRT are similar
        optimized_model = torch_tensorrt.compile(
            fx_graph,
            "dynamo",
            inputs,
            enabled_precisions={torch.half},
            min_block_size=1,
        )
        optimized_model_results = optimized_model(*inputs).detach().cpu()
        torch_model_results = fx_graph(*inputs).detach().cpu()

        max_diff = float(
            torch.max(torch.abs(optimized_model_results - torch_model_results))
        )
        self.assertAlmostEqual(
            max_diff,
            0,
            DECIMALS_OF_AGREEMENT - 1,
            "Scaled_dot_product_flash_attention TRT outputs don't match with the original model.",
        )

    @parameterized.expand(
        [
            (True, False, None),
            (False, True, 0.123),
        ]
    )
    def test_lowering_scaled_dot_product_efficient_attention(
        self, attn, is_causal, scale
    ):
        class TestModule(torch.nn.Module):
            def forward(self, query, key, value, attn_bias=None):
                return torch.ops.aten._scaled_dot_product_efficient_attention.default(
                    query,
                    key,
                    value,
                    attn_bias,
                    False,
                    0.0,
                    is_causal,
                    scale=scale,
                )[0]

        # Operations expected to be removed in the traced graph after decompositions
        unexpected_ops = {
            torch.ops.aten._scaled_dot_product_efficient_attention.default
        }

        inputs = [
            torch.rand(2, 4, 8, 16, dtype=torch.half, device="cuda"),
            torch.rand(2, 4, 8, 16, dtype=torch.half, device="cuda"),
            torch.rand(2, 4, 8, 16, dtype=torch.half, device="cuda"),
        ]
        if attn:
            inputs += [torch.rand(2, 4, 8, 8, dtype=torch.half, device="cuda")]

        fx_graph = torch.fx.symbolic_trace(TestModule())
        unexpected_ops_seen, _ = lower_graph_testing(
            fx_graph, inputs, unexpected_ops=unexpected_ops, min_block_size=1
        )

        self.assertEqual(
            len(unexpected_ops_seen),
            0,
            f"The following unexpected ops were encountered: {unexpected_ops_seen}",
        )

        torch._dynamo.reset()

        # Validate that the results between Torch and Torch-TRT are similar
        optimized_model = torch_tensorrt.compile(
            fx_graph,
            "dynamo",
            inputs,
            enabled_precisions={torch.half},
            min_block_size=1,
        )
        optimized_model_results = optimized_model(*inputs).detach().cpu()
        torch_model_results = fx_graph(*inputs).detach().cpu()

        max_diff = float(
            torch.max(torch.abs(optimized_model_results - torch_model_results))
        )
        self.assertAlmostEqual(
            max_diff,
            0,
            DECIMALS_OF_AGREEMENT - 1,
            "Scaled_dot_product_efficient_attention TRT outputs don't match with the original model.",
        )

    @parameterized.expand(
        [
            (True, False, None),
            (False, True, 0.123),
        ]
    )
    @unittest.skipUnless(
        PLATFORM_SUPPORTS_CUDNN_ATTENTION, "Platform doesn't support cuDNN attention"
    )
    def test_lowering_scaled_dot_product_cudnn_attention(self, attn, is_causal, scale):
        class TestModule(torch.nn.Module):
            def forward(self, query, key, value, attn_bias=None):
                return torch.ops.aten._scaled_dot_product_cudnn_attention.default(
                    query,
                    key,
                    value,
                    attn_bias,
                    False,
                    0.0,
                    is_causal,
                    False,
                    scale=scale,
                )[0]

        # Operations expected to be removed in the traced graph after decompositions
        unexpected_ops = {torch.ops.aten._scaled_dot_product_cudnn_attention.default}

        inputs = [
            torch.rand(2, 4, 8, 16, dtype=torch.half, device="cuda"),
            torch.rand(2, 4, 8, 16, dtype=torch.half, device="cuda"),
            torch.rand(2, 4, 8, 16, dtype=torch.half, device="cuda"),
        ]
        if attn:
            inputs += [torch.rand(2, 4, 8, 8, dtype=torch.half, device="cuda")]

        fx_graph = torch.fx.symbolic_trace(TestModule())
        unexpected_ops_seen, _ = lower_graph_testing(
            fx_graph, inputs, unexpected_ops=unexpected_ops, min_block_size=1
        )

        self.assertEqual(
            len(unexpected_ops_seen),
            0,
            f"The following unexpected ops were encountered: {unexpected_ops_seen}",
        )

        torch._dynamo.reset()

        # Validate that the results between Torch and Torch-TRT are similar
        optimized_model = torch_tensorrt.compile(
            fx_graph,
            "dynamo",
            inputs,
            enabled_precisions={torch.half},
            min_block_size=1,
        )
        optimized_model_results = optimized_model(*inputs).detach().cpu()
        torch_model_results = fx_graph(*inputs).detach().cpu()

        max_diff = float(
            torch.max(torch.abs(optimized_model_results - torch_model_results))
        )
        self.assertAlmostEqual(
            max_diff,
            0,
            DECIMALS_OF_AGREEMENT - 1,
            "Scaled_dot_product_cudnn_attention TRT outputs don't match with the original model.",
        )


if __name__ == "__main__":
    run_tests()
