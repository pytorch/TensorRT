import sys
import unittest

import torch
import torch_tensorrt
from torch.testing._internal.common_cuda import PLATFORM_SUPPORTS_FLASH_ATTENTION
from torch.testing._internal.common_utils import TestCase, run_tests

from ..testing_utilities import DECIMALS_OF_AGREEMENT, lower_graph_testing

isSM8XDevice = torch.cuda.is_available() and torch.cuda.get_device_capability() in [
    (8, 6),
    (8, 7),
    (8, 9),
]


class TestInputAsOutput(TestCase):
    def test_input_as_output(self):
        class InputAsOutput(torch.nn.Module):
            def forward(self, x, y):
                y_new = y + x + 1
                y_new = y_new * 7
                return (y_new, x, y)

        inputs = [
            torch.rand(
                5,
                7,
            ).cuda(),
            torch.rand(
                5,
                7,
            ).cuda(),
        ]

        fx_graph = torch.fx.symbolic_trace(InputAsOutput())
        lower_graph_testing(fx_graph, inputs, min_block_size=1)
        torch._dynamo.reset()

        # Validate that the results between Torch and Torch-TRT are similar
        optimized_model = torch_tensorrt.compile(
            fx_graph,
            "torch_compile",
            inputs,
            min_block_size=1,
            pass_through_build_failures=True,
        )
        optimized_model_results = torch.cat(
            [tensor.detach().cpu() for tensor in optimized_model(*inputs)]
        )
        torch_model_results = torch.cat(
            [tensor.detach().cpu() for tensor in fx_graph(*inputs)]
        )

        max_diff = float(
            torch.max(torch.abs(optimized_model_results - torch_model_results))
        )
        self.assertAlmostEqual(
            max_diff,
            0,
            DECIMALS_OF_AGREEMENT,
            msg=f"InputAsOutput TRT outputs don't match with the original model.",
        )
        torch._dynamo.reset()


class TestLoweringPassMembership(TestCase):
    def insert_at_end(self):
        from torch_tensorrt.dynamo.lowering.passes import (
            ATEN_LOWERING_PASSES,
            _aten_lowering_pass,
            _remove_lowering_pass,
        )

        @_aten_lowering_pass
        def identity_pass(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
            return gm

        self.assertEqual(identity_pass, ATEN_LOWERING_PASSES.passes[-1])

        _remove_lowering_pass(-1)

        self.assertNotIn(identity_pass, ATEN_LOWERING_PASSES.passes)

    def insert_at_index(self):
        from torch_tensorrt.dynamo.lowering.passes import (
            ATEN_LOWERING_PASSES,
            _aten_lowering_pass,
            _remove_lowering_pass,
        )

        @_aten_lowering_pass(index=0)
        def identity_pass(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
            return gm

        self.assertEqual(identity_pass, ATEN_LOWERING_PASSES.passes[0])

        _remove_lowering_pass(0)

        self.assertNotIn(identity_pass, ATEN_LOWERING_PASSES.passes)


class TestPrimBroadcastFusion(TestCase):
    def test_broadcast_fusion(self):
        class BroadcastFusion(torch.nn.Module):
            def forward(self, x):
                return torch.var_mean(x, keepdim=True)[1]

        inputs = [
            torch.rand(
                5,
                7,
            ).cuda(),
        ]

        fx_graph = torch.fx.symbolic_trace(BroadcastFusion())
        expected_ops = {torch.ops.aten.sum.dim_IntList}
        unexpected_ops = {torch.ops.aten.var.default, torch.ops.prims.var.default}

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
        optimized_model_results = torch.cat(
            [tensor.detach().cpu() for tensor in optimized_model(*inputs)]
        )
        torch_model_results = torch.cat(
            [tensor.detach().cpu() for tensor in fx_graph(*inputs)]
        )

        max_diff = float(
            torch.max(torch.abs(optimized_model_results - torch_model_results))
        )
        self.assertAlmostEqual(
            max_diff,
            0,
            DECIMALS_OF_AGREEMENT,
            msg=f"BroadcastFusion TRT outputs don't match with the original model.",
        )
        torch._dynamo.reset()


class TestLowerViewToReshape(TestCase):
    def test_view_to_reshape(self):
        class ViewToReshape(torch.nn.Module):
            def forward(self, input):
                out = torch.ops.aten.view.default(input, (1, 1, -1))
                return out

        inputs = [
            torch.rand((3, 4, 5, 32)).cuda(),
        ]

        fx_graph = torch.fx.symbolic_trace(ViewToReshape())
        expected_ops = {torch.ops.aten.reshape.default}
        unexpected_ops = {
            torch.ops.aten.view.default,
        }

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
        optimized_model_results = torch.cat(
            [tensor.detach().cpu() for tensor in optimized_model(*inputs)]
        )
        torch_model_results = torch.cat(
            [tensor.detach().cpu() for tensor in fx_graph(*inputs)]
        )

        max_diff = float(
            torch.max(torch.abs(optimized_model_results - torch_model_results))
        )
        self.assertAlmostEqual(
            max_diff,
            0,
            DECIMALS_OF_AGREEMENT,
            msg=f"ViewToReshape TRT outputs don't match with the original model.",
        )
        torch._dynamo.reset()


class TestFP32Accumulation(TestCase):
    def test_fp32_acc(self):
        class FP32Acc(torch.nn.Module):
            def forward(self, input, weight):
                out = torch.ops.aten.mm.default(input, weight)
                return out

        inputs = [
            torch.rand((3, 4)).cuda(),
            torch.rand((4, 5)).cuda(),
        ]

        fx_graph = torch.fx.symbolic_trace(FP32Acc())
        expected_ops = {torch.ops.aten._to_copy.default, torch.ops.aten.mm.default}
        unexpected_ops = {}

        unexpected_ops_seen, expected_ops_unseen = lower_graph_testing(
            fx_graph,
            inputs,
            expected_ops=expected_ops,
            unexpected_ops=unexpected_ops,
            min_block_size=1,
            use_fp32_acc=True,
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

    def test_fp32_acc_for_addmm(self):
        class FP32Acc(torch.nn.Module):
            def forward(self, input, mat1, mat2):
                out = torch.ops.aten.addmm.default(input, mat1, mat2)
                return out

        inputs = [
            torch.rand((3, 5)).cuda(),
            torch.rand((3, 4)).cuda(),
            torch.rand((4, 5)).cuda(),
        ]

        fx_graph = torch.fx.symbolic_trace(FP32Acc())
        expected_ops = {
            torch.ops.aten._to_copy.default,
            torch.ops.aten.mm.default,
            torch.ops.aten.add.Tensor,
        }
        unexpected_ops = {}

        unexpected_ops_seen, expected_ops_unseen = lower_graph_testing(
            fx_graph,
            inputs,
            expected_ops=expected_ops,
            unexpected_ops=unexpected_ops,
            min_block_size=1,
            use_fp32_acc=True,
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


class TestLowerEfficientAttention(TestCase):
    def test_lower_efficient_attention(self):
        class EfficientAttention(torch.nn.Module):
            def forward(self, q, k, v):
                attn = torch.ops.aten._scaled_dot_product_efficient_attention.default(
                    q, k, v, None, False
                )
                return attn[0]

        inputs = [
            torch.rand(8, 4, 5, 4).cuda(),
            torch.rand(8, 4, 2, 4).cuda(),
            torch.rand(8, 4, 2, 4).cuda(),
        ]

        fx_graph = torch.fx.symbolic_trace(EfficientAttention())
        expected_ops = {torch.nn.functional.scaled_dot_product_attention}
        unexpected_ops = {
            torch.ops.aten._scaled_dot_product_efficient_attention.default
        }

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
        optimized_model_results = torch.cat(
            [tensor.detach().cpu() for tensor in optimized_model(*inputs)]
        )
        torch_model_results = torch.cat(
            [tensor.detach().cpu() for tensor in fx_graph(*inputs)]
        )

        max_diff = float(
            torch.max(torch.abs(optimized_model_results - torch_model_results))
        )
        self.assertAlmostEqual(
            max_diff,
            0,
            DECIMALS_OF_AGREEMENT,
            msg=f"EfficientAttention TRT outputs don't match with the original model.",
        )
        torch._dynamo.reset()

    def test_efficient_attention_converter(self):
        class EfficientAttention(torch.nn.Module):
            def forward(self, q, k, v):
                attn = torch.ops.aten._scaled_dot_product_efficient_attention.default(
                    q, k, v, None, False
                )
                return attn[0]

        inputs = [
            torch.rand(1, 3, 6, 4).cuda(),
            torch.rand(1, 3, 2, 4).cuda(),
            torch.rand(1, 3, 2, 4).cuda(),
        ]

        fx_graph = torch.fx.symbolic_trace(EfficientAttention())

        # Validate that the results between Torch and Torch-TRT are similar
        optimized_model = torch_tensorrt.compile(
            fx_graph,
            "torch_compile",
            inputs,
            min_block_size=1,
            pass_through_build_failures=True,
        )
        optimized_model_results = torch.cat(
            [tensor.detach().cpu() for tensor in optimized_model(*inputs)]
        )
        torch_model_results = torch.cat(
            [tensor.detach().cpu() for tensor in fx_graph(*inputs)]
        )

        max_diff = float(
            torch.max(torch.abs(optimized_model_results - torch_model_results))
        )
        self.assertAlmostEqual(
            max_diff,
            0,
            DECIMALS_OF_AGREEMENT,
            msg=f"EfficientAttention TRT outputs don't match with the original model.",
        )
        torch._dynamo.reset()


@unittest.skipIf(
    torch.cuda.get_device_properties(torch.cuda.current_device()).major < 8,
    "GPU compute capability is too low to run flash attention, need Ampere (8.0) or greater",
)
@unittest.skipIf(
    sys.platform.startswith("win"),
    "Test not supported on Windows",
)
class TestLowerFlashAttention(TestCase):
    @unittest.skipIf(
        not PLATFORM_SUPPORTS_FLASH_ATTENTION or not isSM8XDevice,
        "Does not support fused SDPA or not SM86+ hardware",
    )
    def test_lower_flash_attention(self):
        class FlashAttention(torch.nn.Module):
            def forward(self, q, k, v):
                attn = torch.ops.aten._scaled_dot_product_flash_attention.default(
                    q,
                    k,
                    v,
                    scale=0.15,
                )
                return attn[0]

        inputs = [
            torch.rand(8, 4, 16, 8).half().cuda(),
            torch.rand(8, 4, 16, 8).half().cuda(),
            torch.rand(8, 4, 16, 8).half().cuda(),
        ]

        fx_graph = torch.fx.symbolic_trace(FlashAttention())
        expected_ops = {torch.nn.functional.scaled_dot_product_attention}
        unexpected_ops = {torch.ops.aten._scaled_dot_product_flash_attention.default}

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
        optimized_model_results = torch.cat(
            [tensor.detach().cpu() for tensor in optimized_model(*inputs)]
        )
        torch_model_results = torch.cat(
            [tensor.detach().cpu() for tensor in fx_graph(*inputs)]
        )

        max_diff = float(
            torch.max(torch.abs(optimized_model_results - torch_model_results))
        )
        # Remove 1 decimal from the requirement for FP16
        self.assertAlmostEqual(
            max_diff,
            0,
            DECIMALS_OF_AGREEMENT - 1,
            msg=f"FlashAttention TRT outputs don't match with the original model.",
        )
        torch._dynamo.reset()

    @unittest.skipIf(
        not PLATFORM_SUPPORTS_FLASH_ATTENTION or not isSM8XDevice,
        "Does not support fused SDPA or not SM86+ hardware",
    )
    def test_flash_attention_converter(self):
        class FlashAttention(torch.nn.Module):
            def forward(self, q, k, v):
                attn = torch.ops.aten._scaled_dot_product_flash_attention.default(
                    q,
                    k,
                    v,
                    scale=0.25,
                )
                return attn[0]

        inputs = [
            torch.rand(1, 3, 6, 8).half().cuda(),
            torch.rand(1, 3, 2, 8).half().cuda(),
            torch.rand(1, 3, 2, 8).half().cuda(),
        ]

        fx_graph = torch.fx.symbolic_trace(FlashAttention())

        # Validate that the results between Torch and Torch-TRT are similar
        optimized_model = torch_tensorrt.compile(
            fx_graph,
            "torch_compile",
            inputs,
            min_block_size=1,
            pass_through_build_failures=True,
        )
        optimized_model_results = torch.cat(
            [tensor.detach().cpu() for tensor in optimized_model(*inputs)]
        )
        torch_model_results = torch.cat(
            [tensor.detach().cpu() for tensor in fx_graph(*inputs)]
        )

        max_diff = float(
            torch.max(torch.abs(optimized_model_results - torch_model_results))
        )
        # Remove 1 decimal from the requirement for FP16
        self.assertAlmostEqual(
            max_diff,
            0,
            DECIMALS_OF_AGREEMENT - 1,
            msg=f"FlashAttention TRT outputs don't match with the original model.",
        )
        torch._dynamo.reset()


if __name__ == "__main__":
    run_tests()
