from unittest import mock

import torch
import torch.nn as nn
from torch.testing._internal.common_utils import TestCase, run_tests
from torch_tensorrt._features import ENABLED_FEATURES
from torch_tensorrt.dynamo._settings import CompilationSettings
from torch_tensorrt.dynamo.conversion.aten_ops_converters import (
    scaled_dot_product_attention_validator,
)
from torch_tensorrt.dynamo.lowering.passes.decompose_unsupported_attention import (
    decompose_unsupported_attention,
)

from ..testing_utilities import lower_graph_testing


class TestDecomposeUnsupportedAttention(TestCase):
    def test_turing_fp32_attention_left_intact_for_pytorch_fallback(self):
        class SDPA(nn.Module):
            def forward(self, q, k, v):
                return torch.ops.aten.scaled_dot_product_attention.default(q, k, v)

        # Also cover MLA shapes: shape-based decomposition must not override a
        # hardware rejection, even when K and V have different head dimensions.
        for value_head_dim in (8, 4):
            with (
                self.subTest(value_head_dim=value_head_dim),
                mock.patch(
                    "torch_tensorrt._features.ENABLED_FEATURES",
                    ENABLED_FEATURES._replace(tensorrt_rtx=True),
                ),
            ):
                inputs = (
                    torch.randn(1, 2, 4, 8),
                    torch.randn(1, 2, 4, 8),
                    torch.randn(1, 2, 4, value_head_dim),
                )
                gm = torch.export.export(SDPA(), inputs).module()
                settings = CompilationSettings(
                    decompose_attention=False,
                    target_compute_capabilities=[(7, 5)],
                )
                sdpa = next(
                    n
                    for n in gm.graph.nodes
                    if n.target == torch.ops.aten.scaled_dot_product_attention.default
                )
                self.assertFalse(scaled_dot_product_attention_validator(sdpa, settings))
                original_graph = str(gm.graph)

                gm = decompose_unsupported_attention(gm, settings)

                self.assertEqual(str(gm.graph), original_graph)
                self.assertFalse(scaled_dot_product_attention_validator(sdpa, settings))
                self.assertEqual(gm(*inputs), SDPA()(*inputs))

    def test_mla_kv_head_dim_mismatch_is_decomposed(self):
        # MLA: K head dim = nope + rope, V head dim = v only.
        b, h, s, d_k, d_v = 1, 2, 4, 6, 4

        class MLA(nn.Module):
            def forward(self, q, k, v):
                return torch.ops.aten.scaled_dot_product_attention.default(q, k, v)

        inputs = [
            torch.randn(b, h, s, d_k),
            torch.randn(b, h, s, d_k),
            torch.randn(b, h, s, d_v),
        ]
        ep = torch.export.export(MLA(), tuple(inputs))
        settings = CompilationSettings(min_block_size=1, decompose_attention=False)
        gm = ep.module()

        sdpa = next(
            n
            for n in gm.graph.nodes
            if n.target == torch.ops.aten.scaled_dot_product_attention.default
        )
        self.assertFalse(scaled_dot_product_attention_validator(sdpa, settings))

        gm = decompose_unsupported_attention(gm, settings)

        targets = {n.target for n in gm.graph.nodes if n.op == "call_function"}
        self.assertNotIn(torch.ops.aten.scaled_dot_product_attention.default, targets)

    def test_equal_kv_shapes_left_intact(self):
        class MHA(nn.Module):
            def forward(self, q, k, v):
                return torch.ops.aten.scaled_dot_product_attention.default(q, k, v)

        inputs = [torch.randn(1, 2, 4, 8) for _ in range(3)]
        ep = torch.export.export(MHA(), tuple(inputs))
        settings = CompilationSettings(min_block_size=1, decompose_attention=False)
        gm = ep.module()
        gm = decompose_unsupported_attention(gm, settings)

        targets = {n.target for n in gm.graph.nodes if n.op == "call_function"}
        self.assertIn(torch.ops.aten.scaled_dot_product_attention.default, targets)

    def test_post_lowering_mla_fully_supported(self):
        class MLA(nn.Module):
            def forward(self, q, k, v):
                return torch.nn.functional.scaled_dot_product_attention(q, k, v)

        inputs = [
            torch.randn(1, 2, 4, 6, device="cuda"),
            torch.randn(1, 2, 4, 6, device="cuda"),
            torch.randn(1, 2, 4, 4, device="cuda"),
        ]
        unexpected_ops = {torch.ops.aten.scaled_dot_product_attention.default}
        fx_graph = torch.export.export(MLA(), tuple(inputs)).module()
        unexpected_ops_seen, _ = lower_graph_testing(
            fx_graph,
            inputs,
            unexpected_ops=unexpected_ops,
            min_block_size=1,
            decompose_attention=False,
        )
        self.assertEqual(
            len(unexpected_ops_seen),
            0,
            f"SDPA should have been decomposed for MLA shapes: {unexpected_ops_seen}",
        )


if __name__ == "__main__":
    run_tests()
