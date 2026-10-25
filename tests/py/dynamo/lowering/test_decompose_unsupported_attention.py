import unittest
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

    @unittest.skipIf(not torch.cuda.is_available(), "builds a TensorRT engine")
    def test_decomposed_attention_runs_in_tensorrt(self):
        # The per-node fallback once left aten.transpose.int and aten._unsafe_view to
        # PyTorch, splitting the model across engines.
        import torch_tensorrt

        class MLA(nn.Module):
            def forward(self, q, k, v):
                return torch.nn.functional.scaled_dot_product_attention(q, k, v)

        inputs = (
            torch.randn(1, 2, 16, 64, device="cuda", dtype=torch.float16),
            torch.randn(1, 2, 16, 64, device="cuda", dtype=torch.float16),
            torch.randn(1, 2, 16, 32, device="cuda", dtype=torch.float16),
        )
        trt_model = torch_tensorrt.dynamo.compile(
            torch.export.export(MLA(), inputs),
            arg_inputs=list(inputs),
            min_block_size=1,
            decompose_attention=False,
            cache_built_engines=False,
            reuse_cached_engines=False,
        )
        torch.testing.assert_close(
            trt_model(*inputs), MLA()(*inputs), rtol=1e-2, atol=1e-2
        )
        self.assertEqual(
            [name for name, _ in trt_model.named_children()], ["_run_on_acc_0"]
        )

    def test_attentions_sharing_a_bf16_query_are_decomposed(self):
        # Wan2.1's image-to-video cross-attention reads one query against text and image keys.
        class TwoAttentions(nn.Module):
            def forward(self, q, text, image):
                q = q.transpose(1, 2)
                return torch.nn.functional.scaled_dot_product_attention(
                    q, text, text
                ) + torch.nn.functional.scaled_dot_product_attention(q, image, image)

        sdpa = torch.ops.aten.scaled_dot_product_attention.default
        settings = CompilationSettings(min_block_size=1, decompose_attention=False)
        for dtype, remaining in ((torch.bfloat16, 0), (torch.float16, 2)):
            with self.subTest(dtype=dtype):
                inputs = (
                    torch.randn(1, 6, 2, 8, dtype=dtype),
                    torch.randn(1, 2, 5, 8, dtype=dtype),
                    torch.randn(1, 2, 3, 8, dtype=dtype),
                )
                gm = torch.export.export(TwoAttentions(), inputs).module()
                gm = decompose_unsupported_attention(gm, settings)
                self.assertEqual(
                    sum(n.target == sdpa for n in gm.graph.nodes), remaining
                )
                if dtype == torch.bfloat16:
                    torch.testing.assert_close(
                        gm(*inputs), TwoAttentions()(*inputs), rtol=1e-2, atol=1e-2
                    )

    def test_attentions_reading_one_tensor_in_different_dtypes_are_kept(self):
        # Only one of the two queries is bfloat16, so they cannot become one tensor in TensorRT.
        class TwoDtypes(nn.Module):
            def forward(self, q, k16, kbf16):
                cast = torch.ops.aten._to_copy.default
                a = torch.nn.functional.scaled_dot_product_attention(
                    cast(q, dtype=torch.float16), k16, k16
                )
                b = torch.nn.functional.scaled_dot_product_attention(
                    cast(q, dtype=torch.bfloat16), kbf16, kbf16
                )
                return a.float() + b.float()

        inputs = (
            torch.randn(1, 2, 6, 8),
            torch.randn(1, 2, 5, 8, dtype=torch.float16),
            torch.randn(1, 2, 5, 8, dtype=torch.bfloat16),
        )
        gm = torch.export.export(TwoDtypes(), inputs).module()
        gm = decompose_unsupported_attention(
            gm, CompilationSettings(min_block_size=1, decompose_attention=False)
        )
        sdpa = torch.ops.aten.scaled_dot_product_attention.default
        self.assertEqual(sum(n.target == sdpa for n in gm.graph.nodes), 2)

    @unittest.skipIf(not torch.cuda.is_available(), "builds a TensorRT engine")
    def test_attentions_sharing_a_query_build(self):
        # TensorRT crashed or failed building these before the attentions were decomposed.
        import torch_tensorrt

        class TwoAttentions(nn.Module):
            def __init__(self, combine):
                super().__init__()
                self.combine = combine

            def forward(self, q, text, image):
                return self.combine(
                    torch.nn.functional.scaled_dot_product_attention(q, text, text),
                    torch.nn.functional.scaled_dot_product_attention(q, image, image),
                )

        inputs = tuple(
            torch.randn(1, 8, n, 128, device="cuda", dtype=torch.bfloat16)
            for n in (256, 512, 257)
        )
        for combine in (torch.add, lambda a, b: torch.cat([a, b], dim=-1)):
            with self.subTest(combine=combine):
                model = TwoAttentions(combine)
                trt_model = torch_tensorrt.dynamo.compile(
                    torch.export.export(model, inputs),
                    arg_inputs=list(inputs),
                    min_block_size=1,
                    decompose_attention=False,
                    cache_built_engines=False,
                    reuse_cached_engines=False,
                )
                torch.testing.assert_close(
                    trt_model(*inputs), model(*inputs), rtol=1e-2, atol=1e-2
                )
                # The decomposed attentions run in TensorRT, not in PyTorch between engines.
                self.assertEqual(
                    [name for name, _ in trt_model.named_children()], ["_run_on_acc_0"]
                )


if __name__ == "__main__":
    run_tests()
