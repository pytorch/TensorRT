# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import operator
import sys

import torch
import torch_tensorrt
from torch.testing._internal.common_utils import TestCase, run_tests

from ..testing_utilities import DECIMALS_OF_AGREEMENT, lower_graph_testing


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


class TestComplexSubgraph(TestCase):
    def test_complex_subgraph(self):
        BATCH = 1
        SEQ_LEN = 2
        HEADS = 1
        DIM = 2

        class RotaryAttention(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.dim = DIM
                self.wq = torch.nn.Linear(self.dim, self.dim)
                self.seq_len = SEQ_LEN

                self.register_buffer(
                    "freqs_ex_tensor",
                    self._freqs_ex_tensor(),
                    persistent=True,
                )

            def rotary_embedding(self, x, dim, freqs_cis=None):
                x_ = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
                x_out_flatten = torch.view_as_real(x_ * freqs_cis)
                return x_out_flatten.type_as(x)

            def _freqs_ex_tensor(self):
                real = torch.tensor([[[[1.0000]], [[2.0000]]]], device="cuda")
                imag = torch.tensor([[[[0.0000]], [[3.0000]]]], device="cuda")

                z = torch.complex(real, imag)
                return z

            def forward(self, x):
                q = self.wq(x)
                freqs_cis = self._freqs_ex_tensor().to(q.device)
                q_out = self.rotary_embedding(q, self.dim, freqs_cis=freqs_cis)
                return q_out

        inputs = [torch.randn(BATCH, SEQ_LEN, HEADS, DIM).cuda()]
        model = RotaryAttention()
        model = model.cuda()

        expected_ops = {torch.ops.aten.mul.Tensor}
        unexpected_ops = {
            torch.ops.aten.view_as_complex.default,
            torch.ops.aten.view_as_real.default,
        }

        unexpected_ops_seen, expected_ops_unseen = lower_graph_testing(
            model,
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
            model,
            "torch_compile",
            inputs,
            min_block_size=1,
            pass_through_build_failures=True,
        )
        optimized_model_results = optimized_model(*inputs)[0].detach().cpu()
        torch_model_results = model(*inputs)[0].detach().cpu()

        max_diff = float(
            torch.max(torch.abs(optimized_model_results - torch_model_results))
        )
        self.assertAlmostEqual(
            max_diff,
            0,
            DECIMALS_OF_AGREEMENT,
            msg=f"ComplexSubgraph TRT outputs don't match with the original model.",
        )
        torch._dynamo.reset()


class TestRemoveSymIntNodes(TestCase):
    def test_remove_sym_nodes(self):
        class ModelContainSymIntNodes(torch.nn.Module):
            def __init__(self, embed_dim: int):
                super().__init__()
                self.cls_token = torch.nn.Parameter(torch.randn(1, 1, embed_dim))
                self.embed_dim = embed_dim
                self.qkv_proj = torch.nn.Linear(self.embed_dim, self.embed_dim * 3)

            def forward(self, x: torch.Tensor):
                batch_size = x.shape[0]
                cls_token = self.cls_token.expand(batch_size, -1, -1)
                x = torch.cat([cls_token, x], dim=1)
                x = self.qkv_proj(x)
                reshaped_qkv = x.reshape(batch_size, x.size(1), 3, 12, -1)
                return reshaped_qkv

        model = ModelContainSymIntNodes(embed_dim=768).cuda().eval()
        inputs = torch.randn(4, 196, 768).cuda()
        torch._dynamo.mark_dynamic(inputs, index=0, min=2, max=32)
        trt_module = torch.compile(
            model,
            backend="tensorrt",
            options={"min_block_size": 1},
        )
        out = trt_module(inputs)
        # if the model can be successfully compiled, we regard the test as passed
        self.assertTrue(True)


class TestNormalizeNegativeSliceStop(TestCase):
    def test_normalizes_negative_symbolic_start_bound(self):
        from torch_tensorrt.dynamo.lowering.passes.normalize_negative_slice_stop import (
            normalize_negative_slice_stop,
        )

        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        x.meta["val"] = torch.empty(2, 5, 3)
        n = graph.placeholder("n")
        neg = graph.call_function(operator.neg, args=(n,))
        sliced = graph.call_function(torch.ops.aten.slice.Tensor, args=(x, -2, neg))
        graph.output(sliced)

        gm = torch.fx.GraphModule({}, graph)
        gm = normalize_negative_slice_stop(gm)

        slice_node = next(
            node
            for node in gm.graph.nodes
            if node.op == "call_function" and node.target == torch.ops.aten.slice.Tensor
        )
        self.assertEqual(slice_node.args[1], 1)

        normalized_start = slice_node.args[2]
        self.assertEqual(normalized_start.op, "call_function")
        self.assertEqual(normalized_start.target, operator.sub)

        dim_size, offset = normalized_start.args
        self.assertEqual(dim_size.target, torch.ops.aten.sym_size.int)
        self.assertEqual(dim_size.args[0], x)
        self.assertEqual(dim_size.args[1], 1)
        self.assertEqual(offset, n)

    def test_normalizes_negative_symbolic_stop_bound(self):
        from torch_tensorrt.dynamo.lowering.passes.normalize_negative_slice_stop import (
            normalize_negative_slice_stop,
        )

        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        x.meta["val"] = torch.empty(2, 5, 3)
        n = graph.placeholder("n")
        neg = graph.call_function(torch.ops.aten.neg.default, args=(n,))
        sliced = graph.call_function(torch.ops.aten.slice.Tensor, args=(x, 1, 0, neg))
        graph.output(sliced)

        gm = torch.fx.GraphModule({}, graph)
        gm = normalize_negative_slice_stop(gm)

        slice_node = next(
            node
            for node in gm.graph.nodes
            if node.op == "call_function" and node.target == torch.ops.aten.slice.Tensor
        )

        normalized_stop = slice_node.args[3]
        self.assertEqual(normalized_stop.op, "call_function")
        self.assertEqual(normalized_stop.target, operator.sub)

        dim_size, offset = normalized_stop.args
        self.assertEqual(dim_size.target, torch.ops.aten.sym_size.int)
        self.assertEqual(dim_size.args[0], x)
        self.assertEqual(dim_size.args[1], 1)
        self.assertEqual(offset, n)


class TestEliminateSymMinInt64Max(TestCase):
    def test_eliminates_noop_sym_min_int64_max(self):
        if not hasattr(torch, "sym_min"):
            self.skipTest("torch.sym_min is not available")

        from torch_tensorrt.dynamo.lowering.passes.eliminate_sym_min_int64_max import (
            eliminate_sym_min_int64_max,
        )

        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        rhs_int64_max = graph.call_function(torch.sym_min, args=(x, sys.maxsize))
        lhs_int64_max = graph.call_function(torch.sym_min, args=(2**63 - 1, x))
        graph.output((rhs_int64_max, lhs_int64_max))

        gm = torch.fx.GraphModule({}, graph)
        gm = eliminate_sym_min_int64_max(gm)

        self.assertFalse(
            any(
                node.op == "call_function" and node.target is torch.sym_min
                for node in gm.graph.nodes
            )
        )

        output_node = next(node for node in gm.graph.nodes if node.op == "output")
        self.assertEqual(output_node.args[0], (x, x))


class TestRewriteEfficientAttention(TestCase):
    def test_force_causal_efficient_attention(self):
        class RewriteEfficientAttention(torch.nn.Module):
            def forward(
                self,
                query,
                key,
                value,
                attn_bias=None,
                compute_log_sumexp=False,
                dropout_p=0.0,
                is_causal=False,
                scale=None,
            ):
                out = torch.ops.aten._scaled_dot_product_efficient_attention.default(
                    query,
                    key,
                    value,
                    attn_bias,
                    compute_log_sumexp,
                    dropout_p,
                    is_causal,
                    scale=scale,
                )
                return out[0]

        attn_bias = torch.zeros(4, 8, 32, 32, device="cuda")
        upper = torch.triu(
            torch.ones((32, 32), dtype=torch.bool, device="cuda"), diagonal=1
        )
        attn_bias = attn_bias.masked_fill(upper, float("-inf"))

        inputs = [
            torch.randn(4, 8, 32, 16).cuda(),
            torch.randn(4, 8, 32, 16).cuda(),
            torch.randn(4, 8, 32, 16).cuda(),
            attn_bias,
            True,
            0.0,
            True,
        ]
        model = RewriteEfficientAttention().cuda()
        pytorch_out = model(*inputs)
        ep = torch.export.export(model, tuple(inputs))
        trt_module = torch_tensorrt.dynamo.compile(
            ep,
            inputs,
            min_block_size=1,
            decompose_attention=False,
            attn_bias_is_causal=True,
        )
        trt_out = trt_module(*inputs)
        torch.testing.assert_close(pytorch_out, trt_out, rtol=1e-2, atol=1e-2)

    def test_force_causal_efficient_attention_with_non_causal_attn_bias(self):
        class RewriteEfficientAttention(torch.nn.Module):
            def forward(
                self,
                query,
                key,
                value,
                attn_bias=None,
                compute_log_sumexp=False,
                dropout_p=0.0,
                is_causal=False,
                scale=None,
            ):
                out = torch.ops.aten._scaled_dot_product_efficient_attention.default(
                    query,
                    key,
                    value,
                    attn_bias,
                    compute_log_sumexp,
                    dropout_p,
                    is_causal,
                    scale=scale,
                )
                return out[0]

        attn_bias = torch.randn(4, 8, 32, 32).cuda()

        inputs = [
            torch.randn(4, 8, 32, 16).cuda(),
            torch.randn(4, 8, 32, 16).cuda(),
            torch.randn(4, 8, 32, 16).cuda(),
            attn_bias,
            True,
            0.0,
            False,
        ]
        model = RewriteEfficientAttention().cuda()
        pytorch_out = model(*inputs)
        ep = torch.export.export(model, tuple(inputs))
        trt_module = torch_tensorrt.dynamo.compile(
            ep,
            inputs,
            min_block_size=1,
            decompose_attention=False,
            attn_bias_is_causal=False,
        )
        trt_out = trt_module(*inputs)
        torch.testing.assert_close(pytorch_out, trt_out, rtol=1e-2, atol=1e-2)


class MaskedAttention(torch.nn.Module):
    """Self-attention whose mask ``mask_fn(positions)`` is built in the graph."""

    def __init__(self, mask_fn):
        super().__init__()
        self.mask_fn = mask_fn

    def forward(self, q, k, v):
        pos = torch.arange(q.shape[-2], device=q.device)
        return torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=self.mask_fn(pos)
        )


def _causal(p):
    return p[None, :] <= p[:, None]


def _sliding_window(window):
    return lambda p: _causal(p) & (p[None, :] > p[:, None] - window)


class TestForceCausalChecksTheMask(TestCase):
    """``force_causal_efficient_attention`` replaces a mask with is_causal=True
    only if the mask is causal: one it can compute at compile time is checked,
    and one that depends on runtime inputs is assumed causal, with a warning."""

    def _lower(self, model, args, dynamic_shapes=None, **settings):
        from torch_tensorrt.dynamo._settings import CompilationSettings
        from torch_tensorrt.dynamo.lowering.passes.force_causal_efficient_attention import (
            force_causal_efficient_attention,
        )

        # The default decompositions turn SDPA with a mask into efficient attention.
        ep = torch.export.export(
            model, args, dynamic_shapes=dynamic_shapes
        ).run_decompositions()
        gm = force_causal_efficient_attention(
            ep.module(), CompilationSettings(**settings)
        )
        (node,) = [
            n
            for n in gm.graph.nodes
            if n.target
            == torch.ops.aten._scaled_dot_product_efficient_attention.default
        ]
        return node.args[3] is None and node.args[6] is True

    @staticmethod
    def _qkv(seq_len=16):
        return tuple(torch.randn(1, 4, seq_len, 16, device="cuda") for _ in range(3))

    def _assert_no_warning(self, fn):
        logger = (
            "torch_tensorrt.dynamo.lowering.passes.force_causal_efficient_attention"
        )
        with self.assertNoLogs(logger, level="WARNING"):
            return fn()

    def test_causal_mask_is_replaced(self):
        forced = self._assert_no_warning(
            lambda: self._lower(MaskedAttention(_causal), self._qkv())
        )
        self.assertTrue(forced)

    def test_sliding_window_is_kept(self):
        self.assertFalse(self._lower(MaskedAttention(_sliding_window(4)), self._qkv()))

    def test_prefix_lm_mask_is_kept(self):
        prefix = lambda p: _causal(p) | (p[None, :] < 4)  # noqa: E731
        self.assertFalse(self._lower(MaskedAttention(prefix), self._qkv()))

    def test_additive_bias_is_kept(self):
        alibi = lambda p: torch.where(  # noqa: E731
            _causal(p), -0.5 * (p[:, None] - p[None, :]).float(), float("-inf")
        )
        self.assertFalse(self._lower(MaskedAttention(alibi), self._qkv()))

    def test_without_the_setting_nothing_is_replaced(self):
        self.assertFalse(
            self._lower(
                MaskedAttention(_causal), self._qkv(), attn_bias_is_causal=False
            )
        )

    def test_mask_from_an_input_is_assumed_causal_with_a_warning(self):
        class M(torch.nn.Module):
            def forward(self, q, k, v, mask):
                return torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, attn_mask=mask
                )

        mask = torch.ones(16, 16, dtype=torch.bool, device="cuda").tril()
        logger = (
            "torch_tensorrt.dynamo.lowering.passes.force_causal_efficient_attention"
        )
        with self.assertLogs(logger, level="WARNING") as logs:
            forced = self._lower(M(), self._qkv() + (mask,))
        self.assertTrue(forced)
        self.assertIn("attn_bias_is_causal=False", "\n".join(logs.output))

    def test_window_with_an_input_padding_mask_is_kept(self):
        """The input-dependent part is unknown, but the window alone already
        masks out causal positions."""

        class M(torch.nn.Module):
            def forward(self, q, k, v, keep):
                pos = torch.arange(q.shape[-2], device=q.device)
                mask = _sliding_window(4)(pos) & keep[None, :]
                return torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, attn_mask=mask
                )

        keep = torch.ones(16, dtype=torch.bool, device="cuda")
        self.assertFalse(self._lower(M(), self._qkv() + (keep,)))

    def test_window_wider_than_the_example_is_checked_at_the_longest_length(self):
        """At the example length (4) the window of 8 looks causal; at the
        longest length compiled for (32) it is not."""
        t = torch.export.Dim("t", min=2, max=32)
        self.assertFalse(
            self._lower(
                MaskedAttention(_sliding_window(8)),
                self._qkv(4),
                dynamic_shapes=({2: t}, {2: t}, {2: t}),
            )
        )

    def test_causal_mask_with_a_dynamic_length_is_replaced(self):
        t = torch.export.Dim("t", min=2, max=32)
        forced = self._assert_no_warning(
            lambda: self._lower(
                MaskedAttention(_causal),
                self._qkv(4),
                dynamic_shapes=({2: t}, {2: t}, {2: t}),
            )
        )
        self.assertTrue(forced)

    def test_sliding_window_compiles_to_the_eager_result(self):
        """End to end with the default settings (the bug: the window was dropped)."""
        model = MaskedAttention(_sliding_window(4)).cuda()
        args = self._qkv()
        ep = torch.export.export(model, args).run_decompositions()
        trt_module = torch_tensorrt.dynamo.compile(
            ep, list(args), min_block_size=1, decompose_attention=False
        )
        torch.testing.assert_close(
            trt_module(*args), model(*args), rtol=1e-2, atol=1e-2
        )


if __name__ == "__main__":
    run_tests()
