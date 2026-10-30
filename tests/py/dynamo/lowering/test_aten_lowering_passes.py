# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import operator
import sys
import tempfile

import torch
import torch_tensorrt
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.testing._internal.common_utils import TestCase, run_tests
from torch_tensorrt.dynamo._settings import CompilationSettings
from torch_tensorrt.dynamo.lowering.passes.force_causal_efficient_attention import (
    force_causal_efficient_attention,
)

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

    class ReturnsInput(torch.nn.Module):
        def __init__(self, copies):
            super().__init__()
            self.copies = copies

        def forward(self, x, y):
            return (x + y,) + (y,) * self.copies

    def _assert_matches_eager(self, mod, **settings):
        inputs = (torch.randn(5, 7).cuda(), torch.randn(5, 7).cuda())
        trt_mod = torch_tensorrt.dynamo.compile(
            torch.export.export(mod, inputs),
            list(inputs),
            min_block_size=1,
            require_full_compilation=True,
            **settings,
        )
        torch.testing.assert_close(trt_mod(*inputs), mod(*inputs))

    def test_input_returned_next_to_a_computed_output(self):
        self._assert_matches_eager(self.ReturnsInput(copies=1))

    def test_input_returned_twice(self):
        self._assert_matches_eager(self.ReturnsInput(copies=2))

    def test_computed_output_returned_twice(self):
        class M(torch.nn.Module):
            def forward(self, x, y):
                z = x + y
                return z, z

        self._assert_matches_eager(M())

    def test_input_returned_twice_from_the_engine_cache(self):
        """An engine loaded from the cache has as many outputs as a fresh build."""
        with tempfile.TemporaryDirectory() as cache_dir:
            for _ in range(2):
                self._assert_matches_eager(
                    self.ReturnsInput(copies=2),
                    immutable_weights=False,
                    cache_built_engines=True,
                    reuse_cached_engines=True,
                    engine_cache_dir=cache_dir,
                )


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


class TestPreDispatchSideEffects(TestCase):
    def test_vmap_survives_assert_removal(self):
        # The dtype casts put _assert_tensor_metadata nodes in the exported graph, and
        # removing them runs dead-code elimination over the whole graph while the vmap's
        # nesting calls (used by nothing, kept for their side effect) are still in it.
        # transformers builds attention masks this way.
        class VmapMask(torch.nn.Module):
            def forward(self, x, y):
                mask = torch.vmap(lambda row: row > 0)(x)
                return (x * mask).to(torch.float16) + y.to(torch.float16)

        model = VmapMask().eval().cuda()
        inputs = (
            torch.randn(4, 8).cuda(),
            torch.randn(4, 8, dtype=torch.float64).cuda(),
        )
        exported = torch.export.export(model, inputs)
        trt_module = torch_tensorrt.dynamo.compile(
            exported,
            arg_inputs=list(inputs),
            min_block_size=1,
            truncate_double=True,
        )
        torch.testing.assert_close(
            trt_module(*inputs), model(*inputs), rtol=1e-3, atol=1e-3
        )


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


class TestPreserveAttentionOutputLayout(TestCase):
    def test_view_after_decomposed_attention(self):
        # Fused attention returns this transposed query's layout, so export drops the
        # .contiguous() below and records a view that only works on that layout. The
        # attention output must stay a tensor of its own (detach_), the in-place
        # update through the view must reach it, and the no_grad region puts all of
        # it in a subgraph.
        class ViewAfterAttention(torch.nn.Module):
            def forward(self, query, key, value):
                with torch.no_grad():
                    query, key, value = (t.transpose(1, 2) for t in (query, key, value))
                    out = torch.nn.functional.scaled_dot_product_attention(
                        query, key, value
                    )
                    out.detach_()
                    flat = out.transpose(1, 2).contiguous()
                    flat = flat.view(flat.shape[0], flat.shape[1], -1)
                    flat.add_(1)
                return out, flat

        inputs = [torch.randn(2, 16, 4, 32).half().cuda() for _ in range(3)]
        model = ViewAfterAttention().cuda()
        ep = torch.export.export(model, tuple(inputs))
        # With a contiguous attention output, export keeps the .contiguous() and
        # this test passes without the pass.
        (attention,) = (
            node
            for module in ep.graph_module.modules()
            if isinstance(module, torch.fx.GraphModule)
            for node in module.graph.nodes
            if node.target == torch.ops.aten.scaled_dot_product_attention.default
        )
        self.assertFalse(attention.meta["val"].is_contiguous())
        trt_module = torch_tensorrt.dynamo.compile(
            ep, inputs, min_block_size=1, decompose_attention=True
        )
        torch.testing.assert_close(
            model(*inputs), trt_module(*inputs), rtol=1e-2, atol=1e-2
        )

    def test_view_after_attention_in_cond_branch(self):
        class CondViewAfterAttention(torch.nn.Module):
            def forward(self, pred, query, key, value):
                def attend(query, key, value):
                    query, key, value = (t.transpose(1, 2) for t in (query, key, value))
                    out = torch.nn.functional.scaled_dot_product_attention(
                        query, key, value
                    )
                    out = out.transpose(1, 2).contiguous()
                    return out.view(out.shape[0], out.shape[1], -1)

                def flatten(query, key, value):
                    return query.flatten(2).clone()

                return torch.cond(pred, attend, flatten, (query, key, value))

        inputs = [torch.randn(2, 16, 4, 32).half().cuda() for _ in range(3)]
        model = CondViewAfterAttention().cuda()
        pred = torch.tensor(True).cuda()
        ep = torch.export.export(model, (pred, *inputs))
        trt_module = torch_tensorrt.dynamo.compile(
            ep, [pred, *inputs], min_block_size=1, decompose_attention=True
        )
        for pred in (True, False):
            pred = torch.tensor(pred).cuda()
            torch.testing.assert_close(
                model(pred, *inputs), trt_module(pred, *inputs), rtol=1e-2, atol=1e-2
            )

    def test_attention_on_data_dependent_rows(self):
        # Boolean indexing gives attention a batch size known only at runtime.
        class MaskedAttention(torch.nn.Module):
            def forward(self, x, mask):
                query = x[mask].transpose(1, 2)
                return torch.nn.functional.scaled_dot_product_attention(
                    query, query, query
                )

        model = MaskedAttention().cuda()
        x = torch.randn(3, 16, 4, 32).half().cuda()
        mask = torch.tensor([True, False, True]).cuda()
        ep = torch.export.export(model, (x, mask))
        trt_module = torch_tensorrt.dynamo.compile(
            ep, [x, mask], min_block_size=1, decompose_attention=True
        )
        for mask in ([True, False, False], [True, True, True]):
            mask = torch.tensor(mask).cuda()
            torch.testing.assert_close(
                model(x, mask), trt_module(x, mask), rtol=1e-2, atol=1e-2
            )


class TestRewriteEfficientAttention(TestCase):
    def test_force_causal_efficient_attention(self):
        class CausalEfficientAttention(torch.nn.Module):
            def forward(self, query, key, value):
                position = torch.arange(query.size(-2), device=query.device)
                attn_bias = torch.zeros(
                    query.size(-2), query.size(-2), device=query.device
                ).masked_fill(position[:, None] < position[None], float("-inf"))
                out = torch.ops.aten._scaled_dot_product_efficient_attention.default(
                    query, key, value, attn_bias.expand(4, 8, -1, -1), False
                )
                return out[0]

        inputs = [torch.randn(4, 8, 32, 16).cuda() for _ in range(3)]
        model = CausalEfficientAttention().cuda()
        ep = torch.export.export(model, tuple(inputs))
        # torch.compile runs the lowering passes under a fake mode.
        with FakeTensorMode():
            gm = force_causal_efficient_attention(
                ep.module(), CompilationSettings(attn_bias_is_causal=True)
            )
        (attention,) = [
            node
            for node in gm.graph.nodes
            if node.target
            == torch.ops.aten._scaled_dot_product_efficient_attention.default
        ]
        self.assertIsNone(attention.args[3])
        self.assertTrue(attention.args[6])

        trt_module = torch_tensorrt.dynamo.compile(
            ep,
            inputs,
            min_block_size=1,
            decompose_attention=False,
            attn_bias_is_causal=True,
        )
        torch.testing.assert_close(
            model(*inputs), trt_module(*inputs), rtol=1e-2, atol=1e-2
        )

    def test_force_causal_efficient_attention_keeps_block_causal_mask(self):
        # Complex RoPE makes the complex lowering retrace scaled_dot_product_attention
        # into efficient attention, with the block-causal mask as its attn_bias.
        seq, block = 98, 49

        class RopeBlockCausalAttention(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer(
                    "freqs", torch.polar(torch.ones(seq, 8), torch.randn(seq, 8))
                )

            def forward(self, x):
                q = torch.view_as_complex(x.reshape(seq, 8, 2)) * self.freqs
                q = torch.view_as_real(q).reshape(1, 1, seq, 16)
                frame = torch.arange(seq, device=x.device) // block
                return torch.nn.functional.scaled_dot_product_attention(
                    q, q, q, attn_mask=frame[:, None] >= frame[None]
                )

        x = torch.randn(seq, 16).cuda()
        model = RopeBlockCausalAttention().cuda().eval()
        trt_module = torch_tensorrt.dynamo.compile(
            torch.export.export(model, (x,)),
            [x],
            min_block_size=1,
            disable_tf32=True,
            attn_bias_is_causal=True,
        )
        torch.testing.assert_close(model(x), trt_module(x), rtol=1e-3, atol=1e-3)

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


if __name__ == "__main__":
    run_tests()
