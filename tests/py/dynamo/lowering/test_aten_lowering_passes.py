# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import math
import operator
import random
import sys
import tempfile
from unittest import mock

import torch
import torch_tensorrt
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.testing._internal.common_utils import TestCase, run_tests
from torch_tensorrt.dynamo._settings import CompilationSettings
from torch_tensorrt.dynamo.lowering.passes.force_causal_efficient_attention import (
    _check_causal,
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
        # torch.compile runs the lowering passes under a fake mode. "proven" replaces
        # only a bias it proves causal, so the bias must be computed with real values.
        with FakeTensorMode():
            gm = force_causal_efficient_attention(
                ep.module(), CompilationSettings(attn_bias_is_causal="proven")
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
            ep, inputs, min_block_size=1, decompose_attention=False
        )
        torch.testing.assert_close(
            model(*inputs), trt_module(*inputs), rtol=1e-2, atol=1e-2
        )

    def test_force_causal_efficient_attention_replaces_by_setting(
        self,
    ):
        upper = torch.ones(32, 32, dtype=torch.bool, device="cuda").triu(1)
        causal = torch.zeros(4, 8, 32, 32, device="cuda").masked_fill(
            upper, float("-inf")
        )

        class InputBias(torch.nn.Module):
            def forward(self, query, attn_bias):
                out = torch.ops.aten._scaled_dot_product_efficient_attention.default(
                    query, query, query, attn_bias, False
                )
                return out[0]

        class BufferBias(InputBias):
            def __init__(self, bias):
                super().__init__()
                self.register_buffer("bias", bias)

            def forward(self, query):
                return super().forward(query, self.bias)

        query = torch.randn(4, 8, 32, 16, device="cuda")
        seq = torch.export.Dim("seq", min=2, max=64)
        dynamic_seq = ({2: seq}, {2: seq, 3: seq})
        # (model, inputs, dynamic shapes, settings, whether the bias is replaced)
        cases = [
            (InputBias(), (query, causal), None, CompilationSettings(), True),
            (
                InputBias(),
                (query, causal),
                None,
                CompilationSettings(attn_bias_is_causal="proven"),
                False,
            ),
            (
                InputBias(),
                (query, causal),
                None,
                CompilationSettings(attn_bias_is_causal=True),
                True,
            ),
            (
                InputBias(),
                (query, causal),
                dynamic_seq,
                CompilationSettings(attn_bias_is_causal=True),
                True,
            ),
            (BufferBias(causal), (query,), None, CompilationSettings(), True),
            (
                BufferBias(causal),
                (query,),
                None,
                CompilationSettings(attn_bias_is_causal=False),
                False,
            ),
            # A sliding window: blocks every position above the diagonal, and some below.
            (
                BufferBias(causal.masked_fill(upper.mT.tril(-4), float("-inf"))),
                (query,),
                None,
                CompilationSettings(attn_bias_is_causal=True),
                False,
            ),
            # Blocked only to -1e9: unknown, so kept with "proven", replaced with True.
            (
                BufferBias(causal.nan_to_num(neginf=-1e9)),
                (query,),
                None,
                CompilationSettings(attn_bias_is_causal="proven"),
                False,
            ),
            (
                BufferBias(causal.nan_to_num(neginf=-1e9)),
                (query,),
                None,
                CompilationSettings(attn_bias_is_causal=True),
                True,
            ),
            # Zeros on the lower triangle, but positions above it are not blocked.
            (
                BufferBias(causal.nan_to_num(neginf=5.0)),
                (query,),
                None,
                CompilationSettings(attn_bias_is_causal=True),
                False,
            ),
        ]
        for model, inputs, dynamic_shapes, settings, replaced in cases:
            with self.subTest(
                type(model).__name__,
                dynamic=dynamic_shapes is not None,
                attn_bias_is_causal=settings.attn_bias_is_causal,
            ):
                ep = torch.export.export(model, inputs, dynamic_shapes=dynamic_shapes)
                gm = force_causal_efficient_attention(ep.module(), settings)
                (attention,) = [
                    node
                    for node in gm.graph.nodes
                    if node.target
                    == torch.ops.aten._scaled_dot_product_efficient_attention.default
                ]
                self.assertEqual(attention.args[3] is None, replaced)

        # A check that fails, for example by running out of memory, and a mask too long
        # to check are not proven: True replaces them, as before, and "proven" keeps
        # them.
        long_seq = 4097
        long_causal = torch.zeros(1, 1, long_seq, long_seq, device="cuda").masked_fill(
            torch.ones(long_seq, long_seq, dtype=torch.bool, device="cuda").triu(1),
            float("-inf"),
        )
        unproven = {
            "failed check": (BufferBias(causal), query),
            "long mask": (
                BufferBias(long_causal),
                torch.randn(1, 1, long_seq, 16, device="cuda"),
            ),
        }
        for name, (model, inputs) in unproven.items():
            for setting, replaced in ((True, True), ("proven", False)):
                with (
                    self.subTest(name, attn_bias_is_causal=setting),
                    mock.patch(
                        "torch_tensorrt.dynamo.lowering.passes.force_causal_efficient_attention._check_causal",
                        side_effect=torch.OutOfMemoryError,
                    ) as check,
                ):
                    gm = force_causal_efficient_attention(
                        torch.export.export(model, (inputs,)).module(),
                        CompilationSettings(attn_bias_is_causal=setting),
                    )
                    (attention,) = [
                        node
                        for node in gm.graph.nodes
                        if node.target
                        == torch.ops.aten._scaled_dot_product_efficient_attention.default
                    ]
                    self.assertEqual(attention.args[3] is None, replaced)
                    self.assertEqual(check.called, name == "failed check")

    def test_attn_bias_is_causal_rejects_other_values(self):
        # None means the default for other settings, and a misspelled mode is truthy.
        for value in (None, "Proven"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                CompilationSettings(attn_bias_is_causal=value)

    def test_causal_check_memory(self):
        seq = 4096
        bias = torch.zeros(seq, seq, dtype=torch.float16, device="cuda").masked_fill(
            torch.ones(seq, seq, dtype=torch.bool, device="cuda").triu(1),
            float("-inf"),
        )
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        before = torch.cuda.memory_allocated()
        self.assertTrue(_check_causal(bias, seq))
        # Indexing out the blocked positions needed several times the bias.
        self.assertLess(torch.cuda.max_memory_allocated() - before, 4 * bias.nbytes)

    def test_force_causal_efficient_attention_keeps_a_bias_with_unused_log_sumexp(
        self,
    ):
        # The converter refuses compute_log_sumexp, so this compiles only if the
        # pass clears the flag that nothing reads.
        class LogSumExp(torch.nn.Module):
            def forward(self, query, attn_bias):
                out = torch.ops.aten._scaled_dot_product_efficient_attention.default(
                    query, query, query, attn_bias, True
                )
                return out[0]

        query = torch.randn(4, 8, 32, 16, device="cuda")
        attn_bias = torch.randn(4, 8, 32, 32, device="cuda")
        model = LogSumExp()
        trt_module = torch_tensorrt.dynamo.compile(
            torch.export.export(model, (query, attn_bias)),
            [query, attn_bias],
            min_block_size=1,
            require_full_compilation=True,
            attn_bias_is_causal="proven",
        )
        torch.testing.assert_close(
            model(query, attn_bias), trt_module(query, attn_bias), rtol=1e-2, atol=1e-2
        )

        # A log_sumexp that is read, or a dropout, keeps the flag and the node in PyTorch.
        class ReadsLogSumExp(torch.nn.Module):
            def forward(self, query, attn_bias):
                out = torch.ops.aten._scaled_dot_product_efficient_attention.default(
                    query, query, query, attn_bias, True
                )
                return out[0], out[1]

        class Dropout(torch.nn.Module):
            def forward(self, query, attn_bias):
                out = torch.ops.aten._scaled_dot_product_efficient_attention.default(
                    query, query, query, attn_bias, True, 0.5
                )
                return out[0]

        for model in (ReadsLogSumExp(), Dropout()):
            with self.subTest(type(model).__name__):
                gm = torch.export.export(model, (query, attn_bias)).module()
                # Lowering drops the unused getitems that export keeps.
                gm.graph.eliminate_dead_code()
                gm = force_causal_efficient_attention(
                    gm, CompilationSettings(attn_bias_is_causal="proven")
                )
                (attention,) = [
                    node
                    for node in gm.graph.nodes
                    if node.target
                    == torch.ops.aten._scaled_dot_product_efficient_attention.default
                ]
                self.assertTrue(attention.args[4])

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

    def test_force_causal_efficient_attention_only_for_lengths_known_to_be_equal(self):
        class Attention(torch.nn.Module):
            def forward(self, query, key, value, attn_bias):
                return torch.ops.aten._scaled_dot_product_efficient_attention.default(
                    query, key, value, attn_bias, False
                )[0]

        q_len, k_len = torch.export.Dim("q_len"), torch.export.Dim("k_len")
        inputs = [torch.randn(1, 2, 8, 16, device="cuda") for _ in range(3)]
        inputs.append(torch.zeros(1, 2, 8, 8, device="cuda"))
        # Separate length symbols can differ at run time even with equal example sizes; one shared symbol cannot.
        for k_dim, forced in ((k_len, False), (q_len, True)):
            with self.subTest(forced=forced):
                gm = torch.export.export(
                    Attention(),
                    tuple(inputs),
                    dynamic_shapes=(
                        {2: q_len},
                        {2: k_dim},
                        {2: k_dim},
                        {2: q_len, 3: k_dim},
                    ),
                ).module()
                gm = force_causal_efficient_attention(
                    gm, CompilationSettings(attn_bias_is_causal=True)
                )
                (node,) = (
                    n
                    for n in gm.graph.nodes
                    if n.target
                    == torch.ops.aten._scaled_dot_product_efficient_attention.default
                )
                self.assertEqual(node.args[3] is None, forced)


class TestCollapseHighRankPermutes(TestCase):
    class Window(torch.nn.Module):
        """Neighborhood attention's 9-D layout."""

        def forward(self, x):
            y = (x + 1).view(*x.shape[:5], 2, 2, *x.shape[6:])
            return y.permute(0, 1, 3, 5, -2, 2, 4, 6, -1).flatten(5, 7) * 2

    class Patches(torch.nn.Module):
        """Qwen2-VL's 10-D image patch layout for 2 images."""

        def forward(self, x):
            p = x.view(2, 1, 2, 3, 2, 2, 2, 2, 2, 2)
            return p.permute(0, 1, 4, 7, 5, 8, 3, 2, 6, 9).reshape(2, 16, 24)

    class WithPatches(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.patches = TestCollapseHighRankPermutes.Patches()

        def forward(self, x):
            return self.patches(x + 1) * 2

    def _compile(self, mod, inputs, dynamic_shapes=None, collapse=True, **settings):
        if collapse:
            settings["collapse_high_rank_permutes"] = True
        trt_mod = torch_tensorrt.dynamo.compile(
            torch.export.export(mod, inputs, dynamic_shapes=dynamic_shapes),
            list(inputs),
            min_block_size=1,
            **settings,
        )
        torch.testing.assert_close(trt_mod(*inputs), mod(*inputs))
        return {n: m for n, m in trt_mod.named_children() if "_run_on" in n}

    def test_off_by_default(self):
        x = torch.randn(2, 2, 3, 8, 8, device="cuda")
        self.assertEqual(
            list(self._compile(self.WithPatches(), (x,), collapse=False)),
            ["_run_on_acc_0", "_run_on_gpu_1", "_run_on_acc_2"],
        )

    def test_steps(self):
        from torch_tensorrt.dynamo.lowering.passes.collapse_high_rank_permutes import (
            _steps,
        )

        patches = [0, 1, 4, 7, 5, 8, 3, 2, 6, 9]
        # Dims 0 and 1 stay together.
        self.assertEqual(
            _steps([2, 2, 5, 3, 5, 4, 5, 6, 7], [0, 1, 3, 5, 7, 2, 4, 6, 8]),
            [([4, 5, 3, 5, 4, 5, 6, 7], [0, 2, 4, 6, 1, 3, 5, 7])],
        )
        # Without the dims of size 1, 8 are left.
        self.assertEqual(
            _steps([1, 1, 2, 3, 8, 2, 14, 8, 2, 14], patches),
            [([2, 3, 8, 2, 14, 8, 2, 14], [2, 5, 3, 6, 1, 0, 4, 7])],
        )
        # 9 are left, so dims 0 and 1 move together first.
        self.assertEqual(
            _steps([2, 1, 2, 3, 8, 2, 14, 8, 2, 14], patches),
            [
                ([4, 3, 8, 2, 14, 8, 2, 14], [0, 2, 5, 3, 6, 1, 4, 7]),
                ([2, 2, 768, 196], [0, 2, 1, 3]),
            ],
        )

    def test_steps_match_one_permute(self):
        from torch_tensorrt.dynamo.lowering.passes.collapse_high_rank_permutes import (
            _steps,
        )

        rng = random.Random(0)
        for _ in range(500):
            sizes = [rng.choice([1, 2, 2, 3]) for _ in range(rng.randint(9, 13))]
            dims = rng.sample(range(len(sizes)), len(sizes))
            x = torch.arange(math.prod(sizes))
            y = x
            for shape, step_dims in _steps(sizes, dims):
                self.assertLessEqual(len(shape), 8)
                y = y.reshape(shape).permute(step_dims)
            self.assertTrue(
                torch.equal(y.flatten(), x.reshape(sizes).permute(dims).flatten())
            )

    def test_window_layout(self):
        x = torch.randn(2, 2, 3, 3, 3, 4, 2, 8, device="cuda")
        self.assertEqual(list(self._compile(self.Window(), (x,))), ["_run_on_acc_0"])

    def test_patch_layout(self):
        x = torch.randn(2, 2, 3, 8, 8, device="cuda")
        self.assertEqual(
            list(self._compile(self.WithPatches(), (x,))), ["_run_on_acc_0"]
        )

    def test_patch_layout_left_to_pytorch(self):
        x = torch.randn(2, 2, 3, 8, 8, device="cuda")
        patches = f"{self.Patches.__module__}.{self.Patches.__qualname__}"
        for settings in (
            {"torch_executed_modules": {patches}},
            {"torch_executed_ops": {"torch.ops.aten.permute.default"}},
            {"torch_executed_ops": {"torch.ops.aten._reshape_copy.default"}},
            {"torch_executed_ops": {"torch.ops.aten.clone.default"}},
        ):
            with self.subTest(settings=settings):
                parts = self._compile(self.WithPatches(), (x,), **settings)
                self.assertEqual(
                    list(parts), ["_run_on_acc_0", "_run_on_gpu_1", "_run_on_acc_2"]
                )
                permutes = [
                    n
                    for n in parts["_run_on_gpu_1"].graph.nodes
                    if n.target == torch.ops.aten.permute.default
                ]
                self.assertEqual(len(permutes), 1)

    def test_torch_compile_and_serialized_engine(self):
        logger = "torch_tensorrt.dynamo.lowering.passes.collapse_high_rank_permutes"
        mod, x = self.WithPatches(), torch.randn(2, 2, 3, 8, 8, device="cuda")
        torch._dynamo.reset()
        compiled = torch.compile(
            mod,
            backend="torch_tensorrt",
            options={"collapse_high_rank_permutes": True, "min_block_size": 1},
        )
        with self.assertLogs(logger, level="DEBUG") as logs, torch.no_grad():
            torch.testing.assert_close(compiled(x), mod(x))
        torch._dynamo.reset()
        self.assertIn("Rewrote 1 permute(s)", "\n".join(logs.output))
        # Without the rewrite this path fails to build the engine.
        with self.assertLogs(logger, level="DEBUG") as logs:
            torch_tensorrt.dynamo.convert_exported_program_to_serialized_trt_engine(
                torch.export.export(mod, (x,)),
                [x],
                min_block_size=1,
                collapse_high_rank_permutes=True,
            )
        self.assertIn("Rewrote 1 permute(s)", "\n".join(logs.output))

    def test_chains_left_alone(self):
        class ReadsReshape(torch.nn.Module):
            def forward(self, x):
                y = (x + 1).view(2, 2, 3, 3, 3, 2, 2, 2, 8)
                z = y.permute(0, 1, 3, 5, 7, 2, 4, 6, 8).flatten(5, 7)
                return z * 2, (y * 3).sum(6)

        class ReadsPermute(torch.nn.Module):
            def forward(self, x):
                y = (x + 1).view(2, 2, 3, 3, 3, 2, 2, 2, 8)
                z = y.permute(0, 1, 3, 5, 7, 2, 4, 6, 8)
                return z.flatten(5, 7) * 2, (z * 3).sum(6)

        class ReadsClone(torch.nn.Module):
            def forward(self, x):
                y = (x + 1).view(2, 2, 3, 3, 3, 2, 2, 2, 8)
                z = y.permute(0, 1, 3, 5, 7, 2, 4, 6, 8).contiguous()
                return z.flatten(5, 7) * 2, (z * 3).sum(6)

        class PermutesProduct(torch.nn.Module):
            def forward(self, x):
                y = (x + 1).view(2, 2, 3, 3, 3, 2, 2, 2, 8) * 3
                return y.permute(0, 1, 3, 5, 7, 2, 4, 6, 8).flatten(5, 7) * 2

        x = torch.randn(2, 2, 3, 3, 3, 4, 2, 8, device="cuda")
        for mod in (ReadsReshape(), ReadsPermute(), ReadsClone(), PermutesProduct()):
            with self.subTest(mod=type(mod).__name__):
                self._compile(mod, (x,))

    def test_dynamic_shape(self):
        x = torch.randn(2, 2, 3, 3, 3, 4, 2, 8, device="cuda")
        batch = torch.export.Dim("batch", min=2, max=8)
        self.assertEqual(
            list(self._compile(self.Window(), (x,), dynamic_shapes=({0: batch},))),
            ["_run_on_acc_0", "_run_on_gpu_1", "_run_on_acc_2"],
        )

    def test_reshape_of_a_node_added_by_lowering(self):
        """The dynamic slice_scatter lowering adds nodes without values."""

        class CacheWindow(torch.nn.Module):
            def forward(self, cache, new):
                n = cache.shape[1]
                out = cache.slice_scatter(new, dim=1, start=n - new.shape[1], end=n)
                y = out.view(2, 2, 3, 3, 3, 2, 2, 2, 8)
                return y.permute(0, 1, 3, 5, 7, 2, 4, 6, 8).flatten(5, 7) * 2

        inputs = (torch.randn(2, 432, 8, device="cuda"), torch.randn(2, 5, 8).cuda())
        length = torch.export.Dim("length", min=2, max=431)
        self.assertEqual(
            list(
                self._compile(CacheWindow(), inputs, dynamic_shapes=({}, {1: length}))
            ),
            ["_run_on_acc_0"],
        )


if __name__ == "__main__":
    run_tests()
