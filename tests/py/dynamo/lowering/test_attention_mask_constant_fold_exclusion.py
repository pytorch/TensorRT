# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch_tensorrt
from parameterized import parameterized
from torch.testing._internal.common_utils import TestCase, run_tests
from torch_tensorrt.dynamo._settings import CompilationSettings
from torch_tensorrt.dynamo.lowering import (
    get_decompositions,
    post_lowering,
)
from torch_tensorrt.dynamo.lowering.constant_fold_exclusions import (
    ATTENTION_MASK_ARANGE_RULE_ID,
    CONSTANT_FOLD_EXCLUSION_META_KEY,
)
from torch_tensorrt.dynamo.lowering.passes.mark_constant_fold_exclusions import (
    mark_constant_fold_exclusions,
)


class TestAttentionMaskConstantFoldExclusion(TestCase):
    class AttentionWithCausalMask(torch.nn.Module):
        def forward(self, query, key, value, attention_mask):
            sequence_length = query.shape[-2]
            row = torch.arange(sequence_length, device=query.device)
            col = torch.arange(sequence_length, device=query.device)
            causal_mask = col.unsqueeze(0) <= row.unsqueeze(1)
            combined_mask = causal_mask & attention_mask
            unrelated_arange = torch.arange(3, device=query.device)
            attention = torch.ops.aten.scaled_dot_product_attention.default(
                query,
                key,
                value,
                combined_mask,
            )
            return attention, unrelated_arange

    def _export(self):
        inputs = (
            torch.randn(1, 2, 8, 16, device="cuda"),
            torch.randn(1, 2, 8, 16, device="cuda"),
            torch.randn(1, 2, 8, 16, device="cuda"),
            torch.ones(8, 8, dtype=torch.bool, device="cuda"),
        )
        return torch.export.export(self.AttentionWithCausalMask(), inputs)

    def _assert_only_attention_aranges_survive(
        self, decompose_attention, disabled_constant_fold_exclusions=()
    ):
        exported_program = self._export().run_decompositions(
            get_decompositions(decompose_attention=decompose_attention)
        )
        gm = post_lowering(
            exported_program.module(),
            CompilationSettings(
                disabled_constant_fold_exclusions=disabled_constant_fold_exclusions
            ),
        )

        arange_nodes = [
            node
            for node in gm.graph.nodes
            if node.op == "call_function"
            and getattr(node.target, "overloadpacket", None) is torch.ops.aten.arange
        ]
        self.assertEqual(len(arange_nodes), 2)
        self.assertTrue(
            all(
                node.meta.get(CONSTANT_FOLD_EXCLUSION_META_KEY, False)
                for node in arange_nodes
            )
        )

    def test_decomposed_attention_mask_aranges_are_not_folded(self):
        self._assert_only_attention_aranges_survive(decompose_attention=True)

    def test_native_attention_mask_aranges_are_not_folded(self):
        self._assert_only_attention_aranges_survive(decompose_attention=False)

    def test_native_attention_rules_can_be_disabled(self):
        exported_program = self._export().run_decompositions(
            get_decompositions(decompose_attention=False)
        )
        gm = post_lowering(
            exported_program.module(),
            CompilationSettings(
                disabled_constant_fold_exclusions={ATTENTION_MASK_ARANGE_RULE_ID}
            ),
        )
        self.assertFalse(
            any(
                node.op == "call_function"
                and getattr(node.target, "overloadpacket", None)
                is torch.ops.aten.arange
                for node in gm.graph.nodes
            )
        )

    def test_decomposed_attention_rules_can_be_disabled(self):
        """post_lowering is the only place a rule has to be disabled.

        The decompositions mark unconditionally while tracing, well before any
        settings object is reachable, so post_lowering revokes those marks
        instead of the caller having to communicate the setting twice.
        """
        exported_program = self._export().run_decompositions(
            get_decompositions(decompose_attention=True)
        )
        gm = post_lowering(
            exported_program.module(),
            CompilationSettings(
                disabled_constant_fold_exclusions={ATTENTION_MASK_ARANGE_RULE_ID}
            ),
        )
        self.assertFalse(
            any(
                node.op == "call_function"
                and getattr(node.target, "overloadpacket", None)
                is torch.ops.aten.arange
                for node in gm.graph.nodes
            )
        )


class TestAttentionMaskArangeRuleCoverage(TestCase):
    """Check every SDPA overload that carries an attention mask."""

    MASKED_ATTENTION_OPS = (
        (torch.ops.aten.scaled_dot_product_attention.default, "attn_mask"),
        (torch.ops.aten._scaled_dot_product_efficient_attention.default, "attn_bias"),
        (torch.ops.aten._scaled_dot_product_cudnn_attention.default, "attn_bias"),
    )

    def _attention_graph(self, target, mask_kwarg, arange_dtype=torch.int64):
        graph = torch.fx.Graph()
        query = graph.placeholder("query")
        key = graph.placeholder("key")
        value = graph.placeholder("value")
        arange = graph.call_function(
            torch.ops.aten.arange.default,
            (8,),
            {} if arange_dtype is None else {"dtype": arange_dtype},
        )
        mask = graph.call_function(torch.ops.aten.unsqueeze.default, (arange, 0))
        if mask_kwarg is None:
            attention = graph.call_function(target, (query, key, value, mask))
        else:
            attention = graph.call_function(
                target, (query, key, value), {mask_kwarg: mask}
            )
        graph.output(attention)
        return torch.fx.GraphModule({}, graph), arange

    def test_mask_aranges_are_marked_for_every_masked_attention_op(self):
        for target, mask_kwarg in self.MASKED_ATTENTION_OPS:
            for passed_as_kwarg in (False, True):
                with self.subTest(target=target, passed_as_kwarg=passed_as_kwarg):
                    gm, arange = self._attention_graph(
                        target, mask_kwarg if passed_as_kwarg else None
                    )
                    mark_constant_fold_exclusions(gm)
                    self.assertTrue(arange.meta.get(CONSTANT_FOLD_EXCLUSION_META_KEY))

    @parameterized.expand([(None,), (torch.int16,)])
    def test_unknown_or_non_allowlisted_integer_dtype_is_not_marked(self, dtype):
        # No tensor metadata is available in this manually constructed graph.
        gm, arange = self._attention_graph(
            torch.ops.aten.scaled_dot_product_attention.default, None, dtype
        )
        mark_constant_fold_exclusions(gm)
        self.assertFalse(arange.meta.get(CONSTANT_FOLD_EXCLUSION_META_KEY))


class TestAttentionMaskCompilation(TestCase):
    class SharedMaskAttention(torch.nn.Module):
        def __init__(
            self, position_dtype, start=0, key_position_dtype=None, cast_dtype=None
        ):
            super().__init__()
            self.position_dtype = position_dtype
            self.start = start
            self.key_position_dtype = key_position_dtype or position_dtype
            self.cast_dtype = cast_dtype

        def forward(self, query, key, value, padding_mask):
            row = torch.arange(
                self.start,
                self.start + query.shape[-2],
                dtype=self.position_dtype,
                device=query.device,
            )
            col = torch.arange(
                self.start,
                self.start + key.shape[-2],
                dtype=self.key_position_dtype,
                device=query.device,
            )
            if self.cast_dtype is not None:
                row = row.to(self.cast_dtype)
                col = col.to(self.cast_dtype)
            mask = (col.unsqueeze(0) <= row.unsqueeze(1)) & padding_mask
            return (
                torch.nn.functional.scaled_dot_product_attention(
                    query, key, value, mask
                ),
                torch.nn.functional.scaled_dot_product_attention(
                    query + 0.5, key, value, mask
                ),
            )

    def _inputs(self, batched_mask=True):
        query = torch.randn(2, 2, 8, 64, dtype=torch.float16, device="cuda")
        key = torch.randn(2, 2, 16, 64, dtype=torch.float16, device="cuda")
        value = torch.randn_like(key)
        shape = (2, 1, 1, 16) if batched_mask else (16,)
        mask = torch.ones(shape, dtype=torch.bool, device="cuda")
        mask[..., 1] = False
        if batched_mask:
            mask[1, ..., 2] = False
        return query, key, value, mask

    def _aranges_after_lowering(self, exported_program, decompose_attention, disabled):
        lowered = exported_program.run_decompositions(
            get_decompositions(decompose_attention=decompose_attention)
        )
        gm = post_lowering(
            lowered.module(),
            CompilationSettings(disabled_constant_fold_exclusions=disabled),
        )
        return [
            node
            for node in gm.graph.nodes
            if node.op == "call_function"
            and getattr(node.target, "overloadpacket", None) is torch.ops.aten.arange
        ]

    def _compile(self, exported_program, inputs, decompose_attention, disabled=()):
        return torch_tensorrt.dynamo.compile(
            exported_program,
            inputs=inputs,
            min_block_size=1,
            require_full_compilation=True,
            decompose_attention=decompose_attention,
            disabled_constant_fold_exclusions=disabled,
            truncate_double=False,
            cache_built_engines=False,
            reuse_cached_engines=False,
        )

    @parameterized.expand(
        [
            (decompose, inferred, disabled)
            for decompose in (False, True)
            for inferred in (False, True)
            for disabled in (False, True)
        ]
    )
    def test_float64_mask_folds_and_compiles(
        self, decompose_attention, inferred_dtype, disable_rule
    ):
        # Adjacent positions above 2**24 are not all representable in FP32.
        # Truncating the range before comparing positions changes this mask.
        model = self.SharedMaskAttention(
            None if inferred_dtype else torch.float64, start=float(2**24)
        ).eval()
        inputs = self._inputs()
        previous_dtype = torch.get_default_dtype()
        try:
            if inferred_dtype:
                torch.set_default_dtype(torch.float64)
            with torch.inference_mode():
                expected = model(*inputs)
                exported_program = torch.export.export(model, inputs)
            disabled = {ATTENTION_MASK_ARANGE_RULE_ID} if disable_rule else ()
            self.assertEqual(
                self._aranges_after_lowering(
                    exported_program, decompose_attention, disabled
                ),
                [],
            )
            with torch.inference_mode():
                compiled = self._compile(
                    exported_program, inputs, decompose_attention, disabled
                )
                torch.testing.assert_close(
                    compiled(*inputs), expected, rtol=0.02, atol=0.02
                )
        finally:
            torch.set_default_dtype(previous_dtype)

    @parameterized.expand(
        [
            (decompose, batched, dtype)
            for decompose in (False, True)
            for batched in (False, True)
            for dtype in (torch.int32, torch.int64)
        ]
    )
    def test_shared_rectangular_mask_is_preserved_and_compiles(
        self, decompose_attention, batched_mask, position_dtype
    ):
        model = self.SharedMaskAttention(position_dtype).eval()
        inputs = self._inputs(batched_mask)
        with torch.inference_mode():
            exported_program = torch.export.export(model, inputs)
            aranges = self._aranges_after_lowering(
                exported_program, decompose_attention, ()
            )
            self.assertEqual(len(aranges), 2)
            self.assertTrue(
                all(
                    ATTENTION_MASK_ARANGE_RULE_ID
                    in node.meta.get(CONSTANT_FOLD_EXCLUSION_META_KEY, ())
                    for node in aranges
                )
            )
            compiled = self._compile(exported_program, inputs, decompose_attention)
            torch.testing.assert_close(
                compiled(*inputs), model(*inputs), rtol=0.02, atol=0.02
            )
            # Reuse the engine with different padding to exercise a runtime mask.
            new_mask = inputs[-1].clone()
            new_mask[..., 1] = True
            new_mask[..., 2] = False
            changed_inputs = (*inputs[:-1], new_mask)
            torch.testing.assert_close(
                compiled(*changed_inputs), model(*changed_inputs), rtol=0.02, atol=0.02
            )

    @parameterized.expand(
        [
            (decompose, dtype, mixed)
            for decompose in (False, True)
            for dtype, mixed in (
                (torch.float16, False),
                (torch.bfloat16, False),
                (torch.float32, False),
                (torch.float32, True),
            )
        ]
    )
    def test_floating_point_mask_ranges_fold_and_compile(
        self, decompose_attention, position_dtype, mixed
    ):
        model = self.SharedMaskAttention(
            position_dtype, key_position_dtype=torch.int64 if mixed else position_dtype
        ).eval()
        inputs = self._inputs()
        with torch.inference_mode():
            exported_program = torch.export.export(model, inputs)
            self.assertEqual(
                self._aranges_after_lowering(exported_program, decompose_attention, ()),
                [],
            )
            compiled = self._compile(exported_program, inputs, decompose_attention)
            torch.testing.assert_close(
                compiled(*inputs), model(*inputs), rtol=0.02, atol=0.02
            )

    @parameterized.expand(
        [(decompose, cast) for decompose in (False, True) for cast in (False, True)]
    )
    def test_float64_intermediates_fold_and_compile(self, decompose_attention, cast):
        if cast:
            model = self.SharedMaskAttention(
                torch.int64, start=2**24, cast_dtype=torch.float64
            )
        else:
            model = self.SharedMaskAttention(
                torch.float64, start=2**24, key_position_dtype=torch.int64
            )
        inputs = self._inputs()
        with torch.inference_mode():
            exported_program = torch.export.export(model.eval(), inputs)
            self.assertEqual(
                self._aranges_after_lowering(exported_program, decompose_attention, ()),
                [],
            )
            compiled = self._compile(exported_program, inputs, decompose_attention)
            torch.testing.assert_close(
                compiled(*inputs), model(*inputs), rtol=0.02, atol=0.02
            )

    @parameterized.expand(
        [
            (decompose, dtype)
            for decompose in (False, True)
            for dtype in (torch.int32, torch.int64, torch.float32, torch.float64)
        ]
    )
    def test_torch_compile_shared_mask(self, decompose_attention, position_dtype):
        torch._dynamo.reset()
        self.addCleanup(torch._dynamo.reset)
        model = self.SharedMaskAttention(position_dtype).eval()
        inputs = self._inputs()
        compiled = torch.compile(
            model,
            backend="tensorrt",
            fullgraph=True,
            options={
                "decompose_attention": decompose_attention,
                "min_block_size": 1,
                "pass_through_build_failures": True,
                "truncate_double": False,
            },
        )
        with torch.inference_mode():
            torch.testing.assert_close(
                compiled(*inputs), model(*inputs), rtol=0.02, atol=0.02
            )


if __name__ == "__main__":
    run_tests()
