import torch
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

    def test_ia_attention_mask_aranges_are_not_folded(self):
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

    def _attention_graph(self, target, mask_kwarg):
        graph = torch.fx.Graph()
        query = graph.placeholder("query")
        key = graph.placeholder("key")
        value = graph.placeholder("value")
        arange = graph.call_function(torch.ops.aten.arange.default, (8,))
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


if __name__ == "__main__":
    run_tests()
