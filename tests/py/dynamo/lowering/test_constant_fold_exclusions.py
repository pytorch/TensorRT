import torch
from torch.testing._internal.common_utils import TestCase, run_tests
from torch_tensorrt.dynamo._settings import (
    CompilationSettings,
    settings_are_compatible,
)
from torch_tensorrt.dynamo.lowering.constant_fold_exclusions import (
    CONSTANT_FOLD_EXCLUSION_META_KEY,
    register_constant_fold_exclusion_rule,
    unregister_constant_fold_exclusion_rule,
)
from torch_tensorrt.dynamo.lowering.passes.constant_folding import constant_fold
from torch_tensorrt.dynamo.lowering.passes.mark_constant_fold_exclusions import (
    mark_constant_fold_exclusions,
)

TEST_RULE_ID_PREFIX = "test_arbitrary_node"


def custom_target():
    return torch.ones(1)


def custom_rule(node):
    return (node,) if node.target is custom_target else ()


class TestConstantFoldExclusionMechanics(TestCase):
    def setUp(self):
        super().setUp()
        self.rule_id = f"{TEST_RULE_ID_PREFIX}:{self.id()}"
        register_constant_fold_exclusion_rule(self.rule_id)(custom_rule)
        self.addCleanup(unregister_constant_fold_exclusion_rule, self.rule_id)

    def _custom_graph(self):
        graph = torch.fx.Graph()
        custom_node = graph.call_function(custom_target)
        graph.output(custom_node)
        return torch.fx.GraphModule({}, graph), custom_node

    def test_disabled_rules_default_to_empty(self):
        self.assertEqual(CompilationSettings().disabled_constant_fold_exclusions, set())
        self.assertEqual(
            CompilationSettings(
                disabled_constant_fold_exclusions=[self.rule_id]
            ).disabled_constant_fold_exclusions,
            {self.rule_id},
        )
        with self.assertRaisesRegex(TypeError, "collection of rule IDs"):
            CompilationSettings(disabled_constant_fold_exclusions=self.rule_id)

    def test_old_serialized_setting_defaults_to_no_disabled_rules(self):
        state = CompilationSettings().__dict__.copy()
        state.pop("disabled_constant_fold_exclusions")
        restored = CompilationSettings.__new__(CompilationSettings)
        restored.__setstate__(state)
        self.assertEqual(restored.disabled_constant_fold_exclusions, set())

    def test_setting_changes_engine_compatibility(self):
        compatible, incompatible_settings = settings_are_compatible(
            CompilationSettings(),
            CompilationSettings(disabled_constant_fold_exclusions={self.rule_id}),
        )
        self.assertFalse(compatible)
        self.assertIn(
            "disabled_constant_fold_exclusions",
            incompatible_settings,
        )

    def test_registered_rule_preserves_an_arbitrary_node(self):
        gm, custom_node = self._custom_graph()
        mark_constant_fold_exclusions(gm)
        self.assertEqual(
            custom_node.meta[CONSTANT_FOLD_EXCLUSION_META_KEY], {self.rule_id}
        )

        constant_fold(gm, CompilationSettings())
        self.assertIn(custom_node, gm.graph.nodes)

    def test_disabled_rule_revokes_an_existing_mark(self):
        gm, custom_node = self._custom_graph()
        custom_node.meta[CONSTANT_FOLD_EXCLUSION_META_KEY] = {self.rule_id}

        mark_constant_fold_exclusions(
            gm,
            CompilationSettings(disabled_constant_fold_exclusions={self.rule_id}),
        )
        self.assertEqual(custom_node.meta[CONSTANT_FOLD_EXCLUSION_META_KEY], set())

    def test_unknown_disabled_rule_is_rejected(self):
        with self.assertRaisesRegex(
            ValueError,
            "Unknown constant-fold exclusion rule IDs",
        ):
            CompilationSettings(disabled_constant_fold_exclusions={"unknown_rule"})


if __name__ == "__main__":
    run_tests()
