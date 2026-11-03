import operator
import sys
from unittest.mock import patch

import torch
from torch.testing._internal.common_utils import TestCase, run_tests
from torch_tensorrt.dynamo._settings import CompilationSettings
from torch_tensorrt.dynamo.lowering.passes.batch_cheap_fx_cleanups import (
    batch_cheap_fx_cleanups,
)
from torch_tensorrt.dynamo.lowering.passes.eliminate_sym_min_int64_max import (
    eliminate_sym_min_int64_max,
)
from torch_tensorrt.dynamo.lowering.passes.normalize_negative_slice_stop import (
    normalize_negative_slice_stop,
)
from torch_tensorrt.dynamo.lowering.passes.pass_utils import (
    clean_up_graph_after_modifications,
    flush_deferred_graph_cleanup,
    set_defer_graph_cleanup,
)
from torch_tensorrt.dynamo.lowering.passes.remove_assert_nodes import (
    remove_assert_nodes,
)
from torch_tensorrt.dynamo.lowering.passes.remove_num_users_is_0_nodes import (
    remove_num_users_is_0_nodes,
)
from torch_tensorrt.dynamo.lowering.passes.replace_fused_rms_norm import (
    replace_fused_rms_norm,
)


def _settings() -> CompilationSettings:
    return CompilationSettings()


def _call_targets(gm: torch.fx.GraphModule) -> list:
    return [n.target for n in gm.graph.nodes if n.op == "call_function"]


def _has_target(gm: torch.fx.GraphModule, target: object) -> bool:
    return any(n.op == "call_function" and n.target == target for n in gm.graph.nodes)


def _dead_add_graph() -> torch.fx.GraphModule:
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    graph.call_function(torch.ops.aten.add.Tensor, args=(x, 1))
    graph.output(x)
    return torch.fx.GraphModule({}, graph)


def _identity_graph() -> torch.fx.GraphModule:
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    graph.output(x)
    return torch.fx.GraphModule({}, graph)


def _assert_orphan_graph() -> torch.fx.GraphModule:
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    unused = graph.call_function(torch.ops.aten.add.Tensor, args=(x, 1.0))
    graph.call_function(
        torch.ops.aten._assert_scalar.default,
        args=(unused, "msg"),
    )
    graph.output(x)
    return torch.fx.GraphModule({}, graph)


def _four_repair_graph() -> torch.fx.GraphModule:
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    x.meta["val"] = torch.empty(2, 5, 3)
    n = graph.placeholder("n")
    unused = graph.call_function(torch.ops.aten.add.Tensor, args=(x, 1.0))
    graph.call_function(
        torch.ops.aten._assert_scalar.default,
        args=(unused, "msg"),
    )
    smin = graph.call_function(torch.sym_min, args=(x, sys.maxsize))
    neg = graph.call_function(operator.neg, args=(n,))
    sliced = graph.call_function(torch.ops.aten.slice.Tensor, args=(x, -2, neg))
    graph.output((smin, sliced))
    return torch.fx.GraphModule({}, graph)


class _CleanupCounter:
    def __init__(self, orig):
        self.n = 0
        self._orig = orig

    def __call__(self, gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
        self.n += 1
        return self._orig(gm)


class TestBatchCheapFxCleanups(TestCase):
    def setUp(self) -> None:
        set_defer_graph_cleanup(False)

    def tearDown(self) -> None:
        set_defer_graph_cleanup(False)

    def test_batch_applies_all_four_repairs(self) -> None:
        if not hasattr(torch, "sym_min"):
            self.skipTest("torch.sym_min is not available")

        gm = _four_repair_graph()
        gm = batch_cheap_fx_cleanups(gm, _settings())

        self.assertFalse(_has_target(gm, torch.ops.aten._assert_scalar.default))
        self.assertFalse(_has_target(gm, torch.ops.aten.add.Tensor))
        self.assertFalse(_has_target(gm, torch.sym_min))

        slice_node = next(
            node
            for node in gm.graph.nodes
            if node.op == "call_function" and node.target == torch.ops.aten.slice.Tensor
        )
        self.assertEqual(slice_node.args[1], 1)
        normalized_start = slice_node.args[2]
        self.assertEqual(normalized_start.target, operator.sub)
        dim_size, offset = normalized_start.args
        self.assertEqual(dim_size.target, torch.ops.aten.sym_size.int)
        self.assertEqual(dim_size.args[1], 1)

        x = next(node for node in gm.graph.nodes if node.op == "placeholder")
        output_node = next(node for node in gm.graph.nodes if node.op == "output")
        self.assertEqual(output_node.args[0], (x, slice_node))

    def test_batch_matches_standalone_passes_in_order(self) -> None:
        if not hasattr(torch, "sym_min"):
            self.skipTest("torch.sym_min is not available")

        batched = batch_cheap_fx_cleanups(_four_repair_graph(), _settings())
        standalone = _four_repair_graph()
        standalone = remove_assert_nodes(standalone, _settings())
        standalone = remove_num_users_is_0_nodes(standalone, _settings())
        standalone = eliminate_sym_min_int64_max(standalone)
        standalone = normalize_negative_slice_stop(standalone)

        self.assertEqual(_call_targets(batched), _call_targets(standalone))
        self.assertEqual(str(batched.graph), str(standalone.graph))

    def test_assert_then_orphan_removed_in_same_batch(self) -> None:
        gm = batch_cheap_fx_cleanups(_assert_orphan_graph(), _settings())

        self.assertFalse(_has_target(gm, torch.ops.aten._assert_scalar.default))
        self.assertFalse(_has_target(gm, torch.ops.aten.add.Tensor))

        x = torch.arange(4.0)
        torch.testing.assert_close(gm(x), x)

    def test_defer_skips_dce_until_flush(self) -> None:
        gm = _dead_add_graph()
        set_defer_graph_cleanup(True)
        gm = clean_up_graph_after_modifications(gm)

        self.assertTrue(_has_target(gm, torch.ops.aten.add.Tensor))

        gm = flush_deferred_graph_cleanup(gm)
        self.assertFalse(_has_target(gm, torch.ops.aten.add.Tensor))

    def test_defer_off_cleans_up_immediately(self) -> None:
        gm = _dead_add_graph()
        set_defer_graph_cleanup(False)
        gm = clean_up_graph_after_modifications(gm)

        self.assertFalse(_has_target(gm, torch.ops.aten.add.Tensor))

    def test_flush_without_pending_cleanup_is_noop(self) -> None:
        gm = _dead_add_graph()
        set_defer_graph_cleanup(True)
        gm = flush_deferred_graph_cleanup(gm)

        self.assertTrue(_has_target(gm, torch.ops.aten.add.Tensor))

    def test_noop_batch_does_not_run_cleanup(self) -> None:
        import torch_tensorrt.dynamo.lowering.passes.pass_utils as pass_utils

        counter = _CleanupCounter(pass_utils._run_graph_cleanup)
        with patch.object(pass_utils, "_run_graph_cleanup", counter):
            gm = batch_cheap_fx_cleanups(_identity_graph(), _settings())

        self.assertEqual(counter.n, 0)
        self.assertEqual(
            [n.op for n in gm.graph.nodes],
            ["placeholder", "output"],
        )

    def test_rms_norm_skip_cleanup_when_unchanged(self) -> None:
        import torch_tensorrt.dynamo.lowering.passes.pass_utils as pass_utils

        counter = _CleanupCounter(pass_utils._run_graph_cleanup)
        with patch.object(pass_utils, "_run_graph_cleanup", counter):
            replace_fused_rms_norm(_identity_graph(), _settings())

        self.assertEqual(counter.n, 0)

    def test_post_lowering_clears_defer_if_a_pass_raises(self) -> None:
        from torch_tensorrt.dynamo.lowering.passes import _aten_lowering_pass as alp
        from torch_tensorrt.dynamo.lowering.passes._aten_lowering_pass import (
            post_lowering,
        )

        def boom(gm: torch.fx.GraphModule, settings: CompilationSettings):
            raise RuntimeError("pass failed")

        with patch.object(alp, "ATEN_POST_LOWERING_PASSES", boom):
            with self.assertRaisesRegex(RuntimeError, "pass failed"):
                post_lowering(_identity_graph(), _settings())

        gm = clean_up_graph_after_modifications(_dead_add_graph())
        self.assertFalse(_has_target(gm, torch.ops.aten.add.Tensor))


if __name__ == "__main__":
    run_tests()
