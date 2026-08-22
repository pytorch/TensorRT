# type: ignore
"""Unit tests for ``lift_mutated_buffers`` and ``inline_lifted_buffers_into_gm``.

``lift_mutated_buffers`` is a pre-compile rewrite that detects mutated
buffers (the trailing ``aten.copy_(get_attr, _)`` pattern that
``ExportedProgram.module()`` generates for each ``BUFFER_MUTATION``) and
lifts each one from a ``get_attr`` to a ``placeholder``. The rebuilt
GraphModule's ``forward`` signature reflects the new placeholder set —
which requires resetting the graph's ``_codegen`` from the
``_PyTreeCodeGen`` baked in by ``ep.module()`` to the default ``CodeGen``.

These tests verify:

* Buffers ARE lifted when mutated.
* Buffers are NOT lifted when only read.
* The rebuilt GraphModule's ``forward`` accepts the new placeholders.
* The rebuilt GraphModule produces the same outputs as the original
  pre-lift gm when both are given the same inputs (buffers + user inputs).
* The original buffer tensors are returned alongside the placeholder
  names for downstream wiring.
* ``inline_lifted_buffers_into_gm`` rewrites the lifted-buffer
  placeholders into ``get_attr`` reads and registers the buffers as
  module state. The result is a plain ``fx.GraphModule`` that
  serializes via ``torch_tensorrt.save`` without an external wrapper.
"""

import inspect
import unittest
from unittest import mock

import torch
from torch.export import export
from torch.testing._internal.common_utils import TestCase, run_tests
from torch_tensorrt.dynamo._settings import CompilationSettings
from torch_tensorrt.dynamo.lowering._buffer_lifting import (
    aliased_input_bindings,
    assert_predicted_kv_aliased,
    inline_lifted_buffers_into_gm,
    lift_mutated_buffers,
)


def _ep_module_decomposed(model, args):
    """Run the prefix of the compile pipeline up through ``ep.module()``."""
    ep = export(model, tuple(args))
    ep = ep.run_decompositions({})
    return ep.module()


class TestLiftMutatedBuffers(TestCase):
    def test_no_mutation_no_lift(self):
        """A module that reads buffers but doesn't mutate them returns
        ``(gm, [])`` — no rewrite happens."""

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("table", torch.arange(16, dtype=torch.float32))

            def forward(self, x):
                return x + self.table.sum()

        gm = _ep_module_decomposed(M(), (torch.zeros(4),))
        new_gm, lifted = lift_mutated_buffers(gm)
        self.assertEqual(lifted, [])
        # The same gm is returned when nothing is lifted.
        self.assertIs(new_gm, gm)

    def test_single_buffer_lifted(self):
        """A buffer that's mutated should be lifted to a placeholder, the
        trailing copy_ removed, and the rebuilt forward should accept it
        as an argument."""

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("cache", torch.zeros(2, 4, 16, 8))

            def forward(self, x):
                self.cache[:, :, 3:4, :] = x
                return self.cache.sum()

        gm = _ep_module_decomposed(M(), (torch.ones(2, 4, 1, 8),))
        new_gm, lifted = lift_mutated_buffers(gm)

        # Exactly one buffer was lifted.
        self.assertEqual(len(lifted), 1)
        ph_name, buf_name, tensor = lifted[0]
        self.assertEqual(buf_name, "cache")
        self.assertEqual(tuple(tensor.shape), (2, 4, 16, 8))
        self.assertEqual(ph_name, "buf_cache")

        # The rebuilt forward should now accept (x, buf_cache).
        sig = inspect.signature(new_gm.forward)
        param_names = list(sig.parameters.keys())
        self.assertEqual(param_names, ["x", "buf_cache"])

        # No get_attr nodes for `cache` remain in the graph.
        for node in new_gm.graph.nodes:
            if node.op == "get_attr":
                self.assertNotEqual(node.target, "cache")
        # No trailing aten.copy_ to the (now removed) cache get_attr.
        for node in new_gm.graph.nodes:
            self.assertNotEqual(node.target, torch.ops.aten.copy_.default)

    def test_nested_buffer_lifted(self):
        """A buffer owned by a submodule should be lifted too.

        ``get_attr`` targets are fully qualified, so this one arrives as
        ``inner.cache``. Resolving it with ``getattr`` reports it as missing and
        skips the rewrite, which leaves the mutation in place."""

        class Inner(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("cache", torch.zeros(2, 4, 16, 8))

            def forward(self, x):
                self.cache[:, :, 3:4, :] = x
                return self.cache.sum()

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.inner = Inner()

            def forward(self, x):
                return self.inner(x)

        gm = _ep_module_decomposed(M(), (torch.ones(2, 4, 1, 8),))
        new_gm, lifted = lift_mutated_buffers(gm)

        self.assertEqual(len(lifted), 1)
        ph_name, buf_name, tensor = lifted[0]
        self.assertEqual(buf_name, "inner.cache")
        self.assertEqual(tuple(tensor.shape), (2, 4, 16, 8))
        self.assertEqual(ph_name, "buf_inner_cache")

        sig = inspect.signature(new_gm.forward)
        self.assertEqual(list(sig.parameters.keys()), ["x", "buf_inner_cache"])

        for node in new_gm.graph.nodes:
            self.assertNotEqual(node.target, torch.ops.aten.copy_.default)

    def test_paired_buffers_lifted(self):
        """Two mutated buffers are both lifted; placeholders appear in a
        stable order so callers can match them positionally."""

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("cache_k", torch.zeros(2, 4, 16, 8))
                self.register_buffer("cache_v", torch.zeros(2, 4, 16, 8))

            def forward(self, x_k, x_v):
                self.cache_k[:, :, 3:4, :] = x_k
                self.cache_v[:, :, 3:4, :] = x_v
                return self.cache_k.sum() + self.cache_v.sum()

        gm = _ep_module_decomposed(
            M(), (torch.ones(2, 4, 1, 8), torch.ones(2, 4, 1, 8))
        )
        new_gm, lifted = lift_mutated_buffers(gm)
        self.assertEqual(len(lifted), 2)
        buf_names = {b for _, b, _ in lifted}
        self.assertEqual(buf_names, {"cache_k", "cache_v"})

        # forward signature should have all 4 params (2 user + 2 buffer).
        sig = inspect.signature(new_gm.forward)
        self.assertEqual(len(sig.parameters), 4)

    def test_rebuilt_forward_matches_original(self):
        """The rebuilt GraphModule, when given (user_args..., buffers...),
        should produce the same outputs as the original ep.module() when
        given the same user_args (with buffers used from internal state)."""

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("cache", torch.zeros(2, 4, 16, 8))

            def forward(self, x):
                self.cache[:, :, 3:4, :] = x
                return self.cache.sum() * 2.0

        x = torch.randn(2, 4, 1, 8)
        gm_original = _ep_module_decomposed(M(), (x.clone(),))
        original_out = gm_original(x.clone())
        # ep.module() returns its outputs as a tuple. Take the first element
        # to compare against the rebuilt gm (whose default CodeGen returns
        # tuples too, but possibly with a different surrounding shape).
        if isinstance(original_out, tuple):
            original_out = original_out[0]

        # Re-create gm for the lift (in-place mutation of the first gm's
        # graph would change its forward behavior).
        gm_for_lift = _ep_module_decomposed(M(), (x.clone(),))
        new_gm, lifted = lift_mutated_buffers(gm_for_lift)
        _, _, buf_tensor = lifted[0]
        # Call rebuilt gm with the original buffer state.
        new_out = new_gm(x.clone(), buf_tensor.clone())
        if isinstance(new_out, tuple):
            new_out = new_out[0]
        self.assertTrue(torch.allclose(new_out, original_out))


class TestInlineLiftedBuffers(TestCase):
    """``inline_lifted_buffers_into_gm`` should register each lifted
    buffer as module state on the gm and rewrite the corresponding
    placeholder node into a ``get_attr`` read. After inlining, the gm's
    forward should accept only the user inputs."""

    def _build_simple_gm(self):
        """Construct an fx GraphModule with two placeholders (x, buf) and
        a body that sums them, matching what ``lift_mutated_buffers``
        would produce."""
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        buf = graph.placeholder("buf_cache")
        out = graph.call_function(torch.add, args=(x, buf))
        graph.output(out)
        gm = torch.fx.GraphModule({}, graph)
        gm.recompile()
        return gm

    def test_inline_registers_buffer_and_rewrites_placeholder(self):
        gm = self._build_simple_gm()
        buf_tensor = torch.tensor([1.0, 2.0, 3.0])

        new_gm = inline_lifted_buffers_into_gm(
            gm, lifted_buffers=[("buf_cache", "cache", buf_tensor)]
        )

        # Buffer registered as module state.
        self.assertTrue(hasattr(new_gm, "cache"))
        self.assertTrue(torch.allclose(new_gm.cache, buf_tensor))

        # Placeholder count is now 1 (only `x`); buffer is a get_attr.
        placeholders = [n for n in new_gm.graph.nodes if n.op == "placeholder"]
        self.assertEqual(len(placeholders), 1)
        self.assertEqual(placeholders[0].name, "x")
        get_attrs = [n for n in new_gm.graph.nodes if n.op == "get_attr"]
        self.assertEqual(len(get_attrs), 1)
        self.assertEqual(get_attrs[0].target, "cache")

        # forward(x) computes x + cache via the inlined get_attr.
        x = torch.tensor([10.0, 20.0, 30.0])
        out = new_gm(x)
        if isinstance(out, tuple):
            out = out[0]
        self.assertTrue(torch.allclose(out, x + buf_tensor))

    def test_inline_is_noop_for_empty_lifted(self):
        gm = self._build_simple_gm()
        ph_before = [n.name for n in gm.graph.nodes if n.op == "placeholder"]
        result = inline_lifted_buffers_into_gm(gm, lifted_buffers=[])
        self.assertIs(result, gm)
        ph_after = [n.name for n in result.graph.nodes if n.op == "placeholder"]
        self.assertEqual(ph_before, ph_after)

    def test_inline_preserves_user_input_order(self):
        """When multiple buffers are inlined, the user inputs come first
        and are unchanged; the buffers become get_attr reads."""
        graph = torch.fx.Graph()
        u1 = graph.placeholder("u1")
        u2 = graph.placeholder("u2")
        b1 = graph.placeholder("buf_a")
        b2 = graph.placeholder("buf_b")
        s1 = graph.call_function(torch.add, args=(u1, b1))
        s2 = graph.call_function(torch.add, args=(u2, b2))
        out = graph.call_function(torch.add, args=(s1, s2))
        graph.output(out)
        gm = torch.fx.GraphModule({}, graph)
        gm.recompile()

        new_gm = inline_lifted_buffers_into_gm(
            gm,
            lifted_buffers=[
                ("buf_a", "a", torch.tensor(1.0)),
                ("buf_b", "b", torch.tensor(2.0)),
            ],
        )
        placeholders = [n.name for n in new_gm.graph.nodes if n.op == "placeholder"]
        self.assertEqual(placeholders, ["u1", "u2"])
        # Numerical: (10 + 1) + (20 + 2) = 33
        out = new_gm(torch.tensor(10.0), torch.tensor(20.0))
        if isinstance(out, tuple):
            out = out[0]
        self.assertEqual(out.item(), 33.0)

    def test_inline_remaps_copyback_targets_for_nested_buffers(self):
        """A nested (dotted) buffer is registered under a flattened
        ``lifted_buf_*`` name (``register_buffer`` rejects "."), so the recorded
        copy-back targets -- which still use the original dotted name -- must be
        remapped through the same mapping. Otherwise the ``BUFFER_MUTATION``
        OutputSpec names a buffer that no longer exists and the ExportedProgram
        verifier rejects the program. A flat name maps to itself and is unchanged.
        """
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        b_state = graph.placeholder("buf_state")
        b_nested = graph.placeholder("buf_nested")
        s = graph.call_function(torch.add, args=(x, b_state))
        out = graph.call_function(torch.add, args=(s, b_nested))
        graph.output(out)
        gm = torch.fx.GraphModule({}, graph)
        gm.recompile()
        # lift_mutated_buffers records copy-back targets under the buffers'
        # original names (dotted for a submodule-owned cache).
        gm.meta["_copyback_mutation_buffers"] = ["state", "layers.0.attn.k_cache"]

        new_gm = inline_lifted_buffers_into_gm(
            gm,
            lifted_buffers=[
                ("buf_state", "state", torch.zeros(4)),
                ("buf_nested", "layers.0.attn.k_cache", torch.zeros(4)),
            ],
        )

        registered = dict(new_gm.named_buffers())
        # Flat name kept; nested name flattened to a valid attribute.
        self.assertIn("state", registered)
        self.assertIn("lifted_buf_layers_0_attn_k_cache", registered)
        # Copy-back targets remapped: flat unchanged, nested -> flattened name.
        self.assertEqual(
            new_gm.meta["_copyback_mutation_buffers"],
            ["state", "lifted_buf_layers_0_attn_k_cache"],
        )
        # Every recorded target now names a buffer that actually exists.
        for name in new_gm.meta["_copyback_mutation_buffers"]:
            self.assertIn(name, registered)


class TestCopyBackClassification(TestCase):
    """``lift_mutated_buffers`` splits mutated buffers into two kinds.

    An *eligible* ``slice_scatter`` / ``index_copy`` KV write (one the converter
    lowers to an ``IKVCacheUpdateLayer`` with in-place aliased I/O) relies on that
    engine aliasing and is NOT recorded for copy-back. Every other mutation --
    including a ``slice_scatter`` / ``index_copy`` that fails the converter's
    eligibility (wrong rank/dim/shape) and is lowered to a non-aliasing scatter --
    has no engine aliasing, so its new value is re-appended as a trailing graph
    output and its buffer name recorded in
    ``gm.meta['_copyback_mutation_buffers']`` for the exporters to reclassify as
    a BUFFER_MUTATION.
    """

    def test_kv_slice_scatter_write_no_copyback(self):
        """A slice-assignment KV write is aliased downstream, not copied back."""

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("cache", torch.zeros(2, 4, 16, 8))

            def forward(self, x):
                self.cache[:, :, 3:4, :] = x
                return self.cache.sum()

        gm = _ep_module_decomposed(M(), (torch.ones(2, 4, 1, 8),))
        new_gm, lifted = lift_mutated_buffers(gm)
        self.assertEqual(len(lifted), 1)
        self.assertEqual(new_gm.meta["_copyback_mutation_buffers"], [])
        # Predicted-KV binding recorded so compile() can assert it actually aliases.
        self.assertEqual(new_gm.meta["_predicted_kv_bindings"], ["buf_cache"])

    def test_kv_index_copy_write_no_copyback(self):
        """An eligible ``index_copy`` KV write (4-D static cache, dim=2, batch 1,
        single-position source) is aliased by the KV converter, so it is not
        copied back."""

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("cache", torch.zeros(1, 4, 16, 8))

            def forward(self, x):
                self.cache.index_copy_(2, torch.tensor([3]), x)
                return self.cache.sum()

        gm = _ep_module_decomposed(M(), (torch.ones(1, 4, 1, 8),))
        new_gm, lifted = lift_mutated_buffers(gm)
        self.assertEqual(len(lifted), 1)
        self.assertEqual(new_gm.meta["_copyback_mutation_buffers"], [])

    def test_non_kv_mutation_recorded_for_copyback(self):
        """A non-KV in-place mutation is recorded for copy-back by buffer name."""

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("state", torch.zeros(4))

            def forward(self, x):
                self.state.add_(x)
                return self.state.sum()

        gm = _ep_module_decomposed(M(), (torch.ones(4),))
        new_gm, lifted = lift_mutated_buffers(gm)
        self.assertEqual(len(lifted), 1)
        self.assertEqual(new_gm.meta["_copyback_mutation_buffers"], ["state"])
        # A non-KV mutation is not predicted to alias, so nothing to assert later.
        self.assertEqual(new_gm.meta["_predicted_kv_bindings"], [])

    def test_copyback_value_appended_as_last_output(self):
        """The non-KV new value is re-attached as a trailing graph output so it
        survives DCE; at the lift stage it is the LAST output and equals the
        updated buffer."""

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("state", torch.zeros(4))

            def forward(self, x):
                self.state.add_(x)
                return self.state.sum()

        x = torch.arange(4, dtype=torch.float32)
        gm = _ep_module_decomposed(M(), (x.clone(),))
        new_gm, lifted = lift_mutated_buffers(gm)
        _, buf_name, buf_tensor = lifted[0]
        self.assertEqual(buf_name, "state")

        out = new_gm(x.clone(), buf_tensor.clone())
        self.assertIsInstance(out, tuple)
        # Last output is the copy-back value == state + x.
        self.assertTrue(torch.allclose(out[-1], buf_tensor + x))

    def test_index_put_is_copyback_not_kv(self):
        """Regression: ``index_put`` has no aliasing converter, so it must fall
        into copy-back rather than being dropped in expectation of aliasing."""

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("state", torch.zeros(4, 8))

            def forward(self, x):
                self.state[torch.tensor([1, 3])] = x
                return self.state.sum()

        gm = _ep_module_decomposed(M(), (torch.ones(2, 8),))
        new_gm, lifted = lift_mutated_buffers(gm)
        self.assertEqual(len(lifted), 1)
        self.assertEqual(new_gm.meta["_copyback_mutation_buffers"], ["state"])

    def test_mixed_kv_and_copyback(self):
        """One KV buffer + one non-KV buffer: only the non-KV one is recorded."""

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("cache", torch.zeros(2, 4, 16, 8))
                self.register_buffer("state", torch.zeros(4))

            def forward(self, x_kv, x_state):
                self.cache[:, :, 3:4, :] = x_kv
                self.state.add_(x_state)
                return self.cache.sum() + self.state.sum()

        gm = _ep_module_decomposed(M(), (torch.ones(2, 4, 1, 8), torch.ones(4)))
        new_gm, lifted = lift_mutated_buffers(gm)
        self.assertEqual(len(lifted), 2)
        self.assertEqual(new_gm.meta["_copyback_mutation_buffers"], ["state"])

    def test_ineligible_index_copy_is_copyback(self):
        """Regression: an ``index_copy`` the KV converter cannot alias (here a 2-D
        cache / dim 0, not the 4-D dim=2 layout ``IKVCacheUpdateLayer`` requires)
        is lowered to a non-aliasing scatter, so its write-back must be preserved
        as copy-back rather than dropped."""

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("cache", torch.zeros(4, 8))

            def forward(self, x):
                self.cache.index_copy_(0, torch.tensor([2]), x)
                return self.cache.sum()

        gm = _ep_module_decomposed(M(), (torch.ones(1, 8),))
        new_gm, lifted = lift_mutated_buffers(gm)
        self.assertEqual(len(lifted), 1)
        self.assertEqual(new_gm.meta["_copyback_mutation_buffers"], ["cache"])

    def test_ineligible_slice_scatter_is_copyback(self):
        """Regression: a ``slice_scatter`` on a KV-shaped 4-D cache but the wrong
        axis (dim 1, not dim 2) is not IKVCacheUpdateLayer-eligible, so it is
        lowered to a non-aliasing scatter and must fall to copy-back rather than
        being dropped."""

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("cache", torch.zeros(2, 16, 4, 8))

            def forward(self, x):
                self.cache[:, 3:4, :, :] = x  # write on dim 1, not the seq dim 2
                return self.cache.sum()

        gm = _ep_module_decomposed(M(), (torch.ones(2, 1, 4, 8),))
        new_gm, lifted = lift_mutated_buffers(gm)
        self.assertEqual(len(lifted), 1)
        self.assertEqual(new_gm.meta["_copyback_mutation_buffers"], ["cache"])

    def test_torch_executed_write_is_copyback_not_kv(self):
        """A KV-shaped write whose op the caller excluded from TensorRT cannot reach a
        converter, so no IKVCacheUpdateLayer can alias it.

        It must still be lifted and copied back. Dropping the buffer instead would
        leave the copy_ with no users, and post_lowering's dead-node pass would erase
        it, silently losing the write."""

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("cache", torch.zeros(2, 4, 16, 8))

            def forward(self, x):
                self.cache[:, :, 3:4, :] = x
                return self.cache.sum()

        args = (torch.ones(2, 4, 1, 8),)
        # Without the exclusion the same write is predicted to alias inside the engine.
        gm = _ep_module_decomposed(M(), args)
        baseline_gm, baseline_lifted = lift_mutated_buffers(gm)
        self.assertEqual(len(baseline_lifted), 1)
        self.assertEqual(baseline_gm.meta["_copyback_mutation_buffers"], [])
        self.assertEqual(len(baseline_gm.meta["_predicted_kv_bindings"]), 1)

        gm = _ep_module_decomposed(M(), args)
        settings = CompilationSettings(
            torch_executed_ops={"torch.ops.aten.slice_scatter.default"}
        )
        new_gm, lifted = lift_mutated_buffers(gm, settings)
        self.assertEqual(len(lifted), 1)
        self.assertEqual(new_gm.meta["_copyback_mutation_buffers"], ["cache"])
        self.assertEqual(new_gm.meta["_predicted_kv_bindings"], [])

        # The write survives as a real graph output, so it cannot be dead-code
        # eliminated the way an orphaned copy_ would be.
        out_node = next(n for n in new_gm.graph.nodes if n.op == "output")
        self.assertEqual(len(out_node.args[0]), 2)
        erasable = [
            n
            for n in new_gm.graph.nodes
            if n is not out_node and not n.users and n.all_input_nodes
        ]
        self.assertEqual(erasable, [])


class _FakeEngine:
    """Stand-in for a compiled TRT submodule exposing an ``aliased_io`` map."""

    def __init__(self, aliased_io):
        self.aliased_io = aliased_io


class _FakeGM:
    """Stand-in for a compiled GraphModule whose children are TRT engines."""

    def __init__(self, children):
        self._children = children

    def named_children(self):
        return list(self._children.items())


class TestPredictedKvAssertion(TestCase):
    """`assert_predicted_kv_aliased` is the ground-truth backstop for the
    pre-conversion KV prediction: every write predicted to alias must actually
    appear in a compiled engine's `aliased_io`, else its write-back would be
    silently dropped."""

    @staticmethod
    def _aliased_in(gm):
        """Ground truth the way ``compile()`` assembles it, off the compiled
        submodules."""
        return aliased_input_bindings(
            getattr(sub, "aliased_io", None) for _name, sub in gm.named_children()
        )

    def test_passes_when_predicted_kv_is_aliased(self):
        gm = _FakeGM(
            {
                "_run_on_acc_0": _FakeEngine(
                    {"out_k": ("buf_k_cache", "kv_cache_update")}
                )
            }
        )
        # buf_k_cache is aliased -> no error.
        assert_predicted_kv_aliased(self._aliased_in(gm), ["buf_k_cache"])

    def test_raises_when_predicted_kv_not_aliased(self):
        # Predicted KV for buf_conv_state, but the engine aliased only buf_k_cache
        # (the converter emitted no IKVCacheUpdateLayer for conv_state) -> must
        # raise rather than silently drop the write-back.
        gm = _FakeGM(
            {
                "_run_on_acc_0": _FakeEngine(
                    {"out_k": ("buf_k_cache", "kv_cache_update")}
                )
            }
        )
        with self.assertRaises(RuntimeError):
            assert_predicted_kv_aliased(self._aliased_in(gm), ["buf_conv_state"])

    def test_aggregates_aliased_io_across_engines(self):
        gm = _FakeGM(
            {
                "_run_on_acc_0": _FakeEngine(
                    {"out_k": ("buf_k_cache", "kv_cache_update")}
                ),
                "_run_on_acc_1": _FakeEngine(
                    {"out_v": ("buf_v_cache", "kv_cache_update")}
                ),
            }
        )
        # Both predicted-KV bindings are aliased across the two engines -> no error.
        assert_predicted_kv_aliased(
            self._aliased_in(gm), ["buf_k_cache", "buf_v_cache"]
        )

    def test_noop_when_no_prediction(self):
        assert_predicted_kv_aliased(self._aliased_in(_FakeGM({})), [])

    def test_no_engines_built_does_not_raise(self):
        """With no engine built, aliased_io is empty for every prediction whatever
        the predictions were worth, so there is nothing to check and raising would
        fail a run on the strength of its own absence of evidence."""
        assert_predicted_kv_aliased(
            self._aliased_in(_FakeGM({})),
            ["buf_k_cache"],
            engines_built=False,
        )

    def test_dryrun_alone_does_not_skip(self):
        """``dryrun`` is not what the skip keys on. The engine converter accepts it,
        never acts on it and builds an engine regardless, so treating it as "no
        engines" there would turn a check the converter deliberately has no opt-out
        for into one with an opt-out kwarg."""
        with self.assertRaises(RuntimeError):
            assert_predicted_kv_aliased(
                self._aliased_in(_FakeGM({})),
                ["buf_k_cache"],
                CompilationSettings(dryrun=True),
            )

    def test_message_names_min_block_size(self):
        """The classification runs before partitioning, so the usual cause is the
        partitioner rejecting the subgraph the write landed in. Without the value in
        force and the remedy, there is nothing in the error to act on."""
        with self.assertRaises(RuntimeError) as caught:
            assert_predicted_kv_aliased(
                self._aliased_in(_FakeGM({})),
                ["buf_k_cache"],
                CompilationSettings(min_block_size=5),
            )
        message = str(caught.exception)
        self.assertIn("min_block_size (5)", message)
        self.assertIn("min_block_size=1", message)


@unittest.skipIf(not torch.cuda.is_available(), "CUDA required")
class TestCompileSeam(TestCase):
    """End-to-end coverage of the wire between the two halves of the design.

    ``lift_mutated_buffers`` classifies each write before partitioning and
    ``assert_predicted_kv_aliased`` re-checks that classification after conversion.
    Every other test in this file drives one half directly, so the two could be
    disconnected in ``_compiler.py`` -- the copy-back list never reaching
    ``trt_gm.meta``, or the cross-check never being called -- without a single
    failure. These go through ``compile()`` so that wiring is observed.
    """

    @staticmethod
    def _compile(model, args, **kwargs):
        import torch_tensorrt

        ep = torch.export.export(model, args, strict=False)
        return torch_tensorrt.dynamo.compile(
            ep,
            inputs=list(args),
            cache_built_engines=False,
            reuse_cached_engines=False,
            **kwargs,
        )

    @staticmethod
    def _kv_model(batch):
        """A model and inputs whose single-position cache write lift classifies as
        engine-aliased, so it drops the ``copy_`` and the cross-check has something
        to be right or wrong about."""

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("cache", torch.zeros(batch, 4, 16, 8).cuda())

            def forward(self, x):
                self.cache[:, :, 3:4, :] = x
                return self.cache.sum()

        return M().cuda().eval(), (torch.ones(batch, 4, 1, 8).cuda(),)

    def test_compile_threads_copyback_buffers_into_trt_gm_meta(self):
        """The copy-back list ``lift_mutated_buffers`` records has to survive
        ``compile_module`` and land on the compiled module's meta -- that is the only
        channel by which ``save()`` learns which trailing outputs to reclassify as
        BUFFER_MUTATIONs."""

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("state", torch.zeros(8).cuda())

            def forward(self, x):
                self.state.add_(x)
                return self.state.sum()

        x = torch.arange(8, dtype=torch.float32).cuda()
        trt_gm = self._compile(M().cuda().eval(), (x,), min_block_size=1)

        self.assertEqual(trt_gm.meta.get("_copyback_mutation_buffers"), ["state"])
        # The recorded name must resolve to a buffer that still exists, else the
        # ExportedProgram verifier rejects the BUFFER_MUTATION downstream.
        self.assertIn("state", dict(trt_gm.named_buffers()))

        # The copy-back value is a real trailing output carrying the post-write
        # buffer, not a stale read.
        outs = trt_gm(x)
        self.assertIsInstance(outs, (tuple, list))
        self.assertEqual(len(outs), 2)
        self.assertTrue(torch.allclose(outs[-1].float().cpu(), x.cpu()))

    def test_compile_runs_the_predicted_kv_cross_check(self):
        """``compile()`` must actually call ``assert_predicted_kv_aliased`` with the
        predictions lift made; without that call a mis-classified write silently
        loses its write-back."""
        from torch_tensorrt.dynamo import _compiler as C

        model, args = self._kv_model(1)
        calls = []
        original = C.assert_predicted_kv_aliased

        def _spy(aliased_in, bindings, settings=None, **kwargs):
            calls.append(list(bindings))
            return original(aliased_in, bindings, settings, **kwargs)

        with mock.patch.object(C, "assert_predicted_kv_aliased", _spy):
            self._compile(model, args, min_block_size=1)

        self.assertEqual(calls, [["buf_cache"]])

    def test_compile_raises_when_a_predicted_kv_write_is_not_aliased(self):
        """The cross-check is load-bearing, not decorative: a write predicted to alias
        whose node the partitioner then leaves out of every engine (here via the
        default ``min_block_size``) fails the compile instead of returning a module
        whose buffer never updates."""
        model, args = self._kv_model(2)

        with self.assertRaisesRegex(RuntimeError, "did not alias them"):
            self._compile(model, args, min_block_size=5)

    def test_compile_does_not_cross_check_a_dryrun(self):
        """A dryrun returns before any engine is built, so every prediction is absent
        from ``aliased_io`` for a reason that says nothing about the prediction.

        ``compile()`` is the only caller that knows this, so it is the only one that
        tells the check -- the check itself cannot read it off ``dryrun``, which the
        engine converter accepts while building anyway. Nothing else observes that
        ``compile()`` says so, and a run whose only job is to report would otherwise
        fail outright."""
        model, args = self._kv_model(1)

        self._compile(model, args, dryrun=True, min_block_size=1)


if __name__ == "__main__":
    run_tests()
