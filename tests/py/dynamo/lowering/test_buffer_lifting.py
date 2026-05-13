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

import torch
from torch.export import export
from torch.testing._internal.common_utils import TestCase, run_tests
from torch_tensorrt.dynamo.lowering._buffer_lifting import (
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


if __name__ == "__main__":
    run_tests()
