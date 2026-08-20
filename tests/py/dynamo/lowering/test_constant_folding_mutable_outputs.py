# Regression tests for https://github.com/pytorch/TensorRT/issues/4466
#
# Constant folding may replace input-independent factories (e.g. torch.zeros)
# with persistent module attributes. If those are returned across a graph break
# and mutated in eager code, later calls must not observe that mutation.

import torch
import torch_tensorrt  # noqa: F401  # Registers the "tensorrt" backend
from torch.fx import Graph, GraphModule
from torch.testing._internal.common_utils import TestCase, run_tests
from torch_tensorrt.dynamo._settings import CompilationSettings
from torch_tensorrt.dynamo.lowering.passes.constant_folding import constant_fold
from torch_tensorrt.dynamo.lowering.passes.reset_folded_constructors import (
    reset_folded_constructors,
)


@torch._dynamo.disable
def mutate_accumulators(values, weight, sample):
    # Runs eagerly after the graph break.
    values += sample
    weight += 1
    return values / weight


def accumulate_with_fresh_state(sample):
    # Input-independent factories; constant folding replaces these with
    # persistent module attributes.
    values = torch.zeros((8,), dtype=torch.float32, device="cuda")
    weight = torch.zeros((8,), dtype=torch.float32, device="cuda")
    return mutate_accumulators(values, weight, sample)


class TestFoldedConstantGraphBreak(TestCase):
    def tearDown(self):
        torch._dynamo.reset()

    def test_state_is_fresh_across_calls(self):
        """End-to-end repro from issue #4466."""
        compiled = torch.compile(
            accumulate_with_fresh_state,
            backend="tensorrt",
            dynamic=False,
            options={"min_block_size": 1},
        )

        first = compiled(torch.ones(8, device="cuda"))
        second = compiled(torch.full((8,), 3.0, device="cuda"))

        # Eager semantics: each call starts from fresh zeros.
        # Before the fix, second would be 2.0 from reused mutated state.
        torch.testing.assert_close(first, torch.ones_like(first))
        torch.testing.assert_close(second, torch.full_like(second, 3.0))

    def test_folded_constructor_outputs_are_reset(self):
        """Folded constants returned as graph outputs must be cloned."""
        g = Graph()
        with g.inserting_after():
            values = g.call_function(
                torch.ops.aten.zeros.default,
                args=((8,),),
                kwargs={"dtype": torch.float32},
            )
            weight = g.call_function(
                torch.ops.aten.zeros.default,
                args=((8,),),
                kwargs={"dtype": torch.float32},
            )
            g.output((values, weight))

        gm = GraphModule(torch.nn.Module(), g)
        gm = constant_fold(gm, CompilationSettings())
        gm = reset_folded_constructors(gm, CompilationSettings())

        # Outputs should be clones of the folded constants, not the attrs themselves.
        output_node = next(n for n in gm.graph.nodes if n.op == "output")
        outs = output_node.args[0]
        self.assertEqual(len(outs), 2)
        for out in outs:
            self.assertEqual(out.op, "call_function")
            self.assertIn(out.target, (torch.clone, torch.ops.aten.clone.default))

        out0, out1 = gm()
        frozen = [
            getattr(gm, name)
            for name in dir(gm)
            if name.startswith("_frozen_param")
            and isinstance(getattr(gm, name), torch.Tensor)
        ]
        self.assertGreaterEqual(len(frozen), 2)

        # Mutating returned tensors must not change stored folded constants.
        out0.add_(1)
        out1.add_(1)
        for t in frozen:
            self.assertTrue(torch.equal(t.detach().cpu(), torch.zeros(8)))

    def test_repeated_folded_outputs_preserve_alias(self):
        """Same folded value returned twice must still share storage (eager semantics)."""
        g = Graph()
        with g.inserting_after():
            values = g.call_function(
                torch.ops.aten.zeros.default,
                args=((8,),),
                kwargs={"dtype": torch.float32},
            )
            g.output((values, values))  # same value twice

        gm = GraphModule(torch.nn.Module(), g)
        gm = constant_fold(gm, CompilationSettings())
        gm = reset_folded_constructors(gm, CompilationSettings())

        output_node = next(n for n in gm.graph.nodes if n.op == "output")
        out0, out1 = output_node.args[0]
        # One clone node reused for both outputs.
        self.assertIs(out0, out1)
        self.assertEqual(out0.op, "call_function")
        self.assertIn(out0.target, (torch.clone, torch.ops.aten.clone.default))

        a, b = gm()
        self.assertEqual(a.data_ptr(), b.data_ptr())
        a.add_(1)
        self.assertTrue(torch.equal(a, b))  # mutation shared across aliases

        frozen = [
            getattr(gm, name)
            for name in dir(gm)
            if name.startswith("_frozen_param")
            and isinstance(getattr(gm, name), torch.Tensor)
        ]
        self.assertGreaterEqual(len(frozen), 1)
        for t in frozen:
            self.assertTrue(torch.equal(t.detach().cpu(), torch.zeros(8)))

    def test_post_partition_frozen_output_is_reset(self):
        """A frozen value exposed as a subgraph output is cloned by the late pass."""
        root = torch.nn.Module()
        root.register_parameter(
            "_frozen_param0",
            torch.nn.Parameter(torch.zeros(8), requires_grad=False),
        )
        g = Graph()
        frozen = g.get_attr("_frozen_param0")
        g.output({"first": frozen, "nested": (frozen,)})

        gm = GraphModule(root, g)
        gm = reset_folded_constructors(gm, CompilationSettings())

        first = gm()
        self.assertEqual(first["first"].data_ptr(), first["nested"][0].data_ptr())
        first["first"].add_(1)
        self.assertTrue(torch.equal(first["first"], first["nested"][0]))
        self.assertTrue(torch.equal(gm._frozen_param0, torch.zeros(8)))

        second = gm()
        self.assertTrue(torch.equal(second["first"], torch.zeros(8)))

    def test_user_owned_output_is_not_reset(self):
        """Graph inputs remain caller-owned and retain copy-by-reference semantics."""
        g = Graph()
        value = g.placeholder("value")
        g.output(value)

        gm = GraphModule(torch.nn.Module(), g)
        gm = reset_folded_constructors(gm, CompilationSettings())

        supplied = torch.zeros(8)
        returned = gm(supplied)
        self.assertEqual(returned.data_ptr(), supplied.data_ptr())
        returned.add_(1)
        self.assertTrue(torch.equal(supplied, torch.ones(8)))


if __name__ == "__main__":
    run_tests()
