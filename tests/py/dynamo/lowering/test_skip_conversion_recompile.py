import unittest
from unittest.mock import patch

import torch
from torch_tensorrt.dynamo._settings import CompilationSettings
from torch_tensorrt.dynamo.lowering import post_lowering
from torch_tensorrt.dynamo.lowering.passes.pass_utils import (
    clean_up_graph_after_modifications,
)


def _count_recompile(fn: object) -> int:
    calls = {"n": 0}
    orig = torch.fx.GraphModule.recompile

    def counting(self: torch.fx.GraphModule) -> None:
        calls["n"] += 1
        return orig(self)

    with patch.object(torch.fx.GraphModule, "recompile", counting):
        fn()
    return calls["n"]


class _AddOne(torch.nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + 1


def _exported_add_one() -> torch.fx.GraphModule:
    return torch.export.export(_AddOne(), (torch.ones(2, 2),)).module()


class TestSkipConversionRecompile(unittest.TestCase):
    def test_post_lowering_skips_recompile(self) -> None:
        gm = _exported_add_one()
        n = _count_recompile(lambda: post_lowering(gm, CompilationSettings()))
        self.assertEqual(n, 0)

    def test_post_lowering_still_eliminates_dead_code(self) -> None:
        gm = _exported_add_one()
        output = next(n for n in reversed(list(gm.graph.nodes)) if n.op == "output")
        inp = next(n for n in gm.graph.nodes if n.op == "placeholder")
        with gm.graph.inserting_before(output):
            gm.graph.call_function(torch.ops.aten.mul.Tensor, args=(inp, inp))
        mul_before = sum(
            1 for n in gm.graph.nodes if n.target is torch.ops.aten.mul.Tensor
        )
        self.assertEqual(mul_before, 1)

        post_lowering(gm, CompilationSettings())

        mul_after = sum(
            1 for n in gm.graph.nodes if n.target is torch.ops.aten.mul.Tensor
        )
        self.assertEqual(mul_after, 0)

    def test_interpreter_runs_without_recompile(self) -> None:
        value = torch.ones(2, 2)
        gm = torch.export.export(_AddOne(), (value,)).module()
        gm = post_lowering(gm, CompilationSettings())
        out = torch.fx.Interpreter(gm).run(value)
        torch.testing.assert_close(out, value + 1)

    def test_clean_up_outside_defer_still_recompiles(self) -> None:
        gm = _exported_add_one()
        n = _count_recompile(lambda: clean_up_graph_after_modifications(gm))
        self.assertGreater(n, 0)

    def test_post_lowering_recompile_true_rebuilds_forward(self) -> None:
        value = torch.ones(2, 2)
        gm = torch.export.export(_AddOne(), (value,)).module()
        n = _count_recompile(
            lambda: post_lowering(gm, CompilationSettings(), recompile=True)
        )
        self.assertGreater(n, 0)
        torch.testing.assert_close(gm(value), value + 1)

    def test_fast_partition_does_not_recompile_input_module(self) -> None:
        from torch_tensorrt.dynamo.partitioning import fast_partition

        gm = _exported_add_one()
        calls = {"n": 0}
        orig = gm.recompile

        def counting(*args: object, **kwargs: object) -> None:
            calls["n"] += 1
            return orig(*args, **kwargs)

        gm.recompile = counting  # type: ignore[method-assign]
        partitioned, _ = fast_partition(
            gm,
            min_block_size=1,
            require_full_compilation=True,
            assume_full_support=True,
            skip_fusion=True,
        )
        self.assertEqual(calls["n"], 0)
        value = torch.ones(2, 2)
        torch.testing.assert_close(partitioned(value), value + 1)


if __name__ == "__main__":
    unittest.main()
