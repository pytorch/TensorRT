import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch.fx._lazy_graph_module import _LazyGraphModule
from torch_tensorrt.dynamo._compiler import compile_module
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


def _change_add_constant(gm: torch.fx.GraphModule, value: int) -> None:
    add = next(
        node for node in gm.graph.nodes if node.target is torch.ops.aten.add.Tensor
    )
    add.args = (add.args[0], value)


class TestSkipConversionRecompile(unittest.TestCase):
    def test_post_lowering_defers_real_recompile(self) -> None:
        gm = _exported_add_one()
        gm.meta["sentinel"] = "preserved"
        n = _count_recompile(lambda: post_lowering(gm, CompilationSettings()))
        self.assertEqual(n, 0)
        lowered = post_lowering(gm, CompilationSettings())
        self.assertIsInstance(lowered, _LazyGraphModule)
        self.assertEqual(lowered.meta["sentinel"], "preserved")
        self.assertTrue(lowered._needs_recompile())

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

        gm = post_lowering(gm, CompilationSettings())

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
        self.assertTrue(gm._needs_recompile())

    def test_clean_up_outside_defer_still_recompiles(self) -> None:
        gm = _exported_add_one()
        n = _count_recompile(lambda: clean_up_graph_after_modifications(gm))
        self.assertGreater(n, 0)

    def test_python_execution_recompiles_lazily(self) -> None:
        value = torch.ones(2, 2)
        gm = torch.export.export(_AddOne(), (value,)).module()
        _change_add_constant(gm, 2)

        gm = post_lowering(gm, CompilationSettings())
        self.assertTrue(gm._needs_recompile())
        torch.testing.assert_close(gm(value), value + 2)
        self.assertFalse(gm._needs_recompile())

    def test_compile_module_early_return_executes_lowered_graph(self) -> None:
        value = torch.ones(2, 2)
        gm = torch.export.export(_AddOne(), (value,)).module()
        _change_add_constant(gm, 2)
        gm = post_lowering(gm, CompilationSettings())

        op_support = SimpleNamespace(fallback_operators={})
        with patch(
            "torch_tensorrt.dynamo._compiler.partitioning.get_graph_converter_support_overview",
            return_value=(0, 1, op_support),
        ):
            result = compile_module(
                gm,
                [],
                settings=CompilationSettings(min_block_size=2),
            )

        self.assertIs(result, gm)
        torch.testing.assert_close(result(value), value + 2)

    def test_fast_partition_does_not_generate_parent_forward(self) -> None:
        from torch_tensorrt.dynamo.partitioning import fast_partition

        gm = post_lowering(_exported_add_one(), CompilationSettings())
        with patch.object(
            gm,
            "_real_recompile",
            wraps=gm._real_recompile,
        ) as real_recompile:
            partitioned, _ = fast_partition(
                gm,
                min_block_size=1,
                require_full_compilation=True,
                assume_full_support=True,
                skip_fusion=True,
            )
            real_recompile.assert_not_called()

        value = torch.ones(2, 2)
        torch.testing.assert_close(partitioned(value), value + 1)


if __name__ == "__main__":
    unittest.main()
