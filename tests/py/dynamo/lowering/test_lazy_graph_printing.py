import io
import logging
import re
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch_tensorrt.dynamo._settings import CompilationSettings
from torch_tensorrt.dynamo.lowering.passes.constant_folding import constant_fold

# Error-path messages and a few complex-only passes may stringify the graph;
# debug logs on the compile hot path must not.
_ALLOW_EAGER_GRAPH_STR = {
    "conversion/truncate_double.py",
    "lowering/passes/_modify_reshape_complex_nodes.py",
}

_EAGER_GRAPH_LOG = re.compile(
    r"\+\s*str\([^)]*\.graph\)"
    r"|f[\"'][^\"']*\{[^}]*\.graph\}"
    r"|logger\.(?:debug|info|warning)\([^)]*str\([^)]*\.graph\)",
    re.MULTILINE,
)


def _dynamo_root() -> Path:
    import torch_tensorrt.dynamo as dynamo

    return Path(dynamo.__file__).resolve().parent


class TestLazyGraphPrinting(unittest.TestCase):
    def test_dynamo_debug_logs_do_not_eagerly_stringify_graphs(self) -> None:
        violations: list[str] = []
        for path in _dynamo_root().rglob("*.py"):
            rel = path.relative_to(_dynamo_root()).as_posix()
            if rel in _ALLOW_EAGER_GRAPH_STR:
                continue
            text = path.read_text(encoding="utf-8")
            if _EAGER_GRAPH_LOG.search(text):
                violations.append(rel)
        self.assertEqual(
            violations,
            [],
            "logger.debug/info must pass gm.graph as a %%s arg so "
            "Graph.__str__ is skipped at INFO: " + ", ".join(violations),
        )

    def test_constant_fold_does_not_stringify_graph_at_info(self) -> None:
        gm = torch.export.export(_AddOne(), (torch.ones(2, 2),)).module()
        logger = logging.getLogger(
            "torch_tensorrt.dynamo.lowering.passes.constant_folding"
        )
        calls = {"n": 0}
        orig = torch.fx.Graph.__str__

        def counting(self: torch.fx.Graph) -> str:
            calls["n"] += 1
            return orig(self)

        buf = io.StringIO()
        handler = logging.StreamHandler(buf)
        handler.setLevel(logging.DEBUG)
        old_level = logger.level
        logger.setLevel(logging.INFO)
        logger.addHandler(handler)
        try:
            with patch.object(torch.fx.Graph, "__str__", counting):
                constant_fold(gm, CompilationSettings())
            self.assertEqual(calls["n"], 0)
        finally:
            logger.removeHandler(handler)
            logger.setLevel(old_level)

    def test_constant_fold_stringifies_graph_at_debug(self) -> None:
        gm = torch.export.export(_AddOne(), (torch.ones(2, 2),)).module()
        logger = logging.getLogger(
            "torch_tensorrt.dynamo.lowering.passes.constant_folding"
        )
        calls = {"n": 0}
        orig = torch.fx.Graph.__str__

        def counting(self: torch.fx.Graph) -> str:
            calls["n"] += 1
            return orig(self)

        buf = io.StringIO()
        handler = logging.StreamHandler(buf)
        handler.setLevel(logging.DEBUG)
        old_level = logger.level
        old_propagate = logger.propagate
        logger.setLevel(logging.DEBUG)
        logger.propagate = False
        logger.addHandler(handler)
        try:
            with patch.object(torch.fx.Graph, "__str__", counting):
                constant_fold(gm, CompilationSettings())
            self.assertGreater(calls["n"], 0)
            self.assertIn("Graph after constant folding", buf.getvalue())
        finally:
            logger.removeHandler(handler)
            logger.setLevel(old_level)
            logger.propagate = old_propagate


class _AddOne(torch.nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + 1


if __name__ == "__main__":
    unittest.main()
