import logging
import unittest
from collections import Counter
from typing import Callable, Dict, List

import torch
import torch_tensorrt.fx.tracer.acc_tracer.acc_ops as acc_ops
import torch_tensorrt.fx.tracer.acc_tracer.acc_tracer as acc_tracer
from torch_tensorrt.fx.passes.graph_opts import common_subexpression_elimination


_LOGGER: logging.Logger = logging.getLogger(__name__)


def debug_print_graph_module(mod_graph: torch.fx.GraphModule) -> None:
    """
    Helper func to print model's graph in plain and tabular format, also print code.
    """
    _LOGGER.info(mod_graph.graph)
    mod_graph.graph.print_tabular()
    _LOGGER.info(mod_graph.code)


@torch.fx.wrap
def _test_op(keys, value):
    return value


class GraphOptsTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def _test_opt_with_module(
        self,
        module: torch.nn.Module,
        inputs: List,
        opt: Callable,
        should_change_graph: bool,
        deleted_ops: Dict = None,
        created_ops: Dict = None,
        rtol: float = None,
        atol: float = None,
    ):
        assert should_change_graph or not bool(deleted_ops or created_ops)
        deleted_ops = deleted_ops or {}
        created_ops = created_ops or {}
        module.eval()

        # Before Opt
        before_results = module(*inputs)
        mod_traced = acc_tracer.trace(module, inputs)
        before_node_list = list(mod_traced.graph.nodes)
        _LOGGER.info("Model before opt.")
        debug_print_graph_module(mod_traced)

        # Apply Opt
        graph_changed = bool(opt(mod_traced))

        # After Opt
        after_results = mod_traced(*inputs)
        after_node_list = list(mod_traced.graph.nodes)
        _LOGGER.info("Model after opt.")
        mod_traced.recompile()
        debug_print_graph_module(mod_traced)

        # Tests
        #  * Numerics
        tol_args = {}
        if rtol is not None:
            tol_args["rtol"] = rtol
        if atol is not None:
            tol_args["atol"] = atol
        torch.testing.assert_close(before_results, after_results, **tol_args)

        #  * opt changes graph
        self.assertEqual(graph_changed, before_node_list != after_node_list)
        self.assertEqual(should_change_graph, graph_changed)

        # * modified nodes
        before_node_set = set(before_node_list)
        after_node_set = set(after_node_list)
        self.assertEqual(
            dict(Counter([node.target for node in before_node_set - after_node_set])),
            deleted_ops,
        )
        self.assertEqual(
            dict(Counter([node.target for node in after_node_set - before_node_set])),
            created_ops,
        )

        return mod_traced

    def test_common_subexpression_elimination(self):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                xx = x + x
                xx2 = x + x
                return xx * xx2 - x

        self._test_opt_with_module(
            module=TestModule(),
            inputs=[torch.rand(3, 2, 1)],
            opt=common_subexpression_elimination,
            should_change_graph=True,
            deleted_ops={acc_ops.add: 1},
        )

    def test_common_subexpression_elimination2(self):
        class TestModule2(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return x + x

        self._test_opt_with_module(
            module=TestModule2(),
            inputs=[torch.rand(3, 2, 1)],
            opt=common_subexpression_elimination,
            should_change_graph=False,
        )

    def test_common_subexpression_elimination3(self):
        class TestModule3(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, a, b, c):
                x = a * b
                y = b - c
                z = a * b
                xy = x + y
                zy = z + y
                return xy - zy

        self._test_opt_with_module(
            module=TestModule3(),
            inputs=[
                torch.rand(3, 2, 1),
                torch.rand(3, 2, 1),
                torch.rand(3, 2, 1),
            ],
            opt=common_subexpression_elimination,
            should_change_graph=True,
            deleted_ops={acc_ops.add: 1, acc_ops.mul: 1},
        )

    def test_common_subexpression_elimination4(self):
        class TestModule3(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, a, b, c):
                x = torch.cat([a, b, c])
                y = torch.cat([a, b, c])
                z = torch.cat([c, b, a])
                return x + y + z

        self._test_opt_with_module(
            module=TestModule3(),
            inputs=[
                torch.rand(3, 2, 1),
                torch.rand(3, 2, 1),
                torch.rand(3, 2, 1),
            ],
            opt=common_subexpression_elimination,
            should_change_graph=True,
            deleted_ops={acc_ops.cat: 1},
        )

    def test_common_subexpression_elimination_string_arg(self):
        class TestModule(torch.nn.Module):
            def forward(self, a):
                x = _test_op(["foo", "bar"], a)
                return x

        self._test_opt_with_module(
            module=TestModule(),
            inputs=[
                torch.rand(3, 2, 1),
            ],
            opt=common_subexpression_elimination,
            should_change_graph=False,
        )
