import logging
import unittest

import torch
import torch_tensorrt.fx.tracer.acc_tracer.acc_tracer as acc_tracer
from torch_tensorrt.fx.passes.lower_basic_pass import (
    fix_clamp_numerical_limits_to_fp16,
)


_LOGGER: logging.Logger = logging.getLogger(__name__)


def debug_print_graph_module(mod_graph: torch.fx.GraphModule) -> None:
    """
    Helper func to print model's graph in plain and tabular format, also print code.
    """
    _LOGGER.info(mod_graph.graph)
    mod_graph.graph.print_tabular()
    _LOGGER.info(mod_graph.code)


class ClampNumericalLimitsTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_clamp_numerical_limits_to_fp16(self):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                y = torch.clamp(x + x, min=-1e8, max=1e8)
                return y

        module = TestModule()
        inputs = [torch.rand(3, 2, 1)]

        module.eval()

        # Before Opt
        before_results = module(*inputs)
        mod_traced = acc_tracer.trace(module, inputs)
        before_node_list = list(mod_traced.graph.nodes)
        clamp_node_before = [node for node in before_node_list if "clamp" in str(node)]
        min_val_before = clamp_node_before[0].kwargs["min"]
        max_val_before = clamp_node_before[0].kwargs["max"]
        _LOGGER.info("Model before opt.")
        debug_print_graph_module(mod_traced)

        # Apply Opt
        module_after_pass = fix_clamp_numerical_limits_to_fp16(mod_traced, inputs)

        # After Opt
        after_results = module_after_pass(*inputs)
        after_node_list = list(mod_traced.graph.nodes)
        clamp_node_after = [node for node in after_node_list if "clamp" in str(node)]
        min_val_after = clamp_node_after[0].kwargs["min"]
        max_val_after = clamp_node_after[0].kwargs["max"]
        _LOGGER.info("Model after opt.")
        mod_traced.recompile()
        debug_print_graph_module(mod_traced)

        # Tests
        #  * Numerics
        tol_args = {"rtol": 1e-2, "atol": 1e-2}
        torch.testing.assert_close(before_results, after_results, **tol_args)

        # graph should not change
        self.assertTrue(before_node_list == after_node_list)

        # values of clamp node changed
        self.assertTrue(min_val_before != min_val_after)
        self.assertTrue(max_val_before != max_val_after)
