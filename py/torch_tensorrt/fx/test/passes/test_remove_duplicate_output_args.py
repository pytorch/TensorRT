# Owner(s): ["oncall: gpu_enablement"]

import logging
import torch
from packaging import version

import torch.fx as fx
import torch.nn as nn

import torch_tensorrt.fx.passes.remove_duplicate_output_args as dedup
from torch_tensorrt._utils import sanitized_torch_version

from torch.testing._internal.common_utils import run_tests, TestCase

_LOGGER = logging.getLogger(__name__)


class TestFx2TrtPasses(TestCase):
    def test_remove_duplicate_output_args(self):
        class Sub(nn.Module):
            def forward(self, x):
                return (x, x)

        class Top(nn.Module):
            def __init__(self):
                super().__init__()
                self.a = Sub()

            def forward(self, x):
                a_res = self.a(x)
                return a_res[0] + a_res[1]

        class Tracer(fx.Tracer):
            def is_leaf_module(self, m, qn):
                if isinstance(m, Sub):  # don't trace into
                    return True
                return False

        top = Top()
        ttop = fx.GraphModule(top, Tracer().trace(top), "top")
        ttop.a = fx.symbolic_trace(ttop.a)

        name_to_processed_subnet = dedup.remove_duplicate_output_args(ttop, ["a"])

        ttop(1)  # run inference should work

        processed_a = name_to_processed_subnet["a"]
        *_, a_output = processed_a.module.graph.nodes
        a_output: fx.Node

        ttop_graph_actual = str(ttop.graph).strip()
        ttop_graph_expected = """
graph():
    %x : [num_users=1] = placeholder[target=x]
    %a : [num_users=2] = call_module[target=a](args = (%x,), kwargs = {})
    %getitem : [num_users=1] = call_function[target=operator.getitem](args = (%a, 0), kwargs = {})
    %getitem_1 : [num_users=1] = call_function[target=operator.getitem](args = (%a, 0), kwargs = {})
    %add : [num_users=1] = call_function[target=operator.add](args = (%getitem, %getitem_1), kwargs = {})
    return add
""".strip()

        if version.parse(sanitized_torch_version()) < version.parse(
            "2.1.0.dev20230620"
        ):
            ttop_graph_expected = ttop_graph_expected.replace("num_users", "#users")

        assert (
            ttop_graph_expected == ttop_graph_actual
        ), f"Unexpected ttop graph: {ttop_graph_actual}"

        ttop_a_graph_actual = str(ttop.a.graph).strip()
        ttop_a_graph_expected = """
graph():
    %x : [num_users=1] = placeholder[target=x]
    return (x,)
""".strip()

        if version.parse(sanitized_torch_version()) < version.parse(
            "2.1.0.dev20230620"
        ):
            ttop_a_graph_expected = ttop_a_graph_expected.replace("num_users", "#users")

        assert (
            ttop_a_graph_expected == ttop_a_graph_actual
        ), f"Unexpected ttop.a graph: {ttop_a_graph_actual}"


if __name__ == "__main__":
    run_tests()
