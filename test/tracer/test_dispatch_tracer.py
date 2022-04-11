import unittest

import torch
from fx2trt_oss.tracer.dispatch_tracer.tracer import make_fx

torch.manual_seed(0)


class DispatchTracerTest(unittest.TestCase):
    def test_leaf_module_list(self):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 10, 1)
                self.relu = torch.nn.ReLU()

            def forward(self, x):
                x = self.conv(x)
                return self.relu(x)

        mod = TestModule()

        def f(x):
            return mod(x)

        a = torch.randn(1, 3, 1, 1)
        ref_output = f(a)
        func = make_fx(f, leaf_module_list={"torch.nn.modules.activation.ReLU"})
        gm = func(a)
        output = gm(a)
        torch.testing.assert_close(output, ref_output)

        # There should be a call module node in the graph.
        call_module_node = None
        for node in gm.graph.nodes:
            if node.op == "call_module":
                call_module_node = node
        self.assertIsNotNone(call_module_node)
        self.assertEqual(call_module_node.target, "ReLU_0")

    def test_non_tensor_input(self):
        def foo(x):
            a = x["a"]
            b = x["b"]
            return a + b

        x = {"a": torch.randn(1), "b": torch.randn(1)}
        ref_output = foo(x)
        func = make_fx(foo)
        gm = func(x)
        output = gm(x)
        torch.testing.assert_close(output, ref_output)
