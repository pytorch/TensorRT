import copy
import unittest

import torch
import torch._dynamo as torchdynamo

import torch._dynamo.config
import torchvision
from functorch.experimental import functionalize
from torch._dynamo.optimizations import backends
from torch._dynamo.optimizations.normalize import normalize_ir

from torch.library import Library
from torch_tensorrt.dynamo.lower import compile
from torch_tensorrt.fx.tracer.dispatch_tracer.tracer import make_fx
from torch_tensorrt.dynamo.utils import LowerPrecision, proxytensor_trace

# TODO(ezyang): remove this after we properly support fake example inputs
torch._dynamo.config.DO_NOT_USE_legacy_non_fake_example_inputs = True

torch.manual_seed(0)

wrap_lib = Library("wrap", "DEF")
"""
There are two methods for setting leaf_module. leaf(op registeration) and leaf(override call_module)
Only leaf(op registeration) can work together with functionalize.
If you do not need funcitonalize, you can choose any of the leaf module methods.

Test coverage:
ProxytensorTracerTest.test_leaf_operator_reg: python_key tracer + functionalize + leaf(op registeration)
DispatchTracerTest.test_leaf_operator_reg: dispatch tracer + functionalize + leaf(op registeration)
DispatchTracerTest.test_leaf: dispatch tracer + leaf(override call_module)
DispatchTracerTest.test_non_tensor_input: dispatch tracer
DispatchTracerTest.test_reference_copy: dispatch tracer + functionalize
DispatchTracerTest.test_reference_copy_torchdynamo: dispatcher tracer + torchdynamo + functionalize
"""


class ProxytensorTracerTest(unittest.TestCase):
    def test_leaf_operator_reg(self):
        class Leaf(torch.nn.Module):
            def forward(self, x, y):
                return x + y + torch.nn.Parameter(torch.ones(5))

        leaf = Leaf()
        wrap_lib.define("wrapped_foo(Tensor x, Tensor y) -> Tensor")
        wrap_lib.impl("wrapped_foo", leaf, "CPU")

        class Bar(torch.nn.Module):
            def __init__(self):
                super(Bar, self).__init__()
                self.foo = torch.ops.wrap.wrapped_foo
                self.other = torch.nn.Parameter(torch.ones(5))

            def forward(self, x, y):
                x = self.foo(x, y)
                x = x + self.other
                return x

        mod = Bar().eval()
        inputs = [torch.ones(5), torch.ones(5)]
        gm = proxytensor_trace(mod, inputs)
        inputs_new = [torch.ones(5) + 5, torch.ones(5) + 8]
        output = gm(*inputs_new)
        ref_output = mod(*inputs_new)
        torch.testing.assert_close(output, ref_output)

    def test_resnet18_dynamo(self):
        mod = torchvision.models.resnet18()
        mod = mod.cuda().half().eval()

        inputs = [torch.ones(32, 3, 224, 224)]
        inputs = [i.cuda().half() for i in inputs]
        ref_output = mod(*inputs)

        torchdynamo.reset()
        dynamo_mod = torchdynamo.optimize(backends.fx2trt_compiler_fp16)(mod)
        dynamo_output = dynamo_mod(*inputs)
        cos_val = torch.nn.functional.cosine_similarity(
            dynamo_output.flatten(), ref_output.flatten(), dim=0, eps=1e-4
        )
        self.assertTrue(cos_val.detach().cpu().numpy() > 0.999)


class DispatchTracerTest(unittest.TestCase):
    def test_leaf_operator_reg(self):
        class Leaf(torch.nn.Module):
            def forward(self, x, y):
                return x + y + torch.nn.Parameter(torch.ones(5))

        leaf = Leaf()
        wrap_lib.define("wrapped_leaf(Tensor x, Tensor y) -> Tensor")
        wrap_lib.impl("wrapped_leaf", leaf, "CPU")

        class Bar(torch.nn.Module):
            def __init__(self):
                super(Bar, self).__init__()
                self.leaf = torch.ops.wrap.wrapped_leaf
                self.other = torch.nn.Parameter(torch.ones(5))

            def forward(self, x, y):
                x = self.leaf(x, y)
                x = x + self.other
                return x

        mod = Bar()

        def f(x, y):
            return mod(x, y)

        gm = make_fx(functionalize(f))(torch.ones(5), torch.ones(5))
        inputs = [torch.ones(5) + 5, torch.ones(5) + 8]
        output = gm(*inputs)
        ref_output = f(*inputs)
        torch.testing.assert_close(output, ref_output)
        # through the op registration method, the module is defined in a call_function
        call_function_node = None
        for node in gm.graph.nodes:
            if (
                node.op == "call_function"
                and node.target == torch.ops.wrap.wrapped_leaf
            ):
                call_function_node = node
        self.assertIsNotNone(call_function_node)

    ## The test is broken on Aug 27 as the leaf node does not work. P525693772
    # def test_leaf(self):
    #     class TestModuleLeaf(torch.nn.Module):
    #         def __init__(self):
    #             super().__init__()
    #             self.conv = torch.nn.Conv2d(3, 10, 1)
    #             self.relu = torch.nn.ReLU(inplace=True)

    #         def forward(self, x):
    #             x = self.conv(x)
    #             return self.relu(x)

    #     class TestModule(torch.nn.Module):
    #         def __init__(self):
    #             super().__init__()

    #             self.relu = torch.nn.ReLU(inplace=True)
    #             self.leaf = TestModuleLeaf()

    #         def forward(self, x):
    #             x = self.leaf(x)
    #             return self.relu(x)

    #     mod = TestModule()

    #     def f(x):
    #         return mod(x)

    #     a = torch.randn(1, 3, 1, 1)
    #     ref_output = f(a)
    #     func = make_fx(f, leaf_module_list={"test_dispatch_tracer.TestModuleLeaf"})
    #     gm = func(a)
    #     output = gm(a)
    #     torch.testing.assert_close(output, ref_output)
    #     import pdb;pdb.set_trace()
    #     # There should be a call module node in the graph.
    #     call_module_node = None
    #     for node in gm.graph.nodes:
    #         if node.op == "call_module":
    #             call_module_node = node
    #     self.assertIsNotNone(call_module_node)
    #     self.assertEqual(call_module_node.target, "TestModuleLeaf_0")

    def test_non_tensor_input(self):
        def foo(x):
            a = x["a"]
            b = x["b"]
            return a + b

        x = {"a": torch.randn(2, 2), "b": torch.randn(2, 2)}
        ref_output = foo(x)
        func = make_fx(foo)
        gm = func(x)
        output = gm(x)
        torch.testing.assert_close(output, ref_output)

    def test_reference_copy(self):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                y[:, 0] = x[:, 0]
                return y

        mod = TestModule()

        def f(x, y):
            return mod(x, y)

        a = torch.ones(2, 2) + 2
        b = torch.ones(2, 2)
        b_copy = torch.ones(2, 2)
        ref_output = f(a, b)
        gm = make_fx(functionalize(f))(a, b)
        output = gm(a, b_copy)
        torch.testing.assert_close(output, ref_output)

    def test_reference_copy_torchdynamo(self):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.relu = torch.nn.ReLU(inplace=True)

            def forward(self, x, y):
                y = y + 3
                y = self.relu(y)
                y[:, 0] = x[:, 0]
                return y

        mod = TestModule()

        def f(x, y):
            return mod(x, y)

        a = torch.ones(2, 2) + 2
        b = torch.ones(2, 2)
        inputs = [a, b]
        ref_output = f(*inputs)

        def compile_dispatch(gm, example_inputs):
            # after normalization, relu in-place is removed
            gm = normalize_ir(gm, example_inputs)
            # dispatch tracer
            nargs = len(example_inputs)

            def fake_signature(fn, nargs):
                """FX gets confused by varargs, de-confuse it"""
                argnames = ",".join(f"arg{i}" for i in range(nargs))
                return eval(f"lambda {argnames}: fn({argnames})", {"fn": fn})

            gm = make_fx(functionalize(fake_signature(gm, nargs)))(*example_inputs)
            return gm

        optimized_mod = torchdynamo.optimize(
            compile_dispatch,
            nopython=True,
        )(mod)
        output = optimized_mod(*inputs)
        torch.testing.assert_close(output, ref_output)
