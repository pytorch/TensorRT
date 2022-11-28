import copy
import unittest

import torch
import torchdynamo
import torchvision
from functorch.experimental import functionalize

from torch.library import Library
from torch_tensorrt.fx.lower import compile
from torch_tensorrt.fx.tracer.dispatch_tracer.tracer import make_fx
from torch_tensorrt.fx.utils import LowerPrecision, proxytensor_trace
from torchdynamo.optimizations import backends
from torchdynamo.optimizations.normalize import normalize_ir

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

    def test_simple(self):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.relu = torch.nn.ReLU(inplace=True)

            def forward(self, x, y):
                y = y + x
                y = y.mul(x)
                y = y + x
                y = y + x
                y = y / x
                y = y + x
                y = y + x
                y = y / x
                y = y + x
                y = self.relu(y)
                return y

        mod = TestModule()
        mod = mod.cuda().half().eval()

        def f(x, y):
            return mod(x, y)

        inputs = [torch.randn(2, 5), torch.ones(2, 5)]
        inputs = [i.cuda().half() for i in inputs]
        ref_output = f(*inputs)

        mod = compile(
            mod,
            inputs,
            max_batch_size=100,
            explicit_batch_dimension=True,
            lower_precision=LowerPrecision.FP16,
            verbose_log=False,
            timing_cache_prefix="",
            save_timing_cache=False,
            cuda_graph_batch_size=-1,
            dynamic_batch=True,
            is_aten=True,
        )
        output = mod(*inputs)
        torch.testing.assert_close(output, ref_output)

    def test_resnet18_aten(self):
        mod = torchvision.models.resnet18()
        mod = mod.cuda().half().eval()

        def f(x):
            return mod(x)

        inputs = [torch.ones(32, 3, 224, 224)]
        inputs = [i.cuda().half() for i in inputs]

        aten_mod = compile(
            mod,
            inputs,
            max_batch_size=32,
            explicit_batch_dimension=True,
            lower_precision=LowerPrecision.FP16,
            verbose_log=False,
            timing_cache_prefix="",
            save_timing_cache=False,
            cuda_graph_batch_size=-1,
            dynamic_batch=False,
            is_aten=True,
        )
        aten_output = aten_mod(*inputs)
        fx_mod = compile(
            mod,
            inputs,
            max_batch_size=32,
            explicit_batch_dimension=True,
            lower_precision=LowerPrecision.FP16,
            verbose_log=False,
            timing_cache_prefix="",
            save_timing_cache=False,
            cuda_graph_batch_size=-1,
            dynamic_batch=False,
            is_aten=False,
        )
        fx_output = fx_mod(*inputs)
        # Kernel selection is tricky in TRT with big variance as shown below:
        # Mismatched elements: 30816 / 32000 (96.3%)
        # Greatest absolute difference: 0.05859375 at index (0, 499) (up to 1e-05 allowed)
        # Greatest relative difference: 3.293713681986265 at index (0, 142) (up to 0.001 allowed)
        # so we choose to use cosine similarity
        cos = torch.nn.CosineSimilarity(dim=0, eps=1e-4)
        cos_val = cos(aten_output.flatten(), fx_output.flatten())
        self.assertTrue(cos_val.cpu().numpy() > 0.999)

    def test_resnet18_dynamo(self):
        mod = torchvision.models.resnet18()
        mod = mod.cuda().half().eval()

        def f(x):
            return mod(x)

        inputs = [torch.ones(32, 3, 224, 224)]
        inputs = [i.cuda().half() for i in inputs]
        torchdynamo.reset()
        dynamo_aten_mod = torchdynamo.optimize(backends.fx2trt_compiler_fp16)(mod)
        dynamo_aten_output = dynamo_aten_mod(*inputs)

        torchdynamo.reset()

        dynamo_mod = torchdynamo.optimize(backends.fx2trt_compiler_fp16)(mod)
        dynamo_output = dynamo_mod(*inputs)

        cos = torch.nn.CosineSimilarity(dim=0, eps=1e-4)
        cos_val = cos(dynamo_output.flatten(), dynamo_aten_output.flatten())

        self.assertTrue(cos_val.cpu().numpy() > 0.999)


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
