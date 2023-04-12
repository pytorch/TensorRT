# Owner(s): ["oncall: gpu_enablement"]

import logging
import unittest

import torch
import torch.fx as fx
import torch.nn as nn
from torch_tensorrt.dynamo.fx_ts_compat.lower import Lowerer, LowerSetting
from torch_tensorrt.fx.passes.lower_basic_pass import replace_mutable_op

logger = logging.getLogger(__name__)


class Fx2trtLowerTests(unittest.TestCase):
    def test_fx2trt_lower(self):
        class _Mod(nn.Module):
            def forward(self, x):
                return (x, 2 * x)

        mod = _Mod()
        mod_traced = fx.symbolic_trace(mod)
        input = [torch.rand(4).cuda()]
        lower = Lowerer.create(LowerSetting())
        lower(mod_traced, input)

    def test_lower_with_batchnorm_act_rewrite(self):
        class MyBatchNorm(nn.BatchNorm2d):
            def forward(self, x):
                self._check_input_dim(x)
                return x + 1

        class TestModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.bn = MyBatchNorm(3)

            def forward(self, x):
                return self.bn(x)

        module = TestModule()
        inputs = [torch.randn(1, 3, 224, 224).cuda()]
        lower = Lowerer.create(LowerSetting(ast_rewriter_allow_list={MyBatchNorm}))
        lower(module, inputs)

    def test_lower_const_fold(self):
        class TestModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.a = nn.Parameter(torch.randn(1))

            def forward(self, x):
                return (torch.sqrt(x), self.a)

        lower = Lowerer.create(LowerSetting())
        lower(TestModule(), [torch.randn([2, 2]).cuda()])

    def test_replace_mutable_op(self):
        class TestModule(torch.nn.Module):
            def forward(self, x, y):
                xf = x.fill_(100)
                yf = y.fill_(200)
                c = torch.cat([xf, yf], dim=1)
                return c

        lower = Lowerer.create(LowerSetting())
        mod_traced = fx.symbolic_trace(TestModule())
        lower(mod_traced, [torch.randn(3, 4).cuda(), torch.randn(3, 4).cuda()])

    def test_replace_mutable_op_dont_apply(self):
        class TestModule(torch.nn.Module):
            def forward(self, x):
                s = x + 1
                t = s.fill_(5)
                p = s + t
                return p

        mod_traced = fx.symbolic_trace(TestModule())
        old_code = mod_traced.code

        transformed = replace_mutable_op(mod_traced)
        new_code = transformed.code

        # s.fill_ shouldn't have been replaced
        # because s is used later
        self.assertEqual(old_code, new_code)

    def test_replace_mutable_op_do_apply(self):
        class TestModule(torch.nn.Module):
            def forward(self, x):
                s = x + 1
                t = s.fill_(5)  # s not used afterwards
                p = x + t
                return p

        mod_traced = fx.symbolic_trace(TestModule())
        old_code = mod_traced.code

        transformed = replace_mutable_op(mod_traced)
        new_code = transformed.code

        # s.fill_ should have been replaced
        # because s is not used afterwards
        self.assertNotEqual(old_code, new_code)
