# Owner(s): ["oncall: gpu_enablement"]

import logging
import unittest

import torch
import torch.fx as fx
import torch.nn as nn
from torch_tensorrt.fx.lower import Lowerer, LowerSetting

logger = logging.getLogger(__name__)


class Fx2trtLowerTests(unittest.TestCase):
    def test_fx2trt_lower(self):
        class _Mod(nn.Module):
            def forward(self, x):
                return (x, 2 * x)

        mod = _Mod()
        mod_traced = fx.symbolic_trace(mod)
        input = [torch.rand(4)]
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
        inputs = [torch.randn(1, 3, 224, 224)]
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
        lower(TestModule(), [torch.randn([2, 2])])
