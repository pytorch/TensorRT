import unittest

import torch

from torch_tensorrt.fx.lower import compile
from torch_tensorrt.fx.utils import LowerPrecision


class LongInputTest(unittest.TestCase):
    def test_long_input(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                out = x + 1
                out = out * 2
                out = out - 1
                return out

        mod = Model().cuda().eval()

        inputs = [torch.randint(-40, 40, (3, 4, 7)).cuda().long()]

        aten_mod = compile(
            mod,
            inputs,
            min_acc_module_size=3,
            explicit_batch_dimension=True,
            verbose_log=True,
            lower_precision=LowerPrecision.FP16,
            truncate_long_and_double=True,
            dynamic_batch=False,
            is_aten=True,
        )

        aten_output = aten_mod(*inputs)[0].detach().cpu()
        torch_output = mod(*inputs).detach().cpu()

        max_diff = float(torch.max(torch.abs(aten_output - torch_output)))

        self.assertAlmostEqual(
            max_diff, 0, 4, msg="Torch outputs don't match with TRT outputs"
        )


class DoubleInputTest(unittest.TestCase):
    def test_double_input(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                out = x + 1
                out = out * 2
                return torch.mean(out, dim=-1)

        mod = Model().cuda().eval()

        inputs = [torch.rand((3, 4, 1)).cuda().double()]

        aten_mod = compile(
            mod,
            inputs,
            min_acc_module_size=3,
            explicit_batch_dimension=True,
            verbose_log=True,
            lower_precision=LowerPrecision.FP32,
            truncate_long_and_double=True,
            dynamic_batch=False,
            is_aten=True,
        )

        aten_output = aten_mod(*inputs)[0].detach().cpu()
        torch_output = mod(*inputs).detach().cpu()

        max_diff = float(torch.max(torch.abs(aten_output - torch_output)))

        self.assertAlmostEqual(
            max_diff, 0, 4, msg="Torch outputs don't match with TRT outputs"
        )
