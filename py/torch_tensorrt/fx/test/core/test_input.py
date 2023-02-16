# Owner(s): ["oncall: gpu_enablement"]

import io
import os

import torch
import torch_tensorrt
from torch.testing._internal.common_utils import run_tests, TestCase
from torch_tensorrt.fx.utils import LowerPrecision


class TestInput(TestCase):
    def test_add_model(self):
        class TestModule(torch.nn.Module):
            def forward(self, x):
                return x + x

        inputs = [torch_tensorrt.Input(shape=(1, 3, 3, 4), dtype=torch.float32)]
        rand_inputs = [torch.randn((1, 3, 3, 4), dtype=torch.float32).cuda()]
        mod = TestModule().cuda().eval()
        ref_output = mod(*rand_inputs)

        trt_mod = torch_tensorrt.compile(
            mod,
            ir="fx",
            inputs=inputs,
            min_acc_module_size=1,
        )
        trt_output = trt_mod(*rand_inputs)

        torch.testing.assert_close(trt_output, ref_output, rtol=1e-04, atol=1e-04)

    def test_conv_model(self):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 6, 1, 1, 1, 1, 1, True)

            def forward(self, x):
                return self.conv(x)

        inputs = [torch_tensorrt.Input(shape=(1, 3, 32, 32), dtype=torch.float32)]
        rand_inputs = [torch.randn((1, 3, 32, 32), dtype=torch.float32).cuda()]
        mod = TestModule().cuda().eval()
        ref_output = mod(*rand_inputs)

        trt_mod = torch_tensorrt.compile(
            mod,
            ir="fx",
            inputs=inputs,
            min_acc_module_size=1,
        )
        trt_output = trt_mod(*rand_inputs)

        torch.testing.assert_close(trt_output, ref_output, rtol=1e-04, atol=1e-04)

    def test_conv_model_with_dyn_shapes(self):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 6, 1, 1, 1, 1, 1, True)

            def forward(self, x):
                return self.conv(x)

        inputs = [
            torch_tensorrt.Input(
                min_shape=(1, 3, 32, 32),
                opt_shape=(8, 3, 32, 32),
                max_shape=(16, 3, 32, 32),
                dtype=torch.float32,
            )
        ]
        rand_inputs = [torch.randn((4, 3, 32, 32), dtype=torch.float32).cuda()]
        mod = TestModule().cuda().eval()
        ref_output = mod(*rand_inputs)

        trt_mod = torch_tensorrt.compile(
            mod,
            ir="fx",
            inputs=inputs,
            min_acc_module_size=1,
        )
        trt_output = trt_mod(*rand_inputs)

        torch.testing.assert_close(trt_output, ref_output, rtol=1e-04, atol=1e-04)


if __name__ == "__main__":
    run_tests()
