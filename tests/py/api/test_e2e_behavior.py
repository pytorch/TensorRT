import unittest
import torch_tensorrt as torchtrt
import torch
import torchvision.models as models
import copy
from typing import Dict


class TestCompileHalf(unittest.TestCase):
    def test_compile_script_half(self):
        self.model = models.resnet18(pretrained=True).eval().to("cuda")
        self.input = torch.randn((1, 3, 224, 224)).to("cuda")
        self.scripted_model = torch.jit.script(self.model)
        self.scripted_model.half()

        compile_spec = {
            "inputs": [torchtrt.Input(shape=self.input.shape, dtype=torch.half)],
            "device": {
                "device_type": torchtrt.DeviceType.GPU,
                "gpu_id": 0,
            },
            "enabled_precisions": {torch.half},
        }

        trt_mod = torchtrt.ts.compile(self.scripted_model, **compile_spec)
        same = (
            (trt_mod(self.input.half()) - self.scripted_model(self.input.half()))
            .abs()
            .max()
        )
        torchtrt.logging.log(torchtrt.logging.Level.Debug, "Max diff: " + str(same))
        self.assertTrue(same < 3e-2)

    def test_compile_script_half_by_default(self):
        self.model = models.resnet18(pretrained=True).eval().to("cuda")
        self.input = torch.randn((1, 3, 224, 224)).to("cuda")
        self.scripted_model = torch.jit.script(self.model)
        self.scripted_model.half()

        compile_spec = {
            "inputs": [torchtrt.Input(shape=self.input.shape)],
            "device": {
                "device_type": torchtrt.DeviceType.GPU,
                "gpu_id": 0,
            },
            "enabled_precisions": {torch.float, torch.half},
        }

        trt_mod = torchtrt.ts.compile(self.scripted_model, **compile_spec)
        same = (
            (trt_mod(self.input.half()) - self.scripted_model(self.input.half()))
            .abs()
            .max()
        )
        torchtrt.logging.log(torchtrt.logging.Level.Debug, "Max diff: " + str(same))
        self.assertTrue(same < 3e-2)


class TestFallbackToTorch(unittest.TestCase):
    def test_fallback(self):
        self.model = models.resnet18(pretrained=True).eval().to("cuda")
        self.input = torch.randn((1, 3, 224, 224)).to("cuda")
        self.scripted_model = torch.jit.script(self.model)

        compile_spec = {
            "inputs": [torchtrt.Input(self.input.shape)],
            "device": {
                "device_type": torchtrt.DeviceType.GPU,
                "gpu_id": 0,
                "allow_gpu_fallback": False,
                "disable_tf32": False,
            },
            "require_full_compilation": False,
            "torch_executed_ops": ["aten::max_pool2d"],
            "min_block_size": 1,
        }

        trt_mod = torchtrt.ts.compile(self.scripted_model, **compile_spec)
        same = (trt_mod(self.input) - self.scripted_model(self.input)).abs().max()
        self.assertTrue(same < 2e-3)

    def test_module_fallback(self):
        self.model = models.resnet18(pretrained=True).eval().to("cuda")
        self.input = torch.randn((1, 3, 224, 224)).to("cuda")
        self.scripted_model = torch.jit.script(self.model)

        compile_spec = {
            "inputs": [torchtrt.Input(self.input.shape)],
            "device": {
                "device_type": torchtrt.DeviceType.GPU,
                "gpu_id": 0,
                "allow_gpu_fallback": False,
                "disable_tf32": False,
            },
            "require_full_compilation": False,
            "torch_executed_modules": ["torchvision.models.resnet.BasicBlock"],
            "min_block_size": 1,
        }

        trt_mod = torchtrt.ts.compile(self.scripted_model, **compile_spec)
        same = (trt_mod(self.input) - self.scripted_model(self.input)).abs().max()
        self.assertTrue(same < 2e-3)


class TestInputTypeDefaultsFP32Model(unittest.TestCase):
    def test_input_use_default_fp32(self):
        self.model = models.resnet18(pretrained=True).eval().to("cuda")
        self.input = torch.randn((1, 3, 224, 224)).to("cuda")

        ts_model = torch.jit.script(self.model)
        trt_mod = torchtrt.ts.compile(
            ts_model,
            inputs=[torchtrt.Input(self.input.shape)],
            enabled_precisions={torch.float, torch.half},
        )
        trt_mod(self.input)

    def test_input_respect_user_setting_fp32_weights_fp16_in(self):
        self.model = models.resnet18(pretrained=True).eval().to("cuda")
        self.input = torch.randn((1, 3, 224, 224)).to("cuda")

        ts_model = torch.jit.script(self.model)
        trt_mod = torchtrt.ts.compile(
            ts_model,
            inputs=[self.input.half()],
            require_full_compilation=True,
            enabled_precisions={torch.float, torch.half},
        )
        trt_mod(self.input.half())

    def test_input_respect_user_setting_fp32_weights_fp16_in_non_constructor(self):
        self.model = models.resnet18(pretrained=True).eval().to("cuda")
        self.input = torch.randn((1, 3, 224, 224)).to("cuda")

        ts_model = torch.jit.script(self.model)
        input_spec = torchtrt.Input(self.input.shape)
        input_spec.dtype = torch.half

        trt_mod = torchtrt.ts.compile(
            ts_model,
            inputs=[input_spec],
            require_full_compilation=True,
            enabled_precisions={torch.float, torch.half},
        )
        trt_mod(self.input.half())


class TestInputTypeDefaultsFP16Model(unittest.TestCase):
    def test_input_use_default_fp16(self):
        self.model = models.resnet18(pretrained=True).eval().to("cuda")
        self.input = torch.randn((1, 3, 224, 224)).to("cuda")

        half_mod = torch.jit.script(self.model)
        half_mod.half()

        trt_mod = torchtrt.ts.compile(
            half_mod,
            inputs=[torchtrt.Input(self.input.shape)],
            enabled_precisions={torch.float, torch.half},
        )
        trt_mod(self.input.half())

    def test_input_use_default_fp16_without_fp16_enabled(self):
        self.model = models.resnet18(pretrained=True).eval().to("cuda")
        self.input = torch.randn((1, 3, 224, 224)).to("cuda")

        half_mod = torch.jit.script(self.model)
        half_mod.half()

        trt_mod = torchtrt.ts.compile(
            half_mod, inputs=[torchtrt.Input(self.input.shape)]
        )
        trt_mod(self.input.half())

    def test_input_respect_user_setting_fp16_weights_fp32_in(self):
        self.model = models.resnet18(pretrained=True).eval().to("cuda")
        self.input = torch.randn((1, 3, 224, 224)).to("cuda")

        half_mod = torch.jit.script(self.model)
        half_mod.half()

        trt_mod = torchtrt.ts.compile(
            half_mod,
            inputs=[self.input],
            require_full_compilation=True,
            enabled_precisions={torch.float, torch.half},
        )
        trt_mod(self.input)

    def test_input_respect_user_setting_fp16_weights_fp32_in_non_constuctor(self):
        self.model = models.resnet18(pretrained=True).eval().to("cuda")
        self.input = torch.randn((1, 3, 224, 224)).to("cuda")

        half_mod = torch.jit.script(self.model)
        half_mod.half()

        input_spec = torchtrt.Input(self.input.shape)
        input_spec.dtype = torch.float

        trt_mod = torchtrt.ts.compile(
            half_mod,
            inputs=[input_spec],
            require_full_compilation=True,
            enabled_precisions={torch.float, torch.half},
        )
        trt_mod(self.input)


if __name__ == "__main__":
    unittest.main()
