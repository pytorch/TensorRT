import unittest
import torch_tensorrt as torchtrt
import torch
import torchvision.models as models
import copy
from typing import Dict

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
