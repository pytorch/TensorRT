import copy
import unittest
from typing import Dict

import torch
import torch_tensorrt as torchtrt
import torchvision.models as models
from utils import COSINE_THRESHOLD, cosine_similarity


class TestCompile(unittest.TestCase):
    def test_compile_traced(self):
        self.model = models.vgg16(pretrained=True).eval().to("cuda")
        self.input = torch.randn((1, 3, 224, 224)).to("cuda")
        self.traced_model = torch.jit.trace(self.model, [self.input])

        compile_spec = {
            "inputs": [
                torchtrt.Input(
                    self.input.shape, dtype=torch.float, format=torch.contiguous_format
                )
            ],
            "device": {
                "device_type": torchtrt.DeviceType.GPU,
                "gpu_id": 0,
            },
            "enabled_precisions": {torch.float},
        }

        trt_mod = torchtrt.ts.compile(self.traced_model, **compile_spec)
        cos_sim = cosine_similarity(self.model(self.input), trt_mod(self.input))
        self.assertTrue(
            cos_sim > COSINE_THRESHOLD,
            msg=f"VGG16 TRT outputs don't match with the original model. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}",
        )

    def test_compile_script(self):
        self.model = models.vgg16(pretrained=True).eval().to("cuda")
        self.input = torch.randn((1, 3, 224, 224)).to("cuda")
        self.scripted_model = torch.jit.script(self.model)
        with torch.no_grad():
            trt_mod = torchtrt.ts.compile(
                self.scripted_model,
                inputs=[self.input],
                device=torchtrt.Device(gpu_id=0),
                enabled_precisions={torch.float},
            )
            cos_sim = cosine_similarity(self.model(self.input), trt_mod(self.input))
            self.assertTrue(
                cos_sim > COSINE_THRESHOLD,
                msg=f"VGG16 TRT outputs don't match with the original model. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}",
            )

    def test_compile_global(self):
        self.model = models.vgg16(pretrained=True).eval().to("cuda")
        self.input = torch.randn((1, 3, 224, 224)).to("cuda")
        self.scripted_model = torch.jit.script(self.model)
        trt_mod = torchtrt.compile(
            self.scripted_model,
            inputs=[self.input],
            device=torchtrt.Device(gpu_id=0),
            enabled_precisions={torch.float},
        )
        cos_sim = cosine_similarity(self.model(self.input), trt_mod(self.input))
        self.assertTrue(
            cos_sim > COSINE_THRESHOLD,
            msg=f"VGG16 TRT outputs don't match with the original model. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}",
        )

    def test_from_torch_tensor(self):
        self.model = models.vgg16(pretrained=True).eval().to("cuda")
        self.input = torch.randn((1, 3, 224, 224)).to("cuda")
        self.traced_model = torch.jit.trace(self.model, [self.input])
        compile_spec = {
            "inputs": [self.input],
            "device": {
                "device_type": torchtrt.DeviceType.GPU,
                "gpu_id": 0,
            },
            "enabled_precisions": {torch.float},
        }

        trt_mod = torchtrt.ts.compile(self.traced_model, **compile_spec)
        cos_sim = cosine_similarity(self.model(self.input), trt_mod(self.input))
        self.assertTrue(
            cos_sim > COSINE_THRESHOLD,
            msg=f"VGG16 TRT outputs don't match with the original model. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}",
        )

    def test_device(self):
        self.model = models.vgg16(pretrained=True).eval().to("cuda")
        self.input = torch.randn((1, 3, 224, 224)).to("cuda")
        self.traced_model = torch.jit.trace(self.model, [self.input])
        compile_spec = {
            "inputs": [self.input],
            "device": torchtrt.Device("gpu:0"),
            "enabled_precisions": {torch.float},
        }

        trt_mod = torchtrt.ts.compile(self.traced_model, **compile_spec)
        cos_sim = cosine_similarity(self.model(self.input), trt_mod(self.input))
        self.assertTrue(
            cos_sim > COSINE_THRESHOLD,
            msg=f"VGG16 TRT outputs don't match with the original model. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}",
        )

    def test_default_device(self):
        self.model = models.vgg16(pretrained=True).eval().to("cuda")
        self.input = torch.randn((1, 3, 224, 224)).to("cuda")
        self.traced_model = torch.jit.trace(self.model, [self.input])
        compile_spec = {"inputs": [self.input], "enabled_precisions": {torch.float}}

        trt_mod = torchtrt.ts.compile(self.traced_model, **compile_spec)
        cos_sim = cosine_similarity(self.model(self.input), trt_mod(self.input))
        self.assertTrue(
            cos_sim > COSINE_THRESHOLD,
            msg=f"VGG16 TRT outputs don't match with the original model. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}",
        )


@unittest.skipIf(
    torchtrt.ENABLED_FEATURES.tensorrt_rtx,
    "aten::adaptive_avg_pool2d is implemented via plugins which is not supported for tensorrt_rtx",
)
class TestCheckMethodOpSupport(unittest.TestCase):
    def test_check_support(self):
        module = models.alexnet(pretrained=True).eval().to("cuda")
        self.module = torch.jit.trace(module, torch.ones((1, 3, 224, 224)).to("cuda"))

        self.assertTrue(torchtrt.ts.check_method_op_support(self.module, "forward"))


class TestModuleIdentification(unittest.TestCase):
    def test_module_type(self):
        nn_module = models.alexnet(pretrained=True).eval().to("cuda")
        ts_module = torch.jit.trace(nn_module, torch.ones([1, 3, 224, 224]).to("cuda"))
        fx_module = torch.fx.symbolic_trace(nn_module)

        self.assertEqual(
            torchtrt._compile._parse_module_type(nn_module),
            torchtrt._compile._ModuleType.nn,
        )
        self.assertEqual(
            torchtrt._compile._parse_module_type(ts_module),
            torchtrt._compile._ModuleType.ts,
        )
        if not torchtrt.ENABLED_FEATURES.tensorrt_rtx:
            self.assertEqual(
                torchtrt._compile._parse_module_type(fx_module),
                torchtrt._compile._ModuleType.fx,
            )


if __name__ == "__main__":
    unittest.main()
