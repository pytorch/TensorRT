import copy
import importlib
import unittest
from typing import Dict

import torch
import torch_tensorrt as torchtrt
from utils import COSINE_THRESHOLD, cosine_similarity

if importlib.util.find_spec("torchvision"):
    import torchvision.models as models

if importlib.util.find_spec("timm"):
    import timm


@unittest.skipIf(
    not torchtrt.ENABLED_FEATURES.torchscript_frontend,
    "TorchScript Frontend is not available",
)
@unittest.skipIf(
    not importlib.util.find_spec("torchvision"), "torchvision not installed"
)
class TestModels(unittest.TestCase):
    def test_resnet18(self):
        self.model = models.resnet18(pretrained=True).eval().to("cuda")
        self.input = torch.randn((1, 3, 224, 224)).to("cuda")

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
            "ir": "ts",
        }

        trt_mod = torchtrt.compile(self.model, **compile_spec)
        cos_sim = cosine_similarity(self.model(self.input), trt_mod(self.input))
        self.assertTrue(
            cos_sim > COSINE_THRESHOLD,
            msg=f"Resnet50 TRT outputs don't match with the original model. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}",
        )

    @unittest.skipIf(
        not importlib.util.find_spec("torchvision"), "torchvision not installed"
    )
    def test_mobilenet_v2(self):
        self.model = models.mobilenet_v2(pretrained=True).eval().to("cuda")
        self.input = torch.randn((1, 3, 224, 224)).to("cuda")

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
            "ir": "ts",
        }

        trt_mod = torchtrt.compile(self.model, **compile_spec)
        cos_sim = cosine_similarity(self.model(self.input), trt_mod(self.input))
        self.assertTrue(
            cos_sim > COSINE_THRESHOLD,
            msg=f"Mobilenet v2 TRT outputs don't match with the original model. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}",
        )

    @unittest.skipIf(not importlib.util.find_spec("timm"), "timm not installed")
    def test_efficientnet_b0(self):
        self.model = (
            timm.create_model("efficientnet_b0", pretrained=True).eval().to("cuda")
        )
        self.input = torch.randn((1, 3, 224, 224)).to("cuda")

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
            "ir": "ts",
        }

        trt_mod = torchtrt.compile(self.model, **compile_spec)
        cos_sim = cosine_similarity(self.model(self.input), trt_mod(self.input))
        self.assertTrue(
            cos_sim > COSINE_THRESHOLD,
            msg=f"EfficientNet-B0 TRT outputs don't match with the original model. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}",
        )

    @unittest.skipIf(
        not importlib.util.find_spec("torchvision"), "torchvision not installed"
    )
    def test_resnet18_half(self):
        self.model = models.resnet18(pretrained=True).eval().to("cuda")
        self.input = torch.randn((1, 3, 224, 224)).to("cuda")
        self.scripted_model = torch.jit.script(self.model)
        self.scripted_model.half()

        compile_spec = {
            "inputs": [
                torchtrt.Input(
                    self.input.shape, dtype=torch.half, format=torch.contiguous_format
                )
            ],
            "device": {
                "device_type": torchtrt.DeviceType.GPU,
                "gpu_id": 0,
            },
            "enabled_precisions": {torch.half},
            "ir": "ts",
        }

        trt_mod = torchtrt.compile(self.scripted_model, **compile_spec)
        cos_sim = cosine_similarity(
            self.model.half()(self.input.half()), trt_mod(self.input.half())
        )
        self.assertTrue(
            cos_sim > COSINE_THRESHOLD,
            msg=f"Resnet50 Half TRT outputs don't match with the original model. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}",
        )

    def test_aten_unbind_dynamic(self):
        class ATenUnbindDynamic(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x):
                x1, x2, x3 = x.unbind(1)
                y = torch.cat([x1, x2, x3], dim=0)
                return y

        self.model = ATenUnbindDynamic().eval().to("cuda")
        self.input = torch.randn((5, 3, 7, 64)).to("cuda")
        self.scripted_model = torch.jit.script(self.model)

        compile_spec = {
            "inputs": [
                torchtrt.Input(
                    min_shape=[1, 3, 1, 64],
                    opt_shape=[5, 3, 32, 64],
                    max_shape=[10, 3, 64, 64],
                    dtype=torch.float,
                    format=torch.contiguous_format,
                )
            ],
            "device": {
                "device_type": torchtrt.DeviceType.GPU,
                "gpu_id": 0,
            },
            "enabled_precisions": {torch.float},
            "ir": "ts",
        }

        trt_mod = torchtrt.compile(self.scripted_model, **compile_spec)
        cos_sim = cosine_similarity(self.model(self.input), trt_mod(self.input))
        self.assertTrue(
            cos_sim > COSINE_THRESHOLD,
            msg=f"ATen Unbind Dynamic TRT outputs don't match with the original model. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}",
        )


if __name__ == "__main__":
    unittest.main()
