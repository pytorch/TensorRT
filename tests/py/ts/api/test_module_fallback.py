import copy
import unittest
from typing import Dict

import torch
import torch_tensorrt as torchtrt
import torchvision.models as models
from torch_tensorrt._utils import is_tensorrt_rtx
from utils import COSINE_THRESHOLD, cosine_similarity


@unittest.skipIf(
    is_tensorrt_rtx(),
    "aten::adaptive_avg_pool2d is implemented via plugins which is not supported for tensorrt_rtx",
)
class TestModuleFallback(unittest.TestCase):
    def test_fallback_resnet18(self):
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
            "torch_executed_modules": ["torchvision.models.resnet.BasicBlock"],
            "ir": "ts",
        }
        trt_mod = torchtrt.compile(self.model, **compile_spec)
        cos_sim = cosine_similarity(self.model(self.input), trt_mod(self.input))
        self.assertTrue(
            cos_sim > COSINE_THRESHOLD,
            msg=f"Resnet18 TRT outputs don't match with the original model. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}",
        )

    def test_fallback_mobilenet_v2(self):
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
            "torch_executed_modules": [
                "torchvision.models.mobilenetv2.ConvBNActivation"
            ],
            "min_block_size": 5,
            "ir": "ts",
        }
        trt_mod = torchtrt.compile(self.model, **compile_spec)
        cos_sim = cosine_similarity(self.model(self.input), trt_mod(self.input))
        self.assertTrue(
            cos_sim > COSINE_THRESHOLD,
            msg=f"Mobilenet V2 TRT outputs don't match with the original model. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}",
        )


if __name__ == "__main__":
    unittest.main()
