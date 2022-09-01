import unittest
import torch_tensorrt as torchtrt
import torch
import torchvision.models as models
import copy
import timm
import custom_models as cm
from typing import Dict
from utils import cosine_similarity, COSINE_THRESHOLD

class TestModelToEngineToModel(unittest.TestCase):
    def test_multiple_engines(self):
        self.resnet18 = models.resnet18(pretrained=True).eval().to("cuda")
        self.resnet50 = models.resnet50(pretrained=True).eval().to("cuda")
        self.input1 = torch.randn((1, 3, 224, 224)).to("cuda")
        self.input2 = torch.randn((1, 3, 224, 224)).to("cuda")

        compile_spec = {
            "inputs": [
                torchtrt.Input(
                    self.input1.shape, dtype=torch.float, format=torch.contiguous_format
                )
            ],
            "device": {
                "device_type": torchtrt.DeviceType.GPU,
                "gpu_id": 0,
            },
            "enabled_precisions": {torch.float},
        }
        rn18_trt_mod = torchtrt.compile(self.resnet18, **compile_spec)
        rn50_trt_mod = torchtrt.compile(self.resnet50, **compile_spec)

        cos_sim = cosine_similarity(self.resnet18(self.input1), rn18_trt_mod(self.input1))
        self.assertTrue(cos_sim > COSINE_THRESHOLD, msg=f"Resnet18 TRT outputs don't match with the original model. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}")

        cos_sim = cosine_similarity(self.resnet50(self.input1), rn50_trt_mod(self.input1))
        self.assertTrue(cos_sim > COSINE_THRESHOLD, msg=f"Resnet50 TRT outputs don't match with the original model. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}")

if __name__ == "__main__":
    unittest.main()
