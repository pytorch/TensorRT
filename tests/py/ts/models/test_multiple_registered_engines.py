import copy
import importlib
import unittest
from typing import Dict

import torch
import torch_tensorrt as torchtrt
from utils import COSINE_THRESHOLD, cosine_similarity

if importlib.util.find_spec("torchvision"):
    import torchvision.models as models


@unittest.skipIf(
    not torchtrt.ENABLED_FEATURES.torchscript_frontend,
    "TorchScript Frontend is not available",
)
@unittest.skipIf(
    not importlib.util.find_spec("torchvision"), "torchvision not installed"
)
@unittest.skipIf(
    torchtrt.ENABLED_FEATURES.tensorrt_rtx,
    "TensorRT RTX does not support plugins which are required for this test",
)
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
            "ir": "ts",
        }
        rn18_trt_mod = torchtrt.compile(self.resnet18, **compile_spec)
        rn50_trt_mod = torchtrt.compile(self.resnet50, **compile_spec)

        cos_sim = cosine_similarity(
            self.resnet18(self.input1), rn18_trt_mod(self.input1)
        )
        self.assertTrue(
            cos_sim > COSINE_THRESHOLD,
            msg=f"Resnet18 TRT outputs don't match with the original model. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}",
        )

        cos_sim = cosine_similarity(
            self.resnet50(self.input1), rn50_trt_mod(self.input1)
        )
        self.assertTrue(
            cos_sim > COSINE_THRESHOLD,
            msg=f"Resnet50 TRT outputs don't match with the original model. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}",
        )


if __name__ == "__main__":
    unittest.main()
