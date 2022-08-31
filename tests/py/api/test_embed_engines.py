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
    def test_resnet50(self):
        self.model = models.resnet50(pretrained=True).eval().to("cuda")
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
        }

        self.scripted_model = torch.jit.script(self.model)
        trt_engine = torchtrt.ts.convert_method_to_trt_engine(self.scripted_model, "forward", **compile_spec)
        trt_mod = torchtrt.ts.embed_engine_in_new_module(trt_engine)
        cos_sim = cosine_similarity(self.model(self.input), trt_mod(self.input))
        self.assertTrue(cos_sim > COSINE_THRESHOLD, msg=f"Resnet50 TRT outputs don't match with the original model. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}")

    def test_efficientnet_b0(self):
        self.model = timm.create_model("efficientnet_b0", pretrained=True).eval().to("cuda")
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
        }

        self.scripted_model = torch.jit.script(self.model)
        trt_engine = torchtrt.ts.convert_method_to_trt_engine(self.scripted_model, "forward", **compile_spec)
        trt_mod = torchtrt.ts.embed_engine_in_new_module(trt_engine)

        cos_sim = cosine_similarity(self.model(self.input), trt_mod(self.input))
        self.assertTrue(cos_sim > COSINE_THRESHOLD, msg=f"EfficientNet-B0 TRT outputs don't match with the original model. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}")

if __name__ == "__main__":
    unittest.main()
