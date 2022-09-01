import unittest
import torch_tensorrt as torchtrt
import torch
import torchvision.models as models
import copy
import timm
import custom_models as cm
from typing import Dict
from utils import cosine_similarity, COSINE_THRESHOLD


class TestModels(unittest.TestCase):
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

        trt_mod = torchtrt.compile(self.model, **compile_spec)
        cos_sim = cosine_similarity(self.model(self.input), trt_mod(self.input))
        self.assertTrue(
            cos_sim > COSINE_THRESHOLD,
            msg=f"Resnet50 TRT outputs don't match with the original model. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}",
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
        }

        trt_mod = torchtrt.compile(self.model, **compile_spec)
        cos_sim = cosine_similarity(self.model(self.input), trt_mod(self.input))
        self.assertTrue(
            cos_sim > COSINE_THRESHOLD,
            msg=f"Mobilenet v2 TRT outputs don't match with the original model. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}",
        )

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
        }

        trt_mod = torchtrt.compile(self.model, **compile_spec)
        cos_sim = cosine_similarity(self.model(self.input), trt_mod(self.input))
        self.assertTrue(
            cos_sim > COSINE_THRESHOLD,
            msg=f"EfficientNet-B0 TRT outputs don't match with the original model. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}",
        )

    def test_bert_base_uncased(self):
        self.model = cm.BertModule().cuda()
        self.input = torch.randint(0, 5, (1, 14), dtype=torch.int32).to("cuda")

        compile_spec = {
            "inputs": [
                torchtrt.Input(
                    self.input.shape,
                    dtype=self.input.dtype,
                    format=torch.contiguous_format,
                ),
                torchtrt.Input(
                    self.input.shape,
                    dtype=self.input.dtype,
                    format=torch.contiguous_format,
                ),
            ],
            "device": {
                "device_type": torchtrt.DeviceType.GPU,
                "gpu_id": 0,
            },
            "enabled_precisions": {torch.float},
            "truncate_long_and_double": True,
        }
        with torchtrt.logging.errors():
            trt_mod = torchtrt.ts.compile(self.model, **compile_spec)

        model_outputs = self.model(self.input, self.input)
        trt_model_outputs = trt_mod(self.input, self.input)
        for out, trt_out in zip(model_outputs, trt_model_outputs):
            cos_sim = cosine_similarity(out, trt_out)
            self.assertTrue(
                cos_sim > COSINE_THRESHOLD,
                msg=f"HF BERT base-uncased TRT outputs don't match with the original model. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}",
            )

    def test_resnet50_half(self):
        self.model = models.resnet50(pretrained=True).eval().to("cuda")
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
        }

        trt_mod = torchtrt.compile(self.scripted_model, **compile_spec)
        cos_sim = cosine_similarity(
            self.model.half()(self.input.half()), trt_mod(self.input.half())
        )
        self.assertTrue(
            cos_sim > COSINE_THRESHOLD,
            msg=f"Resnet50 Half TRT outputs don't match with the original model. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}",
        )


if __name__ == "__main__":
    unittest.main()
