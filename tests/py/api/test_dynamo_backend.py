import unittest
import torch
import timm

import torch_tensorrt as torchtrt
import torchvision.models as models

from transformers import BertModel
from utils import COSINE_THRESHOLD, cosine_similarity


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
            "device": torchtrt.Device("cuda:0"),
            "enabled_precisions": {torch.float},
        }

        trt_mod = torchtrt.dynamo.compile(self.model, **compile_spec)
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
            "device": torchtrt.Device("cuda:0"),
            "enabled_precisions": {torch.float},
        }

        trt_mod = torchtrt.dynamo.compile(self.model, **compile_spec)
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
            "device": torchtrt.Device("cuda:0"),
            "enabled_precisions": {torch.float},
        }

        trt_mod = torchtrt.dynamo.compile(self.model, **compile_spec)
        cos_sim = cosine_similarity(self.model(self.input), trt_mod(self.input))
        self.assertTrue(
            cos_sim > COSINE_THRESHOLD,
            msg=f"EfficientNet-B0 TRT outputs don't match with the original model. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}",
        )

    def test_bert_base_uncased(self):
        self.model = BertModel.from_pretrained("bert-base-uncased").cuda().eval()
        self.input = torch.randint(0, 1, (1, 14), dtype=torch.int32).to("cuda")
        self.input2 = torch.randint(0, 1, (1, 14), dtype=torch.int32).to("cuda")

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
            "device": torchtrt.Device("cuda:0"),
            "enabled_precisions": {torch.float},
            "truncate_long_and_double": True,
            "debug": True,
        }
        trt_mod = torchtrt.dynamo.compile(self.model, **compile_spec)

        model_outputs = self.model(self.input, self.input2)
        trt_model_outputs = trt_mod(self.input, self.input2)
        for key in model_outputs.keys():
            out, trt_out = model_outputs[key], trt_model_outputs[key]
            cos_sim = cosine_similarity(out, trt_out)
            self.assertTrue(
                cos_sim > COSINE_THRESHOLD,
                msg=f"HF BERT base-uncased TRT outputs don't match with the original model. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}",
            )

    def test_resnet18_half(self):
        self.model = models.resnet18(pretrained=True).eval().to("cuda").half()
        self.input = torch.randn((1, 3, 224, 224)).to("cuda").half()

        compile_spec = {
            "inputs": [
                torchtrt.Input(
                    self.input.shape, dtype=torch.half, format=torch.contiguous_format
                )
            ],
            "device": torchtrt.Device("cuda:0"),
            "enabled_precisions": {torch.half},
        }

        trt_mod = torchtrt.dynamo.compile(self.model, **compile_spec)
        cos_sim = cosine_similarity(self.model(self.input), trt_mod(self.input))
        self.assertTrue(
            cos_sim > COSINE_THRESHOLD,
            msg=f"Resnet50 Half TRT outputs don't match with the original model. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}",
        )


if __name__ == "__main__":
    unittest.main()
