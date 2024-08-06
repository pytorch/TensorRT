import copy
import unittest
from typing import Dict

import custom_models as cm
import timm
import torch
import torch_tensorrt as torchtrt
import torchvision.models as models
from utils import COSINE_THRESHOLD, cosine_similarity


@unittest.skipIf(
    not torchtrt.ENABLED_FEATURES.torchscript_frontend,
    "TorchScript Frontend is not available",
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

    @unittest.skip("Layer Norm issue needs to be addressed")
    def test_bert_base_uncased(self):
        self.model = cm.BertModule().cuda()
        self.input = torch.randint(0, 2, (1, 14), dtype=torch.int32).to("cuda")

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
