# type: ignore

import unittest

import pytest
import timm
import torch
import torch_tensorrt as torchtrt
import torchvision.models as models
from torch_tensorrt.dynamo.utils import COSINE_THRESHOLD, cosine_similarity
from transformers import BertModel

assertions = unittest.TestCase()


@pytest.mark.unit
def test_resnet18(ir):
    model = models.resnet18(pretrained=True).eval().to("cuda")
    input = torch.randn((1, 3, 224, 224)).to("cuda")

    compile_spec = {
        "inputs": [
            torchtrt.Input(
                input.shape, dtype=torch.float, format=torch.contiguous_format
            )
        ],
        "device": torchtrt.Device("cuda:0"),
        "enabled_precisions": {torch.float},
        "ir": ir,
        "pass_through_build_failures": True,
        "optimization_level": 1,
        "cache_built_engines": False,
        "reuse_cached_engines": False,
    }

    trt_mod = torchtrt.compile(model, **compile_spec)
    cos_sim = cosine_similarity(model(input), trt_mod(input))
    assertions.assertTrue(
        cos_sim > COSINE_THRESHOLD,
        msg=f"Resnet18 TRT outputs don't match with the original model. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}",
    )

    # Clean up model env
    torch._dynamo.reset()


@pytest.mark.unit
def test_mobilenet_v2(ir):
    model = models.mobilenet_v2(pretrained=True).eval().to("cuda")
    input = torch.randn((1, 3, 224, 224)).to("cuda")

    compile_spec = {
        "inputs": [
            torchtrt.Input(
                input.shape, dtype=torch.float, format=torch.contiguous_format
            )
        ],
        "device": torchtrt.Device("cuda:0"),
        "enabled_precisions": {torch.float},
        "ir": ir,
        "pass_through_build_failures": True,
        "optimization_level": 1,
        "min_block_size": 10,
        "cache_built_engines": False,
        "reuse_cached_engines": False,
    }

    trt_mod = torchtrt.compile(model, **compile_spec)
    cos_sim = cosine_similarity(model(input), trt_mod(input))
    assertions.assertTrue(
        cos_sim > COSINE_THRESHOLD,
        msg=f"Mobilenet v2 TRT outputs don't match with the original model. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}",
    )

    # Clean up model env
    torch._dynamo.reset()


@pytest.mark.unit
def test_efficientnet_b0(ir):
    model = timm.create_model("efficientnet_b0", pretrained=True).eval().to("cuda")
    input = torch.randn((1, 3, 224, 224)).to("cuda")

    compile_spec = {
        "inputs": [
            torchtrt.Input(
                input.shape, dtype=torch.float, format=torch.contiguous_format
            )
        ],
        "device": torchtrt.Device("cuda:0"),
        "enabled_precisions": {torch.float},
        "ir": ir,
        "pass_through_build_failures": True,
        "optimization_level": 1,
        "min_block_size": 10,
        "cache_built_engines": False,
        "reuse_cached_engines": False,
    }

    trt_mod = torchtrt.compile(model, **compile_spec)
    cos_sim = cosine_similarity(model(input), trt_mod(input))
    assertions.assertTrue(
        cos_sim > COSINE_THRESHOLD,
        msg=f"EfficientNet-B0 TRT outputs don't match with the original model. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}",
    )

    # Clean up model env
    torch._dynamo.reset()


@pytest.mark.unit
def test_bert_base_uncased(ir):
    model = BertModel.from_pretrained("bert-base-uncased").cuda().eval()
    input = torch.randint(0, 1, (1, 14), dtype=torch.int32).to("cuda")
    input2 = torch.randint(0, 1, (1, 14), dtype=torch.int32).to("cuda")

    compile_spec = {
        "inputs": [
            torchtrt.Input(
                input.shape,
                dtype=input.dtype,
                format=torch.contiguous_format,
            ),
            torchtrt.Input(
                input.shape,
                dtype=input.dtype,
                format=torch.contiguous_format,
            ),
        ],
        "device": torchtrt.Device("cuda:0"),
        "enabled_precisions": {torch.float},
        "truncate_double": True,
        "ir": ir,
        "pass_through_build_failures": True,
        "optimization_level": 1,
        "min_block_size": 15,
        "cache_built_engines": False,
        "reuse_cached_engines": False,
    }
    trt_mod = torchtrt.compile(model, **compile_spec)

    model_outputs = model(input, input2)
    trt_model_outputs = trt_mod(input, input2)
    for key in model_outputs.keys():
        out, trt_out = model_outputs[key], trt_model_outputs[key]
        cos_sim = cosine_similarity(out, trt_out)
        assertions.assertTrue(
            cos_sim > COSINE_THRESHOLD,
            msg=f"HF BERT base-uncased TRT outputs don't match with the original model. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}",
        )

    # Clean up model env
    torch._dynamo.reset()


@pytest.mark.unit
def test_resnet18_half(ir):
    model = models.resnet18(pretrained=True).eval().to("cuda").half()
    input = torch.randn((1, 3, 224, 224)).to("cuda").half()

    compile_spec = {
        "inputs": [
            torchtrt.Input(
                input.shape, dtype=torch.half, format=torch.contiguous_format
            )
        ],
        "device": torchtrt.Device("cuda:0"),
        "enabled_precisions": {torch.half},
        "ir": ir,
        "pass_through_build_failures": True,
        "optimization_level": 1,
        "cache_built_engines": False,
        "reuse_cached_engines": False,
    }

    trt_mod = torchtrt.compile(model, **compile_spec)
    cos_sim = cosine_similarity(model(input), trt_mod(input))
    assertions.assertTrue(
        cos_sim > COSINE_THRESHOLD,
        msg=f"Resnet18 Half TRT outputs don't match with the original model. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}",
    )

    # Clean up model env
    torch._dynamo.reset()


@pytest.mark.unit
def test_bf16_model(ir):
    class MyModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 16, 3, stride=1, bias=True)
            self.relu = torch.nn.ReLU()

        def forward(self, x):
            out = self.conv(x)
            out = self.relu(out)
            return out

    model = MyModule().eval().cuda().to(torch.bfloat16)
    input = torch.randn((1, 3, 224, 224)).to("cuda").to(torch.bfloat16)

    compile_spec = {
        "inputs": [
            torchtrt.Input(
                input.shape, dtype=torch.bfloat16, format=torch.contiguous_format
            )
        ],
        "device": torchtrt.Device("cuda:0"),
        "enabled_precisions": {torch.float32},
        "ir": ir,
        "pass_through_build_failures": True,
        "min_block_size": 1,
        "cache_built_engines": False,
        "reuse_cached_engines": False,
        "use_explicit_typing": True,
    }

    trt_mod = torchtrt.compile(model, **compile_spec)
    cos_sim = cosine_similarity(model(input), trt_mod(input))

    assertions.assertTrue(
        cos_sim > COSINE_THRESHOLD,
        msg=f"BF16 model TRT outputs don't match with the original model. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}",
    )

    # Clean up model env
    torch._dynamo.reset()


@pytest.mark.unit
def test_bf16_fallback_model(ir):
    class MyModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 16, 3, padding=1, stride=1, bias=True)
            self.relu = torch.nn.ReLU()
            self.conv2 = torch.nn.Conv2d(16, 16, 3, padding=1, stride=1, bias=True)

        def forward(self, x):
            out = self.conv(x)
            out = self.relu(out)
            out = self.conv2(out)
            return out

    model = MyModule().eval().cuda().to(torch.bfloat16)
    input = torch.randn((1, 3, 224, 224)).to("cuda").to(torch.bfloat16)

    compile_spec = {
        "inputs": [
            torchtrt.Input(
                input.shape, dtype=torch.bfloat16, format=torch.contiguous_format
            )
        ],
        "device": torchtrt.Device("cuda:0"),
        "enabled_precisions": {torch.float32},
        "ir": ir,
        "pass_through_build_failures": True,
        "min_block_size": 1,
        "cache_built_engines": False,
        "reuse_cached_engines": False,
        "use_explicit_typing": True,
        "torch_executed_ops": {"torch.ops.aten.relu.default"},
    }

    trt_mod = torchtrt.compile(model, **compile_spec)
    cos_sim = cosine_similarity(model(input), trt_mod(input))

    assertions.assertTrue(
        cos_sim > COSINE_THRESHOLD,
        msg=f"BF16 fallback model TRT outputs don't match with the original model. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}",
    )

    # Clean up model env
    torch._dynamo.reset()
