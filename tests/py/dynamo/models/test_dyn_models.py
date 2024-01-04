import unittest

import pytest
import timm
import torch
from torch_tensorrt.dynamo.utils import COSINE_THRESHOLD, cosine_similarity

import torch_tensorrt as torchtrt

assertions = unittest.TestCase()


@pytest.mark.unit
def test_base_dynamic(ir):
    """
    Tests the model (which is fully convertible) with dynamic shapes
    """

    class MyModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 16, 3, stride=1, bias=True)
            self.relu = torch.nn.ReLU()

        def forward(self, x):
            out = self.conv(x)
            out = self.relu(out)
            return out

    model = MyModule().eval().cuda()
    input = torch.randn((1, 3, 224, 224)).to("cuda")

    compile_spec = {
        "inputs": [
            torchtrt.Input(
                min_shape=(1, 3, 224, 224),
                opt_shape=(4, 3, 224, 224),
                max_shape=(8, 3, 224, 224),
                dtype=torch.float32,
                name="x",
            )
        ],
        "device": torchtrt.Device("cuda:0"),
        "enabled_precisions": {torch.float},
        "ir": ir,
        "pass_through_build_failures": True,
        "optimization_level": 1,
        "min_block_size": 1,
    }

    trt_mod = torchtrt.compile(model, **compile_spec)
    cos_sim = cosine_similarity(model(input), trt_mod(input)[0])
    assertions.assertTrue(
        cos_sim > COSINE_THRESHOLD,
        msg=f"test_base_dynamic model TRT outputs don't match with the pytorch model. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}",
    )

    # Clean up model env
    torch._dynamo.reset()

    with torch.no_grad():
        torch.cuda.empty_cache()


@pytest.mark.unit
def test_base_dynamic_fallback(ir):
    """
    Tests the model (which is fully convertible) with dynamic shapes
    """

    class MyModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 16, 3, stride=1, bias=True)
            self.relu = torch.nn.ReLU()

        def forward(self, x):
            out = self.conv(x)
            out = torch.abs(out)
            out = self.relu(out)
            return out

    model = MyModule().eval().cuda()
    input = torch.randn((1, 3, 224, 224)).to("cuda")

    compile_spec = {
        "inputs": [
            torchtrt.Input(
                min_shape=(1, 3, 224, 224),
                opt_shape=(4, 3, 224, 224),
                max_shape=(8, 3, 224, 224),
                dtype=torch.float32,
                name="x",
            )
        ],
        "device": torchtrt.Device("cuda:0"),
        "enabled_precisions": {torch.float},
        "ir": ir,
        "pass_through_build_failures": True,
        "optimization_level": 1,
        "torch_executed_ops": {"torch.ops.aten.abs.default"},
        "min_block_size": 1,
    }

    trt_mod = torchtrt.compile(model, **compile_spec)
    cos_sim = cosine_similarity(model(input), trt_mod(input)[0])
    assertions.assertTrue(
        cos_sim > COSINE_THRESHOLD,
        msg=f"test_base_dynamic model TRT outputs don't match with the pytorch model. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}",
    )

    # Clean up model env
    torch._dynamo.reset()

    with torch.no_grad():
        torch.cuda.empty_cache()
