# type: ignore

import unittest

import pytest
import timm
import torch
import torch_tensorrt as torchtrt
from torch_tensorrt.dynamo.utils import COSINE_THRESHOLD, cosine_similarity

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

    compile_spec = {
        "device": torchtrt.Device("cuda:0"),
        "enabled_precisions": {torch.float},
        "ir": ir,
        "pass_through_build_failures": True,
        "min_block_size": 1,
    }
    if ir == "torch_compile":
        input_bs4 = torch.randn((4, 3, 224, 224)).to("cuda")
        torch._dynamo.mark_dynamic(input_bs4, 0, min=1, max=8)
        # Compile the model
        trt_model = torch.compile(model, backend="tensorrt", options=compile_spec)
        trt_model(input_bs4)
    elif ir == "dynamo":
        compile_spec["inputs"] = [
            torchtrt.Input(
                min_shape=(1, 3, 224, 224),
                opt_shape=(4, 3, 224, 224),
                max_shape=(8, 3, 224, 224),
                dtype=torch.float32,
                name="x",
            )
        ]
        trt_model = torchtrt.compile(model, **compile_spec)

    input_bs6 = torch.randn((6, 3, 224, 224)).to("cuda")
    cos_sim = cosine_similarity(model(input_bs6), trt_model(input_bs6))
    assertions.assertTrue(
        cos_sim > COSINE_THRESHOLD,
        msg=f"test_dyn_full_compile model TRT outputs don't match with the pytorch model. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}",
    )
    # Clean up model env
    torch._dynamo.reset()

    with torch.no_grad():
        torch.cuda.empty_cache()


@pytest.mark.unit
def test_base_dynamic_fallback(ir):
    """
    Tests the model with dynamic shapes where torch.abs op is forced to run in PyTorch
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

    compile_spec = {
        "device": torchtrt.Device("cuda:0"),
        "enabled_precisions": {torch.float},
        "ir": ir,
        "pass_through_build_failures": True,
        "torch_executed_ops": {"torch.ops.aten.abs.default"},
        "min_block_size": 1,
    }

    if ir == "torch_compile":
        input_bs4 = torch.randn((4, 3, 224, 224)).to("cuda")
        torch._dynamo.mark_dynamic(input_bs4, 0, min=1, max=8)
        # Compile the model
        trt_model = torch.compile(model, backend="tensorrt", options=compile_spec)
        trt_model(input_bs4)
    elif ir == "dynamo":
        compile_spec["inputs"] = [
            torchtrt.Input(
                min_shape=(1, 3, 224, 224),
                opt_shape=(4, 3, 224, 224),
                max_shape=(8, 3, 224, 224),
                dtype=torch.float32,
                name="x",
            )
        ]
        trt_model = torchtrt.compile(model, **compile_spec)

    input_bs6 = torch.randn((6, 3, 224, 224)).to("cuda")
    cos_sim = cosine_similarity(model(input_bs6), trt_model(input_bs6))
    assertions.assertTrue(
        cos_sim > COSINE_THRESHOLD,
        msg=f"test_base_dynamic_fallback model TRT outputs don't match with the pytorch model. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}",
    )

    # Clean up model env
    torch._dynamo.reset()

    with torch.no_grad():
        torch.cuda.empty_cache()


@pytest.mark.unit
def test_view(ir):
    """
    Tests the model (which is fully convertible) with dynamic shapes
    """

    class MyModule(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            input_shape = x.size()
            y = x.view(input_shape[0], -1)
            return y

    model = MyModule().eval().cuda()

    compile_spec = {
        "device": torchtrt.Device("cuda:0"),
        "enabled_precisions": {torch.float},
        "ir": ir,
        "pass_through_build_failures": True,
        "min_block_size": 1,
    }

    if ir == "torch_compile":
        input_bs4 = torch.randn((4, 3, 4)).to("cuda")
        torch._dynamo.mark_dynamic(input_bs4, 0, min=1, max=8)
        # Compile the model
        trt_model = torch.compile(model, backend="tensorrt", options=compile_spec)
        trt_model(input_bs4)
    elif ir == "dynamo":
        compile_spec["inputs"] = [
            torchtrt.Input(
                min_shape=(1, 3, 4),
                opt_shape=(4, 3, 4),
                max_shape=(8, 3, 4),
                dtype=torch.float32,
                name="x",
            )
        ]
        trt_model = torchtrt.compile(model, **compile_spec)

    input_bs6 = torch.randn((6, 3, 4)).to("cuda")
    cos_sim = cosine_similarity(model(input_bs6), trt_model(input_bs6))
    assertions.assertTrue(
        cos_sim > COSINE_THRESHOLD,
        msg=f"test_view model TRT outputs don't match with the pytorch model. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}",
    )

    # Clean up model env
    torch._dynamo.reset()

    with torch.no_grad():
        torch.cuda.empty_cache()


@pytest.mark.unit
def test_resnet_dynamic(ir):
    """
    Tests the Resnet18 model (which is fully convertible) with dynamic shapes
    """
    import torchvision.models as models

    model = models.resnet18(pretrained=True).eval().to("cuda")

    compile_spec = {
        "device": torchtrt.Device("cuda:0"),
        "enabled_precisions": {torch.float},
        "ir": ir,
        "pass_through_build_failures": True,
        "min_block_size": 1,
    }

    if ir == "torch_compile":
        input_bs2 = torch.randn((2, 3, 224, 224)).to("cuda")
        torch._dynamo.mark_dynamic(input_bs2, 0, min=1, max=8)
        # Compile the model
        trt_model = torch.compile(model, backend="tensorrt", options=compile_spec)
        trt_model(input_bs2)
    elif ir == "dynamo":
        compile_spec["inputs"] = [
            torchtrt.Input(
                min_shape=(1, 3, 224, 224),
                opt_shape=(4, 3, 224, 224),
                max_shape=(8, 3, 224, 224),
                dtype=torch.float32,
                name="x",
            )
        ]
        trt_model = torchtrt.compile(model, **compile_spec)

    input_bs6 = torch.randn((6, 3, 224, 224)).to("cuda")
    cos_sim = cosine_similarity(model(input_bs6), trt_model(input_bs6))
    assertions.assertTrue(
        cos_sim > COSINE_THRESHOLD,
        msg=f"test_resnet_dynamic model TRT outputs don't match with the pytorch model. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}",
    )

    # Clean up model env
    torch._dynamo.reset()

    with torch.no_grad():
        torch.cuda.empty_cache()
