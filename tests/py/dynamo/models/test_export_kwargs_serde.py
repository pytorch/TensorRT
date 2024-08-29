# type: ignore
import os
import tempfile
import unittest

import pytest
import timm
import torch
import torch.nn.functional as F
import torch_tensorrt as torchtrt
import torchvision.models as models
from torch import nn
from torch_tensorrt.dynamo._compiler import (
    convert_exported_program_to_serialized_trt_engine,
)
from torch_tensorrt.dynamo.utils import (
    COSINE_THRESHOLD,
    cosine_similarity,
    prepare_inputs,
)

assertions = unittest.TestCase()


@pytest.mark.unit
def test_custom_model():
    class net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 12, 3, padding=1)
            self.bn = nn.BatchNorm2d(12)
            self.conv2 = nn.Conv2d(12, 12, 3, padding=1)
            self.fc1 = nn.Linear(12 * 56 * 56, 10)

        def forward(self, x, b=5, c=None, d=None):
            x = self.conv1(x)
            x = F.relu(x)
            x = self.bn(x)
            x = F.max_pool2d(x, (2, 2))
            x = self.conv2(x)
            x = F.relu(x)
            x = F.max_pool2d(x, (2, 2))
            x = torch.flatten(x, 1)
            x = x + b
            if c is not None:
                x = x * c
            if d is not None:
                x = x - d["value"]
            return self.fc1(x)

    model = net().eval().to("cuda")
    args = [torch.rand((1, 3, 224, 224)).to("cuda")]
    kwargs = {
        "d": {"value": torch.tensor(8).to("cuda")},
    }

    compile_spec = {
        "inputs": args,
        "kwarg_inputs": kwargs,
        "device": torchtrt.Device("cuda:0"),
        "enabled_precisions": {torch.float},
        "pass_through_build_failures": True,
        "optimization_level": 1,
        "min_block_size": 1,
        "ir": "dynamo",
    }

    exp_program = torch.export.export(model, args=tuple(args), kwargs=kwargs)
    trt_gm = torchtrt.dynamo.compile(exp_program, **compile_spec)
    cos_sim = cosine_similarity(model(*args, **kwargs), trt_gm(*args, **kwargs)[0])
    assertions.assertTrue(
        cos_sim > COSINE_THRESHOLD,
        msg=f"CustomKwargs Module TRT outputs don't match with the original model. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}",
    )

    # Save the module
    trt_ep_path = os.path.join(tempfile.gettempdir(), "compiled.ep")
    torchtrt.save(trt_gm, trt_ep_path, inputs=args, kwarg_inputs=kwargs)
    # Clean up model env
    torch._dynamo.reset()


@pytest.mark.unit
def test_custom_model_with_dynamo_trace():
    class net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 12, 3, padding=1)
            self.bn = nn.BatchNorm2d(12)
            self.conv2 = nn.Conv2d(12, 12, 3, padding=1)
            self.fc1 = nn.Linear(12 * 56 * 56, 10)

        def forward(self, x, b=5, c=None, d=None):
            x = self.conv1(x)
            x = F.relu(x)
            x = self.bn(x)
            x = F.max_pool2d(x, (2, 2))
            x = self.conv2(x)
            x = F.relu(x)
            x = F.max_pool2d(x, (2, 2))
            x = torch.flatten(x, 1)
            x = x + b
            if c is not None:
                x = x * c
            if d is not None:
                x = x - d["value"]
            return self.fc1(x)

    model = net().eval().to("cuda")
    args = [torch.rand((1, 3, 224, 224)).to("cuda")]
    kwargs = {
        "d": {"value": torch.tensor(8).to("cuda")},
        "b": torch.tensor(6).to("cuda"),
    }

    compile_spec = {
        "inputs": prepare_inputs(args),
        "kwarg_inputs": prepare_inputs(kwargs),
        "device": torchtrt.Device("cuda:0"),
        "enabled_precisions": {torch.float},
        "pass_through_build_failures": True,
        "optimization_level": 1,
        "min_block_size": 1,
        "ir": "dynamo",
    }

    exp_program = torchtrt.dynamo.trace(model, **compile_spec)
    trt_gm = torchtrt.dynamo.compile(exp_program, **compile_spec)
    cos_sim = cosine_similarity(model(*args, **kwargs), trt_gm(*args, **kwargs)[0])
    assertions.assertTrue(
        cos_sim > COSINE_THRESHOLD,
        msg=f"CustomKwargs Module TRT outputs don't match with the original model. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}",
    )

    # Save the module
    trt_ep_path = os.path.join(tempfile.gettempdir(), "compiled.ep")
    torchtrt.save(trt_gm, trt_ep_path, inputs=args, kwarg_inputs=kwargs)
    # Clean up model env
    torch._dynamo.reset()


@pytest.mark.unit
def test_custom_model_with_dynamo_trace_dynamic():
    class net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 12, 3, padding=1)
            self.bn = nn.BatchNorm2d(12)
            self.conv2 = nn.Conv2d(12, 12, 3, padding=1)
            self.fc1 = nn.Linear(12 * 56 * 56, 10)

        def forward(self, x, b=5, c=None, d=None):
            x = self.conv1(x)
            x = F.relu(x)
            x = self.bn(x)
            x = F.max_pool2d(x, (2, 2))
            x = self.conv2(x)
            x = F.relu(x)
            x = F.max_pool2d(x, (2, 2))
            x = torch.flatten(x, 1)
            x = x + b
            if c is not None:
                x = x * c
            if d is not None:
                x = x - d["value"]
            return self.fc1(x)

    model = net().eval().to("cuda")
    args = [torch.rand((1, 3, 224, 224)).to("cuda")]
    kwargs = {
        "b": torch.tensor(6).to("cuda"),
        "d": {"value": torch.tensor(8).to("cuda")},
    }

    compile_spec = {
        # "arg_inputs": prepare_inputs(args),
        "inputs": [
            torchtrt.Input(
                min_shape=(1, 3, 224, 224),
                opt_shape=(4, 3, 224, 224),
                max_shape=(8, 3, 224, 224),
                dtype=torch.float32,
                name="x",
            )
        ],
        "kwarg_inputs": prepare_inputs(kwargs),
        "device": torchtrt.Device("cuda:0"),
        "enabled_precisions": {torch.float},
        "pass_through_build_failures": True,
        "optimization_level": 1,
        "min_block_size": 1,
        "ir": "dynamo",
    }

    exp_program = torchtrt.dynamo.trace(model, **compile_spec)
    trt_gm = torchtrt.dynamo.compile(exp_program, **compile_spec)
    cos_sim = cosine_similarity(model(*args, **kwargs), trt_gm(*args, **kwargs)[0])
    assertions.assertTrue(
        cos_sim > COSINE_THRESHOLD,
        msg=f"CustomKwargs Module TRT outputs don't match with the original model. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}",
    )

    # Save the module
    trt_ep_path = os.path.join(tempfile.gettempdir(), "compiled.ep")
    torchtrt.save(trt_gm, trt_ep_path, inputs=args, kwarg_inputs=kwargs)
    # Clean up model env
    torch._dynamo.reset()


@pytest.mark.unit
def test_custom_model_with_dynamo_trace_kwarg_dynamic():
    ir = "dynamo"

    class net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 12, 3, padding=1)
            self.bn = nn.BatchNorm2d(12)
            self.conv2 = nn.Conv2d(12, 12, 3, padding=1)
            self.fc1 = nn.Linear(12 * 56 * 56, 10)

        def forward(self, x, b=None, c=None, d=None, e=[]):
            x = self.conv1(x)
            x = F.relu(x)
            x = self.bn(x)
            x = F.max_pool2d(x, (2, 2))
            x = self.conv2(x)
            x = F.relu(x)
            x = F.max_pool2d(x, (2, 2))
            x = torch.flatten(x, 1)
            x = x @ b
            if c is not None:
                x = x * c
            if d is not None:
                x = x - d["value"]
            for n in e:
                x += n
            return x

    model = net().eval().to("cuda")
    args = [torch.rand((1, 3, 224, 224)).to("cuda")]
    kwargs = {
        "b": torch.rand((37632, 10)).to("cuda"),
        "d": {"value": torch.tensor(8).to("cuda")},
        "e": [torch.tensor(8).to("cuda"), torch.tensor(10).to("cuda")],
    }
    model(*args, **kwargs)
    kwarg_torchtrt_input = prepare_inputs(kwargs)
    kwarg_torchtrt_input["b"] = torchtrt.Input(
        min_shape=(37632, 1),
        opt_shape=(37632, 5),
        max_shape=(37632, 10),
        dtype=torch.float32,
        name="b",
    )
    compile_spec = {
        # "arg_inputs": prepare_inputs(args),
        "inputs": [
            torchtrt.Input(
                min_shape=(1, 3, 224, 224),
                opt_shape=(4, 3, 224, 224),
                max_shape=(8, 3, 224, 224),
                dtype=torch.float32,
                name="x",
            ),
        ],
        "kwarg_inputs": kwarg_torchtrt_input,
        "device": torchtrt.Device("cuda:0"),
        "enabled_precisions": {torch.float},
        "pass_through_build_failures": True,
        "optimization_level": 1,
        "min_block_size": 1,
        "ir": "dynamo",
    }

    exp_program = torchtrt.dynamo.trace(model, **compile_spec)
    trt_gm = torchtrt.dynamo.compile(exp_program, **compile_spec)
    cos_sim = cosine_similarity(model(*args, **kwargs), trt_gm(*args, **kwargs)[0])
    assertions.assertTrue(
        cos_sim > COSINE_THRESHOLD,
        msg=f"CustomKwargs Module TRT outputs don't match with the original model. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}",
    )
    # Change the input shape
    kwarg_torchtrt_input["b"] = torch.rand((37632, 1)).to("cuda")
    cos_sim = cosine_similarity(model(*args, **kwargs), trt_gm(*args, **kwargs)[0])
    assertions.assertTrue(
        cos_sim > COSINE_THRESHOLD,
        msg=f"CustomKwargs Module TRT outputs don't match with the original model. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}",
    )
    # Save the module
    trt_ep_path = os.path.join(tempfile.gettempdir(), "compiled.ep")
    torchtrt.save(trt_gm, trt_ep_path, inputs=args, kwarg_inputs=kwargs)
    # Clean up model env
    torch._dynamo.reset()


@pytest.mark.unit
def test_custom_model_with_dynamo_trace_kwarg_dynamic():
    ir = "dynamo"

    class net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 12, 3, padding=1)
            self.bn = nn.BatchNorm2d(12)
            self.conv2 = nn.Conv2d(12, 12, 3, padding=1)
            self.fc1 = nn.Linear(12 * 56 * 56, 10)

        def forward(self, x, b=None, c=None, d=None, e=[]):
            x = self.conv1(x)
            x = F.relu(x)
            x = self.bn(x)
            x = F.max_pool2d(x, (2, 2))
            x = self.conv2(x)
            x = F.relu(x)
            x = F.max_pool2d(x, (2, 2))
            x = torch.flatten(x, 1)
            x = x + b
            if c is not None:
                x = x * c
            if d is not None:
                x = x @ d["value"]
            for n in e:
                x += n
            return x

    model = net().eval().to("cuda")
    args = [torch.rand((1, 3, 224, 224)).to("cuda")]
    kwargs = {
        "b": torch.tensor(1).to("cuda"),
        "e": [torch.tensor(8).to("cuda"), torch.tensor(10).to("cuda")],
        "d": {"value": torch.randn((37632, 5)).to("cuda")},
    }
    model(*args, **kwargs)
    kwarg_torchtrt_input = prepare_inputs(kwargs)
    kwarg_torchtrt_input["d"]["value"] = torchtrt.Input(
        min_shape=(37632, 1),
        opt_shape=(37632, 5),
        max_shape=(37632, 10),
        dtype=torch.float32,
        name="d_value",
    )
    compile_spec = {
        # "arg_inputs": prepare_inputs(args),
        "inputs": [
            torchtrt.Input(
                min_shape=(1, 3, 224, 224),
                opt_shape=(4, 3, 224, 224),
                max_shape=(8, 3, 224, 224),
                dtype=torch.float32,
                name="x",
            ),
        ],
        "kwarg_inputs": kwarg_torchtrt_input,
        "device": torchtrt.Device("cuda:0"),
        "enabled_precisions": {torch.float},
        "pass_through_build_failures": True,
        "optimization_level": 1,
        "min_block_size": 1,
        "ir": "dynamo",
    }

    exp_program = torchtrt.dynamo.trace(model, **compile_spec)
    trt_gm = torchtrt.dynamo.compile(exp_program, **compile_spec)
    cos_sim = cosine_similarity(model(*args, **kwargs), trt_gm(*args, **kwargs)[0])
    assertions.assertTrue(
        cos_sim > COSINE_THRESHOLD,
        msg=f"CustomKwargs Module TRT outputs don't match with the original model. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}",
    )
    # Change the input shape
    kwarg_torchtrt_input["d"]["value"] = torch.rand((37632, 1)).to("cuda")
    cos_sim = cosine_similarity(model(*args, **kwargs), trt_gm(*args, **kwargs)[0])
    assertions.assertTrue(
        cos_sim > COSINE_THRESHOLD,
        msg=f"CustomKwargs Module TRT outputs don't match with the original model. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}",
    )
    # Save the module
    trt_ep_path = os.path.join(tempfile.gettempdir(), "compiled.ep")
    torchtrt.save(trt_gm, trt_ep_path, inputs=args, kwarg_inputs=kwargs)
    # Clean up model env
    torch._dynamo.reset()


@pytest.mark.unit
def test_custom_model_with_dynamo_trace_kwarg_list_dynamic():

    class net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 12, 3, padding=1)
            self.bn = nn.BatchNorm2d(12)
            self.conv2 = nn.Conv2d(12, 12, 3, padding=1)
            self.fc1 = nn.Linear(12 * 56 * 56, 10)

        def forward(self, x, b=None, c=None, d=None, e=[]):
            x = self.conv1(x)
            x = F.relu(x)
            x = self.bn(x)
            x = F.max_pool2d(x, (2, 2))
            x = self.conv2(x)
            x = F.relu(x)
            x = F.max_pool2d(x, (2, 2))
            x = torch.flatten(x, 1)
            x = x @ b
            if c is not None:
                x = x * c
            if d is not None:
                x = x - d["value"]
            for n in e:
                x = x @ n
            return x

    model = net().eval().to("cuda")
    args = [torch.rand((1, 3, 224, 224)).to("cuda")]
    kwargs = {
        "d": {"value": torch.tensor(8).to("cuda")},
        "b": torch.rand((37632, 10)).to("cuda"),
        "e": [torch.randn((10, 10)).to("cuda"), torch.randn((10, 10)).to("cuda")],
    }
    model(*args, **kwargs)
    kwarg_torchtrt_input = prepare_inputs(kwargs)
    kwarg_torchtrt_input["e"][1] = torchtrt.Input(
        min_shape=(10, 1),
        opt_shape=(10, 5),
        max_shape=(10, 10),
        dtype=torch.float32,
        name="e_1",
    )
    compile_spec = {
        # "arg_inputs": prepare_inputs(args),
        "inputs": [
            torchtrt.Input(
                min_shape=(1, 3, 224, 224),
                opt_shape=(4, 3, 224, 224),
                max_shape=(8, 3, 224, 224),
                dtype=torch.float32,
                name="x",
            ),
        ],
        "kwarg_inputs": kwarg_torchtrt_input,
        "device": torchtrt.Device("cuda:0"),
        "enabled_precisions": {torch.float},
        "pass_through_build_failures": True,
        "optimization_level": 1,
        "min_block_size": 1,
        "ir": "dynamo",
    }

    exp_program = torchtrt.dynamo.trace(model, **compile_spec)
    trt_gm = torchtrt.dynamo.compile(exp_program, **compile_spec)
    cos_sim = cosine_similarity(model(*args, **kwargs), trt_gm(*args, **kwargs)[0])
    assertions.assertTrue(
        cos_sim > COSINE_THRESHOLD,
        msg=f"CustomKwargs Module TRT outputs don't match with the original model. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}",
    )
    # Change the input shape
    kwargs["e"][1] = torch.randn([10, 2]).to("cuda")
    cos_sim = cosine_similarity(model(*args, **kwargs), trt_gm(*args, **kwargs)[0])
    assertions.assertTrue(
        cos_sim > COSINE_THRESHOLD,
        msg=f"CustomKwargs Module TRT outputs don't match with the original model. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}",
    )

    torch._dynamo.reset()


def test_custom_model_compile_engine():
    class net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 12, 3, padding=1)
            self.bn = nn.BatchNorm2d(12)
            self.conv2 = nn.Conv2d(12, 12, 3, padding=1)
            self.fc1 = nn.Linear(12 * 56 * 56, 10)

        def forward(self, x, b=5, c=None, d=None):
            x = self.conv1(x)
            x = F.relu(x)
            x = self.bn(x)
            x = F.max_pool2d(x, (2, 2))
            x = self.conv2(x)
            x = F.relu(x)
            x = F.max_pool2d(x, (2, 2))
            x = torch.flatten(x, 1)
            x = x + b
            if c is not None:
                x = x * c
            if d is not None:
                x = x - d["value"]
            return self.fc1(x)

    model = net().eval().to("cuda")
    args = [torch.rand((1, 3, 224, 224)).to("cuda")]
    kwargs = {
        "b": torch.tensor(6).to("cuda"),
        "d": {"value": torch.tensor(8).to("cuda")},
    }

    compile_spec = {
        "inputs": args,
        "kwarg_inputs": kwargs,
        "device": torchtrt.Device("cuda:0"),
        "enabled_precisions": {torch.float},
        "pass_through_build_failures": True,
        "optimization_level": 1,
        "min_block_size": 1,
        "ir": "dynamo",
    }

    exp_program = torch.export.export(model, args=tuple(args), kwargs=kwargs)
    engine = convert_exported_program_to_serialized_trt_engine(
        exp_program, **compile_spec
    )


def test_custom_model_compile_engine_with_pure_kwarg_inputs():
    class net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 12, 3, padding=1)
            self.bn = nn.BatchNorm2d(12)
            self.conv2 = nn.Conv2d(12, 12, 3, padding=1)
            self.fc1 = nn.Linear(12 * 56 * 56, 10)

        def forward(self, x, b=5, c=None, d=None):
            x = self.conv1(x)
            x = F.relu(x)
            x = self.bn(x)
            x = F.max_pool2d(x, (2, 2))
            x = self.conv2(x)
            x = F.relu(x)
            x = F.max_pool2d(x, (2, 2))
            x = torch.flatten(x, 1)
            x = x + b
            if c is not None:
                x = x * c
            if d is not None:
                x = x - d["value"]
            return self.fc1(x)

    model = net().eval().to("cuda")
    kwargs = {
        "x": torch.rand((1, 3, 224, 224)).to("cuda"),
        "b": torch.tensor(6).to("cuda"),
        "d": {"value": torch.tensor(8).to("cuda")},
    }

    compile_spec = {
        "arg_inputs": (),
        "kwarg_inputs": kwargs,
        "device": torchtrt.Device("cuda:0"),
        "enabled_precisions": {torch.float},
        "pass_through_build_failures": True,
        "optimization_level": 1,
        "min_block_size": 1,
        "ir": "dynamo",
    }

    exp_program = torch.export.export(model, args=(), kwargs=kwargs)
    _ = convert_exported_program_to_serialized_trt_engine(exp_program, **compile_spec)
