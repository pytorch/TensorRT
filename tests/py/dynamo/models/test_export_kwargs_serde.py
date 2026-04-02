# type: ignore
import os
import unittest

import pytest
import torch
import torch.nn.functional as F
import torch_tensorrt as torchtrt
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
@pytest.mark.critical
def test_custom_model(tmpdir):
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
        "cache_built_engines": False,
        "reuse_cached_engines": False,
    }

    exp_program = torch.export.export(model, args=tuple(args), kwargs=kwargs)
    trt_gm = torchtrt.dynamo.compile(exp_program, **compile_spec)
    cos_sim = cosine_similarity(model(*args, **kwargs), trt_gm(*args, **kwargs)[0])
    assertions.assertTrue(
        cos_sim > COSINE_THRESHOLD,
        msg=f"CustomKwargs Module TRT outputs don't match with the original model. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}",
    )

    # Save the module
    trt_ep_path = os.path.join(tmpdir, "compiled.ep")
    torchtrt.save(trt_gm, trt_ep_path, retrace=False)
    # Clean up model env
    torch._dynamo.reset()


@pytest.mark.unit
@pytest.mark.critical
def test_custom_model_with_dynamo_trace(tmpdir):
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
        "cache_built_engines": False,
        "reuse_cached_engines": False,
    }

    exp_program = torchtrt.dynamo.trace(model, **compile_spec)
    trt_gm = torchtrt.dynamo.compile(exp_program, **compile_spec)
    cos_sim = cosine_similarity(model(*args, **kwargs), trt_gm(*args, **kwargs)[0])
    assertions.assertTrue(
        cos_sim > COSINE_THRESHOLD,
        msg=f"CustomKwargs Module TRT outputs don't match with the original model. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}",
    )

    # Save the module
    trt_ep_path = os.path.join(tmpdir, "compiled.ep")
    torchtrt.save(trt_gm, trt_ep_path, retrace=False)
    # Clean up model env
    torch._dynamo.reset()


@pytest.mark.unit
@pytest.mark.critical
def test_custom_model_with_dynamo_trace_dynamic(tmpdir):
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
        "cache_built_engines": False,
        "reuse_cached_engines": False,
    }

    exp_program = torchtrt.dynamo.trace(model, **compile_spec)
    trt_gm = torchtrt.dynamo.compile(exp_program, **compile_spec)
    cos_sim = cosine_similarity(model(*args, **kwargs), trt_gm(*args, **kwargs)[0])
    assertions.assertTrue(
        cos_sim > COSINE_THRESHOLD,
        msg=f"CustomKwargs Module TRT outputs don't match with the original model. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}",
    )

    # Save the module
    trt_ep_path = os.path.join(tmpdir, "compiled.ep")
    torchtrt.save(trt_gm, trt_ep_path, retrace=False)
    # Clean up model env
    torch._dynamo.reset()


@pytest.mark.unit
@pytest.mark.critical
def test_custom_model_with_dynamo_trace_kwarg_dynamic(tmpdir):
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
        "cache_built_engines": False,
        "reuse_cached_engines": False,
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
    trt_ep_path = os.path.join(tmpdir, "compiled.ep")
    torchtrt.save(trt_gm, trt_ep_path, retrace=False)
    # Clean up model env
    torch._dynamo.reset()


@pytest.mark.unit
@pytest.mark.critical
def test_custom_model_with_dynamo_trace_kwarg_dynamic(tmpdir):
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
        "cache_built_engines": False,
        "reuse_cached_engines": False,
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
    trt_ep_path = os.path.join(tmpdir, "compiled.ep")
    torchtrt.save(trt_gm, trt_ep_path, retrace=False)
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
        "cache_built_engines": False,
        "reuse_cached_engines": False,
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
        "cache_built_engines": False,
        "reuse_cached_engines": False,
    }

    exp_program = torch.export.export(model, args=tuple(args), kwargs=kwargs)
    engine = convert_exported_program_to_serialized_trt_engine(
        exp_program, **compile_spec
    )


@pytest.mark.unit
@pytest.mark.critical
def test_save_load_pure_kwarg_inputs_retrace_false(tmpdir):
    """
    Test save/load with pure kwarg inputs (no positional args) and retrace=False.
    This exercises the code path where arg_inputs is None during save, which previously
    triggered a TypeError in _extract_tensor when called with None.
    """

    class KwargOnlyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(10, 5)

        def forward(self, *, x):
            return self.linear(x)

    model = KwargOnlyModel().eval().cuda()
    kwargs = {"x": torch.randn(4, 10).cuda()}

    exp_program = torch.export.export(model, args=(), kwargs=kwargs)

    compile_spec = {
        "arg_inputs": (),
        "kwarg_inputs": kwargs,
        "device": torchtrt.Device("cuda:0"),
        "enabled_precisions": {torch.float},
        "min_block_size": 1,
        "ir": "dynamo",
        "cache_built_engines": False,
        "reuse_cached_engines": False,
    }

    trt_gm = torchtrt.dynamo.compile(exp_program, **compile_spec)

    # Save with retrace=False and no positional inputs — the previously broken path
    trt_ep_path = os.path.join(tmpdir, "compiled_kwarg_only.ep")
    torchtrt.save(trt_gm, trt_ep_path, retrace=False)

    # Load and verify outputs match
    deser_trt_module = torchtrt.load(trt_ep_path).module()
    ref_out = model(**kwargs)
    # Pure kwarg model returns a tensor directly (not wrapped in a tuple)
    trt_out = trt_gm(**kwargs)
    deser_out = deser_trt_module(**kwargs)

    cos_sim = cosine_similarity(ref_out, trt_out)
    assertions.assertTrue(
        cos_sim > COSINE_THRESHOLD,
        msg=f"Pure kwarg save/load (retrace=False): TRT output mismatch. Cosine sim: {cos_sim}",
    )
    cos_sim = cosine_similarity(ref_out, deser_out)
    assertions.assertTrue(
        cos_sim > COSINE_THRESHOLD,
        msg=f"Pure kwarg save/load (retrace=False): deserialized output mismatch. Cosine sim: {cos_sim}",
    )
    torch._dynamo.reset()


@pytest.mark.unit
@pytest.mark.critical
def test_save_load_pure_kwarg_inputs_retrace_true(tmpdir):
    """
    Test save/load with pure kwarg inputs (no positional args) and retrace=True.
    Exercises the retrace path when only kwarg_inputs are provided (no positional args).
    """

    class KwargOnlyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(10, 5)

        def forward(self, *, x):
            return self.linear(x)

    model = KwargOnlyModel().eval().cuda()
    kwargs = {"x": torch.randn(4, 10).cuda()}

    exp_program = torch.export.export(model, args=(), kwargs=kwargs)

    compile_spec = {
        "arg_inputs": (),
        "kwarg_inputs": kwargs,
        "device": torchtrt.Device("cuda:0"),
        "enabled_precisions": {torch.float},
        "min_block_size": 1,
        "ir": "dynamo",
        "cache_built_engines": False,
        "reuse_cached_engines": False,
    }

    trt_gm = torchtrt.dynamo.compile(exp_program, **compile_spec)

    trt_ep_path = os.path.join(tmpdir, "compiled_kwarg_only_retrace.ep")
    torchtrt.save(trt_gm, trt_ep_path, kwarg_inputs=kwargs, retrace=True)

    deser_trt_module = torchtrt.load(trt_ep_path).module()
    ref_out = model(**kwargs)
    # Pure kwarg model returns a tensor directly (not wrapped in a tuple)
    trt_out = trt_gm(**kwargs)
    deser_out = deser_trt_module(**kwargs)

    cos_sim = cosine_similarity(ref_out, trt_out)
    assertions.assertTrue(
        cos_sim > COSINE_THRESHOLD,
        msg=f"Pure kwarg save/load (retrace=True): TRT output mismatch. Cosine sim: {cos_sim}",
    )
    cos_sim = cosine_similarity(ref_out, deser_out)
    assertions.assertTrue(
        cos_sim > COSINE_THRESHOLD,
        msg=f"Pure kwarg save/load (retrace=True): deserialized output mismatch. Cosine sim: {cos_sim}",
    )
    torch._dynamo.reset()


@pytest.mark.unit
@pytest.mark.critical
def test_save_load_mixed_args_kwargs_retrace_false(tmpdir):
    """
    Test save/load with mixed positional + kwarg inputs and retrace=False.
    Exercises the _extract_tensor path where arg_inputs is a list of tensors
    (not None, not all Input objects).
    """

    class MixedModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(10, 5)

        def forward(self, x, *, scale):
            return self.linear(x) * scale

    model = MixedModel().eval().cuda()
    args = [torch.randn(4, 10).cuda()]
    kwargs = {"scale": torch.tensor(2.0).cuda()}

    exp_program = torch.export.export(model, args=tuple(args), kwargs=kwargs)

    compile_spec = {
        "inputs": args,
        "kwarg_inputs": kwargs,
        "device": torchtrt.Device("cuda:0"),
        "enabled_precisions": {torch.float},
        "min_block_size": 1,
        "ir": "dynamo",
        "cache_built_engines": False,
        "reuse_cached_engines": False,
    }

    trt_gm = torchtrt.dynamo.compile(exp_program, **compile_spec)

    trt_ep_path = os.path.join(tmpdir, "compiled_mixed.ep")
    torchtrt.save(trt_gm, trt_ep_path, retrace=False)

    deser_trt_module = torchtrt.load(trt_ep_path).module()
    ref_out = model(*args, **kwargs)
    trt_out = trt_gm(*args, **kwargs)
    deser_out = deser_trt_module(*args, **kwargs)

    cos_sim = cosine_similarity(ref_out, trt_out)
    assertions.assertTrue(
        cos_sim > COSINE_THRESHOLD,
        msg=f"Mixed args+kwargs save/load (retrace=False): TRT output mismatch. Cosine sim: {cos_sim}",
    )
    cos_sim = cosine_similarity(ref_out, deser_out)
    assertions.assertTrue(
        cos_sim > COSINE_THRESHOLD,
        msg=f"Mixed args+kwargs save/load (retrace=False): deserialized output mismatch. Cosine sim: {cos_sim}",
    )
    torch._dynamo.reset()


@pytest.mark.unit
@pytest.mark.critical
def test_save_load_input_objects_retrace_false(tmpdir):
    """
    Test save/load where arg_inputs are torch_tensorrt.Input objects (not Tensors) with retrace=False.
    Exercises the all_inputs_are_input_objects=True branch in save().
    """

    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(10, 5)

        def forward(self, x):
            return self.linear(x)

    model = SimpleModel().eval().cuda()
    example = torch.randn(4, 10).cuda()

    exp_program = torch.export.export(model, args=(example,))

    compile_spec = {
        "inputs": [torchtrt.Input(shape=(4, 10), dtype=torch.float)],
        "device": torchtrt.Device("cuda:0"),
        "enabled_precisions": {torch.float},
        "min_block_size": 1,
        "ir": "dynamo",
        "cache_built_engines": False,
        "reuse_cached_engines": False,
    }

    trt_gm = torchtrt.dynamo.compile(exp_program, **compile_spec)

    trt_ep_path = os.path.join(tmpdir, "compiled_input_objects.ep")
    # Pass Input objects — exercises all_inputs_are_input_objects=True path
    torchtrt.save(
        trt_gm,
        trt_ep_path,
        inputs=[torchtrt.Input(shape=(4, 10), dtype=torch.float)],
        retrace=False,
    )

    deser_trt_module = torchtrt.load(trt_ep_path).module()
    ref_out = model(example)
    trt_out = trt_gm(example)
    deser_out = deser_trt_module(example)

    cos_sim = cosine_similarity(ref_out, trt_out)
    assertions.assertTrue(
        cos_sim > COSINE_THRESHOLD,
        msg=f"Input objects save/load (retrace=False): TRT output mismatch. Cosine sim: {cos_sim}",
    )
    cos_sim = cosine_similarity(ref_out, deser_out)
    assertions.assertTrue(
        cos_sim > COSINE_THRESHOLD,
        msg=f"Input objects save/load (retrace=False): deserialized output mismatch. Cosine sim: {cos_sim}",
    )
    torch._dynamo.reset()


# ---------------------------------------------------------------------------
# Helpers shared by use_legacy_exporter combination tests
# ---------------------------------------------------------------------------


def _make_simple_model_and_inputs():
    """Static-shape linear model, args only."""

    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(10, 5)

        def forward(self, x):
            return self.linear(x)

    model = SimpleModel().eval().cuda()
    example = torch.randn(4, 10).cuda()
    return model, example


def _make_dynamic_model_and_inputs():
    """Dynamic-batch linear model, args only."""

    class DynModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(10, 5)

        def forward(self, x):
            return self.linear(x)

    model = DynModel().eval().cuda()
    example = torch.randn(4, 10).cuda()
    batch = torch.export.Dim("batch", min=1, max=8)
    dynamic_shapes = {"x": {0: batch}}
    trt_inputs = [
        torchtrt.Input(
            min_shape=(1, 10),
            opt_shape=(4, 10),
            max_shape=(8, 10),
            dtype=torch.float32,
        )
    ]
    return model, example, dynamic_shapes, trt_inputs


def _compile_simple(model, example):
    ep = torch.export.export(model, args=(example,))
    compile_spec = {
        "inputs": [torchtrt.Input(shape=(4, 10), dtype=torch.float)],
        "device": torchtrt.Device("cuda:0"),
        "enabled_precisions": {torch.float},
        "min_block_size": 1,
        "ir": "dynamo",
        "cache_built_engines": False,
        "reuse_cached_engines": False,
    }
    return torchtrt.dynamo.compile(ep, **compile_spec)


def _compile_dynamic(model, example, dynamic_shapes, trt_inputs):
    ep = torch.export.export(model, args=(example,), dynamic_shapes=dynamic_shapes)
    compile_spec = {
        "inputs": trt_inputs,
        "device": torchtrt.Device("cuda:0"),
        "enabled_precisions": {torch.float},
        "min_block_size": 1,
        "ir": "dynamo",
        "cache_built_engines": False,
        "reuse_cached_engines": False,
    }
    return torchtrt.dynamo.compile(ep, **compile_spec)


def _check_outputs(assertions, ref_out, trt_out, deser_out, label):
    cos_sim = cosine_similarity(ref_out, trt_out)
    assertions.assertTrue(
        cos_sim > COSINE_THRESHOLD,
        msg=f"{label}: TRT output mismatch. Cosine sim: {cos_sim}",
    )
    cos_sim = cosine_similarity(ref_out, deser_out)
    assertions.assertTrue(
        cos_sim > COSINE_THRESHOLD,
        msg=f"{label}: deserialized output mismatch. Cosine sim: {cos_sim}",
    )


# ---------------------------------------------------------------------------
# retrace=False, use_legacy_exporter combinations
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.critical
def test_save_load_retrace_false_legacy_true_static(tmpdir):
    """retrace=False, use_legacy_exporter=True (explicit), static shapes.
    Same effective path as the default; verifies the explicit override is wired through.
    """
    model, example = _make_simple_model_and_inputs()
    trt_gm = _compile_simple(model, example)

    trt_ep_path = os.path.join(tmpdir, "model.ep")
    torchtrt.save(trt_gm, trt_ep_path, retrace=False, use_legacy_exporter=True)

    deser = torchtrt.load(trt_ep_path).module()
    _check_outputs(
        assertions,
        model(example),
        trt_gm(example),
        deser(example),
        "retrace=False, use_legacy_exporter=True, static",
    )
    torch._dynamo.reset()


@pytest.mark.unit
@pytest.mark.critical
def test_save_load_retrace_false_legacy_false_static(tmpdir):
    """retrace=False, use_legacy_exporter=False (override to non-legacy), static shapes.
    Forces torch.export.export on the inlined graph instead of graph surgery."""
    model, example = _make_simple_model_and_inputs()
    trt_gm = _compile_simple(model, example)

    trt_ep_path = os.path.join(tmpdir, "model.ep")
    torchtrt.save(
        trt_gm,
        trt_ep_path,
        inputs=[example],
        retrace=False,
        use_legacy_exporter=False,
    )

    deser = torchtrt.load(trt_ep_path).module()
    _check_outputs(
        assertions,
        model(example),
        trt_gm(example),
        deser(example),
        "retrace=False, use_legacy_exporter=False, static",
    )
    torch._dynamo.reset()


@pytest.mark.unit
@pytest.mark.critical
def test_save_load_retrace_false_default_dynamic(tmpdir):
    """retrace=False, use_legacy_exporter=None (default → legacy), dynamic shapes.
    Verifies the legacy exporter correctly preserves range_constraints for dynamic dims.
    """
    model, example, dynamic_shapes, trt_inputs = _make_dynamic_model_and_inputs()
    trt_gm = _compile_dynamic(model, example, dynamic_shapes, trt_inputs)

    trt_ep_path = os.path.join(tmpdir, "model.ep")
    torchtrt.save(trt_gm, trt_ep_path, retrace=False)

    deser = torchtrt.load(trt_ep_path).module()
    # Run at a different batch size to exercise the dynamic range
    x_alt = torch.randn(2, 10).cuda()
    _check_outputs(
        assertions,
        model(x_alt),
        trt_gm(x_alt),
        deser(x_alt),
        "retrace=False, use_legacy_exporter=None, dynamic",
    )
    torch._dynamo.reset()


@pytest.mark.unit
@pytest.mark.critical
def test_save_load_retrace_false_legacy_true_dynamic(tmpdir):
    """retrace=False, use_legacy_exporter=True (explicit), dynamic shapes."""
    model, example, dynamic_shapes, trt_inputs = _make_dynamic_model_and_inputs()
    trt_gm = _compile_dynamic(model, example, dynamic_shapes, trt_inputs)

    trt_ep_path = os.path.join(tmpdir, "model.ep")
    torchtrt.save(trt_gm, trt_ep_path, retrace=False, use_legacy_exporter=True)

    deser = torchtrt.load(trt_ep_path).module()
    x_alt = torch.randn(2, 10).cuda()
    _check_outputs(
        assertions,
        model(x_alt),
        trt_gm(x_alt),
        deser(x_alt),
        "retrace=False, use_legacy_exporter=True, dynamic",
    )
    torch._dynamo.reset()


@pytest.mark.unit
@pytest.mark.critical
def test_save_load_retrace_false_legacy_false_dynamic(tmpdir):
    """retrace=False, use_legacy_exporter=False (override), dynamic shapes.
    Forces non-legacy (torch.export.export on the inlined graph) even with retrace=False.
    """
    model, example, dynamic_shapes, trt_inputs = _make_dynamic_model_and_inputs()
    trt_gm = _compile_dynamic(model, example, dynamic_shapes, trt_inputs)

    trt_ep_path = os.path.join(tmpdir, "model.ep")
    torchtrt.save(
        trt_gm,
        trt_ep_path,
        inputs=[example],
        dynamic_shapes=dynamic_shapes,
        retrace=False,
        use_legacy_exporter=False,
    )

    deser = torchtrt.load(trt_ep_path).module()
    x_alt = torch.randn(2, 10).cuda()
    _check_outputs(
        assertions,
        model(x_alt),
        trt_gm(x_alt),
        deser(x_alt),
        "retrace=False, use_legacy_exporter=False, dynamic",
    )
    torch._dynamo.reset()


# ---------------------------------------------------------------------------
# retrace=True, use_legacy_exporter combinations (dynamic shapes)
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.critical
def test_save_load_retrace_true_default_dynamic(tmpdir):
    """retrace=True, use_legacy_exporter=None (default → non-legacy), dynamic shapes.
    The has_symbolic_metadata + dynamic_shapes path uses torch.export.export on
    the inlined graph."""
    model, example, dynamic_shapes, trt_inputs = _make_dynamic_model_and_inputs()
    trt_gm = _compile_dynamic(model, example, dynamic_shapes, trt_inputs)

    trt_ep_path = os.path.join(tmpdir, "model.ep")
    torchtrt.save(
        trt_gm,
        trt_ep_path,
        inputs=[example],
        dynamic_shapes=dynamic_shapes,
        retrace=True,
    )

    deser = torchtrt.load(trt_ep_path).module()
    x_alt = torch.randn(2, 10).cuda()
    _check_outputs(
        assertions,
        model(x_alt),
        trt_gm(x_alt),
        deser(x_alt),
        "retrace=True, use_legacy_exporter=None, dynamic",
    )
    torch._dynamo.reset()


@pytest.mark.unit
@pytest.mark.critical
def test_save_load_retrace_true_legacy_true_dynamic(tmpdir):
    """retrace=True, use_legacy_exporter=True (override to legacy), dynamic shapes.
    Forces graph surgery even though retrace=True."""
    model, example, dynamic_shapes, trt_inputs = _make_dynamic_model_and_inputs()
    trt_gm = _compile_dynamic(model, example, dynamic_shapes, trt_inputs)

    trt_ep_path = os.path.join(tmpdir, "model.ep")
    torchtrt.save(
        trt_gm,
        trt_ep_path,
        inputs=[example],
        dynamic_shapes=dynamic_shapes,
        retrace=True,
        use_legacy_exporter=True,
    )

    deser = torchtrt.load(trt_ep_path).module()
    x_alt = torch.randn(2, 10).cuda()
    _check_outputs(
        assertions,
        model(x_alt),
        trt_gm(x_alt),
        deser(x_alt),
        "retrace=True, use_legacy_exporter=True, dynamic",
    )
    torch._dynamo.reset()


@pytest.mark.unit
@pytest.mark.critical
def test_save_load_retrace_true_legacy_false_dynamic(tmpdir):
    """retrace=True, use_legacy_exporter=False (explicit non-legacy), dynamic shapes.
    Same path as the default for this combination; verifies explicit False is wired through.
    """
    model, example, dynamic_shapes, trt_inputs = _make_dynamic_model_and_inputs()
    trt_gm = _compile_dynamic(model, example, dynamic_shapes, trt_inputs)

    trt_ep_path = os.path.join(tmpdir, "model.ep")
    torchtrt.save(
        trt_gm,
        trt_ep_path,
        inputs=[example],
        dynamic_shapes=dynamic_shapes,
        retrace=True,
        use_legacy_exporter=False,
    )

    deser = torchtrt.load(trt_ep_path).module()
    x_alt = torch.randn(2, 10).cuda()
    _check_outputs(
        assertions,
        model(x_alt),
        trt_gm(x_alt),
        deser(x_alt),
        "retrace=True, use_legacy_exporter=False, dynamic",
    )
    torch._dynamo.reset()


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
