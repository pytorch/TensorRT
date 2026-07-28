import importlib
import os
import unittest

import pytest
import torch
import torch.nn as nn
import torch_tensorrt as torchtrt
from torch_tensorrt.dynamo.utils import COSINE_THRESHOLD, cosine_similarity

assertions = unittest.TestCase()

if importlib.util.find_spec("torchvision"):
    import torchvision.models as models


@pytest.mark.unit
@pytest.mark.critical
def test_base_full_compile(ir, tmpdir):
    """
    This tests export serde functionality on a base model
    which is fully TRT convertible
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
                input.shape, dtype=torch.float, format=torch.contiguous_format
            )
        ],
        "ir": ir,
        "min_block_size": 1,
        "cache_built_engines": False,
        "reuse_cached_engines": False,
    }

    exp_program = torch.export.export(model, (input,), strict=False)
    trt_module = torchtrt.dynamo.compile(exp_program, **compile_spec)

    # Reexport
    trt_exp_program = torch.export.export(trt_module, (input,), strict=False)
    trt_ep_path = os.path.join(tmpdir, "trt.ep")

    torch.export.save(trt_exp_program, trt_ep_path)

    deser_trt_module = torchtrt.load(trt_ep_path).module()
    # Check Pyt and TRT exported program outputs
    cos_sim = cosine_similarity(model(input), trt_module(input)[0])
    assertions.assertTrue(
        cos_sim > COSINE_THRESHOLD,
        msg=f"test_base_model_full_compile TRT outputs don't match with the original model. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}",
    )

    # Check Pyt and deserialized TRT exported program outputs
    cos_sim = cosine_similarity(model(input), deser_trt_module(input)[0])
    assertions.assertTrue(
        cos_sim > COSINE_THRESHOLD,
        msg=f"test_base_model_full_compile TRT outputs don't match with the original model. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}",
    )


@pytest.mark.unit
@pytest.mark.critical
def test_base_full_compile_multiple_outputs(ir, tmpdir):
    """
    This tests export serde functionality on a base model
    with multiple outputs which is fully TRT convertible
    """

    trt_ep_path = os.path.join(tmpdir, "trt.ep")

    class MyModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 16, 3, stride=1, bias=True)
            self.relu = torch.nn.ReLU()

        def forward(self, x):
            conv = self.conv(x)
            conv = conv * 0.5
            relu = self.relu(conv)
            return conv, relu

    model = MyModule().eval().cuda()
    input = torch.randn((1, 3, 224, 224)).to("cuda")

    compile_spec = {
        "inputs": [
            torchtrt.Input(
                input.shape, dtype=torch.float, format=torch.contiguous_format
            )
        ],
        "ir": ir,
        "min_block_size": 1,
        "cache_built_engines": False,
        "reuse_cached_engines": False,
    }

    exp_program = torchtrt.dynamo.trace(model, **compile_spec)
    trt_module = torchtrt.dynamo.compile(exp_program, **compile_spec)

    # Reexport
    trt_exp_program = torch.export.export(trt_module, (input,), strict=False)
    torch.export.save(trt_exp_program, trt_ep_path)

    deser_trt_module = torchtrt.load(trt_ep_path).module()
    # Check Pyt and TRT exported program outputs
    outputs_pyt = model(input)
    outputs_trt = trt_module(input)
    for idx in range(len(outputs_pyt)):
        cos_sim = cosine_similarity(outputs_pyt[idx], outputs_trt[idx])
        assertions.assertTrue(
            cos_sim > COSINE_THRESHOLD,
            msg=f"test_base_full_compile_multiple_outputs TRT outputs don't match with the original model. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}",
        )

    # # Check Pyt and deserialized TRT exported program outputs
    outputs_trt_deser = deser_trt_module(input)
    for idx in range(len(outputs_pyt)):
        cos_sim = cosine_similarity(outputs_pyt[idx], outputs_trt_deser[idx])
        assertions.assertTrue(
            cos_sim > COSINE_THRESHOLD,
            msg=f"test_base_full_compile_multiple_outputs deserialized TRT outputs don't match with the original model. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}",
        )


@pytest.mark.unit
@pytest.mark.critical
def test_no_compile(ir, tmpdir):
    """
    This tests export serde functionality on a model
    which won't convert to TRT because of min_block_size=5 constraint
    """

    class MyModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 16, 3, stride=1, bias=True)
            self.relu = torch.nn.ReLU()

        def forward(self, x):
            conv = self.conv(x)
            conv = conv * 0.5
            relu = self.relu(conv)
            return conv, relu

    trt_ep_path = os.path.join(tmpdir, "trt.ep")
    model = MyModule().eval().cuda()
    input = torch.randn((1, 3, 224, 224)).to("cuda")

    compile_spec = {
        "inputs": [
            torchtrt.Input(
                input.shape, dtype=torch.float, format=torch.contiguous_format
            )
        ],
        "ir": ir,
        "cache_built_engines": False,
        "reuse_cached_engines": False,
    }

    exp_program = torchtrt.dynamo.trace(model, **compile_spec)
    trt_module = torchtrt.dynamo.compile(exp_program, **compile_spec)

    # Reexport
    trt_exp_program = torch.export.export(trt_module, (input,), strict=False)
    torch.export.save(trt_exp_program, trt_ep_path)

    deser_trt_module = torchtrt.load(trt_ep_path).module()
    # Check Pyt and TRT exported program outputs
    outputs_pyt = model(input)
    outputs_trt = trt_module(input)
    for idx in range(len(outputs_pyt)):
        cos_sim = cosine_similarity(outputs_pyt[idx], outputs_trt[idx])
        assertions.assertTrue(
            cos_sim > COSINE_THRESHOLD,
            msg=f"test_no_compile TRT outputs don't match with the original model. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}",
        )

    # # Check Pyt and deserialized TRT exported program outputs
    outputs_trt_deser = deser_trt_module(input)
    for idx in range(len(outputs_pyt)):
        cos_sim = cosine_similarity(outputs_pyt[idx], outputs_trt_deser[idx])
        assertions.assertTrue(
            cos_sim > COSINE_THRESHOLD,
            msg=f"test_no_compile deserialized TRT outputs don't match with the original model. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}",
        )


@pytest.mark.unit
def test_hybrid_relu_fallback(ir, tmpdir):
    """
    This tests export save and load functionality on a hybrid
    model with Pytorch and TRT segments. Relu (unweighted) layer is forced to
    fallback
    """

    trt_ep_path = os.path.join(tmpdir, "trt.ep")

    class MyModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 16, 3, stride=1, bias=True)
            self.relu = torch.nn.ReLU()

        def forward(self, x):
            conv = self.conv(x)
            relu = self.relu(conv)
            mul = relu * 0.5
            return mul

    model = MyModule().eval().cuda()
    input = torch.randn((1, 3, 224, 224)).to("cuda")

    compile_spec = {
        "inputs": [
            torchtrt.Input(
                input.shape, dtype=torch.float, format=torch.contiguous_format
            )
        ],
        "ir": ir,
        "min_block_size": 1,
        "torch_executed_ops": {"torch.ops.aten.relu.default"},
        "cache_built_engines": False,
        "reuse_cached_engines": False,
    }

    exp_program = torchtrt.dynamo.trace(model, **compile_spec)
    trt_module = torchtrt.dynamo.compile(exp_program, **compile_spec)

    # Reexport
    trt_exp_program = torch.export.export(trt_module, (input,), strict=False)
    torch.export.save(trt_exp_program, trt_ep_path)

    deser_trt_module = torchtrt.load(trt_ep_path).module()
    outputs_pyt = model(input)
    outputs_trt = trt_module(input)
    for idx in range(len(outputs_pyt)):
        cos_sim = cosine_similarity(outputs_pyt[idx], outputs_trt[idx])
        assertions.assertTrue(
            cos_sim > COSINE_THRESHOLD,
            msg=f"test_hybrid_relu_fallback TRT outputs don't match with the original model. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}",
        )

    outputs_trt_deser = deser_trt_module(input)
    for idx in range(len(outputs_pyt)):
        cos_sim = cosine_similarity(outputs_pyt[idx], outputs_trt_deser[idx])
        assertions.assertTrue(
            cos_sim > COSINE_THRESHOLD,
            msg=f"test_hybrid_relu_fallback deserialized TRT outputs don't match with the original model. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}",
        )


@pytest.mark.unit
@unittest.skipIf(
    not importlib.util.find_spec("torchvision"),
    "torchvision is not installed",
)
def test_resnet18(ir, tmpdir):
    """
    This tests export save and load functionality on Resnet18 model
    """

    trt_ep_path = os.path.join(tmpdir, "trt.ep")

    model = models.resnet18().eval().cuda()
    input = torch.randn((1, 3, 224, 224)).to("cuda")

    compile_spec = {
        "inputs": [
            torchtrt.Input(
                input.shape, dtype=torch.float, format=torch.contiguous_format
            )
        ],
        "ir": ir,
        "min_block_size": 1,
        "cache_built_engines": False,
        "reuse_cached_engines": False,
    }

    exp_program = torchtrt.dynamo.trace(model, **compile_spec)
    trt_module = torchtrt.dynamo.compile(exp_program, **compile_spec)

    # Reexport
    trt_exp_program = torch.export.export(trt_module, (input,), strict=False)
    torch.export.save(trt_exp_program, trt_ep_path)

    deser_trt_module = torchtrt.load(trt_ep_path).module()
    outputs_pyt = model(input)
    outputs_trt = trt_module(input)
    cos_sim = cosine_similarity(outputs_pyt, outputs_trt[0])
    assertions.assertTrue(
        cos_sim > COSINE_THRESHOLD,
        msg=f"test_resnet18 TRT outputs don't match with the original model. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}",
    )

    outputs_trt_deser = deser_trt_module(input)
    cos_sim = cosine_similarity(outputs_pyt, outputs_trt_deser[0])
    assertions.assertTrue(
        cos_sim > COSINE_THRESHOLD,
        msg=f"test_resnet18 deserialized TRT outputs don't match with the original model. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}",
    )


@pytest.mark.unit
def test_hybrid_conv_fallback(ir, tmpdir):
    """
    This tests export save and load functionality on a hybrid
    model where a conv (a weighted layer)  has been forced to fallback to Pytorch.
    """

    trt_ep_path = os.path.join(tmpdir, "trt.ep")

    class MyModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 16, 3, stride=1, bias=True)
            self.relu = torch.nn.ReLU()

        def forward(self, x):
            conv = self.conv(x)
            relu = self.relu(conv)
            mul = relu * 0.5
            return mul

    model = MyModule().eval().cuda()
    input = torch.randn((1, 3, 224, 224)).to("cuda")

    compile_spec = {
        "inputs": [
            torchtrt.Input(
                input.shape, dtype=torch.float, format=torch.contiguous_format
            )
        ],
        "ir": ir,
        "min_block_size": 1,
        "torch_executed_ops": {"torch.ops.aten.convolution.default"},
        "cache_built_engines": False,
        "reuse_cached_engines": False,
    }

    exp_program = torchtrt.dynamo.trace(model, **compile_spec)
    trt_module = torchtrt.dynamo.compile(exp_program, **compile_spec)

    # Reexport
    trt_exp_program = torch.export.export(trt_module, (input,), strict=False)
    torch.export.save(trt_exp_program, trt_ep_path)

    deser_trt_module = torchtrt.load(trt_ep_path).module()
    outputs_pyt = model(input)
    outputs_trt = trt_module(input)

    for idx in range(len(outputs_pyt)):
        cos_sim = cosine_similarity(outputs_pyt[idx], outputs_trt[idx])
        assertions.assertTrue(
            cos_sim > COSINE_THRESHOLD,
            msg=f"test_hybrid_conv_fallback TRT outputs don't match with the original model. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}",
        )

    outputs_trt_deser = deser_trt_module(input)
    for idx in range(len(outputs_pyt)):
        cos_sim = cosine_similarity(outputs_pyt[idx], outputs_trt_deser[idx])
        assertions.assertTrue(
            cos_sim > COSINE_THRESHOLD,
            msg=f"test_hybrid_conv_fallback deserialized TRT outputs don't match with the original model. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}",
        )


@pytest.mark.unit
def test_arange_export(ir, tmpdir):
    """
    This tests export save and load functionality on a arange static graph
    Here the arange output is a static constant (which is registered as input to the graph)
    in the exporter.
    """

    trt_ep_path = os.path.join(tmpdir, "trt.ep")

    class MyModule(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            x_embed = torch.arange(
                1, x.shape[-1] + 1, dtype=torch.float32, device=x.device
            )
            return x_embed

    model = MyModule().eval().cuda()
    input = torch.randn((1, 1, 128, 128)).to("cuda")

    compile_spec = {
        "inputs": [
            torchtrt.Input(
                input.shape, dtype=torch.float, format=torch.contiguous_format
            )
        ],
        "ir": ir,
        "min_block_size": 1,
        "cache_built_engines": False,
        "reuse_cached_engines": False,
    }

    exp_program = torchtrt.dynamo.trace(model, **compile_spec)
    trt_module = torchtrt.dynamo.compile(exp_program, **compile_spec)

    # Reexport
    trt_exp_program = torch.export.export(trt_module, (input,), strict=False)
    torch.export.save(trt_exp_program, trt_ep_path)

    deser_trt_module = torchtrt.load(trt_ep_path).module()
    outputs_pyt = model(input)
    outputs_trt = trt_module(input)

    for idx in range(len(outputs_pyt)):
        cos_sim = cosine_similarity(outputs_pyt[idx], outputs_trt[idx])
        assertions.assertTrue(
            cos_sim > COSINE_THRESHOLD,
            msg=f"test_arange_export TRT outputs don't match with the original model. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}",
        )

    outputs_trt_deser = deser_trt_module(input)
    for idx in range(len(outputs_pyt)):
        cos_sim = cosine_similarity(outputs_pyt[idx], outputs_trt_deser[idx])
        assertions.assertTrue(
            cos_sim > COSINE_THRESHOLD,
            msg=f"test_arange_export deserialized TRT outputs don't match with the original model. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}",
        )


@pytest.mark.unit
@unittest.skipIf(
    not importlib.util.find_spec("torchvision"),
    "torchvision is not installed",
)
def test_resnet18_dynamic(ir, tmpdir):
    """
    This tests export save and load functionality on Resnet18 model with dynamic shapes
    """

    trt_ep_path = os.path.join(tmpdir, "trt.ep")
    model = models.resnet18().eval().cuda()
    input_bs2 = torch.randn((2, 3, 224, 224)).to("cuda")

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
        "ir": ir,
        "min_block_size": 1,
        "cache_built_engines": False,
        "reuse_cached_engines": False,
    }

    dyn_batch = torch.export.Dim("batch", min=1, max=8)
    exp_program = torch.export.export(
        model, (input_bs2,), dynamic_shapes=({0: dyn_batch},), strict=False
    )
    trt_module = torchtrt.dynamo.compile(exp_program, **compile_spec)

    # Save with torch_tensorrt.save() which handles dynamic shapes properly
    torchtrt.save(
        trt_module,
        trt_ep_path,
        output_format="exported_program",
        arg_inputs=[input_bs2],
        dynamic_shapes=({0: dyn_batch},),
        retrace=True,
    )

    # TODO: Enable this serialization issues are fixed
    deser_trt_module = torchtrt.load(trt_ep_path).module()
    outputs_pyt = model(input_bs2)
    outputs_trt = trt_module(input_bs2)
    cos_sim = cosine_similarity(outputs_pyt, outputs_trt)
    assertions.assertTrue(
        cos_sim > COSINE_THRESHOLD,
        msg=f"test_resnet18_dynamic TRT outputs don't match with the original model for batch size=2. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}",
    )

    outputs_trt_deser = deser_trt_module(input_bs2)
    cos_sim = cosine_similarity(outputs_pyt, outputs_trt_deser)
    assertions.assertTrue(
        cos_sim > COSINE_THRESHOLD,
        msg=f"test_resnet18_dynamic TRT outputs don't match with the original model. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}",
    )

    input_bs6 = torch.randn((6, 3, 224, 224)).to("cuda")
    outputs_pyt = model(input_bs6)
    outputs_trt = trt_module(input_bs6)
    cos_sim = cosine_similarity(outputs_pyt, outputs_trt)
    assertions.assertTrue(
        cos_sim > COSINE_THRESHOLD,
        msg=f"test_resnet18_dynamic TRT outputs don't match with the original model. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}",
    )

    outputs_trt_deser = deser_trt_module(input_bs6)
    cos_sim = cosine_similarity(outputs_pyt, outputs_trt_deser)
    assertions.assertTrue(
        cos_sim > COSINE_THRESHOLD,
        msg=f"test_resnet18_dynamic TRT outputs don't match with the original model. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}",
    )


@pytest.mark.unit
@unittest.skipIf(
    not importlib.util.find_spec("torchvision"),
    "torchvision is not installed",
)
def test_resnet18_dynamic_manual_retrace(ir, tmpdir):
    """
    This tests export save and load functionality on Resnet18 model with dynamic shapes
    """

    trt_ep_path = os.path.join(tmpdir, "trt.ep")
    model = models.resnet18().eval().cuda()
    input_bs2 = torch.randn((2, 3, 224, 224)).to("cuda")

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
        "ir": ir,
        "min_block_size": 1,
        "cache_built_engines": False,
        "reuse_cached_engines": False,
    }

    dyn_batch = torch.export.Dim("batch", min=1, max=8)
    exp_program = torch.export.export(
        model, (input_bs2,), dynamic_shapes=({0: dyn_batch},), strict=False
    )
    trt_module = torchtrt.dynamo.compile(exp_program, **compile_spec)

    trt_exp_program = torch.export.export(
        trt_module, (input_bs2,), dynamic_shapes=({0: dyn_batch},), strict=False
    )
    torch.export.save(trt_exp_program, trt_ep_path)

    # TODO: Enable this serialization issues are fixed
    deser_trt_module = torchtrt.load(trt_ep_path).module()
    outputs_pyt = model(input_bs2)
    outputs_trt = trt_module(input_bs2)
    cos_sim = cosine_similarity(outputs_pyt, outputs_trt)
    assertions.assertTrue(
        cos_sim > COSINE_THRESHOLD,
        msg=f"test_resnet18_dynamic TRT outputs don't match with the original model for batch size=2. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}",
    )

    outputs_trt_deser = deser_trt_module(input_bs2)
    cos_sim = cosine_similarity(outputs_pyt, outputs_trt_deser)
    assertions.assertTrue(
        cos_sim > COSINE_THRESHOLD,
        msg=f"test_resnet18_dynamic TRT outputs don't match with the original model. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}",
    )

    input_bs6 = torch.randn((6, 3, 224, 224)).to("cuda")
    outputs_pyt = model(input_bs6)
    outputs_trt = trt_module(input_bs6)
    cos_sim = cosine_similarity(outputs_pyt, outputs_trt)
    assertions.assertTrue(
        cos_sim > COSINE_THRESHOLD,
        msg=f"test_resnet18_dynamic TRT outputs don't match with the original model. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}",
    )

    outputs_trt_deser = deser_trt_module(input_bs6)
    cos_sim = cosine_similarity(outputs_pyt, outputs_trt_deser)
    assertions.assertTrue(
        cos_sim > COSINE_THRESHOLD,
        msg=f"test_resnet18_dynamic TRT outputs don't match with the original model. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}",
    )


@pytest.mark.unit
@unittest.skipIf(
    not importlib.util.find_spec("torchvision"),
    "torchvision is not installed",
)
def test_resnet18_dynamic_fallback(ir, tmpdir):
    """
    This tests export save and load functionality on Resnet18 model with dynamic shapes and fallback
    """

    trt_ep_path = os.path.join(tmpdir, "trt.ep")
    model = models.resnet18().eval().cuda()
    input_bs2 = torch.randn((2, 3, 224, 224)).to("cuda")

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
        "ir": ir,
        "torch_executed_ops": {"torch.ops.aten.convolution.default"},
        "min_block_size": 1,
        "cache_built_engines": False,
        "reuse_cached_engines": False,
    }

    dyn_batch = torch.export.Dim("batch", min=1, max=8)

    exp_program = torch.export.export(
        model, (input_bs2,), dynamic_shapes=({0: dyn_batch},), strict=False
    )
    trt_module = torchtrt.dynamo.compile(exp_program, **compile_spec)

    torchtrt.save(
        trt_module,
        trt_ep_path,
        output_format="exported_program",
        arg_inputs=[input_bs2],
        dynamic_shapes=({0: dyn_batch},),
        retrace=True,
    )

    deser_trt_module = torchtrt.load(trt_ep_path).module()
    outputs_pyt = model(input_bs2)
    outputs_trt = trt_module(input_bs2)
    cos_sim = cosine_similarity(outputs_pyt, outputs_trt)
    assertions.assertTrue(
        cos_sim > COSINE_THRESHOLD,
        msg=f"test_resnet18_dynamic TRT outputs don't match with the original model for batch size=2. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}",
    )

    outputs_trt_deser = deser_trt_module(input_bs2)
    cos_sim = cosine_similarity(outputs_pyt, outputs_trt_deser)
    assertions.assertTrue(
        cos_sim > COSINE_THRESHOLD,
        msg=f"test_resnet18_dynamic TRT outputs don't match with the original model. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}",
    )

    input_bs6 = torch.randn((6, 3, 224, 224)).to("cuda")
    outputs_pyt = model(input_bs6)
    outputs_trt = trt_module(input_bs6)
    cos_sim = cosine_similarity(outputs_pyt, outputs_trt)
    assertions.assertTrue(
        cos_sim > COSINE_THRESHOLD,
        msg=f"test_resnet18_dynamic TRT outputs don't match with the original model. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}",
    )

    outputs_trt_deser = deser_trt_module(input_bs6)
    cos_sim = cosine_similarity(outputs_pyt, outputs_trt_deser)
    assertions.assertTrue(
        cos_sim > COSINE_THRESHOLD,
        msg=f"test_resnet18_dynamic TRT outputs don't match with the original model. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}",
    )


@pytest.mark.unit
def test_bitwise_and_dynamic_fallback(ir, tmpdir):
    """
    This tests export save and load functionality on a bitwise_and model with dynamic shapes and fallback
    """

    trt_ep_path = os.path.join(tmpdir, "trt.ep")

    class bitwise_and(torch.nn.Module):
        def forward(self, lhs_val, rhs_val):
            return torch.ops.aten.bitwise_and.Tensor(lhs_val, rhs_val)

    dyn_dim = torch.export.Dim("dyn_dim", min=3, max=6)
    lhs_4 = torch.randint(0, 2, (2, 4, 2), dtype=bool, device="cuda")
    rhs_4 = torch.randint(0, 2, (4, 2), dtype=bool, device="cuda")
    inputs_4 = (lhs_4, rhs_4)
    torchtrt_inputs = [
        torchtrt.Input(shape=lhs_4.shape, dtype=torch.bool),
        torchtrt.Input(shape=rhs_4.shape, dtype=torch.bool),
    ]
    model = bitwise_and()
    compile_spec = {
        "inputs": torchtrt_inputs,
        "ir": ir,
        "min_block_size": 1,
        "cache_built_engines": False,
        "reuse_cached_engines": False,
    }

    exp_program = torch.export.export(
        model,
        inputs_4,
        dynamic_shapes={"lhs_val": {1: dyn_dim}, "rhs_val": {0: dyn_dim}},
        strict=False,
    )
    trt_module = torchtrt.dynamo.compile(exp_program, **compile_spec)

    # Save with torch_tensorrt.save() which handles dynamic shapes properly
    torchtrt.save(
        trt_module,
        trt_ep_path,
        output_format="exported_program",
        arg_inputs=list(inputs_4),
        dynamic_shapes={"lhs_val": {1: dyn_dim}, "rhs_val": {0: dyn_dim}},
        retrace=True,
    )

    deser_trt_module = torchtrt.load(trt_ep_path).module()
    outputs_pyt = model(*inputs_4)
    outputs_trt = trt_module(*inputs_4)
    cos_sim = cosine_similarity(outputs_pyt, outputs_trt)
    assertions.assertTrue(
        cos_sim > COSINE_THRESHOLD,
        msg=f"test_bitwise_and_dynamic_fallback TRT outputs don't match with the original model with inputs_4. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}",
    )

    outputs_trt_deser = deser_trt_module(*inputs_4)
    cos_sim = cosine_similarity(outputs_pyt, outputs_trt_deser)
    assertions.assertTrue(
        cos_sim > COSINE_THRESHOLD,
        msg=f"test_bitwise_and_dynamic_fallback TRT outputs don't match with the original model with inputs_4. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}",
    )

    lhs_6 = torch.randint(0, 2, (2, 6, 2), dtype=bool, device="cuda")
    rhs_6 = torch.randint(0, 2, (6, 2), dtype=bool, device="cuda")
    inputs_6 = (lhs_6, rhs_6)

    outputs_pyt = model(*inputs_6)
    outputs_trt = trt_module(*inputs_6)
    cos_sim = cosine_similarity(outputs_pyt, outputs_trt)
    assertions.assertTrue(
        cos_sim > COSINE_THRESHOLD,
        msg=f"test_bitwise_and_dynamic_fallback TRT outputs don't match with the original model with inputs_6. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}",
    )

    outputs_trt_deser = deser_trt_module(*inputs_6)
    cos_sim = cosine_similarity(outputs_pyt, outputs_trt_deser)
    assertions.assertTrue(
        cos_sim > COSINE_THRESHOLD,
        msg=f"test_bitwise_and_dynamic_fallback TRT outputs don't match with the original model with inputs_6. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}",
    )


@pytest.mark.unit
def test_bitwise_and_dynamic_fallback_manual_reexport(ir, tmpdir):
    """
    This tests export save and load functionality on a bitwise_and model with dynamic shapes and fallback
    """

    trt_ep_path = os.path.join(tmpdir, "trt.ep")

    class bitwise_and(torch.nn.Module):
        def forward(self, lhs_val, rhs_val):
            return torch.ops.aten.bitwise_and.Tensor(lhs_val, rhs_val)

    dyn_dim = torch.export.Dim("dyn_dim", min=3, max=6)
    lhs_4 = torch.randint(0, 2, (2, 4, 2), dtype=bool, device="cuda")
    rhs_4 = torch.randint(0, 2, (4, 2), dtype=bool, device="cuda")
    inputs_4 = (lhs_4, rhs_4)
    torchtrt_inputs = [
        torchtrt.Input(shape=lhs_4.shape, dtype=torch.bool),
        torchtrt.Input(shape=rhs_4.shape, dtype=torch.bool),
    ]
    model = bitwise_and()
    compile_spec = {
        "inputs": torchtrt_inputs,
        "ir": ir,
        "min_block_size": 1,
        "cache_built_engines": False,
        "reuse_cached_engines": False,
    }

    exp_program = torch.export.export(
        model,
        inputs_4,
        dynamic_shapes={"lhs_val": {1: dyn_dim}, "rhs_val": {0: dyn_dim}},
        strict=False,
    )
    trt_module = torchtrt.dynamo.compile(exp_program, **compile_spec)

    trt_exp_program = torch.export.export(
        trt_module,
        inputs_4,
        dynamic_shapes={"lhs_val": {1: dyn_dim}, "rhs_val": {0: dyn_dim}},
        strict=False,
    )
    torch.export.save(trt_exp_program, trt_ep_path)

    deser_trt_module = torchtrt.load(trt_ep_path).module()
    outputs_pyt = model(*inputs_4)
    outputs_trt = trt_module(*inputs_4)
    cos_sim = cosine_similarity(outputs_pyt, outputs_trt)
    assertions.assertTrue(
        cos_sim > COSINE_THRESHOLD,
        msg=f"test_bitwise_and_dynamic_fallback TRT outputs don't match with the original model with inputs_4. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}",
    )

    outputs_trt_deser = deser_trt_module(*inputs_4)
    cos_sim = cosine_similarity(outputs_pyt, outputs_trt_deser)
    assertions.assertTrue(
        cos_sim > COSINE_THRESHOLD,
        msg=f"test_bitwise_and_dynamic_fallback TRT outputs don't match with the original model with inputs_4. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}",
    )

    lhs_6 = torch.randint(0, 2, (2, 6, 2), dtype=bool, device="cuda")
    rhs_6 = torch.randint(0, 2, (6, 2), dtype=bool, device="cuda")
    inputs_6 = (lhs_6, rhs_6)

    outputs_pyt = model(*inputs_6)
    outputs_trt = trt_module(*inputs_6)
    cos_sim = cosine_similarity(outputs_pyt, outputs_trt)
    assertions.assertTrue(
        cos_sim > COSINE_THRESHOLD,
        msg=f"test_bitwise_and_dynamic_fallback TRT outputs don't match with the original model with inputs_6. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}",
    )

    outputs_trt_deser = deser_trt_module(*inputs_6)
    cos_sim = cosine_similarity(outputs_pyt, outputs_trt_deser)
    assertions.assertTrue(
        cos_sim > COSINE_THRESHOLD,
        msg=f"test_bitwise_and_dynamic_fallback TRT outputs don't match with the original model with inputs_6. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}",
    )


@pytest.mark.unit
def test_random_dynamic_fallback(ir, tmpdir):
    """
    This tests export save and load functionality on a random model with dynamic shapes and fallback
    """

    trt_ep_path = os.path.join(tmpdir, "trt.ep")

    class NeuralNetwork(nn.Module):
        def __init__(self):
            super(NeuralNetwork, self).__init__()
            self.linear_relu_stack = nn.Sequential(
                nn.Linear(28 * 28, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 10),
            )

        def forward(self, x):
            logits = self.linear_relu_stack(x)
            return logits

    model = NeuralNetwork().cuda()
    inputs = [
        torchtrt.Input(
            min_shape=(1, 784),
            opt_shape=(50, 784),
            max_shape=(64, 784),
            dtype=torch.float32,
        )
    ]
    torch_inputs_bs50 = (torch.randn((50, 784), dtype=torch.float32).cuda(),)
    compile_spec = {
        "inputs": inputs,
        "ir": ir,
        "min_block_size": 1,
        "cache_built_engines": False,
        "reuse_cached_engines": False,
    }

    dyn_dim = torch.export.Dim("batch", min=1, max=64)
    exp_program = torch.export.export(
        model, torch_inputs_bs50, dynamic_shapes=({0: dyn_dim},), strict=False
    )
    trt_module = torchtrt.dynamo.compile(exp_program, **compile_spec)

    # Save with torch_tensorrt.save() which handles dynamic shapes properly
    torchtrt.save(
        trt_module,
        trt_ep_path,
        output_format="exported_program",
        arg_inputs=list(torch_inputs_bs50),
        dynamic_shapes=({0: dyn_dim},),
        retrace=True,
    )

    # Test with BS=50
    deser_trt_module = torchtrt.load(trt_ep_path).module()
    outputs_pyt = model(*torch_inputs_bs50)
    outputs_trt = trt_module(*torch_inputs_bs50)
    cos_sim = cosine_similarity(outputs_pyt, outputs_trt)
    assertions.assertTrue(
        cos_sim > COSINE_THRESHOLD,
        msg=f"test_random_dynamic_fallback TRT outputs don't match with the original model with inputs bs=50. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}",
    )

    outputs_trt_deser = deser_trt_module(*torch_inputs_bs50)
    cos_sim = cosine_similarity(outputs_pyt, outputs_trt_deser)
    assertions.assertTrue(
        cos_sim > COSINE_THRESHOLD,
        msg=f"test_random_dynamic_fallback TRT outputs don't match with the original model with inputs bs=50. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}",
    )

    # Test with BS=62
    torch_inputs_bs62 = (torch.randn((62, 784), dtype=torch.float32).cuda(),)
    outputs_pyt = model(*torch_inputs_bs62)
    outputs_trt = trt_module(*torch_inputs_bs62)
    cos_sim = cosine_similarity(outputs_pyt, outputs_trt)
    assertions.assertTrue(
        cos_sim > COSINE_THRESHOLD,
        msg=f"test_random_dynamic_fallback TRT outputs don't match with the original model with inputs bs=62. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}",
    )

    outputs_trt_deser = deser_trt_module(*torch_inputs_bs62)
    cos_sim = cosine_similarity(outputs_pyt, outputs_trt_deser)
    assertions.assertTrue(
        cos_sim > COSINE_THRESHOLD,
        msg=f"test_random_dynamic_fallback TRT outputs don't match with the original model with inputs bs=62. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}",
    )


@pytest.mark.unit
@unittest.skipIf(
    not importlib.util.find_spec("torchvision"),
    "torchvision is not installed",
)
def test_save_with_dynamic_shapes_api(ir, tmpdir):
    """
    This tests the torch_tensorrt.save() API with dynamic_shapes parameter
    to preserve dynamic shape specifications during serialization
    """

    trt_ep_path = os.path.join(tmpdir, "trt_dynamic.ep")
    model = models.resnet18().eval().cuda()
    input_bs2 = torch.randn((2, 3, 224, 224)).to("cuda")

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
        "ir": ir,
        "min_block_size": 1,
        "cache_built_engines": False,
        "reuse_cached_engines": False,
    }

    # Define dynamic shapes
    dyn_batch = torch.export.Dim("batch", min=1, max=8)
    dynamic_shapes = {"x": {0: dyn_batch}}

    # Export with dynamic shapes
    exp_program = torch.export.export(
        model, (input_bs2,), dynamic_shapes=dynamic_shapes, strict=False
    )
    trt_module = torchtrt.dynamo.compile(exp_program, **compile_spec)

    # Use the new torch_tensorrt.save() API with dynamic_shapes parameter
    # Using retrace=True (should work now with fixed meta kernel and symbolic input handling)
    torchtrt.save(
        trt_module,
        trt_ep_path,
        output_format="exported_program",
        arg_inputs=[input_bs2],
        dynamic_shapes=dynamic_shapes,  # Preserve dynamic shapes
        retrace=True,
    )

    # Load and test with different batch sizes
    deser_trt_module = torchtrt.load(trt_ep_path).module()

    # Test with batch size 2
    outputs_pyt = model(input_bs2)
    outputs_trt = trt_module(input_bs2)
    outputs_trt_deser = deser_trt_module(input_bs2)

    cos_sim = cosine_similarity(outputs_pyt, outputs_trt)
    assertions.assertTrue(
        cos_sim > COSINE_THRESHOLD,
        msg=f"test_save_with_dynamic_shapes_api TRT outputs don't match with the original model for batch size=2. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}",
    )

    cos_sim = cosine_similarity(outputs_pyt, outputs_trt_deser)
    assertions.assertTrue(
        cos_sim > COSINE_THRESHOLD,
        msg=f"test_save_with_dynamic_shapes_api deserialized TRT outputs don't match with the original model for batch size=2. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}",
    )

    # Test with batch size 6
    input_bs6 = torch.randn((6, 3, 224, 224)).to("cuda")
    outputs_pyt = model(input_bs6)
    outputs_trt = trt_module(input_bs6)
    outputs_trt_deser = deser_trt_module(input_bs6)

    cos_sim = cosine_similarity(outputs_pyt, outputs_trt)
    assertions.assertTrue(
        cos_sim > COSINE_THRESHOLD,
        msg=f"test_save_with_dynamic_shapes_api TRT outputs don't match with the original model for batch size=6. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}",
    )

    cos_sim = cosine_similarity(outputs_pyt, outputs_trt_deser)
    assertions.assertTrue(
        cos_sim > COSINE_THRESHOLD,
        msg=f"test_save_with_dynamic_shapes_api deserialized TRT outputs don't match with the original model for batch size=6. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}",
    )


@pytest.mark.unit
@unittest.skipIf(
    not importlib.util.find_spec("torchvision"),
    "torchvision is not installed",
)
def test_save_with_input_objects_inferred_dynamic_shapes(ir, tmpdir):
    """
    This tests the torch_tensorrt.save() API with torch_tensorrt.Input objects
    that have min/opt/max shapes. The dynamic_shapes should be inferred automatically.
    This is Method 2 - the recommended approach.
    """

    trt_ep_path = os.path.join(tmpdir, "trt_input_inferred.ep")
    model = models.resnet18().eval().cuda()

    # Define Input objects with dynamic shapes
    compile_inputs = [
        torchtrt.Input(
            min_shape=(1, 3, 224, 224),
            opt_shape=(4, 3, 224, 224),
            max_shape=(8, 3, 224, 224),
            dtype=torch.float32,
            name="x",
        )
    ]

    compile_spec = {
        "arg_inputs": compile_inputs,
        "ir": ir,
        "min_block_size": 1,
        "cache_built_engines": False,
        "reuse_cached_engines": False,
    }

    # Note: We're NOT using torch.export.export here, going directly through compile
    trt_module = torchtrt.compile(model, **compile_spec)

    # Use the new torch_tensorrt.save() API with Input objects
    # Dynamic shapes should be inferred automatically - no explicit dynamic_shapes needed!
    # Note: retrace=False because retracing a TRT-compiled module with dynamic shapes
    # causes issues with unbacked symints from TensorRT runtime
    torchtrt.save(
        trt_module,
        trt_ep_path,
        output_format="exported_program",
        arg_inputs=compile_inputs,  # Pass Input objects, not tensors
        retrace=True,
    )

    # Load and test with different batch sizes
    deser_trt_module = torchtrt.load(trt_ep_path).module()

    # Test with batch size 2
    input_bs2 = torch.randn((2, 3, 224, 224)).to("cuda")
    outputs_pyt = model(input_bs2)
    outputs_trt = trt_module(input_bs2)
    outputs_trt_deser = deser_trt_module(input_bs2)

    cos_sim = cosine_similarity(outputs_pyt, outputs_trt)
    assertions.assertTrue(
        cos_sim > COSINE_THRESHOLD,
        msg=f"test_save_with_input_objects_inferred_dynamic_shapes TRT outputs don't match for batch size=2. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}",
    )

    cos_sim = cosine_similarity(outputs_pyt, outputs_trt_deser)
    assertions.assertTrue(
        cos_sim > COSINE_THRESHOLD,
        msg=f"test_save_with_input_objects_inferred_dynamic_shapes deserialized TRT outputs don't match for batch size=2. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}",
    )

    # Test with batch size 7
    input_bs7 = torch.randn((7, 3, 224, 224)).to("cuda")
    outputs_pyt = model(input_bs7)
    outputs_trt = trt_module(input_bs7)
    outputs_trt_deser = deser_trt_module(input_bs7)

    cos_sim = cosine_similarity(outputs_pyt, outputs_trt)
    assertions.assertTrue(
        cos_sim > COSINE_THRESHOLD,
        msg=f"test_save_with_input_objects_inferred_dynamic_shapes TRT outputs don't match for batch size=7. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}",
    )

    cos_sim = cosine_similarity(outputs_pyt, outputs_trt_deser)
    assertions.assertTrue(
        cos_sim > COSINE_THRESHOLD,
        msg=f"test_save_with_input_objects_inferred_dynamic_shapes deserialized TRT outputs don't match for batch size=7. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}",
    )


@pytest.mark.unit
def test_save_inferred_dynamic_shapes_multiple_dimensions(ir, tmpdir):
    """
    Test automatic dynamic shape inference with multiple dynamic dimensions
    (batch, height, width)

    NOTE: This test is skipped because torch.export cannot properly serialize
    ExportedPrograms with multiple dynamic dimensions when retrace=False (causes
    missing value range info), and retrace=True causes unbacked symint issues with
    TRT-compiled modules. Use test_save_with_dynamic_shapes_api with explicit
    dynamic_shapes parameter instead.
    """

    trt_ep_path = os.path.join(tmpdir, "trt_multi_dim.ep")

    class ConvModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 16, 3, padding=1)

        def forward(self, x):
            return self.conv(x)

    model = ConvModel().eval().cuda()

    # Define Input with 3 dynamic dimensions
    compile_inputs = [
        torchtrt.Input(
            min_shape=(1, 3, 64, 64),
            opt_shape=(4, 3, 256, 256),
            max_shape=(8, 3, 512, 512),
            dtype=torch.float32,
            name="x",
        )
    ]

    compile_spec = {
        "inputs": compile_inputs,
        "ir": ir,
        "min_block_size": 1,
        "cache_built_engines": False,
        "reuse_cached_engines": False,
    }

    # Generate dynamic shape specs
    dyn_batch = torch.export.Dim("batch", min=1, max=8)
    dyn_height = torch.export.Dim("height", min=64, max=512)
    dyn_width = torch.export.Dim("width", min=64, max=512)
    dynamic_shapes = {"x": {0: dyn_batch, 2: dyn_height, 3: dyn_width}}

    trt_module = torchtrt.compile(model, **compile_spec)

    # Save with automatic inference of all 3 dynamic dimensions
    # retrace=True now works correctly with dynamic shapes
    torchtrt.save(
        trt_module,
        trt_ep_path,
        output_format="exported_program",
        arg_inputs=compile_inputs,
        dynamic_shapes=dynamic_shapes,
        retrace=True,
    )

    # Load and test with various sizes
    deser_trt_module = torchtrt.load(trt_ep_path).module()

    # Test different combinations of batch, height, width
    test_shapes = [
        (2, 3, 128, 128),
        (6, 3, 384, 384),
        (1, 3, 64, 64),  # Min
        (8, 3, 512, 512),  # Max
    ]

    for shape in test_shapes:
        input_tensor = torch.randn(shape).to("cuda")
        outputs_pyt = model(input_tensor)
        outputs_trt = trt_module(input_tensor)
        outputs_trt_deser = deser_trt_module(input_tensor)

        cos_sim = cosine_similarity(outputs_pyt, outputs_trt)
        assertions.assertTrue(
            cos_sim > COSINE_THRESHOLD,
            msg=f"test_save_inferred_dynamic_shapes_multiple_dimensions TRT outputs don't match for shape {shape}. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}",
        )

        cos_sim = cosine_similarity(outputs_pyt, outputs_trt_deser)
        assertions.assertTrue(
            cos_sim > COSINE_THRESHOLD,
            msg=f"test_save_inferred_dynamic_shapes_multiple_dimensions deserialized TRT outputs don't match for shape {shape}. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}",
        )


@pytest.mark.unit
def test_save_mixed_static_dynamic_inputs(ir, tmpdir):
    """
    Test saving with mixed static (tensor) and dynamic (Input) inputs

    NOTE: This scenario requires explicit dynamic_shapes because automatic inference
    cannot distinguish between dimensions that should be independent vs. equal.
    """

    trt_ep_path = os.path.join(tmpdir, "trt_mixed.ep")

    class MixedInputModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(10, 5)

        def forward(self, x, bias):
            # x has dynamic batch, bias is a fixed-size tensor that broadcasts
            out = self.linear(x)
            # bias shape [1, 5] broadcasts to [batch, 5]
            return out + bias

    model = MixedInputModel().eval().cuda()

    compile_inputs = [
        torchtrt.Input(
            min_shape=(1, 10),
            opt_shape=(4, 10),
            max_shape=(8, 10),
            dtype=torch.float32,
            name="x",
        ),
        torchtrt.Input(
            shape=(1, 5),  # Fixed size bias
            dtype=torch.float32,
            name="bias",
        ),
    ]

    compile_spec = {
        "inputs": compile_inputs,
        "ir": ir,
        "min_block_size": 1,
        "cache_built_engines": False,
        "reuse_cached_engines": False,
    }

    trt_module = torchtrt.compile(model, **compile_spec)

    # Save with explicit dynamic_shapes
    torchtrt.save(
        trt_module,
        trt_ep_path,
        output_format="exported_program",
        arg_inputs=compile_inputs,
        retrace=True,
    )

    deser_trt_module = torchtrt.load(trt_ep_path).module()

    # Test with different batch sizes for dynamic input
    for batch_size in [2, 6]:
        input_x = torch.randn(batch_size, 10).cuda()
        input_bias = torch.randn(1, 5).cuda()  # Same fixed bias

        outputs_pyt = model(input_x, input_bias)
        outputs_trt_deser = deser_trt_module(input_x, input_bias)

        cos_sim = cosine_similarity(outputs_pyt, outputs_trt_deser)
        assertions.assertTrue(
            cos_sim > COSINE_THRESHOLD,
            msg=f"test_save_mixed_static_dynamic_inputs outputs don't match for batch size {batch_size}. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}",
        )


@pytest.mark.unit
def test_save_with_kwarg_inputs_dynamic(ir, tmpdir):
    """
    Test saving with dynamic shapes in kwarg_inputs

    NOTE: When multiple inputs share the same dynamic dimension (e.g., batch size),
    you must explicitly declare this by sharing a Dim object:

        batch = Dim("batch", min=1, max=8)
        dynamic_shapes = {"x": {0: batch}, "mask": {0: batch}}

    Automatic inference creates separate Dim objects which causes torch.export
    to detect an equality constraint violation.
    """

    trt_ep_path = os.path.join(tmpdir, "trt_kwargs.ep")

    class KwargModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(10, 5)

        def forward(self, x, *, mask):
            # Apply linear transformation and multiply by mask
            out = self.linear(x)
            out = out * mask
            return out

    model = KwargModel().eval().cuda()

    # Create example tensors for export
    example_x = torch.randn(4, 10).cuda()
    example_mask = torch.randn(4, 5).cuda()

    # Define dynamic shapes with shared batch dimension
    # Both inputs share the same batch Dim object to express equality constraint
    batch = torch.export.Dim("batch", min=1, max=8)
    dynamic_shapes = {"x": {0: batch}, "mask": {0: batch}}

    # Step 1: Export with torch.export
    exp_program = torch.export.export(
        model,
        (example_x,),
        {"mask": example_mask},
        dynamic_shapes=dynamic_shapes,
        strict=False,
    )

    # Step 2: Compile with TensorRT using torch_tensorrt.dynamo.compile
    compile_inputs = [
        torchtrt.Input(
            min_shape=(1, 10),
            opt_shape=(4, 10),
            max_shape=(8, 10),
            dtype=torch.float32,
            name="x",
        ),
        torchtrt.Input(
            min_shape=(1, 5),
            opt_shape=(4, 5),
            max_shape=(8, 5),
            dtype=torch.float32,
            name="mask",
        ),
    ]

    compile_spec = {
        "inputs": compile_inputs,
        "ir": ir,
        "min_block_size": 1,
        "cache_built_engines": False,
        "reuse_cached_engines": False,
    }

    trt_module = torchtrt.dynamo.compile(exp_program, **compile_spec)

    # Save with explicit dynamic_shapes
    torchtrt.save(
        trt_module,
        trt_ep_path,
        output_format="exported_program",
        arg_inputs=(example_x,),
        kwarg_inputs={"mask": example_mask},
        dynamic_shapes=dynamic_shapes,
        retrace=True,
    )

    deser_trt_module = torchtrt.load(trt_ep_path).module()

    # Test with different batch sizes
    for batch_size in [2, 6]:
        input_x = torch.randn(batch_size, 10).cuda()
        input_mask = torch.randn(batch_size, 5).cuda()

        outputs_pyt = model(input_x, mask=input_mask)
        outputs_trt_deser = deser_trt_module(input_x, mask=input_mask)

        cos_sim = cosine_similarity(outputs_pyt, outputs_trt_deser)
        assertions.assertTrue(
            cos_sim > COSINE_THRESHOLD,
            msg=f"test_save_with_kwarg_inputs_dynamic outputs don't match for batch size {batch_size}. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}",
        )


@pytest.mark.unit
def test_explicit_dynamic_shapes_takes_precedence(ir, tmpdir):
    """
    Test that explicit dynamic_shapes parameter takes precedence over
    inferred dynamic shapes from Input objects
    """

    trt_ep_path = os.path.join(tmpdir, "trt_precedence.ep")

    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(10, 5)

        def forward(self, x):
            return self.linear(x)

    model = SimpleModel().eval().cuda()
    example_input = torch.randn(4, 10).cuda()

    # Define both Input objects AND explicit dynamic_shapes
    compile_inputs = [
        torchtrt.Input(
            min_shape=(1, 10),
            opt_shape=(4, 10),
            max_shape=(8, 10),
            dtype=torch.float32,
            name="x",
        )
    ]

    # Explicit dynamic_shapes with custom naming
    dyn_batch = torch.export.Dim("custom_batch_name", min=1, max=8)
    dynamic_shapes = {"x": {0: dyn_batch}}

    exp_program = torch.export.export(
        model, (example_input,), dynamic_shapes=dynamic_shapes, strict=False
    )

    compile_spec = {
        "inputs": compile_inputs,
        "ir": ir,
        "min_block_size": 1,
        "cache_built_engines": False,
        "reuse_cached_engines": False,
    }

    trt_module = torchtrt.dynamo.compile(exp_program, **compile_spec)

    # Save with BOTH compile_inputs and explicit dynamic_shapes
    # Explicit should take precedence
    # retrace=True now works correctly with dynamic shapes
    torchtrt.save(
        trt_module,
        trt_ep_path,
        output_format="exported_program",
        arg_inputs=[example_input],
        dynamic_shapes=dynamic_shapes,  # Explicit takes precedence
        retrace=True,
    )

    deser_trt_module = torchtrt.load(trt_ep_path).module()

    # Test with different batch sizes
    for batch_size in [2, 7]:
        input_tensor = torch.randn(batch_size, 10).cuda()
        outputs_pyt = model(input_tensor)
        outputs_trt_deser = deser_trt_module(input_tensor)

        cos_sim = cosine_similarity(outputs_pyt, outputs_trt_deser)
        assertions.assertTrue(
            cos_sim > COSINE_THRESHOLD,
            msg=f"test_explicit_dynamic_shapes_takes_precedence outputs don't match for batch size {batch_size}. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}",
        )


@pytest.mark.unit
def test_save_static_inputs_no_dynamic_inference(ir, tmpdir):
    """
    Test that static Input objects (without min/opt/max) don't trigger
    dynamic shape inference
    """

    trt_ep_path = os.path.join(tmpdir, "trt_static.ep")

    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(10, 5)

        def forward(self, x):
            return self.linear(x)

    model = SimpleModel().eval().cuda()

    # Static Input (single shape, not min/opt/max)
    compile_inputs = [torchtrt.Input(shape=(4, 10), dtype=torch.float32, name="x")]

    compile_spec = {
        "inputs": compile_inputs,
        "ir": ir,
        "min_block_size": 1,
        "cache_built_engines": False,
        "reuse_cached_engines": False,
    }

    trt_module = torchtrt.compile(model, **compile_spec)

    # Save - should NOT infer dynamic shapes (all inputs are static)
    torchtrt.save(
        trt_module,
        trt_ep_path,
        output_format="exported_program",
        arg_inputs=compile_inputs,
        retrace=True,
    )

    deser_trt_module = torchtrt.load(trt_ep_path).module()

    # Should only work with the exact shape
    input_tensor = torch.randn(4, 10).cuda()
    outputs_pyt = model(input_tensor)
    outputs_trt_deser = deser_trt_module(input_tensor)

    cos_sim = cosine_similarity(outputs_pyt, outputs_trt_deser)
    assertions.assertTrue(
        cos_sim > COSINE_THRESHOLD,
        msg=f"test_save_static_inputs_no_dynamic_inference outputs don't match. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}",
    )
