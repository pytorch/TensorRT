import importlib
import os
import tempfile
import unittest

import pytest
import torch
import torch.nn as nn
import torch_tensorrt as torchtrt
from torch_tensorrt.dynamo.utils import COSINE_THRESHOLD, cosine_similarity

assertions = unittest.TestCase()

if importlib.util.find_spec("torchvision"):
    import torchvision.models as models

trt_ep_path = os.path.join(tempfile.gettempdir(), "trt.ep")


@pytest.mark.unit
def test_base_full_compile(ir):
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
def test_base_full_compile_multiple_outputs(ir):
    """
    This tests export serde functionality on a base model
    with multiple outputs which is fully TRT convertible
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
def test_no_compile(ir):
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
def test_hybrid_relu_fallback(ir):
    """
    This tests export save and load functionality on a hybrid
    model with Pytorch and TRT segments. Relu (unweighted) layer is forced to
    fallback
    """

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
def test_resnet18(ir):
    """
    This tests export save and load functionality on Resnet18 model
    """
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
def test_hybrid_conv_fallback(ir):
    """
    This tests export save and load functionality on a hybrid
    model where a conv (a weighted layer)  has been forced to fallback to Pytorch.
    """

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
def test_arange_export(ir):
    """
    This tests export save and load functionality on a arange static graph
    Here the arange output is a static constant (which is registered as input to the graph)
    in the exporter.
    """

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
def test_resnet18_dynamic(ir):
    """
    This tests export save and load functionality on Resnet18 model with dynamic shapes
    """
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
        model, (input_bs2,), dynamic_shapes=({0: dyn_batch},)
    )
    trt_module = torchtrt.dynamo.compile(exp_program, **compile_spec)

    # Reexport with dynamic dimensions
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
def test_resnet18_dynamic_fallback(ir):
    """
    This tests export save and load functionality on Resnet18 model with dynamic shapes and fallback
    """
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
        model, (input_bs2,), dynamic_shapes=({0: dyn_batch},)
    )
    trt_module = torchtrt.dynamo.compile(exp_program, **compile_spec)

    # Reexport with dynamic dimensions
    trt_exp_program = torch.export.export(
        trt_module,
        (input_bs2,),
        strict=False,
        dynamic_shapes=({0: dyn_batch},),
    )
    torch.export.save(trt_exp_program, trt_ep_path)

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
def test_bitwise_and_dynamic_fallback(ir):
    """
    This tests export save and load functionality on a bitwise_and model with dynamic shapes and fallback
    """

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
    )
    trt_module = torchtrt.dynamo.compile(exp_program, **compile_spec)

    # Reexport with dynamic dimensions
    trt_exp_program = torch.export.export(
        trt_module,
        inputs_4,
        strict=False,
        dynamic_shapes={"lhs_val": {1: dyn_dim}, "rhs_val": {0: dyn_dim}},
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
def test_random_dynamic_fallback(ir):
    """
    This tests export save and load functionality on a random model with dynamic shapes and fallback
    """

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
        model, torch_inputs_bs50, dynamic_shapes=({0: dyn_dim},)
    )
    trt_module = torchtrt.dynamo.compile(exp_program, **compile_spec)

    # Reexport with dynamic dimensions
    trt_exp_program = torch.export.export(
        trt_module, torch_inputs_bs50, strict=False, dynamic_shapes=({0: dyn_dim},)
    )
    torch.export.save(trt_exp_program, trt_ep_path)

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
