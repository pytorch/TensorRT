import importlib
import os
import platform
import unittest

import pytest
import torch
import torch_tensorrt as torchtrt
from torch_tensorrt.dynamo.utils import (
    COSINE_THRESHOLD,
    cosine_similarity,
    get_model_device,
)

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
    trt_ep_path = os.path.join(tmpdir, "trt.ep")

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

    exp_program = torchtrt.dynamo.trace(model, **compile_spec)
    trt_module = torchtrt.dynamo.compile(exp_program, **compile_spec)
    torchtrt.save(trt_module, trt_ep_path, retrace=False)

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
    torchtrt.save(trt_module, trt_ep_path, retrace=False)

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
        "cache_built_engines": False,
        "reuse_cached_engines": False,
    }

    exp_program = torchtrt.dynamo.trace(model, **compile_spec)
    trt_module = torchtrt.dynamo.compile(exp_program, **compile_spec)
    torchtrt.save(trt_module, trt_ep_path, retrace=False)

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
@pytest.mark.critical
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
    torchtrt.save(trt_module, trt_ep_path, retrace=False)

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
    torchtrt.save(trt_module, trt_ep_path, retrace=False)

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
@unittest.skipIf(
    not importlib.util.find_spec("torchvision"),
    "torchvision is not installed",
)
def test_resnet18_cpu_offload(ir, tmpdir):
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
        "offload_module_to_cpu": True,
    }

    exp_program = torchtrt.dynamo.trace(model, **compile_spec)
    trt_module = torchtrt.dynamo.compile(exp_program, **compile_spec)
    if ir == "dynamo":
        assertions.assertTrue(
            get_model_device(model).type == "cpu",
            msg="Model should be offloaded to CPU",
        )
        model.cuda()
    torchtrt.save(trt_module, trt_ep_path, retrace=False)

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
@unittest.skipIf(
    not importlib.util.find_spec("torchvision"),
    "torchvision is not installed",
)
def test_resnet18_dynamic(ir, tmpdir):
    """
    This tests export save and load functionality on Resnet18 model
    """

    trt_ep_path = os.path.join(tmpdir, "trt.ep")

    model = models.resnet18().eval().cuda()
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
        "ir": ir,
        "min_block_size": 1,
        "cache_built_engines": False,
        "reuse_cached_engines": False,
    }

    exp_program = torchtrt.dynamo.trace(model, **compile_spec)
    trt_module = torchtrt.dynamo.compile(exp_program, **compile_spec)
    torchtrt.save(trt_module, trt_ep_path, retrace=False)
    # TODO: Enable this serialization issues are fixed
    # deser_trt_module = torchtrt.load(trt_ep_path).module()
    outputs_pyt = model(input)
    outputs_trt = trt_module(input)
    cos_sim = cosine_similarity(outputs_pyt, outputs_trt[0])
    assertions.assertTrue(
        cos_sim > COSINE_THRESHOLD,
        msg=f"test_resnet18 TRT outputs don't match with the original model. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}",
    )


@unittest.skipIf(
    not importlib.util.find_spec("torchvision"), "torchvision not installed"
)
def test_resnet18_torch_exec_ops_serde(ir, tmpdir):
    """
    This tests export save and load functionality on Resnet18 model
    """

    trt_ep_path = os.path.join(tmpdir, "trt.ep")

    model = models.resnet18().eval().cuda()
    input = torch.randn((1, 3, 224, 224)).to("cuda")

    compile_spec = {
        "inputs": [input],
        "ir": ir,
        "min_block_size": 1,
        "cache_built_engines": False,
        "reuse_cached_engines": False,
        "torch_executed_ops": {torch.ops.aten.addmm, "torch.ops.aten.add"},
    }

    exp_program = torchtrt.dynamo.trace(model, **compile_spec)
    trt_module = torchtrt.dynamo.compile(exp_program, **compile_spec)
    torchtrt.save(trt_module, trt_ep_path, retrace=False)
    deser_trt_module = torchtrt.load(trt_ep_path).module()
    outputs_pyt = deser_trt_module(input)
    outputs_trt = trt_module(input)
    cos_sim = cosine_similarity(outputs_pyt, outputs_trt[0])
    assertions.assertTrue(
        cos_sim > COSINE_THRESHOLD,
        msg=f"test_resnet18 TRT outputs don't match with the original model. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}",
    )


@pytest.mark.unit
@pytest.mark.critical
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

    torchtrt.save(trt_module, trt_ep_path, retrace=False)

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
@pytest.mark.critical
def test_hybrid_conv_fallback_cpu_offload(ir, tmpdir):
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
        "offload_module_to_cpu": True,
    }

    exp_program = torchtrt.dynamo.trace(model, **compile_spec)
    trt_module = torchtrt.dynamo.compile(exp_program, **compile_spec)
    model.cuda()
    torchtrt.save(trt_module, trt_ep_path, retrace=False)

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
@pytest.mark.critical
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

    torchtrt.save(trt_module, trt_ep_path, retrace=False)

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
def test_save_load_ts(ir, tmpdir):
    """
    This tests save/load API on Torchscript format (model still compiled using dynamo workflow)
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

    ts_path = os.path.join(tmpdir, "trt.ts")
    model = MyModule().eval().cuda()
    input = torch.randn((1, 3, 224, 224)).to("cuda")

    trt_gm = torchtrt.compile(
        model,
        ir=ir,
        inputs=[input],
        min_block_size=1,
        cache_built_engines=False,
        reuse_cached_engines=False,
    )
    assertions.assertTrue(
        isinstance(trt_gm, torch.fx.GraphModule),
        msg=f"test_save_load_ts output type does not match with torch.fx.GraphModule",
    )
    outputs_trt = trt_gm(input)
    # Save it as torchscript representation
    torchtrt.save(trt_gm, ts_path, output_format="torchscript", inputs=[input])

    trt_ts_module = torchtrt.load(ts_path)
    outputs_trt_deser = trt_ts_module(input)

    cos_sim = cosine_similarity(outputs_trt, outputs_trt_deser)
    assertions.assertTrue(
        cos_sim > COSINE_THRESHOLD,
        msg=f"test_save_load_ts TRT outputs don't match with the original model. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}",
    )


@pytest.mark.unit
@unittest.skipIf(
    platform.system() != "Linux",
    "Save and load in AOT Inductor format is only supported on Linux",
)
def test_save_load_aoti(ir, tmp_path):
    """
    This tests save/load API on the AOTI format
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

    trt_gm = torchtrt.compile(
        model,
        ir=ir,
        inputs=[input],
        min_block_size=1,
        cache_built_engines=False,
        reuse_cached_engines=False,
    )
    assertions.assertTrue(
        isinstance(trt_gm, torch.fx.GraphModule),
        msg=f"test_save_load_ts output type does not match with torch.fx.GraphModule",
    )
    outputs_trt = trt_gm(input)
    print(f"{tmp_path}/trt.pt2")
    torchtrt.save(
        trt_gm,
        f"{tmp_path}/trt.pt2",
        output_format="aot_inductor",
        arg_inputs=[input],
        retrace=True,
    )

    trt_ts_module = torch._inductor.aoti_load_package(f"{tmp_path}/trt.pt2")
    outputs_trt_deser = trt_ts_module(input)

    cos_sim = cosine_similarity(outputs_trt, outputs_trt_deser)
    assertions.assertTrue(
        cos_sim > COSINE_THRESHOLD,
        msg=f"test_save_load_ts TRT outputs don't match with the original model. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}",
    )
