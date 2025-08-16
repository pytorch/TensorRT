# type: ignore
import importlib
import platform
import unittest
from importlib import metadata

import pytest
import torch
import torch_tensorrt as torchtrt
from torch_tensorrt._utils import is_tensorrt_rtx
from torch_tensorrt.dynamo.utils import COSINE_THRESHOLD, cosine_similarity

from packaging.version import Version

if importlib.util.find_spec("torchvision"):
    import timm
    import torchvision.models as models

assertions = unittest.TestCase()


@pytest.mark.unit
@unittest.skipIf(
    not importlib.util.find_spec("torchvision"), "torchvision not installed"
)
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
        "min_block_size": 8,
        "cache_built_engines": False,
        "reuse_cached_engines": False,
    }

    trt_mod = torchtrt.compile(model, **compile_spec)
    cos_sim = cosine_similarity(model(input), trt_mod(input)[0])
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
        "min_block_size": 8,
        "cache_built_engines": False,
        "reuse_cached_engines": False,
    }

    trt_mod = torchtrt.compile(model, **compile_spec)
    cos_sim = cosine_similarity(model(input), trt_mod(input)[0])
    assertions.assertTrue(
        cos_sim > COSINE_THRESHOLD,
        msg=f"Mobilenet v2 TRT outputs don't match with the original model. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}",
    )

    # Clean up model env
    torch._dynamo.reset()


@pytest.mark.unit
@unittest.skipIf(
    not importlib.util.find_spec("timm") or not importlib.util.find_spec("torchvision"),
    "timm or torchvision not installed",
)
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
        "min_block_size": 8,
        "cache_built_engines": False,
        "reuse_cached_engines": False,
    }

    trt_mod = torchtrt.compile(model, **compile_spec)
    cos_sim = cosine_similarity(model(input), trt_mod(input)[0])
    assertions.assertTrue(
        cos_sim > COSINE_THRESHOLD,
        msg=f"EfficientNet-B0 TRT outputs don't match with the original model. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}",
    )

    # Clean up model env
    torch._dynamo.reset()


@pytest.mark.unit
@unittest.skipIf(
    not importlib.util.find_spec("transformers"),
    "transformers is required to run this test",
)
def test_bert_base_uncased(ir):
    from transformers import BertModel

    model = (
        BertModel.from_pretrained("bert-base-uncased", return_dict=False).cuda().eval()
    )
    input = torch.randint(0, 2, (1, 14), dtype=torch.int32).to("cuda")
    input2 = torch.randint(0, 2, (1, 14), dtype=torch.int32).to("cuda")

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
        "min_block_size": 10,
        "cache_built_engines": False,
        "reuse_cached_engines": False,
    }
    trt_mod = torchtrt.compile(model, **compile_spec)
    model_outputs = model(input, input2)
    trt_model_outputs = trt_mod(input, input2)
    assertions.assertTrue(
        len(model_outputs) == len(trt_model_outputs),
        msg=f"Number of outputs for BERT model compilation is different with Pytorch {len(model_outputs)} and TensorRT {len(trt_model_outputs)}. Please check the compilation.",
    )

    for index in range(len(model_outputs)):
        out, trt_out = model_outputs[index], trt_model_outputs[index]
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
        "min_block_size": 8,
        "cache_built_engines": False,
        "reuse_cached_engines": False,
    }

    trt_mod = torchtrt.compile(model, **compile_spec)
    cos_sim = cosine_similarity(model(input), trt_mod(input)[0])
    assertions.assertTrue(
        cos_sim > COSINE_THRESHOLD,
        msg=f"Resnet18 Half TRT outputs don't match with the original model. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}",
    )

    # Clean up model env
    torch._dynamo.reset()


@unittest.skipIf(
    torch.cuda.get_device_capability() < (10, 0),
    "FP4 quantization requires compute capability 10.0 or later",
)
@unittest.skipIf(
    not importlib.util.find_spec("modelopt"),
    "ModelOpt is required to run this test",
)
@pytest.mark.unit
def test_base_fp4_dynamic_shapes(ir):
    import modelopt.torch.quantization as mtq
    from modelopt.torch.quantization.utils import export_torch_mode

    dtype = torch.float16

    class SimpleNetwork(torch.nn.Module):
        def __init__(self):
            super(SimpleNetwork, self).__init__()
            self.linear1 = torch.nn.Linear(
                in_features=64, out_features=32, bias=True, dtype=dtype
            )

        def forward(self, x):
            x = self.linear1(x)
            return x

    def calibrate_loop(model):
        """Simple calibration function for testing."""
        model(dummy_inputs)

    BATCH_SIZE = torch.export.Dim("BATCH_SIZE", min=16, max=128)
    batch_size = 64
    dummy_inputs = torch.ones(batch_size, 64, dtype=dtype).cuda()

    model = SimpleNetwork().eval().cuda()

    quant_cfg = mtq.NVFP4_DEFAULT_CFG
    mtq.quantize(model, quant_cfg, forward_loop=calibrate_loop)
    # model has qdq nodes at this point
    with torch.no_grad():
        with export_torch_mode():
            exp_program = torch.export.export(
                model, (dummy_inputs,), strict=False, dynamic_shapes=({0: BATCH_SIZE},)
            )

            trt_model = torchtrt.dynamo.compile(
                exp_program,
                inputs=[dummy_inputs],
                min_block_size=1,
                cache_built_engines=False,
                reuse_cached_engines=False,
                use_explicit_typing=True,
            )
            batch_size = 128
            input_tensor = torch.ones(batch_size, 64, dtype=dtype).cuda()
            expected_output = model(input_tensor)
            outputs_trt = trt_model(input_tensor)
            abs_diff = torch.abs(expected_output - outputs_trt)
            print(f"max/mean abs_diff: {abs_diff.max().item()=} {abs_diff.mean()=}")
            assert torch.allclose(expected_output, outputs_trt, rtol=0.3, atol=0.3)


@unittest.skipIf(
    torch.cuda.get_device_capability() < (10, 0),
    "FP4 quantization requires compute capability 10.0 or later",
)
@unittest.skipIf(
    not importlib.util.find_spec("modelopt"),
    "ModelOpt is required to run this test",
)
@pytest.mark.unit
def test_base_fp4_static_shapes(ir):
    import modelopt.torch.quantization as mtq
    from modelopt.torch.quantization.utils import export_torch_mode

    dtype = torch.bfloat16

    class SimpleNetwork(torch.nn.Module):
        def __init__(self):
            super(SimpleNetwork, self).__init__()
            self.linear1 = torch.nn.Linear(
                in_features=64, out_features=32, bias=True, dtype=dtype
            )

        def forward(self, x):
            x = self.linear1(x)
            return x

    def calibrate_loop(model):
        """Simple calibration function for testing."""
        model(input_tensor)

    input_tensor = torch.randn(128, 64, dtype=dtype).cuda()

    model = SimpleNetwork().eval().cuda()
    expected_output = model(input_tensor)

    quant_cfg = mtq.NVFP4_DEFAULT_CFG
    mtq.quantize(model, quant_cfg, forward_loop=calibrate_loop)
    # model has qdq nodes at this point
    with torch.no_grad():
        with export_torch_mode():
            exp_program = torch.export.export(model, (input_tensor,), strict=False)
            from torch.fx import passes

            trt_model = torchtrt.dynamo.compile(
                exp_program,
                inputs=[input_tensor],
                min_block_size=1,
                cache_built_engines=False,
                reuse_cached_engines=False,
                use_explicit_typing=True,
            )
            outputs_trt = trt_model(input_tensor)
            abs_diff = torch.abs(expected_output - outputs_trt)
            print(f"max/mean abs_diff: {abs_diff.max().item()=} {abs_diff.mean()=}")
            assert torch.allclose(expected_output, outputs_trt, rtol=0.3, atol=0.3)


@unittest.skipIf(
    torch.cuda.get_device_capability() < (8, 9),
    "FP8 quantization requires compute capability 8.9 or later",
)
@unittest.skipIf(
    not importlib.util.find_spec("modelopt"),
    "ModelOpt is required to run this test",
)
@pytest.mark.unit
def test_base_fp8(ir):
    import modelopt.torch.quantization as mtq
    from modelopt.torch.quantization.utils import export_torch_mode

    class SimpleNetwork(torch.nn.Module):
        def __init__(self):
            super(SimpleNetwork, self).__init__()
            self.linear1 = torch.nn.Linear(in_features=10, out_features=5)
            self.linear2 = torch.nn.Linear(in_features=5, out_features=1)

        def forward(self, x):
            x = self.linear1(x)
            x = torch.nn.ReLU()(x)
            x = self.linear2(x)
            return x

    def calibrate_loop(model):
        """Simple calibration function for testing."""
        model(input_tensor)

    input_tensor = torch.randn(1, 10).cuda()
    model = SimpleNetwork().eval().cuda()
    quant_cfg = mtq.FP8_DEFAULT_CFG

    mtq.quantize(model, quant_cfg, forward_loop=calibrate_loop)
    # model has FP8 qdq nodes at this point
    output_pyt = model(input_tensor)

    with torch.no_grad():
        with export_torch_mode():
            exp_program = torch.export.export(model, (input_tensor,), strict=False)
            trt_model = torchtrt.dynamo.compile(
                exp_program,
                inputs=[input_tensor],
                enabled_precisions={torch.float8_e4m3fn},
                min_block_size=1,
                cache_built_engines=False,
                reuse_cached_engines=False,
            )
            outputs_trt = trt_model(input_tensor)
            assert torch.allclose(output_pyt, outputs_trt, rtol=5e-3, atol=1e-2)


@unittest.skipIf(
    platform.system() != "Linux"
    or not importlib.util.find_spec("modelopt")
    or Version(metadata.version("nvidia-modelopt")) < Version("0.17.0"),
    "modelopt 0.17.0 or later is required, Int8 quantization is supported in modelopt since 0.17.0 or later for linux",
)
@pytest.mark.unit
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_base_int8(ir, dtype):
    import modelopt.torch.quantization as mtq
    from modelopt.torch.quantization.utils import export_torch_mode

    class SimpleNetwork(torch.nn.Module):
        def __init__(self):
            super(SimpleNetwork, self).__init__()
            self.linear1 = torch.nn.Linear(in_features=10, out_features=5)
            self.linear2 = torch.nn.Linear(in_features=5, out_features=1)

        def forward(self, x):
            x = self.linear1(x)
            x = torch.nn.ReLU()(x)
            x = self.linear2(x)
            return x

    def calibrate_loop(model):
        """Simple calibration function for testing."""
        model(input_tensor)

    input_tensor = torch.randn(1, 10).cuda().to(dtype)
    model = SimpleNetwork().eval().cuda().to(dtype)
    quant_cfg = mtq.INT8_DEFAULT_CFG
    # RTX does not support INT8 default quantization(weights+activations), only support INT8 weights only quantization
    if is_tensorrt_rtx():
        quant_cfg["quant_cfg"]["*input_quantizer"] = {"enable": False}
    mtq.quantize(model, quant_cfg, forward_loop=calibrate_loop)
    # model has INT8 qdq nodes at this point
    output_pyt = model(input_tensor)
    with torch.no_grad():
        with export_torch_mode():
            exp_program = torch.export.export(model, (input_tensor,), strict=False)
            with torchtrt.logging.debug():
                trt_model = torchtrt.dynamo.compile(
                    exp_program,
                    inputs=[input_tensor],
                    min_block_size=1,
                    cache_built_engines=False,
                    reuse_cached_engines=False,
                    truncate_double=True,
                    use_explicit_typing=True,
                )
            outputs_trt = trt_model(input_tensor)
            assert output_pyt.dtype == outputs_trt.dtype
            assert outputs_trt.dtype == dtype
            assert torch.allclose(output_pyt, outputs_trt, rtol=5e-3, atol=1e-2)


@unittest.skipIf(
    platform.system() != "Linux"
    or not importlib.util.find_spec("modelopt")
    or Version(metadata.version("nvidia-modelopt")) < Version("0.17.0"),
    "modelopt 0.17.0 or later is required, Int8 quantization is supported in modelopt since 0.17.0 or later for linux",
)
@pytest.mark.unit
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_base_int8_dynamic_shape(ir, dtype):
    import modelopt.torch.quantization as mtq
    from modelopt.torch.quantization.utils import export_torch_mode

    class SimpleNetwork(torch.nn.Module):
        def __init__(self):
            super(SimpleNetwork, self).__init__()
            self.conv = torch.nn.Conv2d(3, 3, 3)
            self.linear = torch.nn.Linear(222, 222)

        def forward(self, x):
            return self.linear(self.conv(x))

    def calibrate_loop(model):
        """Simple calibration function for testing."""
        model(input_tensor)

    BATCH_SIZE = torch.export.Dim("BATCH_SIZE", min=2, max=16)
    batch_size = 8
    input_tensor = torch.randn(batch_size, 3, 224, 224, dtype=dtype).cuda()
    model = SimpleNetwork().eval().cuda().to(dtype)

    quant_cfg = mtq.INT8_DEFAULT_CFG
    # RTX does not support INT8 default quantization(weights+activations), only support INT8 weights only quantization
    if torchtrt.tensorrt_package_name == "tensorrt_rtx":
        quant_cfg["quant_cfg"]["*input_quantizer"] = {"enable": False}
    mtq.quantize(model, quant_cfg, forward_loop=calibrate_loop)

    # model has INT8 qdq nodes at this point
    output_pyt = model(input_tensor)

    with torch.no_grad():
        with export_torch_mode():
            exp_program = torch.export.export(
                model, (input_tensor,), strict=False, dynamic_shapes=({0: BATCH_SIZE},)
            )
            trt_model = torchtrt.dynamo.compile(
                exp_program,
                inputs=[input_tensor],
                min_block_size=1,
                cache_built_engines=False,
                reuse_cached_engines=False,
                truncate_double=True,
                use_explicit_typing=True,
            )
            outputs_trt = trt_model(input_tensor)
            assert torch.allclose(output_pyt, outputs_trt, rtol=5e-2, atol=5e-2)
