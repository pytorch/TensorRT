# type: ignore
import unittest

import modelopt
import pytest
import timm
import torch
import torch_tensorrt as torchtrt
import torchvision.models as models
from torch_tensorrt.dynamo.utils import COSINE_THRESHOLD, cosine_similarity
from transformers import BertModel
from transformers.utils.fx import symbolic_trace as transformers_trace

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
def test_bert_base_uncased(ir):
    model = (
        BertModel.from_pretrained("bert-base-uncased", return_dict=False).cuda().eval()
    )
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
    torch.cuda.get_device_properties(torch.cuda.current_device()).major < 9,
    "FP8 compilation in Torch-TRT is not supported on cards older than Hopper",
)
@pytest.mark.unit
def test_base_fp8(ir):
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

    import modelopt.torch.quantization as mtq
    from modelopt.torch.quantization.utils import export_torch_mode

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
            exp_program = torch.export.export(model, (input_tensor,))
            trt_model = torchtrt.dynamo.compile(
                exp_program,
                inputs=[input_tensor],
                enabled_precisions={torch.float8_e4m3fn},
                min_block_size=1,
                debug=True,
                cache_built_engines=False,
                reuse_cached_engines=False,
            )
            outputs_trt = trt_model(input_tensor)
            assert torch.allclose(output_pyt, outputs_trt, rtol=1e-3, atol=1e-2)


@unittest.skipIf(
    modelopt.__version__ < "0.16.1",
    "Int8 quantization is supported in modelopt since 0.16.1 or later",
)
@pytest.mark.unit
def test_base_int8(ir):
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

    import modelopt.torch.quantization as mtq
    from modelopt.torch.quantization.utils import export_torch_mode

    def calibrate_loop(model):
        """Simple calibration function for testing."""
        model(input_tensor)

    input_tensor = torch.randn(1, 10).cuda()
    model = SimpleNetwork().eval().cuda()

    quant_cfg = mtq.INT8_DEFAULT_CFG
    mtq.quantize(model, quant_cfg, forward_loop=calibrate_loop)
    # model has INT8 qdq nodes at this point
    output_pyt = model(input_tensor)

    with torch.no_grad():
        with export_torch_mode():
            from torch.export._trace import _export

            exp_program = _export(model, (input_tensor,))
            trt_model = torchtrt.dynamo.compile(
                exp_program,
                inputs=[input_tensor],
                enabled_precisions={torch.int8},
                min_block_size=1,
                debug=True,
                cache_built_engines=False,
                reuse_cached_engines=False,
            )
            outputs_trt = trt_model(input_tensor)
            assert torch.allclose(output_pyt, outputs_trt, rtol=1e-3, atol=1e-2)
