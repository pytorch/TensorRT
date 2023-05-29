import torch
import timm
import pytest

import torch_tensorrt as torchtrt
import torchvision.models as models

from transformers import BertModel

from torch_tensorrt.dynamo.common_utils.test_utils import (
    COSINE_THRESHOLD,
    cosine_similarity,
)


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
    }

    trt_mod = torchtrt.compile(model, **compile_spec)
    cos_sim = cosine_similarity(model(input), trt_mod(input))
    assert (
        cos_sim > COSINE_THRESHOLD,
        f"Resnet18 TRT outputs don't match with the original model. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}",
    )

    # Clean up model env
    torch._dynamo.reset()

    with torch.no_grad():
        torch.cuda.empty_cache()


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
    }

    trt_mod = torchtrt.compile(model, **compile_spec)
    cos_sim = cosine_similarity(model(input), trt_mod(input))
    assert (
        cos_sim > COSINE_THRESHOLD,
        f"Mobilenet v2 TRT outputs don't match with the original model. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}",
    )

    # Clean up model env
    torch._dynamo.reset()

    with torch.no_grad():
        torch.cuda.empty_cache()


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
    }

    trt_mod = torchtrt.compile(model, **compile_spec)
    cos_sim = cosine_similarity(model(input), trt_mod(input))
    assert (
        cos_sim > COSINE_THRESHOLD,
        f"EfficientNet-B0 TRT outputs don't match with the original model. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}",
    )

    # Clean up model env
    torch._dynamo.reset()

    with torch.no_grad():
        torch.cuda.empty_cache()


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
        "truncate_long_and_double": True,
        "ir": ir,
    }
    trt_mod = torchtrt.compile(model, **compile_spec)

    model_outputs = model(input, input2)
    trt_model_outputs = trt_mod(input, input2)
    for key in model_outputs.keys():
        out, trt_out = model_outputs[key], trt_model_outputs[key]
        cos_sim = cosine_similarity(out, trt_out)
        assert (
            cos_sim > COSINE_THRESHOLD,
            f"HF BERT base-uncased TRT outputs don't match with the original model. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}",
        )

    # Clean up model env
    torch._dynamo.reset()

    with torch.no_grad():
        torch.cuda.empty_cache()


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
    }

    trt_mod = torchtrt.compile(model, **compile_spec)
    cos_sim = cosine_similarity(model(input), trt_mod(input))
    assert (
        cos_sim > COSINE_THRESHOLD,
        f"Resnet18 Half TRT outputs don't match with the original model. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}",
    )

    # Clean up model env
    torch._dynamo.reset()

    with torch.no_grad():
        torch.cuda.empty_cache()
