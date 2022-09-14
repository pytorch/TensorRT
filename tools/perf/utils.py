import torch
import torch_tensorrt
import custom_models as cm
import torchvision.models as models
import timm

BENCHMARK_MODELS = {
    "vgg16": {"model": models.vgg16(pretrained=True), "path": "script"},
    "resnet50": {
        "model": torch.hub.load("pytorch/vision:v0.9.0", "resnet50", pretrained=True),
        "path": "script",
    },
    "efficientnet_b0": {
        "model": timm.create_model("efficientnet_b0", pretrained=True),
        "path": "script",
    },
    "vit": {
        "model": timm.create_model("vit_base_patch16_224", pretrained=True),
        "path": "script",
    },
    "bert_base_uncased": {"model": cm.BertModule(), "path": "trace"},
}


def precision_to_dtype(pr):
    if pr == "fp32":
        return torch.float
    elif pr == "fp16" or pr == "half":
        return torch.half
    elif pr == "int32":
        return torch.int32
    elif pr == "bool":
        return torch.bool
    else:
        return torch.float32


def parse_inputs(user_inputs, dtype):
    parsed_inputs = user_inputs.split(";")
    torchtrt_inputs = []
    for input in parsed_inputs:
        input_shape = []
        input_shape_and_dtype = input.split("@")
        dtype = (
            precision_to_dtype(input_shape_and_dtype[1])
            if len(input_shape_and_dtype) == 2
            else dtype
        )
        for input_dim in input_shape_and_dtype[0][1:-1].split(","):
            input_shape.append(int(input_dim))
        torchtrt_inputs.append(torch.randint(0, 5, input_shape, dtype=dtype).cuda())

    return torchtrt_inputs


def parse_backends(backends):
    return backends.split(",")


def parse_precisions(precisions):
    return precisions.split(",")
