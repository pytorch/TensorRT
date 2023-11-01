import custom_models as cm
import timm
import torch
import torch_tensorrt
import torchvision.models as models

BENCHMARK_MODELS = {
    "vgg16": {
        "model": models.vgg16(weights=models.VGG16_Weights.DEFAULT),
        "path": ["script", "pytorch"],
    },
    "alexnet": {
        "model": models.alexnet(weights=models.AlexNet_Weights.DEFAULT),
        "path": ["script", "pytorch"],
    },
    "resnet50": {
        "model": models.resnet50(weights=None),
        "path": ["script", "pytorch"],
    },
    "efficientnet_b0": {
        "model": timm.create_model("efficientnet_b0", pretrained=True),
        "path": ["script", "pytorch"],
    },
    "vit": {
        "model": timm.create_model("vit_base_patch16_224", pretrained=True),
        "path": ["script", "pytorch"],
    },
    "vit_large": {
        "model": timm.create_model("vit_giant_patch14_224", pretrained=False),
        "path": ["script", "pytorch"],
    },
    "bert_base_uncased": {
        "model": cm.BertModule(),
        "inputs": cm.BertInputs(),
        "path": ["trace", "pytorch"],
    },
    "sd_unet": {
        "model": cm.StableDiffusionUnet(),
        "path": "pytorch",
    },
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

        if input_shape != [1]:
            if dtype == torch.int32:
                torchtrt_inputs.append(
                    torch.randint(0, 5, input_shape, dtype=dtype).cuda()
                )
            else:
                torchtrt_inputs.append(torch.randn(input_shape, dtype=dtype).cuda())
        else:
            torchtrt_inputs.append(torch.Tensor([1.0]).cuda())

    return torchtrt_inputs


def parse_backends(backends):
    return backends.split(",")


def parse_precisions(precisions):
    return precisions.split(",")
