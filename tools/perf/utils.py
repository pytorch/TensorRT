import copy
import timeit

import custom_models as cm
import numpy as np
import tensorrt as trt
import timm
import torch
import torchvision.models as models
from transformers import AutoModel, AutoModelForCausalLM

BENCHMARK_MODEL_NAMES = {
    "vgg16",
    "alexnet",
    "resnet50",
    "efficientnet_b0",
    "vit",
    "vit_large",
    "bert_base_uncased",
    "sd_unet",
    "meta-llama/Llama-2-7b-chat-hf",
    "gpt2",
    "meta-llama/Meta-Llama-3-8B",
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "apple/DCLM-7B",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "microsoft/Phi-3-mini-4k-instruct",
}


def load_hf_model(model_name_hf):
    print("Loading user-specified HF model: ", model_name_hf)
    model_hf = AutoModelForCausalLM.from_pretrained(
        model_name_hf,
        trust_remote_code=True,
        use_cache=False,
        attn_implementation="eager",
    ).eval()

    return {"model": model_hf}


class ModelStorage:
    def __contains__(self, name: str):
        return name in BENCHMARK_MODEL_NAMES

    def __getitem__(self, name: str):
        assert name in BENCHMARK_MODEL_NAMES

        if name == "vgg16":
            return {
                "model": models.vgg16(weights=models.VGG16_Weights.DEFAULT),
                "path": ["script", "pytorch"],
            }
        elif name == "alexnet":
            return {
                "model": models.alexnet(weights=models.AlexNet_Weights.DEFAULT),
                "path": ["script", "pytorch"],
            }
        elif name == "resnet50":
            return {
                "model": models.resnet50(weights=None),
                "path": ["script", "pytorch"],
            }
        elif name == "efficientnet_b0":
            return {
                "model": timm.create_model("efficientnet_b0", pretrained=True),
                "path": ["script", "pytorch"],
            }
        elif name == "vit":
            return {
                "model": timm.create_model("vit_base_patch16_224", pretrained=True),
                "path": ["script", "pytorch"],
            }
        elif name == "vit_large":
            return {
                "model": timm.create_model("vit_giant_patch14_224", pretrained=False),
                "path": ["script", "pytorch"],
            }
        elif name == "bert_base_uncased":
            return {
                "model": cm.BertModule(),
                "inputs": cm.BertInputs(),
                "path": ["trace", "pytorch"],
            }
        elif name == "sd_unet":
            return {
                "model": cm.StableDiffusionUnet(),
                "path": "pytorch",
            }
        elif name in [
            "gpt2",
            "meta-llama/Meta-Llama-3-8B",
            "meta-llama/Llama-2-7b-chat-hf",
            "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "mistralai/Mistral-7B-Instruct-v0.3",
            "microsoft/Phi-3-mini-4k-instruct",
        ]:
            hf_artifact = load_hf_model(name)
            return {
                "model": hf_artifact["model"],
                "path": "pytorch",
            }
        elif name == "apple/DCLM-7B":
            # Load model directly
            hf_artifact = AutoModel.from_pretrained("apple/DCLM-7B")
            return {
                "model": hf_artifact["model"],
                "path": "pytorch",
            }
        else:
            raise AssertionError(f"Invalid model name {name}")

    def items(self):
        for name in BENCHMARK_MODEL_NAMES:
            yield name, self.__getitem__(name)


BENCHMARK_MODELS = ModelStorage()


def precision_to_dtype(pr):
    if pr == "fp32":
        return torch.float
    elif pr == "fp16" or pr == "half":
        return torch.half
    elif pr == "int32":
        return torch.int32
    elif pr == "int64":
        return torch.int64
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
            if dtype == torch.int32 or dtype == torch.int64:
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


def torch_dtype_from_trt(dtype):
    if dtype == trt.int8:
        return torch.int8
    elif dtype == trt.bool:
        return torch.bool
    elif dtype == trt.int32:
        return torch.int32
    elif dtype == trt.float16:
        return torch.float16
    elif dtype == trt.float32:
        return torch.float32
    else:
        raise TypeError("%s is not supported by torch" % dtype)


def torch_device_from_trt(device):
    if device == trt.TensorLocation.DEVICE:
        return torch.device("cuda")
    elif device == trt.TensorLocation.HOST:
        return torch.device("cpu")
    else:
        return TypeError("%s is not supported by torch" % device)


def export_llm(model, inputs, min_seq_len=1, max_seq_len=16):
    """
    Exports the LLM model into an ExportedProgram with dynamic shapes.
    In the case of guard failures due to some PyTorch kernel implements, we also
    try to re-export the graph by expressing them as runtime assert nodes
    """
    assert isinstance(inputs, list)

    with torch.no_grad():
        # max=1024 has contraint violation error. https://github.com/pytorch/pytorch/issues/125604
        seq_len = torch.export.Dim("seq_len", min=min_seq_len, max=max_seq_len)
        try:
            print("Trying to export the model using torch.export.export()..")
            # strict=False only enables aotautograd tracing and excludes dynamo.
            ep = torch.export.export(
                model, tuple(inputs), dynamic_shapes=({1: seq_len},), strict=False
            )
        except:
            print(
                "Trying torch.export._trace._export to trace the graph since torch.export.export() failed"
            )
            # This API is used to express the constraint violation guards as asserts in the graph.
            ep = torch.export._trace._export(
                model,
                (inputs,),
                dynamic_shapes=({1: seq_len},),
                strict=False,
                allow_complex_guards_as_runtime_asserts=True,
            )

    return ep


def generate(model, input_seq, output_seq_length):
    """
    Greedy decoding of the model. This generates up to max_tokens.
    """

    while input_seq.shape[1] <= output_seq_length:
        outputs = model(input_seq)
        logits = outputs.logits
        next_token_logits = logits[:, -1, :]
        next_tokens = torch.argmax(next_token_logits, dim=-1)
        input_seq = torch.cat([input_seq, next_tokens[:, None]], dim=-1)

    return input_seq


def time_generate(model, inputs, output_seq_length, iterations=10):
    """
    Measure the time for generating a sentence over certain number of iterations
    """
    timings = []
    for _ in range(iterations):
        start_time = timeit.default_timer()
        inputs_copy = copy.copy(inputs)
        _ = generate(model, inputs_copy, output_seq_length)
        torch.cuda.synchronize()
        end_time = timeit.default_timer()
        timings.append(end_time - start_time)

    return timings
