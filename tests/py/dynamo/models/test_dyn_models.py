import unittest

import pytest
import timm
import torch
import torch_tensorrt as torchtrt
from torch_tensorrt.dynamo.utils import COSINE_THRESHOLD, cosine_similarity

assertions = unittest.TestCase()


# @pytest.mark.unit
# def test_base_dynamic(ir):
#     """
#     Tests the model (which is fully convertible) with dynamic shapes
#     """

#     class MyModule(torch.nn.Module):
#         def __init__(self):
#             super().__init__()
#             self.conv = torch.nn.Conv2d(3, 16, 3, stride=1, bias=True)
#             self.relu = torch.nn.ReLU()

#         def forward(self, x):
#             out = self.conv(x)
#             out = self.relu(out)
#             return out

#     model = MyModule().eval().cuda()
#     input = torch.randn((1, 3, 224, 224)).to("cuda")

#     compile_spec = {
#         "inputs": [
#             torchtrt.Input(
#                 min_shape=(1, 3, 224, 224),
#                 opt_shape=(4, 3, 224, 224),
#                 max_shape=(8, 3, 224, 224),
#                 dtype=torch.float32,
#             )
#         ],
#         "device": torchtrt.Device("cuda:0"),
#         "enabled_precisions": {torch.float},
#         "ir": ir,
#         "pass_through_build_failures": True,
#         "optimization_level": 1,
#         "min_block_size": 1,
#     }

#     trt_mod = torchtrt.compile(model, **compile_spec)
#     cos_sim = cosine_similarity(model(input), trt_mod(input)[0])
#     assertions.assertTrue(
#         cos_sim > COSINE_THRESHOLD,
#         msg=f"test_base_dynamic model TRT outputs don't match with the pytorch model. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}",
#     )

#     # Clean up model env
#     torch._dynamo.reset()

#     with torch.no_grad():
#         torch.cuda.empty_cache()


# @pytest.mark.unit
# def test_base_dynamic_fallback(ir):
#     """
#     Tests the model (which is fully convertible) with dynamic shapes
#     """

#     class MyModule(torch.nn.Module):
#         def __init__(self):
#             super().__init__()
#             self.conv = torch.nn.Conv2d(3, 16, 3, stride=1, bias=True)
#             self.relu = torch.nn.ReLU()

#         def forward(self, x):
#             out = self.conv(x)
#             out = torch.abs(out)
#             out = self.relu(out)
#             return out

#     model = MyModule().eval().cuda()
#     input = torch.randn((1, 3, 224, 224)).to("cuda")

#     compile_spec = {
#         "inputs": [
#             torchtrt.Input(
#                 min_shape=(1, 3, 224, 224),
#                 opt_shape=(4, 3, 224, 224),
#                 max_shape=(8, 3, 224, 224),
#                 dtype=torch.float32,
#             )
#         ],
#         "device": torchtrt.Device("cuda:0"),
#         "enabled_precisions": {torch.float},
#         "ir": ir,
#         "pass_through_build_failures": True,
#         "optimization_level": 1,
#         "torch_executed_ops": "torch.ops.aten.abs.default",
#         "min_block_size": 1,
#     }

#     trt_mod = torchtrt.compile(model, **compile_spec)
#     cos_sim = cosine_similarity(model(input), trt_mod(input)[0])
#     assertions.assertTrue(
#         cos_sim > COSINE_THRESHOLD,
#         msg=f"test_base_dynamic model TRT outputs don't match with the pytorch model. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}",
#     )

#     # Clean up model env
#     torch._dynamo.reset()

#     with torch.no_grad():
#         torch.cuda.empty_cache()


@pytest.mark.unit
def test_bert_dynamic_fallback(ir):
    """
    Tests the model (which is fully convertible) with dynamic shapes
    """
    from transformers import BertModel
    from transformers.utils.fx import symbolic_trace as transformers_trace

    model = BertModel.from_pretrained("bert-base-uncased").cuda().eval()
    input = torch.randint(0, 1, (1, 14), dtype=torch.int64).to("cuda")
    input2 = torch.randint(0, 1, (1, 14), dtype=torch.int64).to("cuda")
    model = (
        transformers_trace(model, input_names=["input_ids", "attention_mask"])
        .eval()
        .cuda()
    )
    # import pdb; pdb.set_trace()
    # model = torch.jit.trace(model, [input, input2])
    compile_spec = {
        "inputs": [
            torchtrt.Input(
                min_shape=(1, 14),
                opt_shape=(4, 14),
                max_shape=(8, 14),
                dtype=input.dtype,
                format=torch.contiguous_format,
            ),
            torchtrt.Input(
                min_shape=(1, 14),
                opt_shape=(4, 14),
                max_shape=(8, 14),
                dtype=input.dtype,
                format=torch.contiguous_format,
            ),
        ],
        "device": torchtrt.Device("cuda:0"),
        "enabled_precisions": {torch.float},
        "truncate_long_and_double": True,
        "ir": ir,
        "min_block_size": 1,
    }

    trt_mod = torchtrt.compile(model, **compile_spec)
    model_outputs = model(input, input2)
    trt_model_outputs = trt_mod(input, input2)
    assertions.assertTrue(
        len(model_outputs) == len(trt_model_outputs),
        msg=f"Number of outputs for BERT model compilation is different with Pytorch {len(model_outputs)} and TensorRT {len(trt_model_outputs)}. Please check the compilation.",
    )
    for index, key in enumerate(model_outputs):
        out, trt_out = model_outputs[key], trt_model_outputs[index]
        cos_sim = cosine_similarity(out, trt_out)
        assertions.assertTrue(
            cos_sim > COSINE_THRESHOLD,
            msg=f"HF BERT base-uncased TRT outputs don't match with the original model. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}",
        )

    # Clean up model env
    torch._dynamo.reset()

    with torch.no_grad():
        torch.cuda.empty_cache()
