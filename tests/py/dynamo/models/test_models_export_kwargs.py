# type: ignore
import unittest

import pytest
import timm
import torch
import torch.nn.functional as F
import torch_tensorrt as torchtrt
import torchvision.models as models
from torch import nn
from torch_tensorrt.dynamo.utils import COSINE_THRESHOLD, cosine_similarity
from transformers import BertModel
from transformers.utils.fx import symbolic_trace as transformers_trace

assertions = unittest.TestCase()


@pytest.mark.unit
def test_custom_model():
    class net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 12, 3, padding=1)
            self.bn = nn.BatchNorm2d(12)
            self.conv2 = nn.Conv2d(12, 12, 3, padding=1)
            self.fc1 = nn.Linear(12 * 56 * 56, 10)

        def forward(self, x, b=5, c=None, d=None):
            x = self.conv1(x)
            x = F.relu(x)
            x = self.bn(x)
            x = F.max_pool2d(x, (2, 2))
            x = self.conv2(x)
            x = F.relu(x)
            x = F.max_pool2d(x, (2, 2))
            x = torch.flatten(x, 1)
            x = x + b
            if c is not None:
                x = x * c
            if d is not None:
                x = x - d["value"]
            return self.fc1(x)

    model = net().eval().to("cuda")
    args = [torch.rand((1, 3, 224, 224)).to("cuda")]
    kwargs = {
        "b": torch.tensor(6).to("cuda"),
        "d": {"value": torch.tensor(8).to("cuda")},
    }

    compile_spec = {
        "inputs": args,
        "kwarg_inputs": kwargs,
        "device": torchtrt.Device("cuda:0"),
        "enabled_precisions": {torch.float},
        "pass_through_build_failures": True,
        "optimization_level": 1,
        "min_block_size": 1,
        "ir": "dynamo",
    }
    # TODO: Support torchtrt.compile
    # trt_mod = torchtrt.compile(model, **compile_spec)

    exp_program = torch.export.export(model, args=tuple(args), kwargs=kwargs)
    trt_mod = torchtrt.dynamo.compile(exp_program, **compile_spec)
    cos_sim = cosine_similarity(model(*args, **kwargs), trt_mod(*args, **kwargs)[0])
    assertions.assertTrue(
        cos_sim > COSINE_THRESHOLD,
        msg=f"CustomKwargs Module TRT outputs don't match with the original model. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}",
    )

    # Clean up model env
    torch._dynamo.reset()
