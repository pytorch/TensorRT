import unittest

import pytest
import torch
import torch_tensorrt as torchtrt
from torch_tensorrt.dynamo.utils import COSINE_THRESHOLD, cosine_similarity

assertions = unittest.TestCase()


@pytest.mark.unit
def test_dyn_full_compile(ir):
    """
    Tests the model (which is fully convertible) with dynamic shapes
    """

    class MyModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 16, 3, stride=1, bias=True)
            self.relu = torch.nn.ReLU()

        def forward(self, x):
            torch._check(x.size()[0] >= 1)
            torch._check(x.size()[0] <= 8)
            out = self.conv(x)
            out = self.relu(out)
            return out

    model = MyModule().eval().cuda()
    input_bs4 = torch.randn((4, 3, 224, 224)).to("cuda")
    torch._dynamo.mark_dynamic(input_bs4, 0)
    compile_spec = {
        "inputs": [input_bs4],
        "min_block_size": 1,
        "debug": True,
    }
    # Compile the model
    trt_model = torch.compile(model, backend="tensorrt", options=compile_spec)
    trt_model(input_bs4)

    input_bs6 = torch.randn((6, 3, 224, 224)).to("cuda")
    cos_sim = cosine_similarity(model(input_bs6), trt_model(input_bs6))
    assertions.assertTrue(
        cos_sim > COSINE_THRESHOLD,
        msg=f"test_dyn_full_compile model TRT outputs don't match with the pytorch model. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}",
    )

    # Clean up model env
    torch._dynamo.reset()

    with torch.no_grad():
        torch.cuda.empty_cache()
