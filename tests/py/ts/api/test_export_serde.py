import importlib
import os
import platform
import tempfile
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

@pytest.mark.unit
def test_save_load_ts(ir):
    """
    This tests save/load API on Torchscript format (model still compiled using ts workflow)
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
        ir="ts",
        inputs=[input],
        min_block_size=1,
        cache_built_engines=False,
        reuse_cached_engines=False,
    )
    outputs_trt = trt_gm(input)
    # Save it as torchscript representation
    torchtrt.save(trt_gm, "./trt.ts", output_format="torchscript", inputs=[input])

    trt_ts_module = torchtrt.load("./trt.ts")
    outputs_trt_deser = trt_ts_module(input)

    cos_sim = cosine_similarity(outputs_trt, outputs_trt_deser)
    assertions.assertTrue(
        cos_sim > COSINE_THRESHOLD,
        msg=f"test_save_load_ts TRT outputs don't match with the original model. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}",
    )
