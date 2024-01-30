import unittest

import pytest
import torch
import torch_tensorrt as torchtrt

assertions = unittest.TestCase()


@pytest.mark.unit
def test_output_format(ir):
    """
    This tests output_format type in the compilation setting
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

    trt_ep = torchtrt.compile(model, ir=ir, inputs=[input], min_block_size=1)
    assertions.assertTrue(
        isinstance(trt_ep, torch.export.ExportedProgram),
        msg=f"test_output_format output type does not match with torch.export.ExportedProgram",
    )

    trt_ts = torchtrt.compile(
        model,
        ir=ir,
        inputs=[input],
        min_block_size=1,
        output_format="torchscript",
    )
    assertions.assertTrue(
        isinstance(trt_ts, torch.jit.ScriptModule),
        msg=f"test_output_format output type does not match with torch.jit.ScriptModule",
    )

    trt_gm = torchtrt.compile(
        model,
        ir=ir,
        inputs=[input],
        min_block_size=1,
        output_format="graph_module",
    )
    assertions.assertTrue(
        isinstance(trt_gm, torch.fx.GraphModule),
        msg=f"test_output_format output type does not match with torch.fx.GraphModule",
    )
    # Clean up model env
    torch._dynamo.reset()
