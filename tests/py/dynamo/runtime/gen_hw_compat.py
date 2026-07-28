# This script is used to generate hw_compat.ts file that's used in test_hw_compat.py
# Generate the model on a different hardware compared to the one you're testing on to
# verify HW compatibility feature.

import torch
import torch_tensorrt


class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 16, 3, stride=1, bias=True)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        out = self.conv(x)
        out = self.relu(out)
        return out


model = MyModule().eval().cuda()
inputs = torch.randn((1, 3, 224, 224)).to("cuda")

trt_gm = torch_tensorrt.compile(
    model,
    ir="dynamo",
    inputs=inputs,
    min_block_size=1,
    hardware_compatible=True,
    version_compatible=True,
)
trt_script_model = torch.jit.trace(trt_gm, inputs)
torch.jit.save(trt_script_model, "hw_compat.ts")
