import torch
import torch.nn as nn
import torch_tensorrt as torchtrt
from typing import Tuple, List, Dict

class Normal(nn.Module):
    def __init__(self):
        super(Normal, self).__init__()

    def forward(self, x, y):
        r = x + y
        return r

class TupleInputOutput(nn.Module):
    def __init__(self):
        super(TupleInputOutput, self).__init__()

    def forward(self, z: Tuple[torch.Tensor, torch.Tensor]):
        r1 = z[0] + z[1]
        r2 = z[0] - z[1]
        r = (r1, r2)
        return r

input = torch.randn((1, 3, 224, 224)).to("cuda")
normal_model = Normal()
scripted_model = torch.jit.script(normal_model)

compile_spec = {
    "inputs": [torchtrt.Input(input.shape, dtype=torch.float, format=torch.contiguous_format),
               torchtrt.Input(input.shape, dtype=torch.float, format=torch.contiguous_format)],
    "device": {
        "device_type": torchtrt.DeviceType.GPU,
        "gpu_id": 0,
    },
    "enabled_precisions": {torch.float}
}

trt_mod = torchtrt.ts.compile(scripted_model, **compile_spec)
same = (trt_mod(input, input) - scripted_model(input, input)).abs().max()
print(same.cpu())

# input = torch.randn((1, 3, 224, 224)).to("cuda")
# tuple_model = TupleInputOutput()
# scripted_model = torch.jit.script(tuple_model)

# compile_spec = {
#     "inputs": (torchtrt.Input(input.shape, dtype=torch.float, format=torch.contiguous_format),
#                torchtrt.Input(input.shape, dtype=torch.float, format=torch.contiguous_format)),
#     "device": {
#         "device_type": torchtrt.DeviceType.GPU,
#         "gpu_id": 0,
#     },
#     "enabled_precisions": {torch.float}
# }

# trt_mod = torchtrt.ts.compile(scripted_model, **compile_spec)
# same = (trt_mod((input, input))[0] - scripted_model((input, input))[0]).abs().max()
# print(same.cpu())


