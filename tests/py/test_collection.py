import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Dict

class Model1(nn.Module):
    def __init__(self):
        super(Model1, self).__init__()

    def forward(self, z: Tuple[torch.Tensor, torch.Tensor]):
        r = z[1] + z[0]
        return r, z[1]


class TestModel1(nn.Module):
    def __init__(self):
        super(TestModel, self).__init__()
        self.model1 = Model1()

    def forward(self, z: Tuple[torch.Tensor, torch.Tensor]):
        r2, r1  = self.model1((z[0], z[1]))
        # unsupport ops
        i = r2.size(1)
        j = r2.size(2)
#         r3 = torch.tensor(i) * torch.tensor(j)
        r3 = r2[0,0,0,0]
        k = int(r3) - 5

#         if k > 0:
        r = r1 - k
        result = (r, r1)
#         else:
#             r = r1 - k
#             result = (r1, r)
        return result

class TestModel(nn.Module):
    def __init__(self):
        super(TestModel, self).__init__()

    def forward(self, z: Tuple[torch.Tensor, torch.Tensor]):
        r = z[0] + z[1]
        return r

test_model = TestModel()

ts = torch.jit.script(test_model)
print(ts.graph)

ts.to("cuda").eval()
input_data = torch.randn((16, 3, 32, 32))
input_data = input_data.float().to("cuda")
result = ts((input_data, input_data))
torch.jit.save(ts, "./tuple2_v3.ts")