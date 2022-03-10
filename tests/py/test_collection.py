import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Dict

# class Model1(nn.Module):
#     def __init__(self):
#         super(Model1, self).__init__()

#     def forward(self, z: Tuple[torch.Tensor, torch.Tensor]):
#         r = z[1] + z[0]
#         return r, z[1]


# class TestModel1(nn.Module):
#     def __init__(self):
#         super(TestModel, self).__init__()
#         self.model1 = Model1()

#     def forward(self, z: Tuple[torch.Tensor, torch.Tensor]):
#         r2, r1  = self.model1((z[0], z[1]))
#         # unsupport ops
#         i = r2.size(1)
#         j = r2.size(2)
# #         r3 = torch.tensor(i) * torch.tensor(j)
#         r3 = r2[0,0,0,0]
#         k = int(r3) - 5

# #         if k > 0:
#         r = r1 - k
#         result = (r, r1)
# #         else:
# #             r = r1 - k
# #             result = (r1, r)
#         return result

class Normal(nn.Module):
    def __init__(self):
        super(Normal, self).__init__()

    def forward(self, x, y):
        r = x + y
        return r

class TupleInput(nn.Module):
    def __init__(self):
        super(TupleInput, self).__init__()

    def forward(self, z: Tuple[torch.Tensor, torch.Tensor]):
        r = z[0] + z[1]
        return r

class ListInput(nn.Module):
    def __init__(self):
        super(ListInput, self).__init__()

    def forward(self, z: List[torch.Tensor]):
        r = z[0] + z[1]
        return r


input_data = torch.randn((16, 3, 32, 32))
input_data = input_data.float().to("cuda")

normal_model = Normal()
normal_model_ts = torch.jit.script(normal_model)
print(normal_model_ts.graph)
result = normal_model_ts(input_data, input_data)
normal_model_ts.to("cuda").eval()
torch.jit.save(normal_model_ts, "./normal_model.ts")

tuple_input = TupleInput()
tuple_input_ts = torch.jit.script(tuple_input)
print(tuple_input_ts.graph)
result = tuple_input_ts((input_data, input_data))
tuple_input_ts.to("cuda").eval()
torch.jit.save(tuple_input_ts, "./tuple_input.ts")

list_input = ListInput()
list_input_ts = torch.jit.script(list_input)
print(list_input_ts.graph)
result = list_input_ts([input_data, input_data])
list_input_ts.to("cuda").eval()
torch.jit.save(list_input_ts, "./list_input.ts")