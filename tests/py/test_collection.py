import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Dict

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

class TupleInputOutput(nn.Module):
    def __init__(self):
        super(TupleInputOutput, self).__init__()

    def forward(self, z: Tuple[torch.Tensor, torch.Tensor]):
        r1 = z[0] + z[1]
        r2 = z[0] - z[1]
        r = (r1, r2)
        return r

class ListInputOutput(nn.Module):
    def __init__(self):
        super(ListInputOutput, self).__init__()

    def forward(self, z: List[torch.Tensor]):
        r1 = z[0] + z[1]
        r2 = z[0] - z[1]
        r = [r1, r2]
        return r

class ComplexModel(nn.Module):
    def __init__(self):
        super(ComplexModel, self).__init__()
        self.list_model = ListInputOutput()
        self.tuple_model = TupleInputOutput()

    def forward(self, z: List[torch.Tensor]):
        r1 = z[0] + z[1]
        r2 = z[0] - z[1]
        r3 = (r1, r2)
        r4 = [r2, r1]
        tuple_out = self.tuple_model(r3)
        list_out = self.list_model(r4)
        r = (tuple_out[1], list_out[0])
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

tuple_input = TupleInputOutput()
tuple_input_ts = torch.jit.script(tuple_input)
print(tuple_input_ts.graph)
result = tuple_input_ts((input_data, input_data))
tuple_input_ts.to("cuda").eval()
torch.jit.save(tuple_input_ts, "./tuple_input_output.ts")

list_input = ListInputOutput()
list_input_ts = torch.jit.script(list_input)
print(list_input_ts.graph)
result = list_input_ts([input_data, input_data])
list_input_ts.to("cuda").eval()
torch.jit.save(list_input_ts, "./list_input_output.ts")

complex_model = ComplexModel()
complex_model_ts = torch.jit.script(complex_model)
print(complex_model_ts.graph)
result = complex_model_ts([input_data, input_data])
complex_model_ts.to("cuda").eval()
torch.jit.save(complex_model_ts, "./complex_model.ts")