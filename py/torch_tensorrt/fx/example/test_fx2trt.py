import torch
import torch_tensorrt


class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(5, 3)
        self.relu = torch.nn.functional.relu

    def forward(self, x):
        x = self.linear(x)
        x = self.relu(x)
        return x


model = MyModel().eval()  # torch module needs to be in eval (not training) mode

# torch tensorrt
inputs = [
    torch_tensorrt.Input(
        (2, 5),
        dtype=torch.half,
    )
]
enabled_precisions = {torch.float, torch.half}  # Run with fp16

trt_ts_module = torch_tensorrt.compile(
    model, inputs=inputs, enabled_precisions=enabled_precisions
)

inputs_ts = [torch.ones(2, 5)]
inputs_ts = [i.cuda().half() for i in inputs_ts]
result = trt_ts_module(*inputs_ts)
print(result)

model.cuda().half()
ref = model(*inputs_ts)
print(ref)

# fx2trt
inputs_fx = [torch.ones((2, 5))]

model.cuda().half()
inputs_fx = [i.cuda().half() for i in inputs_fx]

trt_fx_module = torch_tensorrt.compile(
    model, ir="fx", inputs=inputs_fx, enabled_precisions={torch.half}
)
result = trt_fx_module(*inputs_fx)
print(result)

ref = model(*inputs_fx)
print(ref)
