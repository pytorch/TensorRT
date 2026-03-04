"""
.. _autocast_example:

An example of using Torch-TensorRT Autocast
================

This example demonstrates how to use Torch-TensorRT Autocast with PyTorch Autocast to compile a mixed precision model.
"""

import torch
import torch.nn as nn
import torch_tensorrt

# %% Mixed Precision Model
#
# We define a mixed precision model that consists of a few layers, a ``log`` operation, and an ``abs`` operation.
# Among them, the ``fc1``, ``log``, and ``abs`` operations are within PyTorch Autocast context with ``dtype=torch.float16``.


class MixedPytorchAutocastModel(nn.Module):
    def __init__(self):
        super(MixedPytorchAutocastModel, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=1
        )
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(
            in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1
        )
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16 * 8 * 8, 10)

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.relu1(out1)
        out3 = self.pool1(out2)
        out4 = self.conv2(out3)
        out5 = self.relu2(out4)
        out6 = self.pool2(out5)
        out7 = self.flatten(out6)
        with torch.autocast(x.device.type, enabled=True, dtype=torch.float16):
            out8 = self.fc1(out7)
            out9 = torch.log(
                torch.abs(out8) + 1
            )  # log is fp32 due to Pytorch Autocast requirements
        return x, out1, out2, out3, out4, out5, out6, out7, out8, out9


# %%
# Define the model, inputs, and calibration dataloader for Autocast, and then we run the original PyTorch model to get the reference outputs.

model = MixedPytorchAutocastModel().cuda().eval()
inputs = (torch.randn((8, 3, 32, 32), dtype=torch.float32, device="cuda"),)
ep = torch.export.export(model, inputs)
calibration_dataloader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(*inputs), batch_size=2, shuffle=False
)

pytorch_outs = model(*inputs)

# %% Compile the model with Torch-TensorRT Autocast
#
# We compile the model with Torch-TensorRT Autocast by setting ``enable_autocast=True``, ``use_explicit_typing=True``, and
# ``autocast_low_precision_type=torch.bfloat16``. To illustrate, we exclude the ``conv1`` node, all nodes with name
# containing ``relu``, and ``torch.ops.aten.flatten.using_ints`` ATen op from Autocast. In addtion, we also set
# ``autocast_max_output_threshold``, ``autocast_max_depth_of_reduction``, and ``autocast_calibration_dataloader``. Please refer to
# the documentation for more details.

trt_autocast_mod = torch_tensorrt.compile(
    ep.module(),
    arg_inputs=inputs,
    min_block_size=1,
    use_python_runtime=True,
    use_explicit_typing=True,
    enable_autocast=True,
    autocast_low_precision_type=torch.bfloat16,
    autocast_excluded_nodes={"^conv1$", "relu"},
    autocast_excluded_ops={"torch.ops.aten.flatten.using_ints"},
    autocast_max_output_threshold=512,
    autocast_max_depth_of_reduction=None,
    autocast_calibration_dataloader=calibration_dataloader,
)

autocast_outs = trt_autocast_mod(*inputs)

# %% Verify the outputs
#
# We verify both the dtype and values of the outputs of the model are correct.
# As expected, ``fc1`` is in FP16 because of PyTorch Autocast;
# ``pool1``, ``conv2``, and ``pool2`` are in BFP16 because of Torch-TensorRT Autocast;
# the rest remain in FP32. Note that ``log`` is in FP32 because of PyTorch Autocast requirements.

should_be_fp32 = [
    autocast_outs[0],
    autocast_outs[1],
    autocast_outs[2],
    autocast_outs[5],
    autocast_outs[7],
    autocast_outs[9],
]
should_be_fp16 = [
    autocast_outs[8],
]
should_be_bf16 = [autocast_outs[3], autocast_outs[4], autocast_outs[6]]

assert all(
    a.dtype == torch.float32 for a in should_be_fp32
), "Some Autocast outputs are not float32!"
assert all(
    a.dtype == torch.float16 for a in should_be_fp16
), "Some Autocast outputs are not float16!"
assert all(
    a.dtype == torch.bfloat16 for a in should_be_bf16
), "Some Autocast outputs are not bfloat16!"
for i, (a, w) in enumerate(zip(autocast_outs, pytorch_outs)):
    assert torch.allclose(
        a.to(torch.float32), w.to(torch.float32), atol=1e-2, rtol=1e-2
    ), f"Autocast and Pytorch outputs do not match! autocast_outs[{i}] = {a}, pytorch_outs[{i}] = {w}"
print("All dtypes and values match!")
