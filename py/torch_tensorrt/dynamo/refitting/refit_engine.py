import numpy as np
import torch
import torchvision.models as models
from torch_tensorrt._Device import Device
from torch_tensorrt.dynamo._compiler import convert_module_to_trt_engine
from torch_tensorrt.dynamo._refit import refit_trt_engine_from_module
from torch_tensorrt.dynamo.runtime._PythonTorchTensorRTModule import (
    PythonTorchTensorRTModule,
)

np.random.seed(0)
torch.manual_seed(0)


inputs = [torch.rand((1, 3, 224, 224)).to("cuda")]


# class net(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv1 = nn.Conv2d(3, 12, 3, padding=1)
#         self.bn = nn.BatchNorm2d(12)
#         self.relu = nn.ReLU()

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.bn(x)
#         x = self.relu(x)
#         return x


# model = net().eval().to("cuda")
# np.random.seed(1)
# model2 = net().eval().to("cuda")

model = models.resnet18(pretrained=False).eval().to("cuda")
model2 = models.resnet18(pretrained=True).eval().to("cuda")
enabled_precisions = {torch.float}
debug = True
workspace_size = 20 << 30
min_block_size = 1


exp_program = torch.export.export(model, tuple(inputs))
exp_program2 = torch.export.export(model2, tuple(inputs))


serialized_engine = convert_module_to_trt_engine(
    exported_program=exp_program,
    inputs=tuple(inputs),
    enabled_precisions=enabled_precisions,
    debug=debug,
    min_block_size=min_block_size,
    refit=True,
)

trt_module = PythonTorchTensorRTModule(
    engine=serialized_engine,
    input_names=["x"],
    output_names=["output0"],
    target_device=Device._current_device(),
    profiling_enabled=False,
)

output = trt_module.forward(*inputs)
print(output[0].sum().cpu().item())
engine = trt_module.engine
print(model(*inputs)[0].sum().cpu().item())

# ----------------------Refitting------------------------------------
# weights_to_be_fitted = model2.state_dict()


# refit_dict = {
# '[CONVOLUTION]-[aten_ops.convolution.default]-[/conv1/convolution] BIAS': weights_to_be_fitted['conv1.bias']
# ,
# '[SCALE]-[aten_ops._native_batch_norm_legit_no_training.default]-[/bn/_native_batch_norm_legit_no_training] SCALE': weights_to_be_fitted['bn.weight']
# ,
# '[SCALE]-[aten_ops._native_batch_norm_legit_no_training.default]-[/bn/_native_batch_norm_legit_no_training] SHIFT': weights_to_be_fitted['bn.bias']
# ,
# '[CONVOLUTION]-[aten_ops.convolution.default]-[/conv1/convolution] KERNEL': weights_to_be_fitted['conv1.weight']
# }


refit_trt_engine_from_module(
    exported_program=exp_program2,  # New
    inputs=tuple(inputs),
    engine=engine,  # Old
    enabled_precisions=enabled_precisions,
    debug=debug,
    min_block_size=min_block_size,
)

output = trt_module.forward(*inputs)
print(output[0].sum().cpu().item())
engine = trt_module.engine
pytorch_output = model2(*inputs)[0]
print(pytorch_output.sum().cpu().item())
print((output - pytorch_output).mean())
print()


# Iterate over all layers and print weights
# for layer_name in refitter.get_all_weights():
#     # Print kernel weights
#     print_layer_weights(refitter, layer_name)
