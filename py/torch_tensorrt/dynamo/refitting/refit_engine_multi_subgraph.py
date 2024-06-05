import numpy as np
import torch
import torch_tensorrt as trt
import torchvision.models as models

# from torch import nn
from torch_tensorrt.dynamo._refit import refit_module_weights

np.random.seed(0)
torch.manual_seed(0)


inputs = [torch.rand((1, 3, 224, 224)).to("cuda")]

# Small Toy Model ---------------------------------------


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

# Resnet 18 --------------------------------------------

model = models.resnet18(pretrained=False).eval().to("cuda")
model2 = models.resnet18(pretrained=True).eval().to("cuda")

# Bert -----------------------------------------------

# from transformers import BertModel
# inputs = [
#     torch.randint(0, 2, (1, 14), dtype=torch.int32).to("cuda"),
# ]
# model = BertModel.from_pretrained("bert-base-uncased").eval().to("cuda")


enabled_precisions = {torch.float}
debug = True
workspace_size = 20 << 30
min_block_size = 0


exp_program = torch.export.export(model, tuple(inputs))
exp_program2 = torch.export.export(model2, tuple(inputs))

use_python_runtime = False

trt_gm = trt.dynamo.compile(
    exp_program,
    tuple(inputs),
    use_python_runtime=use_python_runtime,
    enabled_precisions=enabled_precisions,
    debug=debug,
    min_block_size=min_block_size,
    refit=True,
    engine_save_path="/home/cehongw/Desktop/torch-trt/TensorRT/py/torch_tensorrt/dynamo/refitting/",
)  # Output is a torch.fx.GraphModule


output = trt_gm.forward(*inputs)
print(output[0].sum().cpu().item())
extra_files_loaded = {"settings": None}
path = "/home/cehongw/Desktop/torch-trt/TensorRT/py/torch_tensorrt/dynamo/refitting/trt_compiled_module.ep"

new_trt_gm = refit_module_weights(
    path,
    new_weight_module=exp_program2,
    inputs=inputs,
)


# compiled_exp_program = torch.export.load("/home/cehongw/Desktop/torch-trt/TensorRT/py/torch_tensorrt/dynamo/refitting/trt_compiled_module.ep"
#                          , extra_files=extra_files_loaded)


# decoded = base64.b64decode(extra_files_loaded["settings"].encode('utf-8'))
# restored_settings = pickle.loads(decoded)

# new_trt_gm = refit_module_weights(
#     compiled_module=compiled_exp_program,
#     new_weight_module=exp_program2,
#     inputs=inputs,
#     settings=restored_settings,
# )

output_refit = new_trt_gm.forward(*inputs)
print(output_refit[0].sum().cpu().item())

pytorch_output = model2(*inputs)[0]
print(pytorch_output.sum().cpu().item())

print((output_refit - pytorch_output).mean())
print()


# Iterate over all layers and print weights
# for layer_name in refitter.get_all_weights():
#     # Print kernel weights
#     print_layer_weights(refitter, layer_name)
