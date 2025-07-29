# %%
import numpy as np
import torch
import torch_tensorrt as torch_trt
import torchvision.models as models
from diffusers import DiffusionPipeline

np.random.seed(5)
torch.manual_seed(5)
inputs = [torch.rand((100, 3, 224, 224)).to("cuda")]

settings = {
    "ir": "dynamo",
    "use_python_runtime": False,
    "enabled_precisions": {torch.float32},
    "immutable_weights": False,
}

model = models.resnet152(pretrained=True).eval().to("cuda")
compiled_module = torch_trt.compile(model, inputs=inputs, **settings)
print((torch.cuda.mem_get_info()[1] - torch.cuda.mem_get_info()[0]) / 1024**3)
compiled_module(*inputs)

breakpoint()
with torch_trt.dynamo.runtime.ResourceAllocatorContext(compiled_module):
    print(
        "Memory used (GB):",
        (torch.cuda.mem_get_info()[1] - torch.cuda.mem_get_info()[0]) / 1024**3,
    )
    breakpoint()
    compiled_module(*inputs)
    print(
        "Memory used (GB):",
        (torch.cuda.mem_get_info()[1] - torch.cuda.mem_get_info()[0]) / 1024**3,
    )
    breakpoint()
