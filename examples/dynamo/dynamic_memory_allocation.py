# %%
import numpy as np
import torch
import torch_tensorrt as torch_trt
import torchvision.models as models
import time
import gc

np.random.seed(5)
torch.manual_seed(5)
inputs = [torch.rand((100, 3, 224, 224)).to("cuda")]

settings = {
    "ir": "dynamo",
    "use_python_runtime": False,
    "enabled_precisions": {torch.float32},
    "immutable_weights": False,
    "lazy_engine_init": True,
    "dynamically_allocate_resources": True

}

model = models.resnet152(pretrained=True).eval().to("cuda")
compiled_module = torch_trt.compile(model, inputs=inputs, **settings)
print((torch.cuda.mem_get_info()[1] - torch.cuda.mem_get_info()[0]) / 1024**3)
compiled_module(*inputs)

time.sleep(30)
with torch_trt.dynamo.runtime.ResourceAllocationStrategy(compiled_module, dynamically_allocate_resources=False):
    print(
        "Memory used (GB):",
        (torch.cuda.mem_get_info()[1] - torch.cuda.mem_get_info()[0]) / 1024**3,
    )
    compiled_module(*inputs)
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(30)
    print(
        "Memory used (GB):",
        (torch.cuda.mem_get_info()[1] - torch.cuda.mem_get_info()[0]) / 1024**3,
    )
    compiled_module(*inputs)
