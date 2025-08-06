import subprocess

import torch
import torch_tensorrt
from torch_tensorrt._utils import is_tensorrt_rtx

print(f"Torch CUDA version: {torch.version.cuda}")
print(f"Torch TensorRT version: {torch_tensorrt.__version__}")
print(f"Is TensorRT RTX: {is_tensorrt_rtx()}")

result = subprocess.run(
    ["systeminfo"],
    capture_output=True,
    text=True,
)
print(result.stdout)
print(result.stderr)
