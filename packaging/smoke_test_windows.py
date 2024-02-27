import glob
import os
import shutil
import site
import subprocess

python_site_packages = site.getsitepackages()[-1]
extra_cudnn = os.path.join(python_site_packages, "nvidia", "cudnn")
extra_torch = os.path.join(python_site_packages, "torch", "lib")

print(
    f"CUDNN DLLS: {list(glob.glob(os.path.join(extra_cudnn, '**', '*.dll'), recursive=True))}"
)
print(
    f"ALL TORCH DLLS: {list(glob.glob(os.path.join(extra_torch, '**', '*.dll'), recursive=True))}"
)
print(f"Current PATH: {os.getenv('PATH')}")

# Move TRT cuDNN DLLs into torch path
for dll in glob.glob(os.path.join(extra_cudnn, "**", "*.dll"), recursive=True):
    shutil.copy(dll, extra_torch)

import torch

print(f"Torch CUDA version: {torch.version.cuda}")

result = subprocess.run(
    ["systeminfo"],
    capture_output=True,
    text=True,
)
print(result.stdout)
print(result.stderr)


import tensorrt  # noqa: F401
