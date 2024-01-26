import glob
import os
import shutil
import site
import subprocess

python_site_packages = site.getsitepackages()[-1]
extra_cudnn = os.path.join(python_site_packages, "nvidia", "cudnn")
extra_torch = os.path.join(python_site_packages, "torch", "lib")

for dll in glob.glob(os.path.join(extra_cudnn, "**", "*.dll"), recursive=True):
    shutil.copy(dll, extra_torch)

result = subprocess.run(
    ['nvidia-smi'],
    capture_output = True,
    text = True,
)
print(result.stdout)
print(result.stderr)


import tensorrt
import torch

import torch_tensorrt
