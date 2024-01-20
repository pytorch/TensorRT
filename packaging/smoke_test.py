import os
import shutil
import site

python_site_packages = site.getsitepackages()[-1]
extra_cudnn = os.path.join(python_site_packages, "nvidia", "cudnn")
print(f"Deleting directory: {extra_cudnn}")
shutil.rmtree(extra_cudnn)


import tensorrt
import torch
import torch_tensorrt
