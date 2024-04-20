import subprocess

import tensorrt  # noqa: F401
import torch

print(f"Torch CUDA version: {torch.version.cuda}")

result = subprocess.run(
    ["systeminfo"],
    capture_output=True,
    text=True,
)
print(result.stdout)
print(result.stderr)
