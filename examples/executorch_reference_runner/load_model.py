import argparse
from pathlib import Path

import torch

# Registers TensorRTBackend with ExecuTorch's backend registry as an import side effect. Nothing
# from this package is referenced below: loading and running a program is ExecuTorch's own API.
import torch_tensorrt_executorch_runtime  # noqa: F401
from executorch.runtime import Runtime

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_path",
    type=Path,
    required=True,
    help="Path to the ExecuTorch .pte model",
)
parser.add_argument("--num_runs", type=int, default=1)
args = parser.parse_args()
if args.num_runs < 1:
    raise ValueError("--num_runs must be at least 1")

model_path = args.model_path
x = torch.ones((2, 3, 4, 4), dtype=torch.float32)

program = Runtime.get().load_program(model_path)
method = program.load_method("forward")
if method is None:
    raise RuntimeError(f"{model_path} has no 'forward' method")
for _ in range(args.num_runs):
    outputs = method.execute((x,))
y = outputs[0]

expected = x + 1
torch.testing.assert_close(y.cpu(), expected)

print("methods:", sorted(program.method_names))
print("output shape:", tuple(y.shape))
print("output device:", y.device)
print("PASS: ExecuTorch TensorRT delegate output matches x + 1")
