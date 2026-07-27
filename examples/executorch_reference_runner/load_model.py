import argparse
from pathlib import Path

import torch
import torch_tensorrt

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

program = torch_tensorrt.load(model_path, format="executorch")
for _ in range(args.num_runs):
    outputs = program.forward(x)
y = outputs[0]

expected = x + 1
torch.testing.assert_close(y.cpu(), expected)

print("methods:", sorted(program.method_names))
print("output shape:", tuple(y.shape))
print("output device:", y.device)
print("PASS: ExecuTorch TensorRT delegate output matches x + 1")
