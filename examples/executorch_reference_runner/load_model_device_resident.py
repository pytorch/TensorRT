"""Run a device-resident .pte and prove the method boundary did not copy.

The program this loads was exported with ``skip_h2d_for_method_inputs`` and
``skip_d2h_for_method_outputs``, so it requires a CUDA input and returns a CUDA
output. Feeding it a CPU tensor is a caller error, not something the runtime
papers over: the delegate would read host memory as if it were device memory.

That contract is what this script checks. ``export_device_resident.py`` already
asserts the serialized program contains no boundary copy operators; this asserts
the runtime half, that a CUDA tensor goes in, a CUDA tensor comes out, and the
values are right.
"""

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
    help="Path to the device-resident ExecuTorch .pte model",
)
parser.add_argument("--num_runs", type=int, default=1)
args = parser.parse_args()
if args.num_runs < 1:
    raise ValueError("--num_runs must be at least 1")

if not torch.cuda.is_available():
    raise RuntimeError(
        "This program keeps its inputs and outputs on the GPU, so it cannot run "
        "without CUDA."
    )

model_path = args.model_path
# The shape export_device_resident.py used, and the value the .expected file
# describes: the reference is cos(erfinv(tanh(1.0))) elementwise.
x = torch.ones((64, 64), dtype=torch.float32, device="cuda")
assert x.is_cuda

program = Runtime.get().load_program(model_path)
method = program.load_method("forward")
if method is None:
    raise RuntimeError(f"{model_path} has no 'forward' method")

for _ in range(args.num_runs):
    outputs = method.execute((x,))
y = outputs[0]

# The point of the whole exercise. Nothing in the Python layer copies a tensor
# now, so if the export flags did not take, this is where it shows up.
if not y.is_cuda:
    raise AssertionError(
        f"FATAL: output came back on {y.device}, so the method boundary still "
        "copies device to host. The skip_d2h_for_method_outputs flag did not take."
    )

expected = torch.cos(torch.erfinv(torch.tanh(x)))
torch.testing.assert_close(y, expected)

print("methods:", sorted(program.method_names))
print("input device:", x.device)
print("output device:", y.device)
print(
    "PASS: device-resident ExecuTorch TensorRT program kept inputs and outputs on the GPU"
)
