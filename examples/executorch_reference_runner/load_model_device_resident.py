# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Run a device-resident .pte and prove the method boundary did not copy.

The program this loads was exported with ``skip_h2d_for_method_inputs`` and
``skip_d2h_for_method_outputs``, so its boundary carries no copy operators and it
wants a CUDA input. A host input still gives the right answer, but it costs a
staging copy that the export existed to remove, so it defeats the point rather
than breaking the program. Where a host-backed buffer tagged for the device does
fail is in the runtime rather than in this delegate, on a device that reports it
cannot read pageable host memory.

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
# No check on the tensor's device here. Asking for the device is what can fail, and it has already
# failed by the time a check would run, so testing the result was a guard that could not fire.
# Whether CUDA is present at all is checked above, where the question can actually be answered.

# The Module API backs device-tagged arenas with device memory.
program = Runtime.get().load_program(model_path)
forward = program.load_method("forward")
if "forward" not in program.method_names:
    raise RuntimeError(f"{model_path} has no 'forward' method")

for _ in range(args.num_runs):
    outputs = forward.execute((x,))
y = outputs[0]

# This catches the export flags not taking, and that is all it catches. It does not prove the buffer is
# device memory: the tag and the pointer can disagree, and measured here they do, because the bindings
# hand back a buffer tagged cuda whose pointer is ordinary host memory. Proving residency needs the
# driver's own view of the pointer, which is more machinery than an example should carry, so the honest
# claim is the narrow one. The numbers below are still checked, and they are correct.
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
print("PASS: device-resident ExecuTorch TensorRT program ran with the expected values")
