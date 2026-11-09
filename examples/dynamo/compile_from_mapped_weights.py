# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""

.. _compile_from_mapped_weights:

Compiling from Memory-Mapped Weights
====================================

To compile a model, its weights normally have to be loaded into host memory first, all of them, before
the first TensorRT engine is built. ``torch_tensorrt.dynamo.tools.map_weights`` instead points the model's
weights at zero-copy views of a safetensors file. ``torch.export`` reads only shapes, so a weight's
pages become resident only when TensorRT reads them, while it builds the engine that uses the weight.
The pages are dropped again once the last engine that reads the weight is built. With the resource
partitioner, the model is built as several engines, each small enough to build within a host memory
budget, so only about one engine's weights are resident at a time.

This example:

1. writes a model's weights once with ``save_weights_for_mapping``, standing in for a checkpoint;
2. with ``torch_tensorrt.dynamo.low_memory``, builds the model on the ``meta`` device, so constructing it
   allocates no weights, maps its weights, and exports and compiles it on the CPU as several engines
   within a CPU memory budget, watching how much of the weights file is resident;
3. checks the compiled model against the original.

How much of the weights file is resident is read from ``/proc/self/smaps``, so it prints on Linux only.
"""

import os
import tempfile
import threading
from typing import Any, Callable, Tuple

import psutil
import torch
import torch_tensorrt
from torch_tensorrt.dynamo.tools import save_weights_for_mapping


def resident_mib(path: str) -> float:
    """MiB of the file at `path` resident in this process (its mappings' Rss in /proc/self/smaps)."""
    if not os.path.exists("/proc/self/smaps"):
        return float("nan")
    total, in_file = 0, False
    for line in open("/proc/self/smaps"):
        fields = line.split()
        if fields and "-" in fields[0] and len(fields) >= 5:  # a mapping's header line
            in_file = fields[-1] == path
        elif in_file and fields[0] == "Rss:":
            total += int(fields[1])
    return total / 1024


def peak_resident_mib(path: str, fn: Callable[[], Any]) -> Tuple[Any, float]:
    """Run ``fn()`` while sampling ``resident_mib(path)``; return its result and the peak. TensorRT
    releases the GIL while it builds an engine, so the samples cover the builds."""
    peak, done = [resident_mib(path)], threading.Event()

    def sample() -> None:
        while not done.is_set():
            peak[0] = max(peak[0], resident_mib(path))
            done.wait(0.005)

    sampler = threading.Thread(target=sample)
    sampler.start()
    try:
        result = fn()
    finally:
        done.set()
        sampler.join()
    return result, peak[0]


class MLP(torch.nn.Module):
    def __init__(self, width: int = 2048, depth: int = 12) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList(
            torch.nn.Linear(width, width) for _ in range(depth)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = torch.relu(layer(x))
        return x


# %%
# Write the weights once
# ----------------------
#
# Any safetensors file maps if its tensors have the names, dtypes and shapes the model uses.
# ``save_weights_for_mapping`` is for checkpoints whose loader casts, e.g. stored in float32 and run in
# bfloat16: it writes the weights as they are after loading, a tied weight once.

torch.manual_seed(0)
reference = MLP().eval()
path = os.path.join(tempfile.mkdtemp(), "mlp.safetensors")
save_weights_for_mapping(reference, path)
print(f"wrote {os.path.getsize(path) / 2**20:.0f} MiB of weights to {path}")

# %%
# Build, map, export and compile within a memory budget
# -----------------------------------------------------
#
# ``low_memory`` builds the model on ``meta``, where it allocates nothing, and maps its weights; every
# tensor must be in the file. Mapping doesn't read the weights, so none of the file is resident yet.
#
# The mapped weights are CPU tensors, so the model and its example input stay on the CPU through
# ``torch.export``. TensorRT takes weights as host pointers and reads each one in place from the mapping:
# there's no GPU copy and no host copy. A weight's pages are dropped once the last engine that reads it
# is built; the tensor stays valid and is re-read from the file if used.
#
# ``enable_resource_partitioning`` splits the model into engines small enough to build within
# ``cpu_memory_budget``, the host memory the process may use. Each engine gets up to a quarter of the
# budget left over after the memory already in use, since building an engine takes a few times its
# weights' size. So a budget of the memory in use now plus the weights' size makes engines of at most
# about a quarter of the weights, and building them one after another keeps only about one engine's
# weights resident.

x = torch.randn(4, 2048)
with (
    torch_tensorrt.dynamo.low_memory(
        lambda: MLP().eval(),
        path,
        cpu_memory_budget=psutil.Process().memory_info().rss + os.path.getsize(path),
    ) as low_mem,
    torch.no_grad(),
):
    print(
        f"mapped {low_mem.report.mapped_bytes / 2**20:.0f} MiB; "
        f"resident: {resident_mib(path):.0f} MiB"
    )
    exported = torch.export.export(low_mem.model, (x,))
    trt_model, peak = peak_resident_mib(
        path,
        lambda: torch_tensorrt.dynamo.compile(
            exported, inputs=[x.cuda()], **low_mem.compile_settings
        ),
    )
    engines = [name for name, _ in trt_model.named_children() if "_run_on_acc" in name]
    # As one engine, without the partitioner, the whole file (192 MiB here) is resident while it builds.
    # With release_weights_after_build=False, all of it would stay resident after compiling.
    print(
        f"compiled into {len(engines)} engines; at most {peak:.0f} MiB of the weights file was "
        f"resident, {resident_mib(path):.0f} MiB is now"
    )

# %%
# Check the result
# ----------------

with torch.no_grad():
    expected = reference(x)
    actual = trt_model(x.cuda()).cpu()
torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)
print(
    f"max abs difference from the original model: {(actual - expected).abs().max().item():.2e}"
)

# %%
# The file must stay in place while the model or its weights are in use: released pages are read from it
# again. On large models this keeps the host's peak near the largest single engine build instead of the
# whole model, e.g. pi0.5 (6.7 GiB of bfloat16 weights): host RSS before compilation went from 8.7 to
# 2.9 GiB.
