# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""

.. _compile_from_mapped_weights:

Compiling from Memory-Mapped Weights
====================================

To compile a model, all of its weights normally stay in host memory while every TensorRT engine is built.
``torch_tensorrt.dynamo.low_memory`` instead writes a loaded model's weights to a file and points the
model's own tensors at zero-copy views of it, which frees the loaded copy. ``torch.export`` reads only
shapes, so a weight's pages become resident only when TensorRT reads them, while it builds the engine that
uses the weight. The pages are dropped again once the last engine that reads the weight is built. With the
resource partitioner, the model is built as several engines, each small enough to build within a host
memory budget, so only about one engine's weights are resident at a time.

This example:

1. loads a model, with random weights standing in for a checkpoint;
2. with ``low_memory``, moves its weights into a file, then exports and compiles it on the CPU as several
   engines within a CPU memory budget, watching how much of the weights file is resident;
3. checks the compiled model against the outputs of the loaded one.

The memory figures are read from ``/proc/self``, so they print on Linux only.
"""

import os
import threading
from typing import Any, Callable, Tuple

import psutil
import torch
import torch_tensorrt


def anonymous_mib() -> float:
    """MiB of this process's memory that no file backs (RssAnon in /proc/self/status)."""
    if not os.path.exists("/proc/self/status"):
        return float("nan")
    for line in open("/proc/self/status"):
        if line.startswith("RssAnon:"):
            return int(line.split()[1]) / 1024
    return float("nan")


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
# Load the model
# --------------
#
# Load it as you usually would, for example with ``from_pretrained``; here its weights are random. The
# expected outputs are computed first, without gradients: an autograd graph would hold the loaded weights
# and keep them in memory.

torch.manual_seed(0)
model = MLP().eval()
x = torch.randn(4, 2048)
with torch.no_grad():
    expected = model(x)
weights_mib = sum(t.nbytes for t in model.state_dict().values()) / 2**20
print(
    f"loaded {weights_mib:.0f} MiB of weights; anonymous memory: {anonymous_mib():.0f} MiB"
)

# %%
# Move the weights into a file, then export and compile within a memory budget
# ----------------------------------------------------------------------------
#
# Entering ``low_memory`` writes the weights to a file in Torch-TensorRT's cache directory and points the
# model's tensors at it, which frees the loaded copy. None of the file is resident until something reads it.
#
# The mapped weights are CPU tensors, so the model and its example input stay on the CPU through
# ``torch.export``. TensorRT takes weights as host pointers and reads each one in place from the mapping:
# there's no GPU copy and no host copy. A weight's pages are dropped once the last engine that reads it
# is built; the tensor stays valid and is re-read from the file if used.
#
# ``enable_resource_partitioning`` splits the model into engines small enough to build within
# ``cpu_memory_budget``, the host memory the process may use. Each engine gets up to a quarter of the
# budget left over after the memory in use when partitioning starts, since building an engine takes a few
# times its weights' size. The budget here is the memory in use now, which counts the loaded weights the
# block frees, so engines get at most about a quarter of the weights, and building them one after another
# keeps only about one engine's weights resident.

with (
    torch_tensorrt.dynamo.low_memory(
        model, cpu_memory_budget=psutil.Process().memory_info().rss
    ) as low_mem,
    torch.no_grad(),
):
    path = os.path.realpath(low_mem.weights_file)
    print(
        f"moved {low_mem.report.mapped_bytes / 2**20:.0f} MiB of weights to {path}; anonymous memory: "
        f"{anonymous_mib():.0f} MiB; resident: {resident_mib(path):.0f} MiB"
    )
    exported = torch.export.export(low_mem.model, (x,))
    trt_model, peak = peak_resident_mib(
        path,
        lambda: torch_tensorrt.dynamo.compile(
            exported, inputs=[x.cuda()], **low_mem.compile_settings
        ),
    )
    engines = [name for name, _ in trt_model.named_children() if "_run_on_acc" in name]
    # As one engine, without the partitioner, the whole file is resident while it builds. With
    # release_weights_after_build=False, all of it would stay resident after compiling.
    print(
        f"compiled into {len(engines)} engines; at most {peak:.0f} MiB of the weights file was "
        f"resident, {resident_mib(path):.0f} MiB is now"
    )

# %%
# Check the result
# ----------------

with torch.no_grad():
    actual = trt_model(x.cuda()).cpu()
torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)
print(
    f"max abs difference from the loaded model: {(actual - expected).abs().max().item():.2e}"
)

# %%
# The model keeps reading its weights from the file, which is deleted once the model no longer uses it. A
# model too large to load at all can be built on the ``meta`` device instead, and its weights mapped from a
# file with ``torch_tensorrt.dynamo.tools.map_weights``; ``save_weights_for_mapping`` converts a checkpoint
# into one without loading it.
