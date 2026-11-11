# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""

.. _compile_from_mapped_weights:

Compiling from Memory-Mapped Weights
====================================

To compile a model, all of its weights normally have to be loaded into host memory first and stay
there while every TensorRT engine is built. With ``torch_tensorrt.dynamo.low_memory(model_weights_on_meta=True)``,
you **construct the model from its config** (for example ``AutoModelForCausalLM.from_config(config)`` or
``MLP()``, **not** ``from_pretrained``): newly registered ``nn.Parameter`` tensors land on ``meta`` so no
weights are allocated in RAM, while computed buffers stay real. Calling ``low_mem.map_weights(model, ...)``
then streams the checkpoint in 64 MiB chunks into a cache file and points the model's tensors at zero-copy
views of it.

``torch.export`` reads only shapes, so a weight's pages become resident only when TensorRT reads them
while building the engine that uses the weight, and are dropped again once the last engine reading that
weight is built. With the resource partitioner, the model is built as several engines within a host
memory budget, so only about one engine's weights are resident at a time.

.. warning::

   Initialize the model from its **config** (``AutoModel.from_config(config)`` or ``MyModel(config)``),
   **not** ``from_pretrained(...)``. Calling ``from_pretrained`` loads the entire checkpoint into memory
   before ``low_memory`` can stream it, defeating ``model_weights_on_meta=True``.

This example:

1. saves a reference checkpoint on disk (standing in for a downloaded safetensors checkpoint);
2. enters ``low_memory(model_weights_on_meta=True)``, constructs ``MLP()`` from its config with zero weight
   allocation, maps weights from the checkpoint via ``low_mem.map_weights``, and compiles within a CPU
   memory budget while watching how much of the weights file is resident;
3. checks the compiled model against the reference outputs.

The memory figures are read from ``/proc/self``, so they print on Linux only.
"""

import os
import tempfile
import threading
from typing import Any, Callable, Tuple

import psutil
import torch
import torch_tensorrt
from torch_tensorrt.dynamo.tools import save_weights_for_mapping


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
# Prepare a reference checkpoint on disk
# --------------------------------------
#
# In real usage this is your existing safetensors file, checkpoint folder, or Hugging Face Hub repo id
# (for Hugging Face models built with ``AutoModel.from_config(config)``, ``map_weights`` finds the
# checkpoint automatically from ``config._name_or_path``). Here we write one to a temporary folder and
# delete the reference model so compilation starts with none of its weights in memory.

checkpoint_dir = tempfile.mkdtemp()
checkpoint_path = os.path.join(checkpoint_dir, "model.safetensors")
torch.manual_seed(0)
x = torch.randn(4, 2048)
reference = MLP().eval()
weights_bytes = sum(t.nbytes for t in reference.state_dict().values())
with torch.no_grad():
    expected = reference(x)
save_weights_for_mapping(reference, checkpoint_path)
del reference

# %%
# Construct from config (not ``from_pretrained``), map weights, and compile
# -------------------------------------------------------------------------
#
# With ``model_weights_on_meta=True``, constructing ``MLP()`` inside ``low_memory`` places every parameter on
# ``meta`` — allocating 0 MiB of weights — instead of loading a checkpoint via ``from_pretrained``.
# ``low_mem.map_weights(model, checkpoint=checkpoint_dir)`` then turns ``meta`` creation off, streams the
# checkpoint in 64 MiB chunks into Torch-TensorRT's cache file, and points ``model``'s tensors at views of
# that file. None of the file is resident until TensorRT reads each partition's weights.

anon_before = anonymous_mib()
with (
    torch_tensorrt.dynamo.low_memory(
        model_weights_on_meta=True,
        cpu_memory_budget=psutil.Process().memory_info().rss + weights_bytes,
    ) as low_mem,
    torch.no_grad(),
):
    model = (
        MLP().eval()
    )  # constructed from config/args: parameters on meta, 0 MiB allocated
    print(
        f"constructed on meta; anonymous memory delta: {anonymous_mib() - anon_before:+.0f} MiB"
    )
    low_mem.map_weights(model, checkpoint=checkpoint_dir)
    path = os.path.realpath(low_mem.weights_file)
    print(
        f"mapped {low_mem.report.mapped_bytes / 2**20:.0f} MiB from checkpoint -> {path}; "
        f"resident: {resident_mib(path):.0f} MiB"
    )
    exported = torch.export.export(model, (x,))
    trt_model, peak = peak_resident_mib(
        path,
        lambda: torch_tensorrt.dynamo.compile(
            exported, inputs=[x.cuda()], **low_mem.compile_settings
        ),
    )
    engines = [name for name, _ in trt_model.named_children() if "_run_on_acc" in name]
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
    f"max abs difference from the reference model: {(actual - expected).abs().max().item():.2e}"
)
os.remove(checkpoint_path)
os.rmdir(checkpoint_dir)
