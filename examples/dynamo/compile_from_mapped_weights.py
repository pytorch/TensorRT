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
With ``release_weights_after_build=True`` the pages are dropped again once the last engine that reads
the weight is built.

This example:

1. writes a model's weights once with ``save_weights_for_mapping``, standing in for a checkpoint;
2. builds the model on the ``meta`` device, so constructing it allocates no weights, and maps its weights;
3. exports and compiles it on the CPU, releasing each weight after its last engine;
4. checks the compiled model against the original.

How much of the weights file is resident is read from ``/proc/self/smaps``, so it prints on Linux only.
"""

import os
import tempfile

import torch
import torch_tensorrt
from torch_tensorrt.dynamo.tools import map_weights, save_weights_for_mapping


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
# Build the model without its weights and map them
# ------------------------------------------------
#
# On ``meta`` the model allocates nothing. ``strict=True`` raises if a tensor in the model's
# ``state_dict`` is missing from the file. Mapping doesn't read the weights, so none of the file is
# resident yet.

with torch.device("meta"):
    model = MLP().eval()

mapping = map_weights(model, path, strict=True)
print(
    f"mapped {mapping.mapped_bytes / 2**20:.0f} MiB, {len(mapping.missing)} missing; "
    f"resident: {resident_mib(path):.0f} MiB"
)

# %%
# Export and compile on the CPU
# -----------------------------
#
# The mapped weights are CPU tensors, so the model and its example input stay on the CPU through
# ``torch.export``. TensorRT takes weights as host pointers and reads each one in place from the mapping:
# there's no GPU copy and no host copy. ``release_weights_after_build=True`` drops a weight's pages once
# the last engine that reads it is built; the tensor stays valid and is re-read from the file if used.

x = torch.randn(4, 2048)
with torch.no_grad():
    exported = torch.export.export(model, (x,))
    trt_model = torch_tensorrt.dynamo.compile(
        exported,
        inputs=[x.cuda()],
        min_block_size=1,
        release_weights_after_build=True,
    )
# With release_weights_after_build=False the whole file (192 MiB here) would still be resident.
print(f"compiled; resident: {resident_mib(path):.0f} MiB of the weights file")

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
