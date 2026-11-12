# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import torch
from torch_tensorrt.dynamo.lowering import torch_disabled_decompositions
from torch_tensorrt.verification import InputSample


class DiffModel(torch.nn.Module):
    """Expose unsupported aten.diff.default for decomposition experiments."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.diff(x, n=1, dim=-1)


def diff_model() -> DiffModel:
    # Keep this unsupported-op simulation local to the isolated MCP worker.
    torch_disabled_decompositions.add(torch.ops.aten.diff.default)
    return DiffModel().cuda().eval()


def diff_inputs(
    seed: int,
    batch_size: int = 32,
    width: int = 4096,
) -> InputSample:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(
        batch_size,
        width,
        generator=generator,
        device="cuda",
        dtype=torch.float16,
    )
    return InputSample(
        args=(x,),
        profile="static",
        sample_id=f"batch-{batch_size}-width-{width}",
        label="Unsupported aten.diff decomposition example",
    )
