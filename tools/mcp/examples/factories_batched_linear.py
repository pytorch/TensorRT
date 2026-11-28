# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import torch
from torch_tensorrt.verification import InputSample


class SharedLinearBranches(torch.nn.Module):
    """Apply one projection to eight independent small batches."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.projection = torch.nn.Linear(
            hidden_size,
            hidden_size,
            bias=False,
            device="cuda",
            dtype=torch.float16,
        )
        self.eval()

    def forward(
        self,
        input0: torch.Tensor,
        input1: torch.Tensor,
        input2: torch.Tensor,
        input3: torch.Tensor,
        input4: torch.Tensor,
        input5: torch.Tensor,
        input6: torch.Tensor,
        input7: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        return (
            self.projection(input0),
            self.projection(input1),
            self.projection(input2),
            self.projection(input3),
            self.projection(input4),
            self.projection(input5),
            self.projection(input6),
            self.projection(input7),
        )


def batched_linear_model(hidden_size: int = 4096) -> SharedLinearBranches:
    return SharedLinearBranches(hidden_size)


def batched_linear_inputs(
    seed: int,
    batch_size: int = 16,
    hidden_size: int = 4096,
) -> InputSample:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    inputs = tuple(
        torch.randn(
            batch_size,
            hidden_size,
            generator=generator,
            device="cuda",
            dtype=torch.float16,
        )
        for _ in range(8)
    )
    return InputSample(
        args=inputs,
        profile="static",
        sample_id=f"branches-8-batch-{batch_size}-hidden-{hidden_size}",
        label="Eight shared-weight linear projections",
    )
