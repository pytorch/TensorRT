# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

"""Offline provider adapter proposing a known-good kernel for the agentic example."""

import math

_NUMEL = 257 * 1025
_BLOCK = 1024
_SOURCE = """import triton
import triton.language as tl


@triton.jit
def generated(arg0, arg1, arg2, out0, N: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    x = tl.load(arg0 + offsets, mask=mask)
    bias = tl.load(arg1 + offsets, mask=mask)
    scale = tl.load(arg2 + offsets, mask=mask)
    shifted = x + bias
    # Unlike tl.maximum, this keeps torch.relu's NaN propagation.
    tl.store(out0 + offsets, tl.where(shifted < 0, 0.0, shifted) * scale, mask=mask)
"""


def author(**_):
    return {
        "action": "propose",
        "candidates": [
            {
                "source": _SOURCE,
                "kernel_name": "generated",
                "constants": [
                    {"name": "N", "value": _NUMEL},
                    {"name": "BLOCK", "value": _BLOCK},
                ],
                "grid": [math.ceil(_NUMEL / _BLOCK)],
                "num_warps": 4,
                "num_stages": None,
            }
        ],
        "resource": None,
        "offset": None,
    }
