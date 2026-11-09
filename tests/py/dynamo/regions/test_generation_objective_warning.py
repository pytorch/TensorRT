# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

"""objective='candidate' warns when it accepts a kernel slower than eager PyTorch."""

from types import SimpleNamespace

import pytest

from torch_tensorrt.dynamo.region_compilation._generation._factory import (
    _slower_than_eager,
)


@pytest.mark.parametrize(
    "objective,latency,eager,warns",
    [
        ("candidate", 13.6, 0.84, True),
        ("candidate", 0.5, 0.84, False),
        ("candidate", 0.84, 0.84, False),
        ("eager", 13.6, 0.84, False),
        ("candidate", 13.6, None, False),
        ("candidate", float("nan"), 0.84, False),
    ],
)
def test_slower_than_eager(objective, latency, eager, warns):
    message = _slower_than_eager(
        {"latency_ms": latency, "reference_latency_ms": eager},
        SimpleNamespace(objective=objective),
    )
    assert bool(message) is warns
    if warns:
        assert "objective='eager'" in message and "13.6 ms vs 0.84 ms" in message
