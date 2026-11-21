# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from pathlib import Path

import pytest
import torch

_TS_TEST_DIR = Path(__file__).parent


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(items):
    """Skip the TorchScript tests on PyTorch builds that no longer ship TorchScript."""
    if hasattr(torch.jit, "script"):
        return
    skip = pytest.mark.skip(
        reason="needs a PyTorch build with TorchScript (torch.jit.script)"
    )
    for item in items:
        if _TS_TEST_DIR in Path(item.path).parents and not item.get_closest_marker(
            "runs_without_torchscript"
        ):
            item.add_marker(skip)
