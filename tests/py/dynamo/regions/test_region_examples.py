# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

"""Keep both public region examples executable; API edge cases live separately."""

import subprocess
import sys
from pathlib import Path

import pytest
import torch
from torch.utils._triton import has_triton

import torch_tensorrt


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(
    ("script", "success"),
    [
        ("execute_in_torch.py", "Both eager and inductor regions match PyTorch"),
        ("fuse_in_plugin.py", "One TensorRT engine includes the fused kernel"),
    ],
)
def test_region_example(script, success):
    pytest.importorskip("triton")
    if script == "execute_in_torch.py" and not has_triton():
        pytest.skip("CUDA Inductor requires Triton support on the device")
    if script == "fuse_in_plugin.py" and not torch_tensorrt.ENABLED_FEATURES.qdp_plugin:
        pytest.skip("TensorRT QDP plugin support is required")
    root = Path(__file__).resolve().parents[4]
    result = subprocess.run(
        [sys.executable, str(root / "examples" / "dynamo" / script)],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert success in result.stdout
