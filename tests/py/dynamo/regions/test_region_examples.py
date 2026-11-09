# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

"""Keep public region examples executable; API edge cases live separately."""

import os
from pathlib import Path

import pytest
import torch
import torch_tensorrt
from torch.utils._triton import has_triton

from ._utils import run_python

ROOT = Path(__file__).resolve().parents[4]
# Each example and the success message it prints.
EXAMPLES = {
    "execute_in_torch.py": "Both eager and inductor regions match PyTorch",
    "fused_op.py": "The supplied custom operator runs in PyTorch",
    "fused_kernel.py": "One TensorRT engine includes the fused kernel",
}


@pytest.mark.parametrize("script", EXAMPLES)
def test_region_example_uses_only_public_api(script):
    # Users copy examples; internal modules, metadata keys and attributes can change.
    source = (ROOT / "examples" / "dynamo" / script).read_text(encoding="utf-8")
    for internal in ("region_compilation", "torch_tensorrt_region", "_torch_tensorrt_"):
        assert internal not in source, f"{script} uses internal {internal!r}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(("script", "success"), EXAMPLES.items())
def test_region_example(script, success):
    if script != "fused_op.py":
        pytest.importorskip("triton")
    if script == "execute_in_torch.py" and not has_triton():
        pytest.skip("CUDA Inductor requires Triton support on the device")
    if script == "fused_kernel.py" and not torch_tensorrt.ENABLED_FEATURES.qdp_plugin:
        pytest.skip("TensorRT QDP plugin support is required")
    output = run_python(ROOT / "examples" / "dynamo" / script, cwd=ROOT, timeout=180)
    assert success in output


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.skipif(os.name != "posix", reason="Kernel generation requires POSIX")
def test_agentic_region_example_runs_with_an_offline_provider(tmp_path, monkeypatch):
    pytest.importorskip("triton")
    if not torch_tensorrt.ENABLED_FEATURES.qdp_plugin:
        pytest.skip("TensorRT QDP plugin support is required")
    root = Path(__file__).resolve().parents[4]
    # The example and its provider subprocesses import the adapter by name.
    path = [str(Path(__file__).resolve().parent), os.environ.get("PYTHONPATH", "")]
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(filter(None, path)))
    output = run_python(
        root / "examples" / "dynamo" / "agentic_region_kernel_generation.py",
        "--provider",
        "_example_kernel_provider:author",
        "--cache-dir",
        tmp_path,
        cwd=root,
        timeout=300,
    )
    assert (
        "matches PyTorch on fresh inputs, and on nonfinite inputs wherever "
        "validation requires it" in output
    )
