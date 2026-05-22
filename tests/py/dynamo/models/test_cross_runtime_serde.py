"""Tests for cross-runtime save/load of .pt2 TRT artifacts.

Verifies that an ExportedProgram saved with the C++ Torch-TensorRT runtime can
be loaded and executed in a Python-only environment (no libtorchtrt*.so), and
that inference results match.
"""

from __future__ import annotations

import os
import subprocess
import sys
import unittest

import pytest
import torch
import torch_tensorrt as torchtrt
from torch_tensorrt.dynamo.utils import COSINE_THRESHOLD, cosine_similarity

assertions = unittest.TestCase()

HELPER_SCRIPT = os.path.join(os.path.dirname(__file__), "_cross_runtime_load_helper.py")


class SmallConvModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 16, 3, stride=1, bias=True)
        self.relu = torch.nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.conv(x))


def _compile_and_save(
    model: torch.nn.Module, inp: torch.Tensor, path: str
) -> torch.Tensor:
    """Compile *model* with TRT, save the artifact, return eager TRT output."""
    compile_spec = {
        "inputs": [
            torchtrt.Input(inp.shape, dtype=torch.float, format=torch.contiguous_format)
        ],
        "ir": "dynamo",
        "min_block_size": 1,
        "cache_built_engines": False,
        "reuse_cached_engines": False,
    }
    exp_program = torchtrt.dynamo.trace(model, **compile_spec)
    trt_module = torchtrt.dynamo.compile(exp_program, **compile_spec)

    with torch.no_grad():
        reference_output = trt_module(inp)

    torchtrt.save(trt_module, path, retrace=False)
    return reference_output


def _tmp_paths(tmpdir):
    base = str(tmpdir)
    return {
        "artifact": os.path.join(base, "trt.ep"),
        "model_state": os.path.join(base, "model_state.pt"),
        "input": os.path.join(base, "input.pt"),
        "output": os.path.join(base, "output.pt"),
    }


def _run_helper(args: list[str]) -> None:
    """Run _cross_runtime_load_helper.py with given args; raise on failure."""
    result = subprocess.run(
        [sys.executable, HELPER_SCRIPT] + args,
        capture_output=True,
        text=True,
        timeout=300,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Python-only subprocess failed (rc={result.returncode}).\n"
            f"--- stdout ---\n{result.stdout}\n"
            f"--- stderr ---\n{result.stderr}"
        )


def _assert_outputs_match(
    reference: torch.Tensor, loaded: torch.Tensor, label: str
) -> None:
    reference = reference[0] if isinstance(reference, (tuple, list)) else reference
    loaded = loaded[0] if isinstance(loaded, (tuple, list)) else loaded
    cos_sim = cosine_similarity(reference.cpu(), loaded.cpu())
    assertions.assertTrue(
        cos_sim > COSINE_THRESHOLD,
        msg=f"{label}: cosine similarity {cos_sim} < {COSINE_THRESHOLD}",
    )


@pytest.mark.unit
def test_save_cpp_load_python(tmpdir):
    """Save with C++ runtime active, load in Python-only subprocess."""
    if not torchtrt.ENABLED_FEATURES.torch_tensorrt_runtime:
        pytest.skip("C++ runtime not available; nothing to cross-test")

    paths = _tmp_paths(tmpdir)

    model = SmallConvModel().eval().cuda()
    inp = torch.randn(1, 3, 32, 32, device="cuda")

    reference_output = _compile_and_save(model, inp, paths["artifact"])

    torch.save(inp, paths["input"])
    _run_helper(
        [
            "load",
            "--artifact",
            paths["artifact"],
            "--input",
            paths["input"],
            "--output",
            paths["output"],
        ]
    )
    python_output = torch.load(paths["output"], weights_only=True)

    _assert_outputs_match(reference_output, python_output, "save_cpp_load_python")


@pytest.mark.unit
def test_save_python_load_python(tmpdir):
    """Save and load entirely in Python-only subprocesses."""
    paths = _tmp_paths(tmpdir)

    model = SmallConvModel().eval().cuda()
    inp = torch.randn(1, 3, 32, 32, device="cuda")

    torch.save(model.state_dict(), paths["model_state"])
    torch.save(inp, paths["input"])

    with torch.no_grad():
        pytorch_output = model(inp)

    _run_helper(
        [
            "save",
            "--model-state",
            paths["model_state"],
            "--input",
            paths["input"],
            "--artifact",
            paths["artifact"],
        ]
    )
    _run_helper(
        [
            "load",
            "--artifact",
            paths["artifact"],
            "--input",
            paths["input"],
            "--output",
            paths["output"],
        ]
    )
    python_output = torch.load(paths["output"], weights_only=True)

    _assert_outputs_match(pytorch_output, python_output, "save_python_load_python")


@pytest.mark.unit
def test_save_python_load_cpp(tmpdir):
    """Save in Python-only subprocess, load in C++ runtime."""
    if not torchtrt.ENABLED_FEATURES.torch_tensorrt_runtime:
        pytest.skip("C++ runtime not available; nothing to cross-test")

    paths = _tmp_paths(tmpdir)

    model = SmallConvModel().eval().cuda()
    inp = torch.randn(1, 3, 32, 32, device="cuda")

    with torch.no_grad():
        pytorch_output = model(inp)

    torch.save(model.state_dict(), paths["model_state"])
    torch.save(inp, paths["input"])
    _run_helper(
        [
            "save",
            "--model-state",
            paths["model_state"],
            "--input",
            paths["input"],
            "--artifact",
            paths["artifact"],
        ]
    )

    loaded_ep = torchtrt.load(paths["artifact"])
    with torch.no_grad():
        cpp_output = loaded_ep.module()(inp)

    _assert_outputs_match(pytorch_output, cpp_output, "save_python_load_cpp")
