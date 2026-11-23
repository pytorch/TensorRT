# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Real CPU TorchScript round trips with public entry points absent."""

import importlib
import importlib.util
import inspect
import io
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[4]


@pytest.fixture
def compatibility():
    path = ROOT / "py/torch_tensorrt/_jit_compat.py"
    spec = importlib.util.spec_from_file_location("trt_jit_compat_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("name", ["load", "save", "script", "trace"])
def test_public_override_is_preserved(compatibility, monkeypatch, name):
    def overridden(*args, **kwargs):
        return "public override"

    monkeypatch.setattr(torch.jit, name, overridden, raising=False)
    assert compatibility.torchscript_api(name) is overridden


@pytest.mark.parametrize("name", ["load", "save", "script", "trace"])
def test_missing_public_api_preserves_original_signature(
    compatibility, monkeypatch, name
):
    monkeypatch.delattr(torch.jit, name, raising=False)
    expected = getattr(
        importlib.import_module(compatibility._IMPLEMENTATIONS[name]), name
    )
    resolved = compatibility.torchscript_api(name)
    assert resolved is expected
    assert inspect.signature(resolved) == inspect.signature(expected)
    assert not hasattr(torch.jit, name)


@pytest.mark.parametrize("frontend", ["script", "trace"])
def test_real_round_trip_without_public_exports(compatibility, monkeypatch, frontend):
    for name in ("load", "save", "script", "trace"):
        monkeypatch.delattr(torch.jit, name, raising=False)
    model = torch.nn.Linear(2, 2).eval()
    inputs = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    compile_model = compatibility.torchscript_api(frontend)
    compiled = (
        compile_model(model)
        if frontend == "script"
        else compile_model(model, (inputs,))
    )
    stream = io.BytesIO()
    compatibility.torchscript_api("save")(
        compiled, stream, _extra_files={"metadata": "preserved"}
    )
    stream.seek(0)
    extra_files = {"metadata": ""}
    loaded = compatibility.torchscript_api("load")(
        stream, map_location="cpu", _extra_files=extra_files
    )
    torch.testing.assert_close(loaded(inputs), model(inputs))
    assert extra_files["metadata"] == b"preserved"
    assert not any(
        hasattr(torch.jit, name) for name in ("load", "save", "script", "trace")
    )
