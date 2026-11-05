# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import functools
import glob
import tempfile

import pytest
import torch
import torch_tensorrt
from torch_tensorrt.dynamo import _compiler
from torch_tensorrt.dynamo.runtime import TorchTensorRTModule


class _TwoEngines(torch.nn.Module):
    """Two TensorRT partitions around an op forced to PyTorch."""

    def __init__(self):
        super().__init__()
        self.a, self.b = torch.nn.Linear(64, 64), torch.nn.Linear(64, 8)

    def forward(self, x):
        return self.b(torch.sin(self.a(x)))


@pytest.mark.unit
def test_exported_program_loads_the_engines_once_all_are_built(tmp_path, monkeypatch):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    built, setups = [], []
    real_convert = _compiler.convert_module
    real_setup = TorchTensorRTModule.setup_engine

    def recording_convert(*args, **kwargs):
        built.append(kwargs.get("name"))
        return real_convert(*args, **kwargs)

    def recording_setup(self):
        setups.append(len(built))
        real_setup(self)

    monkeypatch.setattr(_compiler, "convert_module", recording_convert)
    monkeypatch.setattr(TorchTensorRTModule, "setup_engine", recording_setup)

    torch.manual_seed(0)
    model = _TwoEngines().eval().cuda()
    x = torch.randn(4, 64, device="cuda")
    path = tmp_path / "model.ep"
    torch_tensorrt.dynamo.compile_and_save(
        torch.export.export(model, (x,)),
        path,
        arg_inputs=[x],
        min_block_size=1,
        torch_executed_ops={torch.ops.aten.sin.default},
    )

    # Both engines were built before either was loaded, and the files are gone.
    assert len(built) == 2
    assert setups and all(count == 2 for count in setups)
    assert not glob.glob(str(tmp_path / "*.engine"))
    loaded = torch_tensorrt.load(str(path)).module()
    torch.testing.assert_close(loaded(x), model(x), atol=1e-2, rtol=1e-2)


def test_keywords_go_to_compile_or_save_by_name(monkeypatch):
    calls = []

    @functools.wraps(_compiler.compile)
    def recording_compile(program, **kwargs):
        calls.append(("compile", kwargs))
        return torch.nn.Module()

    @functools.wraps(torch_tensorrt._compile.save)
    def recording_save(module, file_path, **kwargs):
        calls.append(("save", kwargs))

    monkeypatch.setattr(_compiler, "compile", recording_compile)
    monkeypatch.setattr(torch_tensorrt._compile, "save", recording_save)

    _compiler.compile_and_save(
        "program",
        "model.pte",
        output_format="executorch",
        arg_inputs=["x"],
        min_block_size=1,
        external_engine_data="model",
    )
    assert calls == [
        ("compile", {"arg_inputs": ["x"], "min_block_size": 1}),
        ("save", {"output_format": "executorch", "external_engine_data": "model"}),
    ]

    # The other formats trace the compiled module, so their save takes the inputs too.
    calls.clear()
    _compiler.compile_and_save(
        "program", "model.ep", arg_inputs=["x"], min_block_size=1, pickle_protocol=4
    )
    assert calls == [
        ("compile", {"arg_inputs": ["x"], "min_block_size": 1}),
        (
            "save",
            {
                "output_format": "exported_program",
                "pickle_protocol": 4,
                "arg_inputs": ["x"],
            },
        ),
    ]

    with pytest.raises(TypeError, match="min_blok_size"):
        _compiler.compile_and_save("program", "model.ep", min_blok_size=1)


def test_a_fully_supported_graph_with_require_full_compilation(tmp_path, monkeypatch):
    """require_full_compilation's single-engine path returns a bare engine module, which save() doesn't take,
    so compile_and_save partitions instead."""
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    model = torch.nn.Sequential(torch.nn.Linear(64, 64), torch.nn.ReLU()).eval().cuda()
    x = torch.randn(4, 64, device="cuda")
    path = tmp_path / "model.ep"
    torch_tensorrt.dynamo.compile_and_save(
        torch.export.export(model, (x,)),
        path,
        arg_inputs=[x],
        min_block_size=1,
        require_full_compilation=True,
    )
    assert path.stat().st_size > 0
    assert not glob.glob(str(tmp_path / "*.engine"))


def test_engine_files_are_deleted_when_compilation_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    real_convert = _compiler.convert_module
    calls = []

    def failing_second(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("build failed")
        return real_convert(*args, **kwargs)

    monkeypatch.setattr(_compiler, "convert_module", failing_second)
    x = torch.randn(4, 64, device="cuda")
    with pytest.raises(RuntimeError, match="build failed"):
        torch_tensorrt.dynamo.compile_and_save(
            torch.export.export(_TwoEngines().eval().cuda(), (x,)),
            tmp_path / "model.ep",
            arg_inputs=[x],
            min_block_size=1,
            torch_executed_ops={torch.ops.aten.sin.default},
        )
    assert len(calls) == 2
    assert not glob.glob(str(tmp_path / "*.engine"))
