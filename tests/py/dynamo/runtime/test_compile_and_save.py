# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import glob
import platform
import tempfile

import pytest
import torch
import torch_tensorrt
from torch_tensorrt._utils import _cache_root
from torch_tensorrt.dynamo import _compiler
from torch_tensorrt.dynamo.runtime import TorchTensorRTModule

pytestmark = pytest.mark.skipif(
    platform.system() == "Windows",
    reason="compile_and_save raises NotImplementedError on Windows",
)


class _TwoEngines(torch.nn.Module):
    """Two TensorRT partitions around an op forced to PyTorch."""

    def __init__(self):
        super().__init__()
        self.a, self.b = torch.nn.Linear(64, 64), torch.nn.Linear(64, 8)

    def forward(self, x):
        return self.b(torch.sin(self.a(x)))


def _engine_files(root):
    return glob.glob(str(root / "**" / "*.engine"), recursive=True)


@pytest.mark.unit
def test_exported_program_loads_the_engines_once_all_are_built(tmp_path, monkeypatch):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    offloaded = _cache_root() / "offloaded_engines"
    built, setups = [], []
    real_convert = _compiler.convert_module
    real_setup = TorchTensorRTModule.setup_engine

    def recording_convert(*args, **kwargs):
        built.append(kwargs.get("name"))
        return real_convert(*args, **kwargs)

    def recording_setup(self):
        setups.append((len(built), len(glob.glob(str(offloaded / "*.engine")))))
        real_setup(self)

    monkeypatch.setattr(_compiler, "convert_module", recording_convert)
    monkeypatch.setattr(TorchTensorRTModule, "setup_engine", recording_setup)

    torch.manual_seed(0)
    model = _TwoEngines().eval().cuda()
    x = torch.randn(4, 64, device="cuda")
    path = tmp_path / "model.ep"
    torch_tensorrt.compile_and_save(
        torch.export.export(model, (x,)),
        path,
        arg_inputs=[x],
        min_block_size=1,
        torch_executed_ops={torch.ops.aten.sin.default},
    )

    # Both engines were built, into the cache directory, before either was loaded, and the files are gone.
    assert len(built) == 2
    assert setups and all(entry == (2, 2) for entry in setups)
    assert not _engine_files(tmp_path)
    loaded = torch_tensorrt.load(str(path)).module()
    torch.testing.assert_close(loaded(x), model(x), atol=1e-2, rtol=1e-2)


def test_arguments_are_packed_for_compile_and_save(monkeypatch):
    calls = []
    monkeypatch.setattr(
        torch_tensorrt._compile,
        "dynamo_compile",
        lambda program, **kwargs: calls.append(("compile", kwargs))
        or torch.nn.Module(),
    )
    monkeypatch.setattr(
        torch_tensorrt._compile,
        "save",
        lambda module, path, **kwargs: calls.append(("save", kwargs)),
    )

    torch_tensorrt.compile_and_save(
        "program",
        "model.pte",
        output_format="executorch",
        arg_inputs=["x"],
        save_options={"external_engine_data": "model"},
        min_block_size=1,
    )
    assert calls == [
        ("compile", {"arg_inputs": ["x"], "kwarg_inputs": None, "min_block_size": 1}),
        ("save", {"output_format": "executorch", "external_engine_data": "model"}),
    ]

    # The other formats re-export the compiled module, so their save gets the inputs too.
    calls.clear()
    torch_tensorrt.compile_and_save(
        "program", "model.ep", arg_inputs=["x"], save_options={"pickle_protocol": 4}
    )
    assert calls == [
        ("compile", {"arg_inputs": ["x"], "kwarg_inputs": None}),
        (
            "save",
            {
                "output_format": "exported_program",
                "pickle_protocol": 4,
                "arg_inputs": ["x"],
                "kwarg_inputs": None,
            },
        ),
    ]

    with pytest.raises(ValueError, match="torchscript"):
        torch_tensorrt.compile_and_save(
            "program", "model.ts", output_format="torchscript", arg_inputs=["x"]
        )


def test_a_fully_supported_graph_with_require_full_compilation(tmp_path, monkeypatch):
    """require_full_compilation's single-engine path returns a bare engine module, which save() doesn't take,
    so compile_and_save partitions instead."""
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    model = torch.nn.Sequential(torch.nn.Linear(64, 64), torch.nn.ReLU()).eval().cuda()
    x = torch.randn(4, 64, device="cuda")
    path = tmp_path / "model.ep"
    torch_tensorrt.compile_and_save(
        torch.export.export(model, (x,)),
        path,
        arg_inputs=[x],
        min_block_size=1,
        require_full_compilation=True,
    )
    assert path.stat().st_size > 0
    assert not _engine_files(tmp_path)


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
        torch_tensorrt.compile_and_save(
            torch.export.export(_TwoEngines().eval().cuda(), (x,)),
            tmp_path / "model.ep",
            arg_inputs=[x],
            min_block_size=1,
            torch_executed_ops={torch.ops.aten.sin.default},
        )
    assert len(calls) == 2
    assert not _engine_files(tmp_path)
