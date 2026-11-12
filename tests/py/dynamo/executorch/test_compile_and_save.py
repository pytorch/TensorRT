# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import glob
import os
import platform
import stat
import tempfile

import pytest
import torch

pytest.importorskip("executorch.exir")

import torch_tensorrt  # noqa: E402
from executorch.exir._serialize._cord import FileBackedData  # noqa: E402
from executorch.exir._serialize._named_data_store import NamedDataStore  # noqa: E402
from torch_tensorrt.dynamo.runtime import TorchTensorRTModule  # noqa: E402
from torch_tensorrt.dynamo.runtime._offloaded_engines import (  # noqa: E402
    offload_engines_for_save,
)

pytestmark = pytest.mark.skipif(
    platform.system() == "Windows",
    reason="save() and compile_and_save keep engines in files only on Linux",
)


class _TwoEngines(torch.nn.Module):
    """Two TensorRT partitions around an op forced to PyTorch."""

    def __init__(self):
        super().__init__()
        self.a, self.b = torch.nn.Linear(64, 64), torch.nn.Linear(64, 8)

    def forward(self, x):
        return self.b(torch.sin(self.a(x)))


def _assert_in_one_private_directory(paths, root, modes):
    """The engine files were in one directory under ``root`` that only its owner could read, now removed."""
    directories = {os.path.dirname(path) for path in paths}
    assert len(directories) == 1 and all(path.endswith(".engine") for path in paths)
    directory = directories.pop()
    assert os.path.dirname(directory) == str(root)
    assert os.path.basename(directory).startswith("torch_tensorrt_engines_")
    assert modes == {0o700}
    assert not os.path.exists(directory)


def _engine_files(root):
    return glob.glob(str(root / "**" / "*.engine"), recursive=True)


def _two_engine_export():
    torch.manual_seed(0)
    x = torch.randn(4, 64, device="cuda")
    return torch.export.export(_TwoEngines().eval().cuda(), (x,)), {
        "arg_inputs": [x],
        "min_block_size": 1,
        "torch_executed_ops": {torch.ops.aten.sin.default},
    }


def _two_engine_module():
    exported, settings = _two_engine_export()
    return torch_tensorrt.dynamo.compile(exported, **settings)


def _record_streamed_engines(monkeypatch):
    """Record the content of each engine the program gets from a file, the modes of their directories, and the
    keys of the engines it gets as bytes."""
    streamed, modes, copied = {}, set(), set()
    real_add = NamedDataStore.add_named_data

    def recording_add(self, key, data, *args, **kwargs):
        if isinstance(data, FileBackedData):
            with open(data._path, "rb") as f:
                streamed[data._path] = f.read()
            modes.add(stat.S_IMODE(os.stat(os.path.dirname(data._path)).st_mode))
        elif isinstance(data, bytes):
            copied.add(key)
        return real_add(self, key, data, *args, **kwargs)

    monkeypatch.setattr(NamedDataStore, "add_named_data", recording_add)
    return streamed, modes, copied


@pytest.mark.unit
@pytest.mark.parametrize("folder_in_save_options", [False, True])
def test_engines_stream_from_their_files_into_the_program(
    tmp_path, monkeypatch, folder_in_save_options
):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    root = tmp_path / "engines" if folder_in_save_options else tmp_path
    save_options = {"engine_offload_dir": root} if folder_in_save_options else None
    setups = []
    real_setup = TorchTensorRTModule.setup_engine

    def counting_setup(self):
        setups.append(self.name)
        real_setup(self)

    monkeypatch.setattr(TorchTensorRTModule, "setup_engine", counting_setup)
    streamed, modes, _ = _record_streamed_engines(monkeypatch)

    exported, settings = _two_engine_export()
    pte_path = tmp_path / "model.pte"
    torch_tensorrt.compile_and_save(
        exported,
        pte_path,
        output_format="executorch",
        save_options=save_options,
        **settings,
    )

    # No engine was ever set up, and each one reached the program as its own file.
    assert setups == []
    assert len(streamed) == 2
    _assert_in_one_private_directory(streamed, root, modes)
    pte = pte_path.read_bytes()
    assert all(pte.find(engine) >= 0 for engine in streamed.values())
    assert not _engine_files(tmp_path)


@pytest.mark.unit
def test_save_streams_engines_from_files_in_engine_offload_dir(tmp_path, monkeypatch):
    trt_gm = _two_engine_module()
    monkeypatch.setenv("HOME", str(tmp_path))
    engine_dir = tmp_path / "engines"
    engine_dir.mkdir()
    (engine_dir / "notes.txt").write_text("notes")
    (engine_dir / "other.engine").write_text("other")
    streamed, modes, _ = _record_streamed_engines(monkeypatch)

    pte_path = tmp_path / "model.pte"
    torch_tensorrt.save(
        trt_gm,
        str(pte_path),
        output_format="executorch",
        engine_offload_dir="~/engines",
    )

    # Both engines streamed from one private directory in the folder ("~" expanded), now removed.
    assert len(streamed) == 2
    _assert_in_one_private_directory(streamed, engine_dir.resolve(), modes)
    pte = pte_path.read_bytes()
    assert all(pte.find(engine) >= 0 for engine in streamed.values())
    # The folder and its other files stay.
    assert sorted(os.listdir(engine_dir)) == ["notes.txt", "other.engine"]
    assert (engine_dir / "notes.txt").read_text() == "notes"
    assert (engine_dir / "other.engine").read_text() == "other"


@pytest.mark.unit
def test_a_save_without_engine_offload_dir_copies_engines_in_memory(
    tmp_path, monkeypatch
):
    trt_gm = _two_engine_module()
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    streamed, _, copied = _record_streamed_engines(monkeypatch)
    torch_tensorrt.save(trt_gm, str(tmp_path / "model.pte"), output_format="executorch")
    assert streamed == {} and len(copied) == 2


@pytest.mark.unit
def test_engines_compiled_inside_a_save_are_set_up(tmp_path, monkeypatch):
    """Inside save() only the save writes engines to files. A compilation there sets up its
    engines as usual, unlike in compile_and_save."""
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    with offload_engines_for_save(str(tmp_path)):
        trt_gm = _two_engine_module()
        written = _engine_files(tmp_path)
    engines = [
        module.engine
        for module in trt_gm.modules()
        if isinstance(module, TorchTensorRTModule)
    ]
    assert len(engines) == 2 and all(engine is not None for engine in engines)
    assert written == []
