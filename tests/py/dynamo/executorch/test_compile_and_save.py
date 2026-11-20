# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import errno
import glob
import os
import tempfile
import types

import pytest
import torch

pytest.importorskip("executorch.exir")

import executorch.exir._serialize._cord as cord  # noqa: E402
import torch_tensorrt  # noqa: E402
from executorch.exir import ExecutorchProgramManager  # noqa: E402
from executorch.exir._serialize._cord import Cord, FileBackedData  # noqa: E402
from executorch.exir._serialize._named_data_store import NamedDataStore  # noqa: E402
from torch_tensorrt._compile import _takes_consume  # noqa: E402
from torch_tensorrt._utils import _cache_root  # noqa: E402
from torch_tensorrt.dynamo.runtime import TorchTensorRTModule  # noqa: E402

# pytorch/executorch#25099 adds the move with Cord.file_backed_buffers as well as consume, so a
# rename of consume, which turns the move back into a copy, fails these tests instead of skipping them.
_needs_move = pytest.mark.skipif(
    not _takes_consume(ExecutorchProgramManager.save)
    and not hasattr(Cord, "file_backed_buffers"),
    reason="needs an ExecuTorch that moves file-backed data (pytorch/executorch#25099)",
)


def _leave_no_room_for_a_copy(monkeypatch):
    """The save moves the engines only when a copy would not fit."""
    monkeypatch.setattr(
        "torch_tensorrt._compile.shutil.disk_usage",
        lambda path: types.SimpleNamespace(free=0),
    )


class _TwoEngines(torch.nn.Module):
    """Two TensorRT partitions around an op forced to PyTorch."""

    def __init__(self):
        super().__init__()
        self.a, self.b = torch.nn.Linear(64, 64), torch.nn.Linear(64, 8)

    def forward(self, x):
        return self.b(torch.sin(self.a(x)))


@pytest.mark.unit
def test_engines_stream_from_their_files_into_the_program(tmp_path, monkeypatch):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    setups, streamed, owned = [], {}, []
    real_setup = TorchTensorRTModule.setup_engine
    real_add = NamedDataStore.add_named_data

    def counting_setup(self):
        setups.append(self.name)
        real_setup(self)

    def recording_add(self, key, data, *args, **kwargs):
        if isinstance(data, FileBackedData):
            with open(data._path, "rb") as f:
                streamed[data._path] = f.read()
            # Data that owns its file is what a consuming save may move.
            owned.append(data._finalizer is not None)
        else:
            streamed[key] = data
        return real_add(self, key, data, *args, **kwargs)

    monkeypatch.setattr(TorchTensorRTModule, "setup_engine", counting_setup)
    monkeypatch.setattr(NamedDataStore, "add_named_data", recording_add)

    torch.manual_seed(0)
    x = torch.randn(4, 64, device="cuda")
    pte_path = tmp_path / "model.pte"
    torch_tensorrt.compile_and_save(
        torch.export.export(_TwoEngines().eval().cuda(), (x,)),
        pte_path,
        output_format="executorch",
        arg_inputs=[x],
        min_block_size=1,
        torch_executed_ops={torch.ops.aten.sin.default},
    )

    # No engine was ever set up, and each one reached the program as its own file, from the cache directory.
    assert setups == []
    assert len(streamed) == 2
    assert owned and all(owned)
    offloaded = str(_cache_root() / "offloaded_engines")
    assert all(
        path.endswith(".engine") and os.path.dirname(path) == offloaded
        for path in streamed
    )
    pte = pte_path.read_bytes()
    assert all(pte.find(engine) >= 0 for engine in streamed.values())
    # The files are gone once the program is saved.
    assert not glob.glob(str(tmp_path / "**" / "*.engine"), recursive=True)


@_needs_move
@pytest.mark.unit
@pytest.mark.parametrize("external_engine_data", [None, "engines"])
def test_engines_move_into_the_program_without_a_second_copy(
    tmp_path, monkeypatch, external_engine_data
):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    _leave_no_room_for_a_copy(monkeypatch)
    engines, copied = {}, []
    real_add = NamedDataStore.add_named_data
    real_write = FileBackedData.write_to_file

    def recording_add(self, key, data, *args, **kwargs):
        if isinstance(data, FileBackedData):
            with open(data._path, "rb") as f:
                engines[data._path] = f.read()
        return real_add(self, key, data, *args, **kwargs)

    def recording_write(self, outfile):
        copied.append(self._path)
        real_write(self, outfile)

    monkeypatch.setattr(NamedDataStore, "add_named_data", recording_add)
    monkeypatch.setattr(FileBackedData, "write_to_file", recording_write)
    torch.manual_seed(0)
    x = torch.randn(4, 64, device="cuda")
    pte_path = tmp_path / "model.pte"
    torch_tensorrt.compile_and_save(
        torch.export.export(_TwoEngines().eval().cuda(), (x,)),
        pte_path,
        output_format="executorch",
        arg_inputs=[x],
        min_block_size=1,
        torch_executed_ops={torch.ops.aten.sin.default},
        save_options={"external_engine_data": external_engine_data},
    )
    # ExecuTorch copies data it can't move through write_to_file, so no call there means each
    # engine was moved.
    assert len(engines) == 2 and copied == []
    out = tmp_path / ("model.pte" if external_engine_data is None else "engines.ptd")
    data = out.read_bytes()
    assert all(data.find(engine) >= 0 for engine in engines.values())


@_needs_move
@pytest.mark.unit
def test_an_etrecord_makes_the_save_copy_the_engines(tmp_path, monkeypatch):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    _leave_no_room_for_a_copy(monkeypatch)
    left = []
    real_write = FileBackedData.write_to_file

    def recording_write(self, outfile):
        real_write(self, outfile)
        left.append(os.path.getsize(self._path))

    monkeypatch.setattr(FileBackedData, "write_to_file", recording_write)
    torch.manual_seed(0)
    x = torch.randn(4, 64, device="cuda")
    pte_path = tmp_path / "model.pte"
    torch_tensorrt.compile_and_save(
        torch.export.export(_TwoEngines().eval().cuda(), (x,)),
        pte_path,
        output_format="executorch",
        arg_inputs=[x],
        min_block_size=1,
        torch_executed_ops={torch.ops.aten.sin.default},
        save_options={"generate_etrecord": True},
    )
    assert pte_path.stat().st_size > 0
    assert (tmp_path / "model_etrecord.bin").stat().st_size > 0
    # The ETRecord reads the engines after the program is written, so they stayed in their files.
    assert left and all(size > 0 for size in left)


@_needs_move
@pytest.mark.unit
def test_a_failed_move_leaves_no_partial_output(tmp_path, monkeypatch):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    _leave_no_room_for_a_copy(monkeypatch)
    real_sync, syncs = cord._sync, []

    def full_on_second(fd):
        # The first engine is moved; the disk fills before the second one.
        syncs.append(fd)
        if len(syncs) == 2:
            raise OSError(errno.ENOSPC, "No space left on device")
        real_sync(fd)

    monkeypatch.setattr(cord, "_sync", full_on_second)
    torch.manual_seed(0)
    x = torch.randn(4, 64, device="cuda")
    pte_path = tmp_path / "model.pte"
    with pytest.raises(OSError, match="No space left"):
        torch_tensorrt.compile_and_save(
            torch.export.export(_TwoEngines().eval().cuda(), (x,)),
            pte_path,
            output_format="executorch",
            arg_inputs=[x],
            min_block_size=1,
            torch_executed_ops={torch.ops.aten.sin.default},
        )
    assert not glob.glob(str(tmp_path / "*.partial")) and not pte_path.exists()
