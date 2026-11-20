# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import glob
import logging
import os
import platform
import re
import stat
import tempfile

import pytest
import torch

pytest.importorskip("executorch.exir")

import torch_tensorrt  # noqa: E402
from executorch.exir._serialize._cord import FileBackedData  # noqa: E402
from executorch.exir._serialize._named_data_store import NamedDataStore  # noqa: E402
from torch_tensorrt import _compile  # noqa: E402
from torch_tensorrt.dynamo.runtime import TorchTensorRTModule  # noqa: E402
from torch_tensorrt.dynamo.runtime._offloaded_engines import (  # noqa: E402
    engines_go_to_files,
    offload_engines_for_save,
)
from torch_tensorrt.dynamo.runtime._TorchTensorRTModule import (  # noqa: E402
    REQUIRES_OUTPUT_ALLOCATOR_IDX,
)
from torch_tensorrt.executorch import _export_utils  # noqa: E402

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
def test_engine_with_data_dependent_shape_only_inside_exports_a_pte(
    tmp_path, monkeypatch
):
    """compile_and_save keeps each engine in its file, so it loads a marked one to ask
    TensorRT whether the engine needs an output allocator."""

    class MaskedWrite(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("mask", torch.tensor([False, True, True, False]))

        def forward(self, value, extra):
            value = value.clone()
            value[self.mask] = value[self.mask] + extra
            return value * 2

    class MaskedRows(torch.nn.Module):
        def forward(self, value, extra):
            return value[value[:, 0] > 0] + extra[0]

    class FirstTwoMaskedRows(torch.nn.Module):
        def forward(self, value, extra):
            rows = torch.nonzero(value[:, 0] != 0)
            torch._check(rows.shape[0] >= 2)
            return value[rows[:2, 0]] + extra[0]

    class MaskedWriteAfterNonzero(MaskedWrite):
        def forward(self, value, extra):
            rows = torch.nonzero(value[:, 0] > 0)
            return super().forward(value + value[rows[:, 0]].sum(), extra)

    validated, staged_code_current = [], []
    real_validate = _export_utils.validate_engine_program
    real_replace = _export_utils.replace_execute_engine

    def recording_validate(program, resolved=None):
        validated.append((program, resolved))
        return real_validate(program, resolved)

    def recording_replace(program, resolved=None):
        program = real_replace(program, resolved)
        module = program.graph_module
        staged_code_current.append(
            module.code == module.graph.python_code(root_module="self").src
        )
        return program

    monkeypatch.setattr(_export_utils, "validate_engine_program", recording_validate)
    monkeypatch.setattr(_export_utils, "replace_execute_engine", recording_replace)

    inputs = [torch.randn(4, 8, device="cuda"), torch.randn(2, 8, device="cuda")]

    def save(model, **options):
        output = tmp_path / f"{type(model).__name__}.pte"
        torch_tensorrt.compile_and_save(
            torch.export.export(model.eval().cuda(), tuple(inputs)),
            output,
            output_format="executorch",
            arg_inputs=inputs,
            min_block_size=1,
            pass_through_build_failures=True,
            **options,
        )
        return output

    assert save(MaskedWrite()).stat().st_size > 0
    # The engine was marked, and validation cleared the mark in the record the staged
    # copy takes, not in the program it was given.
    program, resolved = validated[0]
    no_op = torch.ops.tensorrt.no_op_placeholder_for_execute_engine.default
    (placeholder,) = [node for node in program.graph.nodes if node.target is no_op]
    assert placeholder.args[1 + REQUIRES_OUTPUT_ALLOCATOR_IDX] == "1"
    assert resolved[placeholder.name][REQUIRES_OUTPUT_ALLOCATOR_IDX] == "0"
    assert staged_code_current == [True]
    refusal = re.escape(
        "ExecuTorch export does not support TensorRT engines that require an output "
        "allocator (data-dependent output shapes) for node"
    )
    # An output shape that depends on the data still needs an output allocator.
    with pytest.raises(RuntimeError, match=refusal):
        save(MaskedRows())
    # So does one whose length export proved but TensorRT computes at run time.
    with pytest.raises(RuntimeError, match=refusal):
        save(FirstTwoMaskedRows())
    # So does an engine with a data-dependent input shape, which the live engine check
    # cannot answer either: nonzero runs in PyTorch and feeds the engine.
    with pytest.raises(RuntimeError, match=refusal):
        save(
            MaskedWriteAfterNonzero(),
            torch_executed_ops={torch.ops.aten.nonzero.default},
        )


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
def test_engine_offload_dir_warns_where_it_does_nothing(tmp_path, monkeypatch, caplog):
    caplog.set_level(logging.WARNING, logger="torch_tensorrt._compile")
    try:
        torch_tensorrt.save(
            torch.nn.Linear(1, 1),
            str(tmp_path / "model.ep"),
            output_format="exported_program",
            engine_offload_dir=tmp_path,
        )
    except Exception:
        pass  # a Linear is not a compiled program; only the warning matters here
    in_files = []
    monkeypatch.setattr(
        _compile,
        "_lower_and_write_executorch",
        lambda *args, **kwargs: in_files.append(engines_go_to_files()),
    )
    monkeypatch.setattr(_compile, "_release_lowering_caches", lambda program: None)
    monkeypatch.setattr(_compile.platform, "system", lambda: "Windows")
    _compile._save_as_executorch(
        None, str(tmp_path / "model.pte"), engine_offload_dir=tmp_path
    )

    assert in_files == [False]
    messages = [record.getMessage() for record in caplog.records]
    assert any(
        message.startswith("engine_offload_dir= is only used with")
        for message in messages
    )
    assert any("engine_offload_dir= is ignored on Windows" in m for m in messages)


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
