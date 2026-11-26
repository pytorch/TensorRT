# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import glob
import logging
import os
import platform
import re
import stat
import tempfile
import weakref
from unittest.mock import MagicMock

import pytest
import torch

pytest.importorskip("executorch.exir")

import torch_tensorrt  # noqa: E402
from executorch.exir._serialize._cord import FileBackedData  # noqa: E402
from executorch.exir._serialize._named_data_store import NamedDataStore  # noqa: E402
from torch_tensorrt import _compile  # noqa: E402
from torch_tensorrt.dynamo.runtime import TorchTensorRTModule, _TRTEngine  # noqa: E402
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


def _record_engine_loads(monkeypatch):
    """Record a weak reference to each engine loaded from its serialized record."""
    loads = []
    real_load = _TRTEngine._reconstruct_trt_engine

    def recording_load(serialized_info):
        engine = real_load(serialized_info)
        loads.append(weakref.ref(engine))
        return engine

    monkeypatch.setattr(_TRTEngine, "_reconstruct_trt_engine", recording_load)
    return loads


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
    streamed, modes, copied = _record_streamed_engines(monkeypatch)
    loads = _record_engine_loads(monkeypatch)

    exported, settings = _two_engine_export()
    pte_path = tmp_path / "model.pte"
    torch_tensorrt.compile_and_save(
        exported,
        pte_path,
        output_format="executorch",
        save_options=save_options,
        **settings,
    )

    # No engine was ever set up or loaded, and each one reached the program as its own file.
    assert setups == [] and loads == []
    assert len(streamed) == 2 and not copied
    _assert_in_one_private_directory(streamed, root, modes)
    pte = pte_path.read_bytes()
    assert all(pte.find(engine) >= 0 for engine in streamed.values())
    assert not _engine_files(tmp_path)


class _MaskedWrite(torch.nn.Module):
    """An engine marked for an output allocator whose output shapes are all static."""

    def __init__(self):
        super().__init__()
        self.register_buffer("mask", torch.tensor([False, True, True, False]))

    def forward(self, value, extra):
        value = value.clone()
        value[self.mask] = value[self.mask] + extra
        return value * 2


def _masked_write_inputs():
    torch.manual_seed(0)
    return [torch.randn(4, 8, device="cuda"), torch.randn(2, 8, device="cuda")]


@pytest.mark.unit
def test_engine_with_data_dependent_shape_only_inside_exports_a_pte(
    tmp_path, monkeypatch
):
    """compile_and_save keeps each engine in its file, so it loads a marked one to ask
    TensorRT whether the engine needs an output allocator."""

    class MaskedRows(torch.nn.Module):
        def forward(self, value, extra):
            return value[value[:, 0] > 0] + extra[0]

    class FirstTwoMaskedRows(torch.nn.Module):
        def forward(self, value, extra):
            rows = torch.nonzero(value[:, 0] != 0)
            torch._check(rows.shape[0] >= 2)
            return value[rows[:2, 0]] + extra[0]

    class MaskedWriteAfterNonzero(_MaskedWrite):
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
    loads = _record_engine_loads(monkeypatch)

    inputs = _masked_write_inputs()

    def save(model, name=None, dynamic_shapes=None, **options):
        output = tmp_path / f"{name or type(model).__name__}.pte"
        options.setdefault("arg_inputs", inputs)
        torch_tensorrt.compile_and_save(
            torch.export.export(
                model.eval().cuda(), tuple(inputs), dynamic_shapes=dynamic_shapes
            ),
            output,
            output_format="executorch",
            min_block_size=1,
            pass_through_build_failures=True,
            **options,
        )
        return output

    assert save(_MaskedWrite()).stat().st_size > 0
    assert len(loads) == 1 and loads[0]() is None
    # The engine was marked, and validation cleared the mark in the record the staged
    # copy takes, not in the program it was given.
    program, resolved = validated[0]
    no_op = torch.ops.tensorrt.no_op_placeholder_for_execute_engine.default
    (placeholder,) = [node for node in program.graph.nodes if node.target is no_op]
    assert placeholder.args[1 + REQUIRES_OUTPUT_ALLOCATOR_IDX] == "1"
    assert resolved[placeholder.name][REQUIRES_OUTPUT_ALLOCATOR_IDX] == "0"
    assert staged_code_current == [True]
    # The output shapes follow a dynamic input dimension, and are still known.
    columns = torch.export.Dim("columns", min=2, max=16)
    dynamic = save(
        _MaskedWrite(),
        name="dynamic",
        dynamic_shapes=({1: columns}, {1: columns}),
        arg_inputs=[
            torch_tensorrt.Input(
                min_shape=(rows, 2), opt_shape=(rows, 8), max_shape=(rows, 16)
            )
            for rows in (4, 2)
        ],
    )
    assert dynamic.stat().st_size > 0
    assert len(loads) == 2 and loads[1]() is None
    refusal = re.escape(
        "ExecuTorch export does not support TensorRT engines that require an output "
        "allocator (data-dependent output shapes) for node"
    )
    # An output shape that depends on the data still needs an output allocator.
    with pytest.raises(RuntimeError, match=refusal) as refused:
        save(MaskedRows())
    # The C++ engine raises on the unknown dimension, and the refusal keeps that error as its
    # cause; the Python engine reports the dimension as -1. Neither keeps the loaded engine.
    if torch_tensorrt.ENABLED_FEATURES.torch_tensorrt_runtime:
        assert refused.value.__cause__ is not None
    assert loads[-1]() is None
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
def test_a_masked_write_program_runs_like_the_eager_model(tmp_path):
    pytest.importorskip("torch_tensorrt_executorch_runtime")
    from executorch.runtime import Runtime

    model, inputs = _MaskedWrite().eval().cuda(), _masked_write_inputs()
    pte = tmp_path / "model.pte"
    torch_tensorrt.compile_and_save(
        torch.export.export(model, tuple(inputs)),
        pte,
        output_format="executorch",
        arg_inputs=inputs,
        min_block_size=1,
        pass_through_build_failures=True,
    )
    # The program plans its inputs in host memory, so the caller passes host tensors.
    method = Runtime.get().load_program(pte).load_method("forward")
    (actual,) = method.execute([tensor.cpu() for tensor in inputs])
    torch.testing.assert_close(actual.cpu(), model(*inputs).cpu())


@pytest.mark.unit
@pytest.mark.parametrize("external_engine_data", [None, "engines"])
def test_save_streams_engines_from_files_in_engine_offload_dir(
    tmp_path, monkeypatch, external_engine_data
):
    trt_gm = _two_engine_module()
    monkeypatch.setenv("HOME", str(tmp_path))
    engine_dir = tmp_path / "engines"
    engine_dir.mkdir()
    (engine_dir / "notes.txt").write_text("notes")
    (engine_dir / "other.engine").write_text("other")
    streamed, modes, copied = _record_streamed_engines(monkeypatch)

    pte_path = tmp_path / "model.pte"
    torch_tensorrt.save(
        trt_gm,
        str(pte_path),
        output_format="executorch",
        engine_offload_dir="~/engines",
        external_engine_data=external_engine_data,
    )

    # Both engines streamed from one private directory in the folder ("~" expanded), now removed.
    assert len(streamed) == 2 and not copied
    _assert_in_one_private_directory(streamed, engine_dir.resolve(), modes)
    data_path = (
        tmp_path / f"{external_engine_data}.ptd" if external_engine_data else pte_path
    )
    data = data_path.read_bytes()
    assert all(data.find(engine) >= 0 for engine in streamed.values())
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
@pytest.mark.parametrize("path", ["exported_program", "fx_legacy", "fx_retrace"])
def test_save_forwards_engine_offload_dir_on_every_path(tmp_path, monkeypatch, path):
    # save() reaches the ExecuTorch save by a separate branch for each source shape and retrace value.
    import torch_tensorrt.dynamo._exporter as exporter

    program = torch.export.export(torch.nn.Identity(), (torch.ones(1),))
    monkeypatch.setattr(_compile.platform, "system", lambda: "Linux")
    monkeypatch.setattr(_compile, "_has_executorch_exir", lambda: True)
    monkeypatch.setattr(exporter, "export", MagicMock(return_value=program))
    monkeypatch.setattr(torch.export, "export", MagicMock(return_value=program))
    save_as_executorch = MagicMock()
    monkeypatch.setattr(_compile, "_save_as_executorch", save_as_executorch)
    torch_tensorrt.save(
        program if path == "exported_program" else program.module(),
        str(tmp_path / "model.pte"),
        output_format="executorch",
        arg_inputs=[torch.ones(1)] if path == "fx_retrace" else None,
        retrace=path != "fx_legacy",
        engine_offload_dir=tmp_path,
    )
    save_as_executorch.assert_called_once()
    assert save_as_executorch.call_args.kwargs["engine_offload_dir"] == tmp_path


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
