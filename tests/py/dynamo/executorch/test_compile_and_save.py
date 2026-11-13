# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import glob
import os
import re
import tempfile

import pytest
import torch

pytest.importorskip("executorch.exir")

import torch_tensorrt  # noqa: E402
from executorch.exir._serialize._cord import FileBackedData  # noqa: E402
from executorch.exir._serialize._named_data_store import NamedDataStore  # noqa: E402
from torch_tensorrt._utils import _cache_root  # noqa: E402
from torch_tensorrt.dynamo.runtime import TorchTensorRTModule  # noqa: E402
from torch_tensorrt.dynamo.runtime._TorchTensorRTModule import (  # noqa: E402
    REQUIRES_OUTPUT_ALLOCATOR_IDX,
)
from torch_tensorrt.executorch import _export_utils  # noqa: E402


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
    setups, streamed = [], {}
    real_setup = TorchTensorRTModule.setup_engine
    real_add = NamedDataStore.add_named_data

    def counting_setup(self):
        setups.append(self.name)
        real_setup(self)

    def recording_add(self, key, data, *args, **kwargs):
        if isinstance(data, FileBackedData):
            with open(data._path, "rb") as f:
                streamed[data._path] = f.read()
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
    offloaded = str(_cache_root() / "offloaded_engines")
    assert all(
        path.endswith(".engine") and os.path.dirname(path) == offloaded
        for path in streamed
    )
    pte = pte_path.read_bytes()
    assert all(pte.find(engine) >= 0 for engine in streamed.values())
    # The files are gone once the program is saved.
    assert not glob.glob(str(tmp_path / "**" / "*.engine"), recursive=True)


@pytest.mark.unit
def test_engine_with_data_dependent_shape_only_inside_exports_a_pte(
    tmp_path, monkeypatch
):
    """compile_and_save keeps each engine in its file, so the shapes export traced
    decide whether the engine needs an output allocator."""

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
    # So does an engine with a data-dependent input shape, which the live engine check
    # cannot answer either: nonzero runs in PyTorch and feeds the engine.
    with pytest.raises(RuntimeError, match=refusal):
        save(
            MaskedWriteAfterNonzero(),
            torch_executed_ops={torch.ops.aten.nonzero.default},
        )
