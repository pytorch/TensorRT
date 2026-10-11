# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""CPU-only checks for writing TensorRT engines to an ExecuTorch data file (.ptd)."""

import importlib
import inspect
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

pytest.importorskip("executorch.exir")

import torch  # noqa: E402
import torch_tensorrt  # noqa: E402
from executorch.exir.backend.compile_spec_schema import CompileSpec  # noqa: E402
from torch_tensorrt.executorch.backend import (  # noqa: E402
    EXTERNAL_ENGINE_DATA_COMPILE_SPEC_KEY,
)

_KEY = EXTERNAL_ENGINE_DATA_COMPILE_SPEC_KEY


@pytest.fixture
def program():
    return torch.export.export(torch.nn.Identity(), (torch.ones(1),))


@pytest.fixture
def lowering(monkeypatch):
    import executorch.exir
    import torch_tensorrt.executorch._export_utils as export_utils

    export_module = importlib.import_module("torch_tensorrt.executorch._export")
    monkeypatch.setattr(export_module.platform, "system", lambda: "Linux")
    lower = MagicMock(return_value=SimpleNamespace(methods=()))
    monkeypatch.setattr(executorch.exir, "to_edge_transform_and_lower", lower)
    monkeypatch.setattr(
        export_utils, "validate_engine_program", lambda program, resolved: 1
    )
    monkeypatch.setattr(export_utils, "stage_exported_program", lambda program: program)
    monkeypatch.setattr(
        export_utils, "replace_execute_engine", lambda program, resolved: program
    )
    return lower


def _trt_specs(lowering):
    chains = lowering.call_args.kwargs["partitioner"]
    if not isinstance(chains, dict):
        chains = {"forward": chains}
    return {name: chain[0].compile_specs for name, chain in chains.items()}


@pytest.mark.unit
def test_external_engine_data_is_a_keyword_on_both_entry_points():
    parameter = inspect.signature(torch_tensorrt.executorch.export).parameters[
        "external_engine_data"
    ]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is None
    from torch_tensorrt._compile import _EXECUTORCH_SAVE_OPTIONS

    assert _EXECUTORCH_SAVE_OPTIONS["external_engine_data"] is None


@pytest.mark.unit
def test_export_points_every_method_at_the_data_file(program, lowering):
    source = {
        "forward": program,
        "decode": torch.export.export(torch.nn.Identity(), (torch.ones(1),)),
    }
    torch_tensorrt.executorch.export(source, external_engine_data="engines")
    specs = _trt_specs(lowering)
    assert set(specs) == {"forward", "decode"}
    for method_specs in specs.values():
        assert [s.value for s in method_specs if s.key == _KEY] == [b"engines"]


@pytest.mark.unit
def test_export_leaves_engines_embedded_by_default(program, lowering):
    torch_tensorrt.executorch.export(program)
    assert not any(s.key == _KEY for s in _trt_specs(lowering)["forward"])


@pytest.mark.unit
@pytest.mark.parametrize(
    "value",
    ["", ".", "..", "a/b", "../engines", "a\\b", "a\0b", "aoti_cuda_blob.ptd", 1, b"x"],
)
def test_export_rejects_a_name_that_is_not_a_plain_file_name(program, lowering, value):
    with pytest.raises(ValueError, match="external_engine_data must be"):
        torch_tensorrt.executorch.export(program, external_engine_data=value)
    lowering.assert_not_called()


@pytest.mark.unit
@pytest.mark.parametrize("value", [None, "engines"])
def test_export_refuses_the_raw_compile_spec(program, lowering, value):
    with pytest.raises(ValueError, match="Pass external_engine_data= instead"):
        torch_tensorrt.executorch.export(
            program,
            compile_specs=[CompileSpec(_KEY, b"engines")],
            external_engine_data=value,
        )
    lowering.assert_not_called()


@pytest.mark.unit
def test_save_forwards_the_name_to_export(monkeypatch, tmp_path):
    import torch_tensorrt._compile as compile_module

    seen = {}

    def fake_save_as_executorch(module, file_path, **kwargs):
        seen.update(kwargs)

    monkeypatch.setattr(compile_module, "_save_as_executorch", fake_save_as_executorch)
    monkeypatch.setattr(compile_module, "_has_executorch_exir", lambda: True)
    program = torch.export.export(torch.nn.Identity(), (torch.ones(1),))
    torch_tensorrt.save(
        program,
        str(tmp_path / "model.pte"),
        output_format="executorch",
        external_engine_data="engines",
    )
    assert seen["external_engine_data"] == "engines"
