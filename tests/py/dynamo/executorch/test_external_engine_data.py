# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""CPU-only checks for writing TensorRT engines to an ExecuTorch data file (.ptd)."""

import importlib
import inspect
import logging
from types import SimpleNamespace
from unittest.mock import MagicMock, create_autospec

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
    [
        "",
        ".",
        "..",
        "a/b",
        "../engines",
        "/abs",
        "a\\b",
        "a\0b",
        "D:engines",
        "aoti_cuda_blob.ptd",
    ],
)
def test_export_rejects_a_name_that_is_not_a_plain_file_name(program, lowering, value):
    with pytest.raises(ValueError, match="external_engine_data must be"):
        torch_tensorrt.executorch.export(program, external_engine_data=value)
    lowering.assert_not_called()


@pytest.mark.unit
def test_the_name_the_docstring_gives_for_sharing_with_cuda_is_accepted(
    program, lowering
):
    assert "``aoti_cuda_blob``" in inspect.getdoc(torch_tensorrt.executorch.export)
    torch_tensorrt.executorch.export(program, external_engine_data="aoti_cuda_blob")
    specs = _trt_specs(lowering)["forward"]
    assert [s.value for s in specs if s.key == _KEY] == [b"aoti_cuda_blob"]


@pytest.mark.unit
@pytest.mark.parametrize("value", [1, b"x"])
def test_export_rejects_a_name_that_is_not_a_str(program, lowering, value):
    with pytest.raises(TypeError, match="external_engine_data must be a str or None"):
        torch_tensorrt.executorch.export(program, external_engine_data=value)
    lowering.assert_not_called()


@pytest.mark.unit
@pytest.mark.parametrize("value", ["../outside", "/abs", "x.ptd"])
def test_the_partitioner_rejects_a_name_that_is_not_a_plain_file_name(value):
    from torch_tensorrt.executorch.partitioner import TensorRTPartitioner

    with pytest.raises(ValueError, match="external_engine_data must be"):
        TensorRTPartitioner([CompileSpec(_KEY, value.encode())])


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
@pytest.mark.parametrize("path", ["exported_program", "fx_legacy", "fx_retrace"])
def test_save_forwards_the_name_to_export(monkeypatch, tmp_path, program, path):
    # save() reaches export() by a separate branch for each source shape and retrace value.
    import torch_tensorrt._compile as compile_module
    import torch_tensorrt.dynamo._exporter as exporter

    monkeypatch.setattr(compile_module.platform, "system", lambda: "Linux")
    monkeypatch.setattr(compile_module, "_has_executorch_exir", lambda: True)
    source = program if path == "exported_program" else program.module()
    monkeypatch.setattr(exporter, "export", MagicMock(return_value=program))
    monkeypatch.setattr(torch.export, "export", MagicMock(return_value=program))
    result = SimpleNamespace(_tensor_data={}, write_to_file=MagicMock())
    export = MagicMock(
        return_value=SimpleNamespace(to_executorch=MagicMock(return_value=result))
    )
    monkeypatch.setattr(torch_tensorrt.executorch, "export", export)
    torch_tensorrt.save(
        source,
        str(tmp_path / "model.pte"),
        output_format="executorch",
        arg_inputs=[torch.ones(1)] if path == "fx_retrace" else None,
        retrace=path != "fx_legacy",
        external_engine_data="engines",
    )
    export.assert_called_once()
    assert export.call_args.kwargs["external_engine_data"] == "engines"


@pytest.mark.unit
def test_save_checks_the_name_before_the_model_and_input_checks(tmp_path):
    with pytest.raises(ValueError, match="external_engine_data must be"):
        torch_tensorrt.save(
            torch.fx.symbolic_trace(torch.nn.Identity()),
            str(tmp_path / "model.pte"),
            output_format="executorch",
            external_engine_data="../outside",
        )


@pytest.mark.unit
@pytest.mark.parametrize("value", [None, "engines"])
def test_save_warns_and_does_not_forward_for_other_formats(
    monkeypatch, tmp_path, program, caplog, value
):
    sink = create_autospec(torch.export.save)
    monkeypatch.setattr(torch.export, "save", sink)
    with caplog.at_level(logging.WARNING):
        torch_tensorrt.save(
            program, str(tmp_path / "model.ep"), external_engine_data=value
        )
    sink.assert_called_once()
    assert "external_engine_data" not in sink.call_args.kwargs
    warnings = [
        r.message for r in caplog.records if "external_engine_data=" in r.message
    ]
    assert len(warnings) == (0 if value is None else 1)
    if warnings:
        assert "will be ignored" in warnings[0]
