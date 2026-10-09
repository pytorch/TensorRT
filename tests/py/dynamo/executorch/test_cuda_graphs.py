# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""CPU-only checks for the typed CUDA graph replay export option."""

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
from torch_tensorrt.executorch.partitioner import (  # noqa: E402
    CUDA_GRAPHS_COMPILE_SPEC_KEY,
    normalize_use_cuda_graphs,
)

_KEY = CUDA_GRAPHS_COMPILE_SPEC_KEY
_VALUES = [(None, None), (False, b"0"), (True, b"1")]
_INVALID = [0, 1, -1, 0.0, 1.0, "true", "false", b"1", [], {}]


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


@pytest.mark.unit
@pytest.mark.parametrize("api", [torch_tensorrt.save, torch_tensorrt.executorch.export])
def test_use_cuda_graphs_is_explicit_typed_keyword(api):
    parameter = inspect.signature(api).parameters["use_cuda_graphs"]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is None
    assert parameter.annotation == "bool | None"


@pytest.mark.unit
@pytest.mark.parametrize("value,expected", _VALUES)
def test_normalize_use_cuda_graphs(value, expected):
    assert normalize_use_cuda_graphs(value) == expected


@pytest.mark.unit
@pytest.mark.parametrize("value", _INVALID)
def test_normalize_use_cuda_graphs_rejects_non_bools(value):
    with pytest.raises(TypeError, match="use_cuda_graphs must be a bool or None"):
        normalize_use_cuda_graphs(value)


@pytest.mark.unit
@pytest.mark.parametrize("value,expected", _VALUES)
@pytest.mark.parametrize("methods", [False, True])
@pytest.mark.parametrize("mapped_specs", [False, True])
def test_export_applies_choice_to_every_method(
    program, lowering, value, expected, methods, mapped_specs
):
    source = (
        {
            "forward": program,
            "decode": torch.export.export(torch.nn.Identity(), (torch.ones(1),)),
        }
        if methods
        else program
    )
    caller = CompileSpec("target_device", b"cuda:1")
    specs = [caller]
    compile_specs = {"forward": specs} if mapped_specs else specs
    torch_tensorrt.executorch.export(
        source, compile_specs=compile_specs, use_cuda_graphs=value
    )
    chains = lowering.call_args.kwargs["partitioner"]
    if not methods:
        chains = {"forward": chains}
    assert set(chains) == ({"forward", "decode"} if methods else {"forward"})
    for name, chain in chains.items():
        actual = chain[0].compile_specs
        assert [s.value for s in actual if s.key == _KEY] == (
            [] if expected is None else [expected]
        )
        if name == "forward" or not mapped_specs:
            assert caller in actual
    assert specs == [caller]


@pytest.mark.unit
def test_export_omitted_choice_leaves_spec_absent(program, lowering):
    torch_tensorrt.executorch.export(program)
    specs = lowering.call_args.kwargs["partitioner"][0].compile_specs
    assert not any(spec.key == _KEY for spec in specs)


@pytest.mark.unit
@pytest.mark.parametrize("value", _INVALID)
def test_export_rejects_non_bools(program, lowering, value):
    with pytest.raises(TypeError, match="use_cuda_graphs must be a bool or None"):
        torch_tensorrt.executorch.export(program, use_cuda_graphs=value)
    lowering.assert_not_called()


@pytest.mark.unit
@pytest.mark.parametrize("value", [None, False, True])
@pytest.mark.parametrize("raw", [b"0", b"1", b"invalid"])
@pytest.mark.parametrize("mapped", [False, True])
def test_export_refuses_raw_key_even_without_keyword(
    program, lowering, value, raw, mapped
):
    source = {
        "forward": program,
        "decode": torch.export.export(torch.nn.Identity(), (torch.ones(1),)),
    }
    specs = [CompileSpec(_KEY, raw)]
    with pytest.raises(ValueError, match="Pass use_cuda_graphs= instead"):
        torch_tensorrt.executorch.export(
            source,
            compile_specs={"decode": specs} if mapped else specs,
            use_cuda_graphs=value,
        )
    lowering.assert_not_called()


@pytest.mark.unit
@pytest.mark.parametrize("value", _INVALID)
def test_save_validates_before_model_and_input_checks(tmp_path, value):
    with pytest.raises(TypeError, match="use_cuda_graphs must be a bool or None"):
        torch_tensorrt.save(
            torch.fx.symbolic_trace(torch.nn.Identity()),
            str(tmp_path / "model.pte"),
            output_format="executorch",
            use_cuda_graphs=value,
        )


@pytest.mark.unit
@pytest.mark.parametrize("value", [None, False, True])
@pytest.mark.parametrize("path", ["exported_program", "fx_legacy", "fx_retrace"])
def test_public_save_forwards_choice_to_export(
    monkeypatch, tmp_path, program, value, path
):
    import torch_tensorrt._compile as compile_module
    import torch_tensorrt.dynamo._exporter as exporter

    monkeypatch.setattr(compile_module.platform, "system", lambda: "Linux")
    source = program if path == "exported_program" else program.module()
    monkeypatch.setattr(exporter, "export", MagicMock(return_value=program))
    monkeypatch.setattr(torch.export, "export", MagicMock(return_value=program))
    result = SimpleNamespace(_tensor_data={}, write_to_file=MagicMock())
    edge = SimpleNamespace(to_executorch=MagicMock(return_value=result))
    export = MagicMock(return_value=edge)
    monkeypatch.setattr(torch_tensorrt.executorch, "export", export)
    output = tmp_path / "model.pte"
    torch_tensorrt.save(
        source,
        str(output),
        output_format="executorch",
        arg_inputs=[torch.ones(1)] if path == "fx_retrace" else None,
        retrace=path != "fx_legacy",
        use_cuda_graphs=value,
    )
    export.assert_called_once()
    assert export.call_args.kwargs["use_cuda_graphs"] is value
    result.write_to_file.assert_called_once()
    assert output.exists()


@pytest.mark.unit
@pytest.mark.parametrize("value", [None, False, True, "ignored"])
def test_save_non_executorch_warning_and_no_forwarding(
    monkeypatch, tmp_path, program, caplog, value
):
    sink = create_autospec(torch.export.save)
    monkeypatch.setattr(torch.export, "save", sink)
    with caplog.at_level(logging.WARNING):
        torch_tensorrt.save(program, str(tmp_path / "model.ep"), use_cuda_graphs=value)
    sink.assert_called_once()
    assert "use_cuda_graphs" not in sink.call_args.kwargs
    warnings = [r.message for r in caplog.records if "use_cuda_graphs=" in r.message]
    assert len(warnings) == (0 if value is None else 1)
    if warnings:
        assert "will be ignored" in warnings[0]


@pytest.mark.unit
def test_save_supported_options_include_use_cuda_graphs(tmp_path):
    from torch_tensorrt._compile import _EXECUTORCH_SAVE_OPTIONS

    assert "use_cuda_graphs" in _EXECUTORCH_SAVE_OPTIONS
    with pytest.raises(TypeError, match="unexpected keyword argument") as error:
        torch_tensorrt.save(
            torch.nn.Identity(),
            str(tmp_path / "model.pte"),
            output_format="executorch",
            use_cuda_graph=True,
        )
    assert "'use_cuda_graphs'" in str(error.value)
