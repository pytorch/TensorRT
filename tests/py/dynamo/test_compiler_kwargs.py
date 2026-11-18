# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import importlib

import pytest
import torch
import torch_tensorrt
from torch_tensorrt.dynamo import _compiler
from torch_tensorrt.dynamo.utils import parse_dynamo_kwargs


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("debug", True),
        ("enable_cross_compile_for_windows", True),
        ("enabled_precisions", {torch.float16}),
        ("make_refittable", True),
        ("refit", True),
        ("truncate_long_and_double", True),
        ("use_explicit_typing", True),
        ("use_python_runtime", False),
    ],
)
def test_removed_compiler_options_raise(option, value):
    with pytest.raises(TypeError, match=rf"`{option}` has been removed"):
        _compiler.compile(None, **{option: value})


def test_all_removed_compiler_options_are_reported():
    with pytest.raises(TypeError) as exc_info:
        _compiler.compile(None, debug=True, refit=True)

    message = str(exc_info.value)
    assert "`debug` has been removed" in message
    assert "`refit` has been removed" in message


def test_top_level_compile_rejects_removed_enabled_precisions():
    with pytest.raises(TypeError, match=r"`enabled_precisions` has been removed"):
        torch_tensorrt.compile(
            torch.nn.Identity(),
            ir="dynamo",
            enabled_precisions={torch.float16},
        )


def test_top_level_cross_compile_rejects_removed_enabled_precisions():
    compile_module = importlib.import_module("torch_tensorrt._compile")

    with pytest.raises(TypeError, match=r"`enabled_precisions` has been removed"):
        compile_module.cross_compile_for_windows.__wrapped__(
            torch.nn.Identity(),
            "model.ep",
            enabled_precisions={torch.float16},
        )


def test_top_level_engine_conversion_rejects_removed_enabled_precisions():
    with pytest.raises(TypeError, match=r"`enabled_precisions` has been removed"):
        torch_tensorrt.convert_method_to_trt_engine(
            torch.nn.Identity(),
            inputs=[torch.randn(1)],
            enabled_precisions={torch.float16},
        )


def test_misspelled_compiler_option_suggests_correction():
    with pytest.raises(
        TypeError,
        match=(
            r"compile\(\) got an unexpected keyword argument 'min_blok_size'\. "
            r"Did you mean 'min_block_size'\?"
        ),
    ):
        _compiler.compile(None, min_blok_size=1)


def test_unknown_compiler_option_is_rejected_without_suggestion():
    with pytest.raises(
        TypeError,
        match=r"compile\(\) got an unexpected keyword argument 'not_an_option'\.",
    ):
        _compiler.compile(None, not_an_option=True)


def test_torch_compile_misspelled_option_suggests_correction():
    with pytest.raises(
        TypeError,
        match=(
            r"torch\.compile\(\) got an unexpected keyword argument "
            r"'min_blok_size'\. Did you mean 'min_block_size'\?"
        ),
    ):
        parse_dynamo_kwargs({"options": {"min_blok_size": 1}})


def test_torch_compile_removed_option_raises():
    with pytest.raises(TypeError, match=r"`use_python_runtime` has been removed"):
        parse_dynamo_kwargs({"options": {"use_python_runtime": False}})


def test_top_level_compile_rejects_misspelled_option_before_tracing(monkeypatch):
    compile_module = importlib.import_module("torch_tensorrt._compile")

    def fail_if_called(*args, **kwargs):
        raise AssertionError("Tracing should not run for invalid compiler options")

    monkeypatch.setattr(compile_module, "dynamo_trace", fail_if_called)

    with pytest.raises(
        TypeError,
        match=(
            r"compile\(\) got an unexpected keyword argument 'min_blok_size'\. "
            r"Did you mean 'min_block_size'\?"
        ),
    ):
        compile_module.compile(
            torch.nn.Identity(),
            ir="dynamo",
            inputs=[torch.randn(1)],
            min_blok_size=1,
        )


def test_top_level_compile_does_not_forward_strict_to_dynamo_compile(monkeypatch):
    compile_module = importlib.import_module("torch_tensorrt._compile")
    trace_kwargs = {}
    compiler_kwargs = {}
    exported_program = object()
    compiled_module = object()

    def fake_trace(*args, **kwargs):
        trace_kwargs.update(kwargs)
        return exported_program

    def fake_compile(*args, **kwargs):
        compiler_kwargs.update(kwargs)
        return compiled_module

    monkeypatch.setattr(compile_module, "dynamo_trace", fake_trace)
    monkeypatch.setattr(compile_module, "dynamo_compile", fake_compile)

    result = compile_module.compile(
        torch.nn.Identity(),
        ir="dynamo",
        inputs=[torch.randn(1)],
        strict=True,
    )

    assert result is compiled_module
    assert trace_kwargs["strict"] is True
    assert "strict" not in compiler_kwargs


def test_cross_compile_routes_strict_only_to_tracer(monkeypatch):
    compile_module = importlib.import_module("torch_tensorrt._compile")
    trace_kwargs = {}
    compiler_kwargs = {}
    exported_program = object()
    compiled_module = object()

    def fake_trace(*args, **kwargs):
        trace_kwargs.update(kwargs)
        return exported_program

    def fake_compile(*args, **kwargs):
        compiler_kwargs.update(kwargs)
        return compiled_module

    monkeypatch.setattr(compile_module, "dynamo_trace", fake_trace)
    monkeypatch.setattr(
        compile_module, "dynamo_cross_compile_for_windows", fake_compile
    )
    monkeypatch.setattr(
        compile_module,
        "dynamo_save_cross_compiled_exported_program",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(compile_module.platform, "system", lambda: "Linux")
    monkeypatch.setattr(compile_module.platform, "architecture", lambda: ("64bit", ""))

    compile_module.cross_compile_for_windows.__wrapped__(
        torch.nn.Identity(),
        "model.ep",
        inputs=[torch.randn(1)],
        strict=True,
    )

    assert trace_kwargs["strict"] is True
    assert "strict" not in compiler_kwargs


def test_engine_conversion_routes_strict_only_to_tracer(monkeypatch):
    compile_module = importlib.import_module("torch_tensorrt._compile")
    trace_kwargs = {}
    compiler_kwargs = {}
    exported_program = object()
    serialized_engine = b"engine"

    def fake_trace(*args, **kwargs):
        trace_kwargs.update(kwargs)
        return exported_program

    def fake_convert(*args, **kwargs):
        compiler_kwargs.update(kwargs)
        return serialized_engine

    monkeypatch.setattr(compile_module, "dynamo_trace", fake_trace)
    monkeypatch.setattr(
        compile_module,
        "dynamo_convert_exported_program_to_serialized_trt_engine",
        fake_convert,
    )

    result = compile_module.convert_method_to_trt_engine(
        torch.nn.Identity(),
        inputs=[torch.randn(1)],
        strict=True,
    )

    assert result is serialized_engine
    assert trace_kwargs["strict"] is True
    assert "strict" not in compiler_kwargs


def test_misspelled_inputs_option_suggests_correction():
    with pytest.raises(
        TypeError,
        match=(
            r"compile\(\) got an unexpected keyword argument 'inptus'\. "
            r"Did you mean 'inputs'\?"
        ),
    ):
        _compiler.compile(None, inptus=[])


def test_cross_compile_misspelled_option_suggests_correction():
    with pytest.raises(
        TypeError,
        match=(
            r"cross_compile_for_windows\(\) got an unexpected keyword argument "
            r"'min_blok_size'\. Did you mean 'min_block_size'\?"
        ),
    ):
        _compiler.validate_compiler_kwargs(
            {"min_blok_size": 1},
            _compiler.cross_compile_for_windows,
        )
