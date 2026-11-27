# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import importlib

import pytest
import torch
import torch_tensorrt


@pytest.mark.parametrize("api", ["compile", "convert_method_to_trt_engine"])
@pytest.mark.parametrize("model_type", ["module", "graph_module", "exported_program"])
def test_removed_fx_frontend_has_migration_error(api, model_type):
    model = torch.nn.ReLU()
    inputs = [torch.randn(2, 3)]
    if model_type == "graph_module":
        model = torch.fx.symbolic_trace(model)
    elif model_type == "exported_program":
        model = torch.export.export(model, tuple(inputs))

    with pytest.raises(
        ValueError, match="FX frontend support has been removed.*dynamo"
    ):
        getattr(torch_tensorrt, api)(model, ir="fx", inputs=inputs)


@pytest.mark.parametrize("ir", ["default", "dynamo"])
@pytest.mark.parametrize("model_type", ["module", "graph_module", "exported_program"])
def test_dynamo_frontend_still_accepts_supported_models(monkeypatch, ir, model_type):
    compiler = importlib.import_module("torch_tensorrt._compile")
    model = torch.nn.ReLU()
    inputs = [torch.randn(2, 3)]
    if model_type == "graph_module":
        model = torch.fx.symbolic_trace(model)
    exported = torch.export.export(model, tuple(inputs))
    if model_type == "exported_program":
        model = exported

    traced = []
    compiled = []

    def trace(module, *args, **kwargs):
        traced.append(module)
        return exported

    def compile_program(program, **kwargs):
        compiled.append(program)
        return program.module()

    monkeypatch.setattr(compiler, "dynamo_trace", trace)
    monkeypatch.setattr(compiler, "dynamo_compile", compile_program)
    result = torch_tensorrt.compile(model, ir=ir, inputs=inputs)

    assert traced == ([] if model_type == "exported_program" else [model])
    assert compiled == [exported]
    torch.testing.assert_close(result(*inputs), torch.relu(inputs[0]))


def test_torch_compile_frontend_is_selected(monkeypatch):
    compiler = importlib.import_module("torch_tensorrt._compile")
    model = torch.fx.symbolic_trace(torch.nn.ReLU())
    calls = []

    def compile_model(module, **kwargs):
        calls.append((module, kwargs))
        return module

    monkeypatch.setattr(compiler, "torch_compile", compile_model)
    assert torch_tensorrt.compile(model, ir="torch_compile", min_block_size=1) is model
    assert calls == [(model, {"min_block_size": 1})]
