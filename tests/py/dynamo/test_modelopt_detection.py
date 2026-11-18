# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""CPU checks for optional-dependency isolation in quantized-model detection."""

import ast
import builtins
import importlib
import importlib.machinery
import importlib.util
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[3]


def load_function(path, name, namespace):
    # Execute the real detector without importing unrelated native TRT modules.
    tree = ast.parse(path.read_text(), filename=str(path))
    function = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name
    )
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"),
        namespace,
    )
    return namespace[name]


@pytest.fixture(params=["standard", "mutable"])
def detector(request):
    namespace = {"torch": torch, "importlib": importlib}
    standard = load_function(
        ROOT / "py/torch_tensorrt/dynamo/utils.py",
        "is_quantized_by_modelopt",
        namespace,
    )
    if request.param == "standard":
        return standard
    return load_function(
        ROOT / "py/torch_tensorrt/dynamo/runtime/_MutableTorchTensorRTModule.py",
        "_is_modelopt_quantized",
        namespace,
    )


@pytest.fixture
def broken_modelopt(monkeypatch):
    imports = []
    original_import = builtins.__import__
    original_spec = importlib.util.find_spec

    def find_spec(name, *args, **kwargs):
        if name == "modelopt":
            return importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
        return original_spec(name, *args, **kwargs)

    def import_module(name, *args, **kwargs):
        if name == "modelopt" or name.startswith("modelopt."):
            imports.append(name)
            raise AttributeError("module 'torch.jit' has no attribute 'script'")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", find_spec)
    monkeypatch.setattr(builtins, "__import__", import_module)
    return imports


@pytest.mark.parametrize("nested", [False, True])
def test_plain_model_does_not_initialize_modelopt(detector, broken_modelopt, nested):
    model = torch.nn.Linear(2, 2)
    if nested:
        model = torch.nn.Sequential(model, torch.nn.ModuleList([torch.nn.ReLU()]))
    assert not detector(model)
    assert broken_modelopt == []


@pytest.mark.parametrize("subclass", [False, True])
def test_quantizers_and_user_subclasses_are_recognized(
    detector, broken_modelopt, subclass
):
    quantizer = type(
        "TensorQuantizer",
        (torch.nn.Module,),
        {"__module__": "modelopt.torch.quantization.nn.modules.tensor_quantizer"},
    )
    if subclass:
        quantizer = type(
            "UserQuantizer", (quantizer,), {"__module__": "user.quantization"}
        )
    model = torch.nn.ModuleDict({"nested": torch.nn.Sequential(quantizer())})
    assert detector(model)
    assert broken_modelopt == []


@pytest.mark.parametrize(
    "namespace", ["user.quantization", "modelopt.torch.quantization_extra"]
)
def test_unrelated_quantizer_names_are_not_modelopt(
    detector, broken_modelopt, namespace
):
    unrelated = type("TensorQuantizer", (torch.nn.Module,), {"__module__": namespace})
    assert not detector(torch.nn.Sequential(unrelated()))
    assert broken_modelopt == []
