# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Compiled pointer types must match actual TensorRT bindings, not just FX metadata."""

from types import SimpleNamespace

import pytest
import tensorrt as trt
import tensorrt.plugin as trtp
import torch

import torch_tensorrt.kernels as ttk
from torch_tensorrt.kernels import _common, _cutile, _register, _triton

from .conftest import register_once, skip_no_qdp


def _desc(dtype):
    shape = trtp.ShapeExprs(1)
    shape[0] = 8
    return trtp.TensorDesc(shape, dtype=dtype)


@skip_no_qdp
@pytest.mark.parametrize(
    "spelling", ["fp64", "float64", torch.float64, "i16", "complex64"]
)
@pytest.mark.parametrize("parameter", ["x", "out"])
def test_cutile_rejects_unsupported_pointer_dtype_before_compiling(
    monkeypatch, spelling, parameter
):
    monkeypatch.setattr(
        _cutile,
        "compile_cutile_to_ptx",
        lambda *args, **kwargs: pytest.fail(
            "unsupported signature reached the compiler"
        ),
    )

    def meta(x: torch.Tensor) -> torch.Tensor:
        return torch.empty_like(x)

    signature = {"x": "fp32", "out": "fp32"}
    signature[parameter] = spelling
    with pytest.raises(ValueError, match="no exact TensorRT representation"):
        ttk.cutile_op(
            "dtype_safety::unsupported", object(), signature, meta, grid=lambda i, o: 1
        )


@skip_no_qdp
def test_truncated_engine_bindings_cannot_pass_an_fp64_metadata_guard():
    from tensorrt.plugin._lib import QDP_REGISTRY

    from torch_tensorrt.dynamo.conversion.plugins._generate_plugin import (
        _generate_plugin,
    )

    op = "dtype_safety::truncated_descriptor"

    def meta(x: torch.Tensor) -> torch.Tensor:
        return torch.empty_like(x)

    def register():
        _register._register_pytorch_op(op, meta, None)
        _generate_plugin(op)

    register_once(op, register)
    input_desc = _desc(trt.float32)
    (output_desc,) = QDP_REGISTRY[op].register_func(input_desc)
    assert input_desc.dtype == output_desc.dtype == trt.float32

    # Truncation can leave the original FX tensor metadata at float64 even
    # though the plugin descriptors above describe float32 engine storage.
    value = torch.empty(8, dtype=torch.float64)
    node = SimpleNamespace(
        args=[SimpleNamespace(meta={"val": value})], meta={"val": value}
    )
    validator = _common.make_dtype_capability_validator(
        op, "cuTile", [torch.float64], [torch.float64]
    )
    assert not validator(node, SimpleNamespace(truncate_double=True))


@pytest.fixture(params=["cutile", "cutile_custom", "triton"])
def aot_callback(request, monkeypatch):
    captured = {}
    monkeypatch.setattr(
        _register,
        "register_precompiled_qdp_plugin",
        lambda **kwargs: captured.update(kwargs),
    )
    monkeypatch.setattr(
        _cutile, "compile_cutile_to_ptx", lambda *args: (b"ptx", "kernel", (128, 1, 1))
    )
    monkeypatch.setattr(_triton, "_device_arch", lambda device=None: 90)
    monkeypatch.setattr(
        _triton,
        "compile_triton_to_ptx",
        lambda *args, **kwargs: _triton.CompiledTritonArtifact(
            b"ptx", "kernel", 4, 0, SimpleNamespace(arch=90), 90
        ),
    )

    def meta(x: torch.Tensor) -> torch.Tensor:
        return torch.empty_like(x, dtype=torch.float16)

    def launch(*args):
        raise RuntimeError("launch callback reached")

    op = "dtype_safety::descriptor_check"
    if request.param == "triton":
        kernel = SimpleNamespace(
            arg_names=["x", "out"],
            params=[
                SimpleNamespace(name=name, is_constexpr=False) for name in ("x", "out")
            ],
        )
        ttk.triton_op(op, kernel, {"x": "*fp32", "out": "*fp16"}, {}, launch, meta)
    else:
        options = (
            {"aot_fn": launch} if request.param == "cutile_custom" else {"grid": launch}
        )
        ttk.cutile_op(op, object(), {"x": "fp32", "out": "fp16"}, meta, **options)
    return captured["aot_fn"]


@skip_no_qdp
@pytest.mark.parametrize(
    "inputs,outputs", [(trt.float16, trt.float16), (trt.float32, trt.float32)]
)
def test_aot_rejects_actual_descriptor_dtype_before_user_launch(
    aot_callback, inputs, outputs
):
    with pytest.raises(
        ValueError, match="descriptor has dtype.*compiled kernel requires"
    ):
        aot_callback([_desc(inputs)], [_desc(outputs)], 0)


@skip_no_qdp
@pytest.mark.parametrize("kind", ["input", "output"])
def test_aot_requires_every_tensor_descriptor(aot_callback, kind):
    inputs, outputs = [_desc(trt.float32)], [_desc(trt.float16)]
    (inputs if kind == "input" else outputs).clear()
    with pytest.raises(ValueError, match=f"expected 1 {kind} descriptor"):
        aot_callback(inputs, outputs, 0)


@skip_no_qdp
def test_aot_accepts_matching_mixed_dtype_bindings(aot_callback):
    with pytest.raises(RuntimeError, match="launch callback reached"):
        aot_callback([_desc(trt.float32)], [_desc(trt.float16)], 0)
