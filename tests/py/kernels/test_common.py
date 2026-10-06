# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Behavior shared by the compiled-kernel frontend capability guards."""

from types import SimpleNamespace

import pytest
import torch

from torch_tensorrt.kernels import _common, _cutile, _triton

from .conftest import register_once, skip_no_qdp


def _validator(frontend, user_validator=None):
    if frontend == "cutile":
        layout = _cutile.validate_cutile_config(
            "test_common::op", {"x": "fp32", "out": "fp32"}, {}, (1, 1)
        )
        return _cutile.make_dtype_capability_validator(
            "test_common::op", layout, user_validator
        )

    layout = _triton.analyze_signature({"x": "*fp32", "out": "*fp32"}, (1, 1))
    artifact = _triton.CompiledTritonArtifact(
        b"ptx", "kernel", 4, 0, SimpleNamespace(arch=90), 90
    )
    return _triton.make_dtype_capability_validator(
        "test_common::op", layout, user_validator, artifact=artifact
    )


@pytest.fixture
def reject_target_check(monkeypatch):
    monkeypatch.setattr(
        _triton,
        "validate_target",
        lambda *args, **kwargs: pytest.fail(
            "Declined conversion checked its GPU target"
        ),
    )


@pytest.mark.parametrize("frontend", ["triton", "cutile"])
def test_frontend_declines_missing_positional_arguments(frontend, reject_target_check):
    node = SimpleNamespace(args=None, meta={"val": torch.empty(1)})

    assert _validator(frontend)(node) is False


@pytest.mark.parametrize("frontend", ["triton", "cutile"])
def test_frontend_user_rejection_precedes_metadata_and_target_checks(
    frontend, reject_target_check
):
    class UnavailableMetadata:
        @property
        def args(self):
            pytest.fail("Explicitly declined conversion inspected input metadata")

        @property
        def meta(self):
            pytest.fail("Explicitly declined conversion inspected output metadata")

    node = UnavailableMetadata()
    settings = SimpleNamespace(device=SimpleNamespace(gpu_id=2))
    seen = []

    def decline(actual_node, actual_settings):
        seen.append((actual_node, actual_settings))
        return False

    assert _validator(frontend, decline)(node, settings) is False
    assert seen == [(node, settings)]


class _FakeNode:
    """Minimal stand-in for the torch.fx.Node a capability validator receives."""

    def __init__(self, arg_dtypes, out_dtype):
        self.args = [
            SimpleNamespace(
                meta={"val": torch.empty(2, dtype=d)} if d is not None else {}
            )
            for d in arg_dtypes
        ]
        if isinstance(out_dtype, list):
            produced = [
                torch.empty(2, dtype=d) if d is not None else None for d in out_dtype
            ]
        else:
            produced = (
                torch.empty(2, dtype=out_dtype) if out_dtype is not None else None
            )
        self.meta = {"val": produced}


@pytest.mark.parametrize(
    "inputs, output, expected",
    [
        pytest.param([torch.float32], torch.float16, True, id="mixed-dtype-match"),
        pytest.param([torch.float16], torch.float16, False, id="input-dtype"),
        pytest.param([torch.float32], torch.float32, False, id="output-dtype"),
        pytest.param([None], torch.float16, False, id="missing-input-metadata"),
        pytest.param([torch.float32], None, False, id="missing-output-metadata"),
        pytest.param([], torch.float16, False, id="missing-input"),
        pytest.param(
            [torch.float32, torch.float32], torch.float16, False, id="extra-input"
        ),
        pytest.param(
            [torch.float32], [torch.float16, torch.float16], False, id="extra-output"
        ),
    ],
)
def test_dtype_guard(inputs, output, expected):
    validate = _common.make_dtype_capability_validator(
        "test_common::op", "test", [torch.float32], [torch.float16]
    )
    assert validate(_FakeNode(inputs, output)) is expected


@pytest.mark.parametrize("frontend", ["triton", "cutile"])
def test_frontend_dtype_guard(frontend, monkeypatch):
    monkeypatch.setattr(_triton, "validate_target", lambda *a, **k: None)
    validate = _validator(frontend)
    assert validate(_FakeNode([torch.float32], torch.float32)) is True
    assert validate(_FakeNode([torch.float16], torch.float32)) is False
    assert (
        validate(SimpleNamespace(args=_FakeNode([torch.float32], None).args)) is False
    )


@skip_no_qdp
@pytest.mark.parametrize(
    "preserve_input_dtype", [False, True], ids=["cast", "preserve"]
)
def test_generated_descriptor_preserves_meta_dtype(preserve_input_dtype):
    import tensorrt as trt
    import tensorrt.plugin as trtp
    from tensorrt.plugin._lib import QDP_REGISTRY

    from torch_tensorrt.dynamo.conversion.plugins._generate_plugin import (
        _generate_plugin,
    )
    from torch_tensorrt.kernels._register import _register_pytorch_op

    op_name = f"ttk_test::descriptor_dtype_{preserve_input_dtype}"

    def meta(x: torch.Tensor) -> torch.Tensor:
        assert x.dtype == (torch.float16 if preserve_input_dtype else torch.float32)
        return (
            torch.empty_like(x)
            if preserve_input_dtype
            else torch.empty_like(x, dtype=torch.float16)
        )

    def register():
        _register_pytorch_op(op_name, meta, None)
        _generate_plugin(op_name)

    register_once(op_name, register)
    shape = trtp.ShapeExprs(1)
    shape[0] = 16
    input_desc = trtp.TensorDesc(
        shape, dtype=trt.float16 if preserve_input_dtype else trt.float32
    )
    (output_desc,) = QDP_REGISTRY[op_name].register_func(input_desc)
    assert output_desc.dtype == trt.float16
    assert output_desc.ndim == input_desc.ndim
