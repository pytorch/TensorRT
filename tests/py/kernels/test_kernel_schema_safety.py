# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Schema inputs cannot replace the helpers used by generated plugin callbacks."""

import pytest
import torch

import torch_tensorrt.kernels as ttk
from torch_tensorrt.kernels import _cutile, _register, _triton

from .conftest import skip_no_qdp


@pytest.mark.parametrize(
    "name",
    [
        "_fn",
        "_generic_plugin_desc",
        "_generic_plugin_impl",
        "_user_aot_fn",
        "_kernel_name",
        "_ptx_str",
        "_trtp",
        "isinstance",
        "tuple",
        "len",
    ],
)
def test_schema_rejects_inputs_that_shadow_callback_dependencies(name):
    with pytest.raises(ValueError, match="invalid argument names"):
        _register.analyze_op_schema(lambda x: x, f"(Tensor {name}) -> Tensor")


@skip_no_qdp
@pytest.mark.parametrize("frontend", ["triton", "cutile"])
@pytest.mark.parametrize("name", ["_kernel_name", "_generic_plugin_desc", "len"])
def test_frontend_rejects_unsafe_schema_before_compilation_or_registration(
    monkeypatch, frontend, name
):
    def unexpected(*args, **kwargs):
        pytest.fail("Unsafe schema reached compilation or registration")

    monkeypatch.setattr(_triton, "compile_triton_to_ptx", unexpected)
    monkeypatch.setattr(_cutile, "compile_cutile_to_ptx", unexpected)
    monkeypatch.setattr(_register, "_register_pytorch_op", unexpected)

    def meta(x: torch.Tensor) -> torch.Tensor:
        return torch.empty_like(x)

    kwargs = dict(
        op_name="schema_safety::unsafe",
        kernel=object(),
        meta_fn=meta,
        grid=lambda inputs, outputs: 1,
        schema=f"(Tensor {name}) -> Tensor",
    )
    with pytest.raises(ValueError, match="invalid argument names"):
        if frontend == "triton":
            ttk.triton_op(
                **kwargs, signature={name: "*fp32", "out": "*fp32"}, constexprs={}
            )
        else:
            ttk.cutile_op(**kwargs, signature={name: "fp32", "out": "fp32"})


@skip_no_qdp
@pytest.mark.parametrize("name", ["inputs", "result", "launch_params", "extra_args"])
def test_aot_callback_preserves_inputs_named_after_local_temporaries(name):
    info = _register.analyze_op_schema(lambda x: x, f"(Tensor {name}) -> Tensor")
    tensor, outputs, launch, extras = object(), (object(),), object(), object()
    seen = []

    def aot(inputs, actual_outputs, tactic):
        seen.append((inputs, actual_outputs, tactic))
        return launch, extras

    callback = _register._make_aot_impl(info, "ptx", "kernel", aot)

    assert callback(tensor, outputs, 3) == ("kernel", "ptx", launch, extras)
    assert seen == [([tensor], outputs, 3)]
