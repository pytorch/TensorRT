# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Behavior shared by the compiled-kernel frontend capability guards."""

import contextlib
from types import SimpleNamespace

import pytest
import torch

from torch_tensorrt.kernels import _common, _cutile, _register, _triton

from .conftest import (
    assert_ran_in_engine,
    compile_op,
    register_once,
    skip_no_cuda,
    skip_no_qdp,
)


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


@pytest.mark.parametrize("target_tied", [False, True])
def test_hardware_compatible_engines_decline_target_tied_kernels(target_tied):
    seen = []
    validate = _common.make_dtype_capability_validator(
        "test_common::op",
        "kernel",
        [torch.float32],
        [torch.float32],
        target_validator=seen.append if target_tied else None,
    )
    settings = SimpleNamespace(hardware_compatible=True)
    assert validate(_FakeNode([torch.float32], torch.float32), settings) is (
        not target_tied
    )
    assert seen == []


def test_unspecified_target_uses_current_device():
    seen = []
    validate = _common.make_dtype_capability_validator(
        "test_common::op",
        "kernel",
        [torch.float32],
        [torch.float32],
        target_validator=seen.append,
    )
    settings = SimpleNamespace(device=SimpleNamespace(gpu_id=-1))
    assert validate(_FakeNode([torch.float32], torch.float32), settings)
    assert seen == [None]


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


@pytest.mark.parametrize("extra_bytes, accepted", [(0, True), (16, False)])
def test_kernels_above_48_kib_of_shared_memory_are_declined(
    extra_bytes, accepted, caplog
):
    shared_mem = 48 * 1024 + extra_bytes
    validate = _common.make_dtype_capability_validator(
        "test_common::op",
        "kernel",
        [torch.float32],
        [torch.float32],
        shared_mem=shared_mem,
    )
    with caplog.at_level("WARNING", logger=_common.__name__):
        assert validate(_FakeNode([torch.float32], torch.float32)) is accepted
    assert bool(caplog.records) is not accepted
    if not accepted:
        assert "49168 bytes" in caplog.text

    # Without an eager_fn, registration must fail exactly where conversion declines.
    expectation = (
        contextlib.nullcontext()
        if accepted
        else pytest.raises(RuntimeError, match="49168 bytes")
    )
    with expectation:
        _common.require_eager_fn_above_shared_mem_limit(
            "test_common::op", "kernel", shared_mem, None
        )

    # With one, registration warns once that the op will stay in PyTorch.
    caplog.clear()
    with caplog.at_level("WARNING", logger=_common.__name__):
        _common.require_eager_fn_above_shared_mem_limit(
            "test_common::op", "kernel", shared_mem, lambda x: x
        )
    assert ("never lowered to TensorRT" in caplog.text) is not accepted


# Adds 1 to each element; the optional static array is the only thing that varies.
_SHARED_MEM_PTX = """
.version 7.0
.target sm_70
.address_size 64
{static_decl}
.visible .entry add_one(.param .u64 x, .param .u64 out)
{{
 .reg .b32 %r<5>; .reg .b64 %rd<6>; .reg .f32 %f<3>;
 ld.param.u64 %rd1, [x]; ld.param.u64 %rd2, [out];
 cvta.to.global.u64 %rd1, %rd1; cvta.to.global.u64 %rd2, %rd2;
 mov.u32 %r1, %ctaid.x; mov.u32 %r2, %ntid.x; mov.u32 %r3, %tid.x;
 mad.lo.s32 %r4, %r1, %r2, %r3; mul.wide.s32 %rd3, %r4, 4;
 add.s64 %rd4, %rd1, %rd3; add.s64 %rd5, %rd2, %rd3;
 ld.global.f32 %f1, [%rd4];
 {static_use}
 add.f32 %f2, %f1, 0f3F800000; st.global.f32 [%rd5], %f2;
 ret;
}}
"""


def _register_shared_mem_op(op_name, static_bytes, shared_mem_by_tactic):
    from typing import List, Tuple

    import tensorrt.plugin as trtp

    import torch_tensorrt.kernels as ttk

    ptx = _SHARED_MEM_PTX.format(
        static_decl=(
            f".shared .align 4 .b8 sbuf[{static_bytes}];" if static_bytes else ""
        ),
        static_use="st.volatile.shared.f32 [sbuf], %f1;" if static_bytes else "",
    )

    def aot(inputs, outputs, tactic):
        params = trtp.KernelLaunchParams()
        params.grid_x = trtp.cdiv(inputs[0].shape_expr.numel(), 128)
        params.block_x = 128
        params.shared_mem = shared_mem_by_tactic[tactic]
        return params

    ttk.ptx_op(
        op_name,
        ptx.encode(),
        "add_one",
        meta_fn=lambda x: torch.empty_like(x),
        eager_fn=lambda x: x + 1,
        aot_fn=aot,
        schema="(Tensor x) -> Tensor",
    )
    if 0 not in shared_mem_by_tactic:

        @trtp.autotune(op_name)
        def _tactics(
            x: trtp.TensorDesc, outputs: Tuple[trtp.TensorDesc]
        ) -> List[trtp.AutoTuneCombination]:
            return [
                trtp.AutoTuneCombination(
                    "FP32, FP32", "LINEAR", list(shared_mem_by_tactic)
                )
            ]


@skip_no_cuda
@skip_no_qdp
@pytest.mark.parametrize(
    "static_bytes, shared_mem_by_tactic, builds, message",
    [
        (
            0,
            {0: 48 * 1024 + 16},
            False,
            "49168 bytes of dynamic shared memory on top of 0",
        ),
        (16 * 1024, {0: 32 * 1024}, True, None),
        (
            16 * 1024,
            {0: 32 * 1024 + 16},
            False,
            "32784 bytes of dynamic shared memory on top of 16384",
        ),
        # TensorRT can drop a custom tactic that cannot launch and build with another.
        (0, {1: 64 * 1024, 2: 32 * 1024}, True, "TensorRT cannot use tactic 1"),
    ],
)
def test_aot_launch_over_48_kib_of_shared_memory_fails_the_engine_build(
    static_bytes, shared_mem_by_tactic, builds, message, caplog, capfd
):
    op_name = "test_common::smem_{}_{}".format(
        static_bytes, "_".join(map(str, shared_mem_by_tactic.values()))
    )
    register_once(
        op_name,
        lambda: _register_shared_mem_op(op_name, static_bytes, shared_mem_by_tactic),
    )
    x = torch.randn(1024, device="cuda")
    if not builds:
        with pytest.raises(RuntimeError, match="could not build the engine"):
            compile_op(op_name, [x])
        # TensorRT prints the exception raised in the AOT callback to stderr.
        assert message in capfd.readouterr().err
        return
    trt_module = compile_op(op_name, [x])
    assert_ran_in_engine(trt_module, op_name)
    torch.testing.assert_close(trt_module(x), x + 1)
    if message:
        assert message in caplog.text


@skip_no_cuda
def test_unreadable_static_shared_memory_warns_and_is_read_again(monkeypatch, caplog):
    ptx = _SHARED_MEM_PTX.format(
        static_decl=".shared .align 4 .b8 sbuf[16384];",
        static_use="st.volatile.shared.f32 [sbuf], %f1;",
    )
    load_ptx = _common.load_ptx
    calls = []

    def fails_once(ptx, kernel_name):
        calls.append(kernel_name)
        if len(calls) == 1:
            raise ImportError("no cuda-python")
        return load_ptx(ptx, kernel_name)

    monkeypatch.setattr(_common, "load_ptx", fails_once)
    _register._static_shared_mem.cache_clear()
    params = SimpleNamespace(shared_mem=32 * 1024 + 16)
    with caplog.at_level("WARNING", logger=_register.__name__):
        _register._check_aot_shared_mem("add_one", ptx, params, 0)
    assert "Could not read the static shared memory" in caplog.text
    with pytest.raises(ValueError, match="on top of 16384"):
        _register._check_aot_shared_mem("add_one", ptx, params, 0)


_SCALAR_BETWEEN_POINTERS = _common.analyze_signature(
    {"x": "*fp32", "n": "i32", "out": "*fp32"}, (1, 1), frontend="kernel"
)


@pytest.mark.parametrize(
    "alignment,expected",
    [
        (1, {"x": 1, "out": 1}),
        (16, {"x": 16, "out": 16}),
        ({"x": 16}, {"x": 16, "out": 1}),
    ],
)
def test_pointer_alignment_resolves_per_pointer(alignment, expected):
    assert (
        _common.normalize_pointer_alignment(
            "ns::op", _SCALAR_BETWEEN_POINTERS, alignment, frontend="kernel"
        )
        == expected
    )


@pytest.mark.parametrize("alignment", [0, 3, 512, True, 2.0, {"n": 16}, {"x": 6}])
def test_invalid_pointer_alignment_is_rejected(alignment):
    with pytest.raises(ValueError, match="kernel 'ns::op' pointer_alignment"):
        _common.normalize_pointer_alignment(
            "ns::op", _SCALAR_BETWEEN_POINTERS, alignment, frontend="kernel"
        )
