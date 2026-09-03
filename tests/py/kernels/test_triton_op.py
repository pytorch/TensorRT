"""Tests for triton_op (Triton kernel -> AOT QDP plugin path)."""

import inspect
import re
import subprocess
import sys
import textwrap
import types

import pytest
import torch

import torch_tensorrt
import torch_tensorrt.kernels as ttk

from .conftest import (
    register_once,
    skip_no_cuda,
    skip_no_qdp,
    skip_no_triton,
)

# A minimal PTX ``.entry`` carrying the two trailing scratch params Triton adds
# to every kernel, used as the stub compiler's output in the no-GPU unit tests.
FAKE_PTX = """//
.version 9.3
.target sm_90
.address_size 64

.visible .entry my_kernel(
\t.param .u64 my_kernel_param_0,
\t.param .u32 my_kernel_param_1,
\t.param .u64 my_kernel_param_2,
\t.param .u64 my_kernel_param_3,
\t.param .u64 my_kernel_param_4
)
{
\tret;
}
"""


def _identity_meta(x: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(x)


class _FakeKernel:
    """Small stand-in exposing the declaration metadata used by triton_op."""

    def __init__(self, arg_names, constexpr_names=()):
        self.arg_names = list(arg_names)
        constexpr_names = set(constexpr_names)
        self.params = [
            type(
                "_Param",
                (),
                {"name": name, "is_constexpr": name in constexpr_names},
            )()
            for name in self.arg_names
        ]


@pytest.fixture
def captured_registration(monkeypatch):
    """Stub compile + registration; returns the kwargs triton_op forwards on."""
    from torch_tensorrt.kernels import _register, _triton

    monkeypatch.setattr(
        _triton,
        "compile_triton_to_ptx",
        lambda *a, **k: (b"// ptx bytes", "my_kernel_entry", 4, 0),
    )
    captured = {}
    monkeypatch.setattr(
        _register,
        "register_precompiled_qdp_plugin",
        lambda *a, **k: captured.update(k),
    )
    return captured


# ---- No-GPU, no-Triton: PTX post-processing helpers ----


def test_parse_ptx_version():
    from torch_tensorrt.kernels._triton import _parse_ptx_version

    assert _parse_ptx_version(FAKE_PTX) == 93
    assert _parse_ptx_version("// no version here") is None


@pytest.mark.parametrize("version", ["3.5.0", "3.5.0+git.abc123", "3.8.0"])
def test_triton_version_guard_accepts_supported_releases(version):
    from torch_tensorrt.kernels._triton import _require_supported_triton_version

    _require_supported_triton_version(types.SimpleNamespace(__version__=version))


@pytest.mark.parametrize("version", ["2.3.0", "3.4.0"])
def test_triton_version_guard_rejects_older_releases(version):
    from torch_tensorrt.kernels._triton import _require_supported_triton_version

    with pytest.raises(
        ImportError,
        match=rf"requires Triton >=3\.5\.0; found {re.escape(version)}",
    ):
        _require_supported_triton_version(types.SimpleNamespace(__version__=version))


@pytest.mark.parametrize("version", [None, "development"])
def test_triton_version_guard_rejects_unknown_versions(version):
    from torch_tensorrt.kernels._triton import _require_supported_triton_version

    with pytest.raises(ImportError, match="could not determine.*Triton version"):
        _require_supported_triton_version(types.SimpleNamespace(__version__=version))


def test_triton_import_rejects_old_version_before_compiler_import(monkeypatch):
    from torch_tensorrt.kernels._triton import _triton_import

    old_triton = types.ModuleType("triton")
    old_triton.__version__ = "3.4.0"
    monkeypatch.setitem(sys.modules, "triton", old_triton)
    monkeypatch.delitem(sys.modules, "triton.compiler", raising=False)

    with pytest.raises(
        ImportError,
        match=r"requires Triton >=3\.5\.0.*pip install 'triton>=3\.5\.0'",
    ):
        _triton_import()


# ---- No-GPU: triton_op plumbing (compile + register mocked out) ----


@skip_no_qdp
def test_triton_op_forwards_to_registrar(captured_registration):
    """triton_op must compile then use the shared precompiled-PTX registrar."""
    captured = captured_registration
    sig = {"x_ptr": "*fp32", "y_ptr": "*fp32"}
    ttk.triton_op(
        "ttk_test::triton_forward",
        kernel=_FakeKernel(sig),
        signature=sig,
        constexprs={},
        grid=lambda inputs, outputs: (1,),
        meta_fn=_identity_meta,
        supports_dynamic_shapes=True,
    )

    assert captured["op_name"] == "ttk_test::triton_forward"
    assert captured["ptx"] == b"// ptx bytes"
    assert captured["kernel_name"] == "my_kernel_entry"
    assert callable(captured["aot_fn"])
    assert captured["use_aot_if_available"] is True
    assert captured["supports_dynamic_shapes"] is True
    # A dtype capability validator is always installed, even with none passed.
    assert callable(captured["capability_validator"])


def test_triton_op_exposes_one_validated_aot_path():
    """The high-level API owns AOT construction instead of accepting a bypass."""
    parameters = inspect.signature(ttk.triton_op).parameters
    assert "aot_fn" not in parameters
    assert "requires_output_allocator" not in parameters


# ---- No-GPU: signature validation (misuse must not reach the GPU) ----


def _sig_error(**overrides):
    """Call triton_op with a deliberately broken config, return the ValueError."""
    kwargs = dict(
        signature={"x_ptr": "*fp32", "y_ptr": "*fp32"},
        constexprs={},
        grid=lambda i, o: (1,),
        meta_fn=_identity_meta,
    )
    kwargs.update(overrides)
    if "kernel" not in overrides:
        kwargs["kernel"] = _FakeKernel(
            [*kwargs["signature"], *kwargs["constexprs"]],
            kwargs["constexprs"],
        )
    with pytest.raises(ValueError) as excinfo:
        ttk.triton_op("ttk_test::triton_invalid", **kwargs)
    return str(excinfo.value)


@skip_no_qdp
def test_scalar_without_extra_args_fn_rejected():
    """A scalar in the signature with no extra_args_fn would silently read zero."""
    message = _sig_error(
        signature={"x_ptr": "*fp32", "n": "i32", "y_ptr": "*fp32"},
        extra_args_fn=None,
    )
    assert "extra_args_fn" in message
    assert "n" in message


@skip_no_qdp
def test_extra_args_fn_without_scalars_rejected():
    message = _sig_error(extra_args_fn=lambda i, o: [1])
    assert "no scalar parameters" in message


@skip_no_qdp
def test_signature_arity_must_match_meta_fn():
    """Two tensor inputs + one output needs three pointers, not two."""

    def _meta2(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.empty_like(x)

    message = _sig_error(meta_fn=_meta2)
    assert "2 tensor input(s)" in message


@skip_no_qdp
def test_interleaved_scalar_rejected():
    message = _sig_error(
        signature={"x_ptr": "*fp32", "n": "i32", "y_ptr": "*fp32", "m": "i32"},
        extra_args_fn=lambda i, o: [1],
    )
    assert "begin and end with pointer" in message


@skip_no_qdp
def test_scalar_run_must_be_contiguous():
    def _meta2(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.empty_like(x)

    message = _sig_error(
        meta_fn=_meta2,
        signature={
            "x_ptr": "*fp32",
            "n": "i32",
            "y_ptr": "*fp32",
            "m": "i32",
            "z_ptr": "*fp32",
        },
        extra_args_fn=lambda i, o: [1, 2],
    )
    assert "interleaves" in message


@skip_no_qdp
def test_signature_order_must_match_kernel_declaration():
    """A reordered dict must not disguise a different compiled kernel ABI."""
    message = _sig_error(
        kernel=_FakeKernel(["x_ptr", "y_ptr", "n"]),
        signature={"x_ptr": "*fp32", "n": "i32", "y_ptr": "*fp32"},
        extra_args_fn=lambda i, o: [1],
    )
    assert "exactly match the kernel declaration" in message
    assert "['x_ptr', 'y_ptr', 'n']" in message


@skip_no_qdp
def test_signature_must_cover_every_runtime_parameter():
    message = _sig_error(
        kernel=_FakeKernel(["x_ptr", "stride", "y_ptr"]),
        signature={"x_ptr": "*fp32", "y_ptr": "*fp32"},
    )
    assert "['x_ptr', 'stride', 'y_ptr']" in message


@skip_no_qdp
def test_kernel_declaration_metadata_is_required():
    kernel = type("_Kernel", (), {"arg_names": ["x_ptr", "y_ptr"]})()
    message = _sig_error(kernel=kernel)
    assert "complete arg_names and KernelParam metadata" in message


@skip_no_qdp
def test_kernel_declaration_metadata_must_be_consistent():
    kernel = _FakeKernel(["x_ptr", "y_ptr"])
    kernel.params[1].name = "z_ptr"
    message = _sig_error(kernel=kernel)
    assert "inconsistent declaration metadata" in message


@skip_no_qdp
def test_kernel_constexpr_markers_must_be_boolean():
    kernel = _FakeKernel(["x_ptr", "y_ptr"])
    kernel.params[1].is_constexpr = "false"
    message = _sig_error(kernel=kernel)
    assert "is_constexpr marker must be bool" in message


@skip_no_qdp
def test_constexpr_keys_must_be_parameter_names():
    message = _sig_error(
        kernel=_FakeKernel(["x_ptr", "y_ptr"]),
        constexprs={1: 16},
    )
    assert "constexpr keys must be non-empty parameter names" in message


@skip_no_qdp
def test_constexprs_must_exactly_match_declaration():
    message = _sig_error(
        kernel=_FakeKernel(["x_ptr", "BLOCK", "y_ptr"], ["BLOCK"]),
        signature={"x_ptr": "*fp32", "y_ptr": "*fp32"},
        constexprs={},
    )
    assert "missing constexpr values" in message


@skip_no_qdp
@pytest.mark.parametrize("scalar_type", ["fp16", "fp32", "fp64", "i8", "i16", "i64"])
def test_unsupported_scalar_signature_type_rejected(scalar_type):
    message = _sig_error(
        signature={"x_ptr": "*fp32", "value": scalar_type, "y_ptr": "*fp32"},
        extra_args_fn=lambda i, o: [1],
    )
    assert f"scalar type '{scalar_type}'" in message
    assert "SymInt32" in message


@skip_no_qdp
@pytest.mark.parametrize("values", [None, [], [1, 2]])
def test_extra_args_count_must_exactly_match_signature(captured_registration, values):
    ttk.triton_op(
        "ttk_test::triton_scalar_count",
        kernel=_FakeKernel(["x_ptr", "n", "y_ptr"]),
        signature={"x_ptr": "*fp32", "n": "i32", "y_ptr": "*fp32"},
        constexprs={},
        grid=lambda i, o: (1,),
        meta_fn=_identity_meta,
        extra_args_fn=lambda i, o: values,
    )

    with pytest.raises(ValueError, match="returned .* value"):
        captured_registration["aot_fn"](["in"], ["out"], 0)


@skip_no_qdp
@pytest.mark.parametrize("invalid_value", [1.5, True, "1"])
def test_extra_arg_value_must_be_symint32_compatible(
    captured_registration, invalid_value
):
    ttk.triton_op(
        "ttk_test::triton_scalar_value_type",
        kernel=_FakeKernel(["x_ptr", "n", "y_ptr"]),
        signature={"x_ptr": "*fp32", "n": "i32", "y_ptr": "*fp32"},
        constexprs={},
        grid=lambda i, o: (1,),
        meta_fn=_identity_meta,
        extra_args_fn=lambda i, o: [invalid_value],
    )

    with pytest.raises(TypeError, match="must be an int or trtp.SymInt32"):
        captured_registration["aot_fn"](["in"], ["out"], 0)


@skip_no_qdp
@pytest.mark.parametrize("invalid_value", [-(2**31) - 1, 2**31])
def test_extra_arg_value_must_fit_signed_i32(captured_registration, invalid_value):
    ttk.triton_op(
        "ttk_test::triton_scalar_range",
        kernel=_FakeKernel(["x_ptr", "n", "y_ptr"]),
        signature={"x_ptr": "*fp32", "n": "i32", "y_ptr": "*fp32"},
        constexprs={},
        grid=lambda i, o: (1,),
        meta_fn=_identity_meta,
        extra_args_fn=lambda i, o: [invalid_value],
    )

    with pytest.raises(ValueError, match="outside the signed i32 range"):
        captured_registration["aot_fn"](["in"], ["out"], 0)


@skip_no_qdp
def test_extra_args_result_must_be_iterable(captured_registration):
    ttk.triton_op(
        "ttk_test::triton_scalar_iterable",
        kernel=_FakeKernel(["x_ptr", "n", "y_ptr"]),
        signature={"x_ptr": "*fp32", "n": "i32", "y_ptr": "*fp32"},
        constexprs={},
        grid=lambda i, o: (1,),
        meta_fn=_identity_meta,
        extra_args_fn=lambda i, o: 1,
    )

    with pytest.raises(TypeError, match="must return an iterable"):
        captured_registration["aot_fn"](["in"], ["out"], 0)


@skip_no_qdp
def test_explicit_tensor_schema_drives_arity(captured_registration):
    """An explicit tensor schema supports an otherwise unannotated meta_fn."""

    def _meta(x):
        return torch.empty_like(x)

    ttk.triton_op(
        "ttk_test::triton_schema_arity",
        kernel=_FakeKernel(["x_ptr", "y_ptr"]),
        signature={"x_ptr": "*fp32", "y_ptr": "*fp32"},
        constexprs={},
        grid=lambda i, o: (1,),
        meta_fn=_meta,
        schema="(Tensor x) -> Tensor",
    )
    assert captured_registration["schema"] == "(Tensor x) -> Tensor"


@skip_no_qdp
def test_triton_schema_rejects_scalar_torch_attributes():
    """Torch attributes are not kernel extras and must never be silently dropped."""

    def _meta(x, n):
        return torch.empty_like(x)

    message = _sig_error(
        meta_fn=_meta,
        schema="(Tensor x, int n) -> Tensor",
        signature={"x_ptr": "*fp32", "n": "i32", "y_ptr": "*fp32"},
        extra_args_fn=lambda i, o: [1],
    )
    assert "Tensor-only Torch schemas" in message
    assert "scalar Torch attributes ['n']" in message


def test_schema_analysis_prefers_schema_over_hints():
    from torch_tensorrt.kernels._register import analyze_op_schema

    def _meta(x, n):
        return torch.empty_like(x)

    with pytest.raises(ValueError, match="complete meta_fn type hints"):
        analyze_op_schema(_meta, require_complete_hints=True)

    # An explicit schema is authoritative and supports unannotated functions.
    info = analyze_op_schema(_meta, "(Tensor x, int n) -> Tensor")
    assert (len(info.tensor_arg_names), info.num_outputs) == (1, 1)

    def _multi_meta(x):
        return torch.empty_like(x), torch.empty_like(x)

    info = analyze_op_schema(_multi_meta, "(Tensor x) -> (Tensor, Tensor)")
    assert (len(info.tensor_arg_names), info.num_outputs) == (1, 2)
    with pytest.raises(ValueError, match="could not parse"):
        analyze_op_schema(_meta, "not a schema")


def test_analyze_signature_splits_on_convention():
    from torch_tensorrt.kernels._triton import analyze_signature

    layout = analyze_signature({"x_ptr": "*fp16", "n": "i32", "y_ptr": "*fp16"}, (1, 1))
    assert [p.name for p in layout.inputs] == ["x_ptr"]
    assert [p.name for p in layout.scalars] == ["n"]
    assert [p.name for p in layout.outputs] == ["y_ptr"]
    assert layout.inputs[0].dtype == torch.float16
    assert layout.outputs[0].dtype == torch.float16


@pytest.mark.parametrize("pointer_type", ["**fp32", "*fp32:16", "*fp32:anything"])
def test_malformed_pointer_spelling_rejected(pointer_type):
    from torch_tensorrt.kernels._triton import analyze_signature

    with pytest.raises(ValueError, match="malformed pointer type"):
        analyze_signature({"x_ptr": pointer_type, "y_ptr": "*fp32"}, (1, 1))


@pytest.mark.parametrize("pointer_type", ["fp64", "i16", "fp8e5", "weird"])
def test_untested_pointer_dtype_rejected(pointer_type):
    from torch_tensorrt.kernels._triton import analyze_signature

    with pytest.raises(ValueError, match="unsupported pointer element type"):
        analyze_signature({"x_ptr": "*fp32", "y_ptr": f"*{pointer_type}"}, (1, 1))


# ---- No-GPU: dtype capability validator ----


class _FakeNode:
    """Minimal stand-in for the torch.fx.Node a capability validator receives."""

    def __init__(self, arg_dtypes, out_dtype):
        self.args = [
            type(
                "_Arg",
                (),
                {"meta": ({"val": torch.empty(2, dtype=d)} if d is not None else {})},
            )()
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


def _validator_for(signature, user_validator=None):
    from torch_tensorrt.kernels._triton import (
        analyze_signature,
        make_dtype_capability_validator,
    )

    layout = analyze_signature(signature, (1, 1))
    return make_dtype_capability_validator("ns::op", layout, user_validator)


def test_dtype_validator_accepts_matching_dtypes():
    validate = _validator_for({"x_ptr": "*fp32", "y_ptr": "*fp32"})
    assert validate(_FakeNode([torch.float32], torch.float32), None) is True


def test_dtype_validator_rejects_mismatched_input():
    """The fp16-into-an-fp32-kernel case, which used to return silent garbage."""
    validate = _validator_for({"x_ptr": "*fp32", "y_ptr": "*fp32"})
    assert validate(_FakeNode([torch.float16], torch.float16), None) is False


def test_dtype_validator_rejects_mismatched_output():
    validate = _validator_for({"x_ptr": "*fp32", "y_ptr": "*fp16"})
    assert validate(_FakeNode([torch.float32], torch.float32), None) is False


def test_dtype_validator_composes_with_user_validator():
    validate = _validator_for(
        {"x_ptr": "*fp32", "y_ptr": "*fp32"}, user_validator=lambda n, s: False
    )
    # dtypes match, but the user's predicate still gets to veto.
    assert validate(_FakeNode([torch.float32], torch.float32), None) is False


def test_dtype_validator_rejects_missing_input_metadata():
    validate = _validator_for({"x_ptr": "*fp32", "y_ptr": "*fp32"})
    assert validate(_FakeNode([None], torch.float32), None) is False


def test_dtype_validator_rejects_missing_output_metadata():
    validate = _validator_for({"x_ptr": "*fp32", "y_ptr": "*fp32"})
    assert validate(_FakeNode([torch.float32], None), None) is False


def test_dtype_validator_rejects_missing_input_argument():
    validate = _validator_for({"x_ptr": "*fp32", "y_ptr": "*fp32"})
    assert validate(_FakeNode([], torch.float32), None) is False


def test_dtype_validator_rejects_extra_input_argument():
    validate = _validator_for({"x_ptr": "*fp32", "y_ptr": "*fp32"})
    assert (
        validate(_FakeNode([torch.float32, torch.float32], torch.float32), None)
        is False
    )


def test_dtype_validator_rejects_missing_node_metadata():
    validate = _validator_for({"x_ptr": "*fp32", "y_ptr": "*fp32"})
    node = types.SimpleNamespace(args=_FakeNode([torch.float32], None).args)
    assert validate(node, None) is False


def test_dtype_validator_rejects_wrong_output_count():
    validate = _validator_for({"x_ptr": "*fp32", "y_ptr": "*fp32"})
    assert (
        validate(_FakeNode([torch.float32], [torch.float32, torch.float32]), None)
        is False
    )


@skip_no_qdp
def test_generated_descriptor_uses_fake_output_dtype():
    """QDP must not inherit input 0's dtype for a mixed-dtype output."""
    import tensorrt as trt
    import tensorrt.plugin as trtp
    from tensorrt.plugin._lib import QDP_REGISTRY

    from torch_tensorrt.dynamo.conversion.plugins._generate_plugin import (
        _generate_plugin,
    )
    from torch_tensorrt.kernels._register import _register_pytorch_op

    op_name = "ttk_test::triton_desc_fp32_to_fp16"

    def _meta(x: torch.Tensor) -> torch.Tensor:
        return torch.empty_like(x, dtype=torch.float16)

    def _register() -> None:
        _register_pytorch_op(op_name, _meta, None)
        _generate_plugin(op_name)

    register_once(op_name, _register)

    shape = trtp.ShapeExprs(2)
    shape[0], shape[1] = 4, 256
    input_desc = trtp.TensorDesc(shape, dtype=trt.float32)
    (output_desc,) = QDP_REGISTRY[op_name].register_func(input_desc)

    assert output_desc.dtype == trt.float16
    assert output_desc.ndim == input_desc.ndim


@skip_no_qdp
def test_generated_descriptor_propagates_input_dtype_to_meta():
    """An empty_like meta function must see the QDP input descriptor dtype."""
    import tensorrt as trt
    import tensorrt.plugin as trtp
    from tensorrt.plugin._lib import QDP_REGISTRY

    from torch_tensorrt.dynamo.conversion.plugins._generate_plugin import (
        _generate_plugin,
    )
    from torch_tensorrt.kernels._register import _register_pytorch_op

    op_name = "ttk_test::triton_desc_preserve_fp16"

    def _meta(x: torch.Tensor) -> torch.Tensor:
        return torch.empty_like(x)

    def _register() -> None:
        _register_pytorch_op(op_name, _meta, None)
        _generate_plugin(op_name)

    register_once(op_name, _register)

    shape = trtp.ShapeExprs(1)
    shape[0] = 16
    input_desc = trtp.TensorDesc(shape, dtype=trt.float16)
    (output_desc,) = QDP_REGISTRY[op_name].register_func(input_desc)

    assert output_desc.dtype == trt.float16


# ---- No-GPU: PTX ISA capping ----

# FAKE_PTX declares .version 9.3 and 5 entry params for this 3-param signature.
CAPPING_SIG = {"a": "*fp32", "b": "i32", "c": "*fp32"}


def _stub_triton(monkeypatch, requested, omitted_metadata=(), **extra_metadata):
    """Replace triton.compile with a stub recording each requested ptx_version."""
    from torch_tensorrt.kernels import _triton

    metadata = {
        "name": "my_kernel",
        "num_warps": 4,
        "shared": 0,
        "global_scratch_size": 0,
        "profile_scratch_size": 0,
        "num_ctas": 1,
        "warp_size": 32,
        "launch_cooperative_grid": False,
        "launch_pdl": False,
        "tmem_size": 0,
        "tensordesc_meta": [],
        **extra_metadata,
    }
    for field in omitted_metadata:
        metadata.pop(field)
    fake_metadata = type("_M", (), metadata)()

    compiled_results = []

    class _Compiled:
        def __init__(self, ptx):
            self.asm = {"ptx": ptx}
            self.metadata = fake_metadata

    class _FakeTriton:
        class compiler:
            ASTSource = staticmethod(lambda **kw: object())

        @staticmethod
        def compile(src, options=None):
            ptx_version = (options or {}).get("ptx_version")
            requested.append(ptx_version)
            # Make each compilation artifact distinct so callers can verify that
            # compile_triton_to_ptx returns the final (possibly capped) artifact,
            # rather than rewriting it or retaining an earlier uncapped result.
            emitted_version = 93 if ptx_version is None else ptx_version
            ptx = FAKE_PTX.replace(
                ".version 9.3",
                f".version {emitted_version // 10}.{emitted_version % 10}",
            )
            ptx += f"// requested ptx_version={ptx_version!r}\n"
            compiled = _Compiled(ptx)
            compiled_results.append(compiled)
            return compiled

    monkeypatch.setattr(_triton, "_triton_import", lambda: _FakeTriton)
    return compiled_results


@pytest.mark.parametrize(
    "driver_max, default_isa, expected",
    [
        # Triton's ISA is known up front and too new: one compile, already capped.
        (91, 93, [91]),
        # Driver is new enough: one compile at Triton's default.
        (95, 93, [None]),
        # Triton's ISA can't be predicted: compile, notice 9.3 > 9.1, recompile.
        (91, None, [None, 91]),
        # Unpredictable ISA and unknown driver: no capping at all.
        (None, None, [None]),
    ],
)
def test_ptx_isa_capped_to_driver(monkeypatch, driver_max, default_isa, expected):
    from torch_tensorrt.kernels import _triton

    monkeypatch.setattr(_triton, "_driver_max_ptx_version", lambda: driver_max)
    monkeypatch.setattr(_triton, "_triton_default_ptx_version", lambda: default_isa)
    requested = []
    compiled_results = _stub_triton(monkeypatch, requested)

    ptx, _, _, _ = _triton.compile_triton_to_ptx(object(), CAPPING_SIG, {})

    assert requested == expected
    assert ptx == compiled_results[-1].asm["ptx"].encode("utf-8")


def test_nonzero_scratch_raises(monkeypatch):
    """Triton scratch buffers can't be fed by the AOT launcher — fail loudly."""
    from torch_tensorrt.kernels import _triton

    monkeypatch.setattr(_triton, "_driver_max_ptx_version", lambda: None)
    monkeypatch.setattr(_triton, "_triton_default_ptx_version", lambda: None)
    _stub_triton(monkeypatch, [], global_scratch_size=128, profile_scratch_size=0)

    with pytest.raises(RuntimeError, match="scratch memory"):
        _triton.compile_triton_to_ptx(object(), CAPPING_SIG, {})


@pytest.mark.parametrize(
    ("metadata", "match"),
    [
        ({"num_ctas": 2}, "num_ctas=2"),
        ({"warp_size": 64}, "warp_size=64"),
        ({"launch_cooperative_grid": True}, "launch_cooperative_grid=True"),
        ({"launch_pdl": True}, "launch_pdl=True"),
        ({"tmem_size": 16}, "tmem_size=16"),
        ({"tensordesc_meta": [object()]}, "tensordesc_meta is non-empty"),
    ],
)
def test_unsupported_launch_metadata_raises(monkeypatch, metadata, match):
    from torch_tensorrt.kernels import _triton

    monkeypatch.setattr(_triton, "_driver_max_ptx_version", lambda: None)
    _stub_triton(monkeypatch, [], **metadata)

    with pytest.raises(RuntimeError, match=match):
        _triton.compile_triton_to_ptx(object(), CAPPING_SIG, {})


def test_missing_launch_metadata_fails_closed(monkeypatch):
    from torch_tensorrt.kernels import _triton

    monkeypatch.setattr(_triton, "_driver_max_ptx_version", lambda: None)
    _stub_triton(monkeypatch, [], omitted_metadata=("global_scratch_size",))

    with pytest.raises(RuntimeError, match="missing.*global_scratch_size"):
        _triton.compile_triton_to_ptx(object(), CAPPING_SIG, {})


@pytest.mark.parametrize(
    ("metadata", "match"),
    [
        ({"num_warps": 0}, "invalid num_warps"),
        ({"num_warps": 33}, "invalid num_warps"),
        ({"shared": -1}, "invalid shared memory"),
        ({"shared": 2**31}, "invalid shared memory"),
    ],
)
def test_launch_metadata_ranges_are_validated(monkeypatch, metadata, match):
    from torch_tensorrt.kernels import _triton

    monkeypatch.setattr(_triton, "_driver_max_ptx_version", lambda: None)
    _stub_triton(monkeypatch, [], **metadata)

    with pytest.raises(RuntimeError, match=match):
        _triton.compile_triton_to_ptx(object(), CAPPING_SIG, {})


@pytest.mark.parametrize(
    "field",
    [
        "num_warps",
        "shared",
        "global_scratch_size",
        "profile_scratch_size",
        "num_ctas",
        "warp_size",
        "tmem_size",
    ],
)
@pytest.mark.parametrize("value", [None, True, 1.5, "1"])
def test_launch_numeric_metadata_is_not_coerced(monkeypatch, field, value):
    from torch_tensorrt.kernels import _triton

    monkeypatch.setattr(_triton, "_driver_max_ptx_version", lambda: None)
    _stub_triton(monkeypatch, [], **{field: value})

    with pytest.raises(RuntimeError, match=rf"'{field}'.*must be an integer"):
        _triton.compile_triton_to_ptx(object(), CAPPING_SIG, {})


@pytest.mark.parametrize("field", ["launch_cooperative_grid", "launch_pdl"])
@pytest.mark.parametrize("value", [None, 0, "false"])
def test_launch_boolean_metadata_is_not_coerced(monkeypatch, field, value):
    from torch_tensorrt.kernels import _triton

    monkeypatch.setattr(_triton, "_driver_max_ptx_version", lambda: None)
    _stub_triton(monkeypatch, [], **{field: value})

    with pytest.raises(RuntimeError, match=rf"'{field}'.*must be bool"):
        _triton.compile_triton_to_ptx(object(), CAPPING_SIG, {})


@pytest.mark.parametrize("value", [None, {}, ""])
def test_tensor_descriptor_metadata_type_is_validated(monkeypatch, value):
    from torch_tensorrt.kernels import _triton

    monkeypatch.setattr(_triton, "_driver_max_ptx_version", lambda: None)
    _stub_triton(monkeypatch, [], tensordesc_meta=value)

    with pytest.raises(RuntimeError, match="'tensordesc_meta'.*list or tuple"):
        _triton.compile_triton_to_ptx(object(), CAPPING_SIG, {})


@skip_no_qdp
@pytest.mark.parametrize("grid", [(), (1, 2, 3, 4)])
def test_grid_dimension_count_is_validated(captured_registration, grid):
    """TensorRT launches accept exactly one through three grid dimensions."""

    ttk.triton_op(
        "ttk_test::triton_grid_count",
        kernel=_FakeKernel(["x_ptr", "y_ptr"]),
        signature={"x_ptr": "*fp32", "y_ptr": "*fp32"},
        constexprs={},
        grid=lambda i, o: grid,
        meta_fn=_identity_meta,
    )

    with pytest.raises(ValueError, match="dimension"):
        captured_registration["aot_fn"](["in"], ["out"], 0)


@skip_no_qdp
@pytest.mark.parametrize("dimension", [True, 1.5, "1"])
def test_grid_dimension_type_is_validated(captured_registration, dimension):
    ttk.triton_op(
        "ttk_test::triton_grid_type",
        kernel=_FakeKernel(["x_ptr", "y_ptr"]),
        signature={"x_ptr": "*fp32", "y_ptr": "*fp32"},
        constexprs={},
        grid=lambda i, o: (dimension,),
        meta_fn=_identity_meta,
    )

    with pytest.raises(TypeError, match="must be an int or TensorRT symbolic"):
        captured_registration["aot_fn"](["in"], ["out"], 0)


@skip_no_qdp
@pytest.mark.parametrize("dimension", [0, -1, 2**31])
def test_grid_dimension_range_is_validated(captured_registration, dimension):
    ttk.triton_op(
        "ttk_test::triton_grid_range",
        kernel=_FakeKernel(["x_ptr", "y_ptr"]),
        signature={"x_ptr": "*fp32", "y_ptr": "*fp32"},
        constexprs={},
        grid=lambda i, o: (dimension,),
        meta_fn=_identity_meta,
    )

    with pytest.raises(ValueError, match="positive|signed i32"):
        captured_registration["aot_fn"](["in"], ["out"], 0)


# ---- GPU integration: real Triton kernel through triton_op ----

try:
    import triton
    import triton.language as tl

    @triton.jit
    def _ttk_add_one_kernel(x_ptr, n, y_ptr, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        off = pid * BLOCK + tl.arange(0, BLOCK)
        mask = off < n
        tl.store(y_ptr + off, tl.load(x_ptr + off, mask=mask) + 1, mask=mask)

    @triton.jit
    def _ttk_fp32_to_fp16_kernel(x_ptr, n, y_ptr, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        off = pid * BLOCK + tl.arange(0, BLOCK)
        mask = off < n
        # The fp16 output pointer makes tl.store perform the intended cast.
        tl.store(y_ptr + off, tl.load(x_ptr + off, mask=mask) + 1, mask=mask)

except ImportError:
    triton = None


def _register_add_one(op_name: str) -> None:
    import tensorrt.plugin as trtp

    BLOCK = 256

    def _meta(x: torch.Tensor) -> torch.Tensor:
        return torch.empty_like(x)

    def _eager(x: torch.Tensor) -> torch.Tensor:
        y = torch.empty_like(x)
        grid = lambda meta: (triton.cdiv(x.numel(), meta["BLOCK"]),)
        _ttk_add_one_kernel[grid](x, x.numel(), y, BLOCK=BLOCK)
        return y

    def _register() -> None:
        ttk.triton_op(
            op_name,
            kernel=_ttk_add_one_kernel,
            signature={"x_ptr": "*fp32", "n": "i32", "y_ptr": "*fp32"},
            constexprs={"BLOCK": BLOCK},
            grid=lambda inputs, outputs: (
                trtp.cdiv(inputs[0].shape_expr.numel(), BLOCK),
            ),
            meta_fn=_meta,
            extra_args_fn=lambda inputs, outputs: [
                trtp.SymInt32(inputs[0].shape_expr.numel())
            ],
            eager_fn=_eager,
            supports_dynamic_shapes=True,
        )

    register_once(op_name, _register)


def _register_fp32_to_fp16(op_name: str) -> None:
    import tensorrt.plugin as trtp

    BLOCK = 256

    def _meta(x: torch.Tensor) -> torch.Tensor:
        return torch.empty_like(x, dtype=torch.float16)

    def _eager(x: torch.Tensor) -> torch.Tensor:
        y = torch.empty_like(x, dtype=torch.float16)
        grid = lambda meta: (triton.cdiv(x.numel(), meta["BLOCK"]),)
        _ttk_fp32_to_fp16_kernel[grid](x, x.numel(), y, BLOCK=BLOCK)
        return y

    def _register() -> None:
        ttk.triton_op(
            op_name,
            kernel=_ttk_fp32_to_fp16_kernel,
            signature={"x_ptr": "*fp32", "n": "i32", "y_ptr": "*fp16"},
            constexprs={"BLOCK": BLOCK},
            grid=lambda inputs, outputs: (
                trtp.cdiv(inputs[0].shape_expr.numel(), BLOCK),
            ),
            meta_fn=_meta,
            extra_args_fn=lambda inputs, outputs: [
                trtp.SymInt32(inputs[0].shape_expr.numel())
            ],
            eager_fn=_eager,
        )

    register_once(op_name, _register)


@skip_no_cuda
@skip_no_qdp
@skip_no_triton
class TestTritonOpIntegration:
    def test_ptx_is_embedded_unmodified(self, monkeypatch):
        """We embed exactly what triton.compile emitted -- no PTX rewriting.

        TensorRT sizes the kernel argument buffer from the kernel's own declared
        ABI and zero-fills the slots it wasn't given, so Triton's trailing
        zero-sized scratch params need no special handling.
        """
        import triton

        from torch_tensorrt.kernels import _triton

        real_compile = triton.compile
        compiled_results = []

        def recording_compile(*args, **kwargs):
            compiled = real_compile(*args, **kwargs)
            compiled_results.append(compiled)
            return compiled

        # The production helper may compile once or may recompile with a driver
        # PTX cap. Capture the actual final Triton result in either case instead
        # of comparing it with an independently compiled, always-uncapped result.
        monkeypatch.setattr(triton, "compile", recording_compile)

        sig = {"x_ptr": "*fp32", "n": "i32", "y_ptr": "*fp32"}
        ptx, name, num_warps, _shared = _triton.compile_triton_to_ptx(
            _ttk_add_one_kernel, sig, {"BLOCK": 256}
        )
        final_compilation = compiled_results[-1]
        assert ptx == final_compilation.asm["ptx"].encode("utf-8")
        assert name == final_compilation.metadata.name
        assert num_warps >= 1

    def test_num_warps_is_honored(self):
        from torch_tensorrt.kernels._triton import compile_triton_to_ptx

        sig = {"x_ptr": "*fp32", "n": "i32", "y_ptr": "*fp32"}
        _, _, warps, _ = compile_triton_to_ptx(
            _ttk_add_one_kernel, sig, {"BLOCK": 256}, num_warps=8, num_stages=2
        )
        assert warps == 8

    def test_dtype_mismatch_falls_back_instead_of_returning_garbage(self):
        """fp16 into an fp32-compiled kernel must not silently produce nonsense."""
        _register_add_one("ttk_test::triton_add_one_dtype")

        class M(torch.nn.Module):
            def forward(self, x):
                return torch.ops.ttk_test.triton_add_one_dtype(x)

        x = torch.randn(4, 256, device="cuda", dtype=torch.float16)
        trt = torch_tensorrt.compile(
            M().cuda().eval(),
            inputs=[x],
            enabled_precisions={torch.float16},
            min_block_size=1,
        )
        with torch.no_grad():
            # The plugin is declined, so this runs in PyTorch — and is correct.
            assert torch.allclose(trt(x), x + 1, atol=1e-2, rtol=1e-2)

    def test_mixed_output_dtype_lowers_and_returns_fp16(self):
        """An fp32-input/fp16-output kernel gets an fp16 TensorRT binding."""
        op_name = "ttk_test::triton_fp32_to_fp16"
        _register_fp32_to_fp16(op_name)

        class M(torch.nn.Module):
            def forward(self, x):
                return torch.ops.ttk_test.triton_fp32_to_fp16(x)

        x = torch.linspace(-2, 2, 1024, device="cuda", dtype=torch.float32)
        compiled = torch_tensorrt.compile(
            M().cuda().eval(),
            inputs=[x],
            enabled_precisions={torch.float32, torch.float16},
            min_block_size=1,
        )
        assert not any(
            node.op == "call_function" and "triton_fp32_to_fp16" in str(node.target)
            for node in compiled.graph.nodes
        ), "The mixed-dtype Triton op remained as a PyTorch fallback node"

        with torch.no_grad():
            actual = compiled(x)
        assert actual.dtype == torch.float16
        torch.testing.assert_close(actual, (x + 1).to(torch.float16))

    def test_register_and_eager(self):
        _register_add_one("ttk_test::triton_add_one_eager")
        x = torch.randn(1024, device="cuda")
        assert torch.allclose(
            torch.ops.ttk_test.triton_add_one_eager(x), x + 1, atol=1e-4, rtol=1e-4
        )

    def test_trt_compile_dynamic_shapes(self):
        _register_add_one("ttk_test::triton_add_one_dyn")

        class M(torch.nn.Module):
            def forward(self, x):
                return torch.ops.ttk_test.triton_add_one_dyn(x)

        inputs = [
            torch_tensorrt.Input(
                min_shape=(1, 128),
                opt_shape=(1, 512),
                max_shape=(1, 2048),
                dtype=torch.float32,
            )
        ]
        trt = torch_tensorrt.compile(
            M().cuda().eval(),
            inputs=inputs,
            enabled_precisions={torch.float32},
            min_block_size=1,
        )
        engine_modules = [
            (name, module)
            for name, module in trt.named_modules()
            if name.startswith("_run_on_acc_")
        ]
        assert len(engine_modules) == 1, (
            "The single-op graph must be lowered into one TensorRT engine; "
            f"compiled graph was:\n{trt.graph}"
        )
        assert not any(
            node.op == "call_function" and "triton_add_one_dyn" in str(node.target)
            for node in trt.graph.nodes
        ), "The Triton custom op remained as a PyTorch fallback node"

        from tensorrt.plugin._lib import QDP_REGISTRY

        descriptor = QDP_REGISTRY["ttk_test::triton_add_one_dyn"]
        assert descriptor.aot_impl_func is not None

        import tensorrt as trt_api

        serialized_engine = engine_modules[0][1].serialized_engine
        runtime = trt_api.Runtime(trt_api.Logger(trt_api.Logger.ERROR))
        engine = runtime.deserialize_cuda_engine(serialized_engine)
        assert engine is not None
        layer_info = engine.create_engine_inspector().get_engine_information(
            trt_api.LayerInformationFormat.JSON
        )
        assert "triton_add_one_dyn" in layer_info

        for size in [128, 512, 2048]:
            x = torch.randn(1, size, device="cuda")
            with torch.no_grad():
                assert torch.allclose(trt(x), x + 1, atol=1e-2, rtol=1e-2)

    def test_serialized_engine_runs_without_triton_in_fresh_process(self, tmp_path):
        """The AOT engine embeds PTX and needs no Triton/Python callback at runtime."""
        op_name = "ttk_test::triton_add_one_serialized"
        _register_add_one(op_name)

        class M(torch.nn.Module):
            def forward(self, x):
                return torch.ops.ttk_test.triton_add_one_serialized(x)

        x = torch.arange(1024, device="cuda", dtype=torch.float32).reshape(4, 256)
        compiled = torch_tensorrt.compile(
            M().cuda().eval(),
            inputs=[x],
            enabled_precisions={torch.float32},
            min_block_size=1,
        )
        engine_modules = [
            module
            for name, module in compiled.named_modules()
            if name.startswith("_run_on_acc_")
        ]
        assert len(engine_modules) == 1

        engine_path = tmp_path / "triton_add_one.plan"
        engine_path.write_bytes(engine_modules[0].serialized_engine)

        child = textwrap.dedent("""
            import importlib.abc
            import sys

            import torch

            # This Torch build probes/imports Triton during CUDA's own lazy
            # initialization. Complete that unrelated setup first, then remove
            # and block Triton for engine deserialization and execution.
            torch.cuda.init()
            for module_name in list(sys.modules):
                if module_name == "triton" or module_name.startswith("triton."):
                    del sys.modules[module_name]

            class BlockTriton(importlib.abc.MetaPathFinder):
                def find_spec(self, fullname, path=None, target=None):
                    if fullname == "triton" or fullname.startswith("triton."):
                        raise ImportError("Triton is intentionally unavailable")
                    return None

            sys.meta_path.insert(0, BlockTriton())

            import tensorrt as trt

            logger = trt.Logger(trt.Logger.ERROR)
            runtime = trt.Runtime(logger)
            engine = runtime.deserialize_cuda_engine(open(sys.argv[1], "rb").read())
            assert engine is not None
            context = engine.create_execution_context()
            assert context is not None

            inputs = [
                engine.get_tensor_name(i)
                for i in range(engine.num_io_tensors)
                if engine.get_tensor_mode(engine.get_tensor_name(i))
                == trt.TensorIOMode.INPUT
            ]
            outputs = [
                engine.get_tensor_name(i)
                for i in range(engine.num_io_tensors)
                if engine.get_tensor_mode(engine.get_tensor_name(i))
                == trt.TensorIOMode.OUTPUT
            ]
            assert len(inputs) == len(outputs) == 1

            x = torch.arange(1024, device="cuda", dtype=torch.float32).reshape(4, 256)
            y = torch.empty_like(x)
            assert context.set_tensor_address(inputs[0], x.data_ptr())
            assert context.set_tensor_address(outputs[0], y.data_ptr())
            assert context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
            torch.cuda.synchronize()
            torch.testing.assert_close(y, x + 1)
            assert not any(
                name == "triton" or name.startswith("triton.")
                for name in sys.modules
            )
            print("standalone-ok")
            """)
        result = subprocess.run(
            [sys.executable, "-c", child, str(engine_path)],
            text=True,
            capture_output=True,
            timeout=120,
            check=False,
        )
        assert result.returncode == 0, (
            f"fresh-process stdout:\n{result.stdout}\n"
            f"fresh-process stderr:\n{result.stderr}"
        )
        assert "standalone-ok" in result.stdout
