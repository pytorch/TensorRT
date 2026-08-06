"""Tests for triton_op (Triton kernel -> AOT QDP plugin path)."""

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

# A minimal PTX ``.entry`` with two extra trailing (scratch) params, used by the
# no-GPU unit tests for the PTX post-processing helpers.
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
        _register, "register_qdp_plugin", lambda *a, **k: captured.update(k)
    )
    return captured


# ---- No-GPU, no-Triton: PTX post-processing helpers ----


def test_parse_ptx_version():
    from torch_tensorrt.kernels._triton import _parse_ptx_version

    assert _parse_ptx_version(FAKE_PTX) == 93
    assert _parse_ptx_version("// no version here") is None


def test_parse_entry_params_counts_all():
    from torch_tensorrt.kernels._triton import _parse_entry_params

    match, params = _parse_entry_params(FAKE_PTX, "my_kernel")
    assert match is not None
    assert len(params) == 5


def test_strip_trailing_scratch_params_keeps_only_runtime_args():
    from torch_tensorrt.kernels._triton import (
        _parse_entry_params,
        _strip_trailing_scratch_params,
    )

    match, params = _parse_entry_params(FAKE_PTX, "my_kernel")
    stripped = _strip_trailing_scratch_params(FAKE_PTX, match, params, keep=3)
    _, params = _parse_entry_params(stripped, "my_kernel")
    assert len(params) == 3
    assert "my_kernel_param_3" not in stripped
    assert "my_kernel_param_4" not in stripped
    # The kept params and body are untouched.
    assert "my_kernel_param_2" in stripped
    assert "ret;" in stripped


def test_strip_trailing_scratch_params_noop_when_nothing_to_strip():
    from torch_tensorrt.kernels._triton import (
        _parse_entry_params,
        _strip_trailing_scratch_params,
    )

    match, params = _parse_entry_params(FAKE_PTX, "my_kernel")
    assert _strip_trailing_scratch_params(FAKE_PTX, match, params, keep=5) == FAKE_PTX
    # Unknown kernel name -> no match -> left untouched.
    missing, params = _parse_entry_params(FAKE_PTX, "other")
    assert _strip_trailing_scratch_params(FAKE_PTX, missing, params, 1) == FAKE_PTX


# ---- No-GPU: triton_op plumbing (compile + register mocked out) ----


@skip_no_qdp
def test_triton_op_forwards_to_registrar(captured_registration):
    """triton_op must compile then forward the PTX + TritonSpec to the registrar."""
    from torch_tensorrt.kernels._triton_spec import TritonSpec

    captured = captured_registration
    sig = {"x_ptr": "*fp32", "y_ptr": "*fp32"}
    ttk.triton_op(
        "ttk_test::triton_forward",
        kernel=object(),
        signature=sig,
        constexprs={},
        grid=lambda inputs, outputs: (1,),
        meta_fn=_identity_meta,
        supports_dynamic_shapes=True,
    )

    assert captured["op_name"] == "ttk_test::triton_forward"
    assert captured["precompiled_ptx"] == b"// ptx bytes"
    # triton_op builds a TritonSpec, not a CudaPythonSpec.
    assert isinstance(captured["spec"], TritonSpec)
    assert captured["spec"].kernel_name == "my_kernel_entry"
    assert captured["spec"].signature == sig
    assert captured["use_aot_if_available"] is True
    assert captured["supports_dynamic_shapes"] is True
    # A dtype capability validator is always installed, even with none passed.
    assert callable(captured["capability_validator"])


@skip_no_qdp
def test_triton_op_aot_override_used(captured_registration):
    """A user-supplied aot_fn overrides the derived one."""
    captured = captured_registration
    sentinel = object()

    ttk.triton_op(
        "ttk_test::triton_override",
        kernel=object(),
        signature={"x_ptr": "*fp32", "y_ptr": "*fp32"},
        constexprs={},
        grid=lambda i, o: (1,),
        meta_fn=_identity_meta,
        aot_fn=lambda *a: sentinel,
    )

    assert captured["spec"].aot_fn is not None
    # The override is forwarded verbatim (not wrapped in the derived launcher).
    assert captured["spec"].aot_fn("i", "o", 0) is sentinel


# ---- No-GPU: signature validation (misuse must not reach the GPU) ----


def _sig_error(**overrides):
    """Call triton_op with a deliberately broken config, return the ValueError."""
    kwargs = dict(
        kernel=object(),
        signature={"x_ptr": "*fp32", "y_ptr": "*fp32"},
        constexprs={},
        grid=lambda i, o: (1,),
        meta_fn=_identity_meta,
    )
    kwargs.update(overrides)
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
def test_explicit_schema_drives_arity(captured_registration):
    """An explicit schema, not meta_fn hints, decides what the arity check sees."""

    def _meta(x, n):  # unannotated — hints alone would count both as tensors
        return torch.empty_like(x)

    ttk.triton_op(
        "ttk_test::triton_schema_arity",
        kernel=object(),
        signature={"x_ptr": "*fp32", "n": "i32", "y_ptr": "*fp32"},
        constexprs={},
        grid=lambda i, o: (1,),
        meta_fn=_meta,
        extra_args_fn=lambda i, o: [1],
        schema="(Tensor x, int n) -> Tensor",
    )
    assert captured_registration["schema"] == "(Tensor x, int n) -> Tensor"


def test_tensor_arity_prefers_schema_over_hints():
    from torch_tensorrt.kernels._register import tensor_arity

    def _meta(x, n):
        return torch.empty_like(x)

    # Unannotated params default to Tensor...
    assert tensor_arity(_meta) == (2, 1)
    # ...but an explicit schema is what actually gets registered.
    assert tensor_arity(_meta, "(Tensor x, int n) -> Tensor") == (1, 1)
    assert tensor_arity(_meta, "(Tensor x) -> (Tensor, Tensor)") == (1, 2)
    # An unparseable schema disables the check rather than failing registration.
    assert tensor_arity(_meta, "not a schema") is None


def test_analyze_signature_splits_on_convention():
    from torch_tensorrt.kernels._triton import analyze_signature

    layout = analyze_signature({"x_ptr": "*fp16", "n": "i32", "y_ptr": "*fp16"}, (1, 1))
    assert [p.name for p in layout.inputs] == ["x_ptr"]
    assert [p.name for p in layout.scalars] == ["n"]
    assert [p.name for p in layout.outputs] == ["y_ptr"]
    assert layout.inputs[0].dtype == torch.float16
    assert layout.outputs[0].dtype == torch.float16


def test_analyze_signature_tolerates_alignment_suffix_and_unknown_dtype():
    from torch_tensorrt.kernels._triton import analyze_signature

    layout = analyze_signature({"x_ptr": "*fp32:16", "y_ptr": "*weird"}, (1, 1))
    assert layout.inputs[0].dtype == torch.float32
    # Unknown element types disable checking rather than rejecting the kernel.
    assert layout.outputs[0].dtype is None


# ---- No-GPU: dtype capability validator ----


class _FakeNode:
    """Minimal stand-in for the torch.fx.Node a capability validator receives."""

    def __init__(self, arg_dtypes, out_dtype):
        self.args = [
            type("_Arg", (), {"meta": {"val": torch.empty(2, dtype=d)}})()
            for d in arg_dtypes
        ]
        self.meta = {"val": torch.empty(2, dtype=out_dtype)}


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


def test_dtype_validator_skips_unknown_dtypes():
    validate = _validator_for({"x_ptr": "*weird", "y_ptr": "*weird"})
    assert validate(_FakeNode([torch.float16], torch.bfloat16), None) is True


# ---- No-GPU: PTX ISA capping ----

# FAKE_PTX declares .version 9.3 and 5 entry params for this 3-param signature.
CAPPING_SIG = {"a": "*fp32", "b": "i32", "c": "*fp32"}


def _stub_triton(monkeypatch, requested, **extra_metadata):
    """Replace triton.compile with a stub recording each requested ptx_version."""
    from torch_tensorrt.kernels import _triton

    fake_metadata = type(
        "_M", (), {"name": "my_kernel", "num_warps": 4, "shared": 0, **extra_metadata}
    )()

    class _Compiled:
        asm = {"ptx": FAKE_PTX}
        metadata = fake_metadata

    class _FakeTriton:
        class compiler:
            ASTSource = staticmethod(lambda **kw: object())

        @staticmethod
        def compile(src, options=None):
            requested.append((options or {}).get("ptx_version"))
            return _Compiled()

    monkeypatch.setattr(_triton, "_triton_import", lambda: _FakeTriton)


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
    _stub_triton(monkeypatch, requested)

    _triton.compile_triton_to_ptx(object(), CAPPING_SIG, {})

    assert requested == expected


def test_nonzero_scratch_raises(monkeypatch):
    """Triton scratch buffers can't be fed by the AOT launcher — fail loudly."""
    from torch_tensorrt.kernels import _triton

    monkeypatch.setattr(_triton, "_driver_max_ptx_version", lambda: None)
    monkeypatch.setattr(_triton, "_triton_default_ptx_version", lambda: None)
    _stub_triton(monkeypatch, [], global_scratch_size=128, profile_scratch_size=0)

    with pytest.raises(RuntimeError, match="scratch memory"):
        _triton.compile_triton_to_ptx(object(), CAPPING_SIG, {})


@skip_no_qdp
def test_grid_beyond_three_dims_rejected(captured_registration):
    """TRT launches take at most grid_x/y/z; extra dims must not be dropped."""
    captured = captured_registration

    ttk.triton_op(
        "ttk_test::triton_grid4",
        kernel=object(),
        signature={"x_ptr": "*fp32", "y_ptr": "*fp32"},
        constexprs={},
        grid=lambda i, o: (1, 2, 3, 4),
        meta_fn=_identity_meta,
    )

    with pytest.raises(ValueError, match="4 dimension"):
        captured["spec"].aot_fn(["in"], ["out"], 0)


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

    register_once(_register)


@skip_no_cuda
@skip_no_qdp
@skip_no_triton
class TestTritonOpIntegration:
    def test_compile_produces_matched_param_count(self):
        """The compiled+processed PTX entry must have exactly len(signature) params."""
        from torch_tensorrt.kernels._triton import (
            _parse_entry_params,
            compile_triton_to_ptx,
        )

        sig = {"x_ptr": "*fp32", "n": "i32", "y_ptr": "*fp32"}
        ptx, name, num_warps, _shared = compile_triton_to_ptx(
            _ttk_add_one_kernel, sig, {"BLOCK": 256}
        )
        _, params = _parse_entry_params(ptx.decode("utf-8"), name)
        assert len(params) == len(sig)
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
        for size in [128, 512, 2048]:
            x = torch.randn(1, size, device="cuda")
            with torch.no_grad():
                assert torch.allclose(trt(x), x + 1, atol=1e-2, rtol=1e-2)
