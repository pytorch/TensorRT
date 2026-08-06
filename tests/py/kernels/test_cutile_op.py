"""Tests for cutile_op (cuTile kernel -> AOT QDP plugin path)."""

import pytest
import torch

import torch_tensorrt
import torch_tensorrt.kernels as ttk

from .conftest import (
    assert_ran_in_engine,
    compile_op,
    cutile_max_ptx_version,
    register_once,
    skip_no_cuda,
    skip_no_cutile,
    skip_no_qdp,
)

# A cuTile-shaped PTX entry: one rank-1 input array and one rank-1 output array,
# each expanded to (ptr, extent, stride). The body carries a vector-register
# ``{...}`` pair so the CUBIN extractor's brace matching is actually exercised.
FAKE_PTX = """//
.version 9.3
.target sm_100
.address_size 64

.visible .entry relu_kernel(
\t.param .u64 relu_kernel_param_0,
\t.param .u32 relu_kernel_param_1,
\t.param .u32 relu_kernel_param_2,
\t.param .u64 relu_kernel_param_3,
\t.param .u32 relu_kernel_param_4,
\t.param .u32 relu_kernel_param_5
)
.reqntid 128
{
\tmov.b64 {%r1, %r2}, %rd0;
\tret;
}
"""


def _fake_cubin(ptx: str = FAKE_PTX) -> bytes:
    """An ELF-shaped blob with the PTX stored the way cuTile stores it."""
    header = b"\x7fELF" + b"\x00" * 60
    body = b"\x00".join(line.encode("utf-8") for line in ptx.splitlines())
    return header + body + b"\x00\x00trailing-elf-junk"


# A complete, valid module the driver will JIT — used to exercise the ISA check
# without compiling a real kernel.
MINIMAL_PTX = (
    ".version 9.0\n.target sm_50\n.address_size 64\n"
    ".visible .entry k()\n{\n\tret;\n}\n"
)


def _identity_meta(x: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(x)


SIG_1IN_1OUT = {"x": "fp32", "out": "fp32"}


@pytest.fixture
def captured_registration(monkeypatch):
    """Stub compile + registration; returns the kwargs cutile_op forwards on."""
    from torch_tensorrt.kernels import _cutile, _register

    monkeypatch.setattr(
        _cutile,
        "compile_cutile_to_ptx",
        lambda *a, **k: (b"// ptx bytes", "relu_kernel", 128),
    )
    captured = {}
    monkeypatch.setattr(
        _register, "register_qdp_plugin", lambda *a, **k: captured.update(k)
    )
    return captured


# ---- No-GPU, no-cuda-tile: PTX post-processing helpers ----


def test_extract_ptx_from_cubin():
    from torch_tensorrt.kernels._cutile import extract_ptx_from_cubin

    ptx = extract_ptx_from_cubin(_fake_cubin())
    assert ptx is not None
    assert ptx.startswith(".version 9.3")
    # Brace matching, not a plain find('}'): the vector-register pair in the
    # body must not terminate the extraction early.
    assert "ret;" in ptx
    assert "trailing-elf-junk" not in ptx


def test_extract_ptx_from_cubin_rejects_non_elf():
    from torch_tensorrt.kernels._cutile import extract_ptx_from_cubin

    assert extract_ptx_from_cubin(b"not an elf") is None
    assert extract_ptx_from_cubin(b"\x7fELF" + b"\x00" * 200) is None


def test_parse_entry():
    from torch_tensorrt.kernels._cutile import parse_entry

    match, name, params = parse_entry(FAKE_PTX)
    assert match is not None
    assert name == "relu_kernel"
    assert len(params) == 6

    missing, name, params = parse_entry("// nothing here")
    assert missing is None and name == "" and params == []


def test_reorder_entry_params():
    from torch_tensorrt.kernels._cutile import parse_entry, reorder_entry_params

    # cuTile order (x_ptr, x_ext, x_str, out_ptr, out_ext, out_str) ->
    # TRT order (x_ptr, x_ext, x_str, out_ext, out_str, out_ptr).
    reordered = reorder_entry_params(FAKE_PTX, (0, 1, 2, 4, 5, 3))
    _, _, params = parse_entry(reordered)
    assert [p.split()[-1] for p in params] == [
        "relu_kernel_param_0",
        "relu_kernel_param_1",
        "relu_kernel_param_2",
        "relu_kernel_param_4",
        "relu_kernel_param_5",
        "relu_kernel_param_3",
    ]
    # Only the declaration list moves; the body is untouched.
    assert "mov.b64 {%r1, %r2}, %rd0;" in reordered


def test_reorder_entry_params_rejects_wrong_arity():
    from torch_tensorrt.kernels._cutile import reorder_entry_params

    with pytest.raises(RuntimeError, match="reorder expects"):
        reorder_entry_params(FAKE_PTX, (0, 1, 2))
    with pytest.raises(RuntimeError, match="no '.entry'"):
        reorder_entry_params("// no entry", ())


def test_parse_reqntid():
    from torch_tensorrt.kernels._cutile import parse_reqntid

    assert parse_reqntid(FAKE_PTX) == 128
    assert parse_reqntid("// none") is None


def test_ptx_version_helpers():
    from torch_tensorrt.kernels._cutile import (
        cap_ptx_version,
        parse_ptx_version,
        set_ptx_version,
    )

    assert parse_ptx_version(FAKE_PTX) == 93
    assert parse_ptx_version("// no version") is None
    assert parse_ptx_version(set_ptx_version(FAKE_PTX, 90)) == 90
    # Only lowered when the emitted ISA is newer than the driver accepts.
    # Purely textual: lower only when the emitted ISA exceeds the ceiling.
    assert parse_ptx_version(cap_ptx_version(FAKE_PTX, 90)) == 90
    assert parse_ptx_version(cap_ptx_version(FAKE_PTX, 95)) == 93
    assert cap_ptx_version("// no version", 90) == "// no version"


def test_driver_check_needs_no_triton(monkeypatch):
    """The ISA check must not depend on Triton, which torch-tensorrt lacks.

    Triton is not a declared dependency anywhere in torch-tensorrt; it only
    happens to be present because PyTorch pulls it in on Linux. Relying on it
    would silently disable the check, and an ISA the driver refuses surfaces as
    an opaque onShapeChange failure at engine runtime.
    """
    import builtins

    from torch_tensorrt.kernels import _cutile

    real_import = builtins.__import__

    def _no_triton(name, *args, **kwargs):
        if name.split(".")[0] == "triton":
            raise ImportError("triton is not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_triton)
    try:
        _cutile.verify_driver_accepts_ptx("ns::op", "k", MINIMAL_PTX)
    finally:
        monkeypatch.undo()


@skip_no_cuda
def test_loadable_ptx_is_accepted():
    from torch_tensorrt.kernels._cutile import verify_driver_accepts_ptx

    assert verify_driver_accepts_ptx("ns::op", "k", MINIMAL_PTX) is None


@skip_no_cuda
def test_too_new_isa_raises_instead_of_being_downgraded():
    """A refused ISA is an environment mismatch, reported rather than patched.

    Rewriting the .version header over a body the compiler emitted for a newer
    ISA may or may not assemble depending on which instructions it contains, so
    it would turn a clear driver/toolchain gap into a conditional one.
    """
    from torch_tensorrt.kernels._cutile import (
        set_ptx_version,
        verify_driver_accepts_ptx,
    )

    too_new = set_ptx_version(MINIMAL_PTX, 129)  # 12.9: no driver accepts this
    with pytest.raises(RuntimeError, match="too old to load"):
        verify_driver_accepts_ptx("ns::op", "k", too_new)


@skip_no_cuda
def test_invalid_ptx_is_reported_distinctly_from_a_version_gap():
    """Malformed PTX must not be described as a driver-too-old problem."""
    from torch_tensorrt.kernels._cutile import verify_driver_accepts_ptx

    with pytest.raises(RuntimeError, match="not merely a version gap"):
        verify_driver_accepts_ptx("ns::op", "k", ".version 9.0\n.visible .entry k( x")


@skip_no_cuda
def test_max_ptx_version_still_lowers_the_header_when_asked():
    """The explicit opt-out stays available for callers who know it is safe."""
    from torch_tensorrt.kernels._cutile import cap_ptx_version, parse_ptx_version

    assert parse_ptx_version(cap_ptx_version(set_ptx_version_(129), 90)) == 90


def set_ptx_version_(version):
    from torch_tensorrt.kernels._cutile import set_ptx_version

    return set_ptx_version(MINIMAL_PTX, version)


# ---- No-GPU: parameter permutation ----


SIG_2IN_1OUT = {"a": "fp32", "b": "fp32", "out": "fp32"}


def _order_for(signature, arity, ndim=1):
    from torch_tensorrt.kernels._cutile import cutile_param_order

    return cutile_param_order(_validate(signature, arity=arity, ndim=ndim))


@pytest.mark.parametrize(
    "signature, arity, ndim, expected",
    [
        # (ptr, extent, stride) per array; the output pointer moves to last.
        (SIG_1IN_1OUT, (1, 1), 1, (0, 1, 2, 4, 5, 3)),
        # Both input pointers first, then all extents/strides, then the output.
        (SIG_2IN_1OUT, (2, 1), 1, (0, 3, 1, 2, 4, 5, 7, 8, 6)),
        # Rank 2 gives each array two extents and two strides.
        (SIG_1IN_1OUT, (1, 1), 2, (0, 1, 2, 3, 4, 6, 7, 8, 9, 5)),
    ],
)
def test_param_order(signature, arity, ndim, expected):
    assert _order_for(signature, arity, ndim) == expected


def test_param_order_is_a_permutation_with_mixed_ranks():
    order = _order_for({"a": "fp32", "b": "fp32[2]", "out": "fp32"}, (2, 1))
    assert sorted(order) == list(range(len(order)))


# ---- No-GPU: signature validation ----


def _validate(signature, arity=(1, 1), constants=None, ndim=1, **kwargs):
    from torch_tensorrt.kernels._cutile import validate_cutile_config

    return validate_cutile_config(
        "ns::op", signature, constants or {}, arity, ndim, **kwargs
    )


def test_signature_splits_into_inputs_and_outputs():
    layout = _validate({"a": "fp32", "b": "fp16", "out": "bf16"}, arity=(2, 1))
    assert [p.name for p in layout.inputs] == ["a", "b"]
    assert [p.name for p in layout.outputs] == ["out"]
    assert [p.dtype for p in layout.inputs] == [torch.float32, torch.float16]
    assert layout.outputs[0].dtype == torch.bfloat16


def test_signature_accepts_torch_dtypes_and_rank_suffix():
    layout = _validate({"x": torch.int32, "out": "float32[3]"}, arity=(1, 1))
    assert layout.inputs[0].dtype == torch.int32
    assert layout.inputs[0].ndim == 1
    assert layout.outputs[0].dtype == torch.float32
    assert layout.outputs[0].ndim == 3


@pytest.mark.parametrize(
    "kwargs, message",
    [
        # The signature must describe exactly the op's tensors...
        (dict(signature=SIG_1IN_1OUT, arity=(2, 1)), "2 tensor input"),
        (dict(signature={}, arity=(0, 0)), "signature is empty"),
        # ...name a dtype cuTile can actually be compiled for...
        (dict(signature={"x": "weird", "out": "fp32"}), "unknown element type"),
        # ...and keep arrays and ct.Constant parameters apart.
        (
            dict(signature=SIG_1IN_1OUT, constants={"x": 128}),
            "both signature and constants",
        ),
        (
            dict(signature=SIG_1IN_1OUT, constants={"tile_size": 1.5}),
            "only int values",
        ),
        (
            dict(signature=SIG_1IN_1OUT, constants={"tile_size": True}),
            "only int values",
        ),
        (dict(signature=SIG_1IN_1OUT, ndim=0), "rank >= 1"),
        # Exactly one of grid= / aot_fn= builds the launch.
        (dict(signature=SIG_1IN_1OUT, has_grid=False), "needs a grid="),
        (
            dict(signature=SIG_1IN_1OUT, derived_launch=False, has_grid=True),
            "both grid= and aot_fn=",
        ),
        # An unreadable arity must ask for schema=, not guess the split.
        (dict(signature=SIG_1IN_1OUT, arity=None), "Pass schema="),
    ],
)
def test_invalid_configurations_are_rejected(kwargs, message):
    with pytest.raises(ValueError, match=message):
        _validate(**kwargs)


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
    from torch_tensorrt.kernels._cutile import make_dtype_capability_validator

    layout = _validate(signature)
    return make_dtype_capability_validator("ns::op", layout, user_validator)


def test_dtype_validator_accepts_matching_dtypes():
    validate = _validator_for(SIG_1IN_1OUT)
    assert validate(_FakeNode([torch.float32], torch.float32), None) is True


def test_dtype_validator_rejects_mismatched_input():
    """fp16 into an fp32-compiled kernel would otherwise return silent garbage."""
    validate = _validator_for(SIG_1IN_1OUT)
    assert validate(_FakeNode([torch.float16], torch.float16), None) is False


def test_dtype_validator_rejects_mismatched_output():
    validate = _validator_for({"x": "fp32", "out": "fp16"})
    assert validate(_FakeNode([torch.float32], torch.float32), None) is False


def test_dtype_validator_composes_with_user_validator():
    validate = _validator_for(SIG_1IN_1OUT, user_validator=lambda n, s: False)
    assert validate(_FakeNode([torch.float32], torch.float32), None) is False


# ---- No-GPU: cutile_op plumbing (compile + register mocked out) ----


@skip_no_qdp
def test_cutile_op_forwards_to_registrar(captured_registration):
    """cutile_op must compile, then hand the PTX to the ptx_op funnel."""
    captured = captured_registration
    ttk.cutile_op(
        "ttk_test::cutile_forward",
        kernel=object(),
        signature=SIG_1IN_1OUT,
        grid=lambda inputs, outputs: (1,),
        meta_fn=_identity_meta,
        constants={"tile_size": 128},
        supports_dynamic_shapes=True,
    )

    assert captured["op_name"] == "ttk_test::cutile_forward"
    assert captured["precompiled_ptx"] == b"// ptx bytes"
    assert captured["spec"].kernel_name == "relu_kernel"
    assert captured["supports_dynamic_shapes"] is True
    # A dtype capability validator is always installed, even with none passed.
    assert callable(captured["capability_validator"])


@skip_no_qdp
def test_cutile_op_aot_override_used(captured_registration):
    """A user-supplied aot_fn overrides the derived one."""
    captured = captured_registration
    sentinel = object()

    ttk.cutile_op(
        "ttk_test::cutile_override",
        kernel=object(),
        signature=SIG_1IN_1OUT,
        meta_fn=_identity_meta,
        aot_fn=lambda *a: sentinel,
    )

    assert captured["spec"].aot_fn("i", "o", 0) is sentinel


@skip_no_qdp
def test_block_size_conflicting_with_reqntid_rejected(captured_registration):
    """.reqntid is a hard requirement — a different block_size must not be honored."""
    with pytest.raises(ValueError, match="reqntid"):
        ttk.cutile_op(
            "ttk_test::cutile_block_conflict",
            kernel=object(),
            signature=SIG_1IN_1OUT,
            grid=lambda i, o: (1,),
            meta_fn=_identity_meta,
            block_size=256,
        )


@skip_no_qdp
def test_missing_reqntid_requires_block_size(monkeypatch):
    from torch_tensorrt.kernels import _cutile, _register

    monkeypatch.setattr(
        _cutile, "compile_cutile_to_ptx", lambda *a, **k: (b"ptx", "k", None)
    )
    monkeypatch.setattr(_register, "register_qdp_plugin", lambda *a, **k: None)

    with pytest.raises(ValueError, match="block_size"):
        ttk.cutile_op(
            "ttk_test::cutile_no_reqntid",
            kernel=object(),
            signature=SIG_1IN_1OUT,
            grid=lambda i, o: (1,),
            meta_fn=_identity_meta,
        )


@skip_no_qdp
def test_grid_beyond_three_dims_rejected(captured_registration):
    """TRT launches take at most grid_x/y/z; extra dims must not be dropped."""
    captured = captured_registration

    ttk.cutile_op(
        "ttk_test::cutile_grid4",
        kernel=object(),
        signature=SIG_1IN_1OUT,
        grid=lambda i, o: (1, 2, 3, 4),
        meta_fn=_identity_meta,
    )

    with pytest.raises(ValueError, match="4 dimension"):
        captured["spec"].aot_fn(["in"], ["out"], 0)


# ---- PTX parameter-count diagnosis (no cuda-tile needed) ----


def test_param_count_mismatch_message_suggests_rank():
    from torch_tensorrt.kernels._cutile import _diagnose_param_count

    layout = _validate(SIG_1IN_1OUT)
    # 10 params over 2 arrays = 5 per array = rank 2.
    assert "ndim=2" in _diagnose_param_count(layout, 10)


def test_param_count_mismatch_message_flags_runtime_scalars():
    from torch_tensorrt.kernels._cutile import _diagnose_param_count

    layout = _validate(SIG_1IN_1OUT)
    assert "ct.Constant" in _diagnose_param_count(layout, 7)


# ---- AOT extra arguments ----
#
# The real trtp.SymInt32 only supports arithmetic inside a live plugin's
# expression builder, so these substitute an int-backed stand-in and check the
# extents / strides / ordering the launch would be given.


class _FakeShapeExpr(list):
    def numel(self):
        total = 1
        for dim in self:
            total *= dim
        return total


class _FakeDesc:
    def __init__(self, *shape):
        self.shape_expr = _FakeShapeExpr(shape)


class _FakeSymInt32(int):
    def __mul__(self, other):
        return _FakeSymInt32(int(self) * int(other))


class _FakeTrtp:
    SymInt32 = _FakeSymInt32

    @staticmethod
    def SymIntExprs(count):
        return [None] * count


@pytest.fixture
def stub_trtp(monkeypatch):
    from torch_tensorrt.kernels import _cutile

    monkeypatch.setattr(_cutile, "_trtp", lambda: _FakeTrtp)
    return _cutile


def test_extra_args_rank1_flattens(stub_trtp):
    layout = _validate(SIG_1IN_1OUT)
    extent, stride = stub_trtp._extents_and_strides(_FakeDesc(4, 256), layout.inputs[0])
    assert (int(extent), int(stride)) == (1024, 1)


def test_extra_args_rank2_is_row_major(stub_trtp):
    layout = _validate(SIG_1IN_1OUT, ndim=2)
    values = stub_trtp._extents_and_strides(_FakeDesc(4, 256), layout.inputs[0])
    assert [int(v) for v in values] == [4, 256, 256, 1]


def test_extra_args_rank_mismatch_rejected(stub_trtp):
    layout = _validate(SIG_1IN_1OUT, ndim=2)
    with pytest.raises(ValueError, match="rank 2"):
        stub_trtp._extents_and_strides(_FakeDesc(4, 8, 16), layout.inputs[0])


def test_extra_args_rejects_tensor_count_mismatch(stub_trtp):
    """A short/long tensor list must raise, not silently emit too few extras.

    Zipping would truncate to the shorter list and leave the kernel reading
    whatever occupied the unfilled parameter slots — the exact silent misbind
    this module exists to prevent.
    """
    layout = _validate({"a": "fp32", "b": "fp32", "out": "fp32"}, arity=(2, 1))
    with pytest.raises(RuntimeError, match="1 input tensor.*describes 2"):
        stub_trtp.build_extra_args([_FakeDesc(8)], [_FakeDesc(8)], layout)
    with pytest.raises(RuntimeError, match="2 output tensor.*describes 1"):
        stub_trtp.build_extra_args(
            [_FakeDesc(8), _FakeDesc(8)], [_FakeDesc(8), _FakeDesc(8)], layout
        )


def test_extra_args_order_is_inputs_then_outputs(stub_trtp):
    layout = _validate({"a": "fp32", "b": "fp32", "out": "fp32"}, arity=(2, 1))
    extra = stub_trtp.build_extra_args(
        [_FakeDesc(2, 4), _FakeDesc(8)], [_FakeDesc(8)], layout
    )
    # (extent, stride) per array, inputs first — six slots for three rank-1
    # arrays, and rank 1 collapses the (2, 4) input to its element count.
    assert [int(v) for v in extra] == [8, 1, 8, 1, 8, 1]


# ---- GPU integration: real cuTile kernels through cutile_op ----

TILE = 128

try:
    import cuda.tile as ct

    @ct.kernel
    def _ttk_add_one_kernel(x, out, tile_size: ct.Constant[int]):
        pid = ct.bid(0)
        tile = ct.load(x, index=(pid,), shape=(tile_size,))
        ct.store(out, index=(pid,), tile=tile + 1.0)

    @ct.kernel
    def _ttk_reglu_kernel(gate, up, out, tile_size: ct.Constant[int]):
        pid = ct.bid(0)
        g = ct.load(gate, index=(pid,), shape=(tile_size,))
        u = ct.load(up, index=(pid,), shape=(tile_size,))
        ct.store(out, index=(pid,), tile=ct.maximum(g, 0.0) * u)

    @ct.kernel
    def _ttk_scale2d_kernel(x, out, tile_m: ct.Constant[int], tile_n: ct.Constant[int]):
        m, n = ct.bid(0), ct.bid(1)
        tile = ct.load(x, index=(m, n), shape=(tile_m, tile_n))
        ct.store(out, index=(m, n), tile=tile * 2.0)

except ImportError:
    ct = None


def _register_add_one(
    op_name: str, tile_size: int = TILE, with_eager: bool = True
) -> None:
    import tensorrt.plugin as trtp

    def _meta(x: torch.Tensor) -> torch.Tensor:
        return torch.empty_like(x)

    def _eager(x: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(x)
        flat_x = x.contiguous().reshape(-1)
        flat_out = out.reshape(-1)
        ct.launch(
            torch.cuda.current_stream().cuda_stream,
            (ct.cdiv(flat_x.numel(), tile_size), 1, 1),
            _ttk_add_one_kernel,
            (flat_x, flat_out, tile_size),
        )
        return out

    register_once(
        lambda: ttk.cutile_op(
            op_name,
            kernel=_ttk_add_one_kernel,
            signature={"x": "fp32", "out": "fp32"},
            grid=lambda inputs, outputs: (
                trtp.cdiv(inputs[0].shape_expr.numel(), tile_size),
            ),
            meta_fn=_meta,
            constants={"tile_size": tile_size},
            eager_fn=_eager if with_eager else None,
            supports_dynamic_shapes=True,
            max_ptx_version=cutile_max_ptx_version(),
        )
    )


def _register_reglu(op_name: str) -> None:
    import tensorrt.plugin as trtp

    def _meta(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        return torch.empty_like(gate)

    register_once(
        lambda: ttk.cutile_op(
            op_name,
            kernel=_ttk_reglu_kernel,
            signature={"gate": "fp32", "up": "fp32", "out": "fp32"},
            grid=lambda inputs, outputs: (
                trtp.cdiv(inputs[0].shape_expr.numel(), TILE),
            ),
            meta_fn=_meta,
            constants={"tile_size": TILE},
            supports_dynamic_shapes=True,
            max_ptx_version=cutile_max_ptx_version(),
        )
    )


def _register_scale2d(op_name: str, tile_m: int = 16, tile_n: int = 64) -> None:
    import tensorrt.plugin as trtp

    def _meta(x: torch.Tensor) -> torch.Tensor:
        return torch.empty_like(x)

    register_once(
        lambda: ttk.cutile_op(
            op_name,
            kernel=_ttk_scale2d_kernel,
            signature={"x": "fp32", "out": "fp32"},
            grid=lambda inputs, outputs: (
                trtp.cdiv(inputs[0].shape_expr[0], tile_m),
                trtp.cdiv(inputs[0].shape_expr[1], tile_n),
            ),
            meta_fn=_meta,
            constants={"tile_m": tile_m, "tile_n": tile_n},
            ndim=2,
            supports_dynamic_shapes=True,
            max_ptx_version=cutile_max_ptx_version(),
        )
    )


@skip_no_cuda
@skip_no_qdp
@skip_no_cutile
class TestCuTileOpIntegration:
    def test_compile_reorders_entry_into_trt_launch_order(self):
        """The compiled PTX entry must be permuted into TRT's argument order.

        cuTile emits ``(x_ptr, x_ext, x_str, out_ptr, out_ext, out_str)``; TRT
        launches with the output pointer last, after the extra arguments. This
        pins the reorder that makes the two agree — the one thing that produces
        wrong numbers rather than an error when it is off.
        """
        from torch_tensorrt.kernels._cutile import (
            compile_cutile_to_ptx,
            parse_entry,
            validate_cutile_config,
        )

        layout = validate_cutile_config(
            "ns::op", {"x": "fp32", "out": "fp32"}, {"tile_size": TILE}, (1, 1), 1
        )
        ptx, name, reqntid = compile_cutile_to_ptx(
            "ns::op",
            _ttk_add_one_kernel,
            layout,
            {"tile_size": TILE},
            max_ptx_version=cutile_max_ptx_version(),
        )
        _, entry_name, params = parse_entry(ptx.decode("utf-8"))
        assert entry_name == name
        # Two rank-1 arrays: (ptr, extent, stride) each.
        assert len(params) == 6
        assert reqntid is not None and reqntid >= 1

        # cuTile names params ``<kernel>_param_<declaration index>``, so the
        # trailing digits spell out the permutation that was applied.
        assert [p.split("_param_")[-1] for p in params] == [
            "0",
            "1",
            "2",
            "4",
            "5",
            "3",
        ]
        # The two pointer params bracket the extras: input ptr first, output last.
        assert ".u64" in params[0] and ".u64" in params[-1]
        assert all(".u32" in p for p in params[1:-1])

    def test_register_and_eager(self):
        _register_add_one("ttk_test::cutile_add_one_eager")
        x = torch.randn(1024, device="cuda")
        assert torch.allclose(
            torch.ops.ttk_test.cutile_add_one_eager(x), x + 1, atol=1e-4, rtol=1e-4
        )

    def test_trt_compile_static_shapes(self):
        op = "ttk_test::cutile_add_one_static"
        _register_add_one(op)
        x = torch.randn(4, 256, device="cuda")
        trt = compile_op(op, [x])
        assert_ran_in_engine(trt, op)
        with torch.no_grad():
            assert torch.allclose(trt(x), x + 1, atol=1e-4, rtol=1e-4)

    def test_runs_in_engine_without_an_eager_impl(self):
        """Register with no eager_fn: falling back to PyTorch cannot even run.

        Removing the fallback is what makes a passing result mean the cuTile
        kernel executed inside the engine — with an eager_fn present, a declined
        op returns the same numbers and the assertion proves nothing.
        """
        op = "ttk_test::cutile_add_one_trt_only"
        _register_add_one(op, with_eager=False)
        x = torch.randn(4, 256, device="cuda")
        trt = compile_op(op, [x])
        assert_ran_in_engine(trt, op)
        with torch.no_grad():
            assert torch.equal(trt(x), x + 1)

    def test_two_inputs_bind_in_the_right_order(self):
        """ReGLU: relu(gate) * up is asymmetric, so swapped pointers show up."""
        op = "ttk_test::cutile_reglu"
        _register_reglu(op)
        gate = torch.randn(4, 256, device="cuda")
        up = torch.randn(4, 256, device="cuda")
        trt = compile_op(op, [gate, up])
        assert_ran_in_engine(trt, op)
        with torch.no_grad():
            assert torch.allclose(
                trt(gate, up), torch.relu(gate) * up, atol=1e-5, rtol=1e-5
            )

    def test_rank2_extents_and_strides(self):
        """A rank-2 kernel indexes real 2-D tiles, so extents/strides must be right."""
        op = "ttk_test::cutile_scale2d"
        _register_scale2d(op)
        x = torch.randn(64, 256, device="cuda")
        trt = compile_op(op, [x])
        assert_ran_in_engine(trt, op)
        with torch.no_grad():
            assert torch.allclose(trt(x), x * 2, atol=1e-5, rtol=1e-5)

    def test_trt_compile_dynamic_shapes(self):
        op = "ttk_test::cutile_add_one_dyn"
        _register_add_one(op)
        trt = compile_op(
            op,
            [
                torch_tensorrt.Input(
                    min_shape=(1, 128),
                    opt_shape=(1, 512),
                    max_shape=(1, 2048),
                    dtype=torch.float32,
                )
            ],
        )
        assert_ran_in_engine(trt, op)
        for size in [128, 512, 2048]:
            x = torch.randn(1, size, device="cuda")
            with torch.no_grad():
                assert torch.allclose(trt(x), x + 1, atol=1e-4, rtol=1e-4)

    def test_dtype_mismatch_falls_back_instead_of_returning_garbage(self):
        """fp16 into an fp32-compiled kernel must not silently produce nonsense."""
        _register_add_one("ttk_test::cutile_add_one_dtype")
        x = torch.randn(4, 256, device="cuda", dtype=torch.float16)
        trt = compile_op(
            "ttk_test::cutile_add_one_dtype",
            [x],
            enabled_precisions={torch.float16},
        )
        # The mirror image of _assert_ran_in_engine: the op must have been
        # declined and left in the graph rather than compiled into the engine
        # against a kernel that would reinterpret its half-precision bytes.
        assert any(
            node.op == "call_function" and "cutile_add_one_dtype" in str(node.target)
            for node in trt.graph.nodes
        ), "the fp16 op was lowered to an fp32-compiled plugin"
        with torch.no_grad():
            # Declined, so this runs in PyTorch via eager_fn — and is correct.
            assert torch.allclose(trt(x), x + 1, atol=1e-2, rtol=1e-2)
