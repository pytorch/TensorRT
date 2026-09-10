"""Tests for cutile_op (cuTile kernel -> AOT QDP plugin path)."""

import struct
from types import SimpleNamespace

import pytest
import torch

import torch_tensorrt
import torch_tensorrt.kernels as ttk

from .conftest import (
    assert_ran_in_engine,
    compile_op,
    register_once,
    skip_no_cuda,
    skip_no_cutile,
    skip_no_qdp,
)

SIG_1IN_1OUT = {"x": "fp32", "out": "fp32"}
SIG_2IN_1OUT = {"a": "fp32", "b": "fp32", "out": "fp32"}


FAKE_PTX = """//
.version 9.3
.target sm_100
.address_size 64

.visible .entry relu_kernel(
    .param .u64 relu_kernel_param_0,
    .param .u32 relu_kernel_param_1,
    .param .u32 relu_kernel_param_2,
    .param .u64 relu_kernel_param_3,
    .param .u32 relu_kernel_param_4,
    .param .u32 relu_kernel_param_5
)
.reqntid 128, 1, 1
{
    mov.b64 {%r1, %r2}, %rd0;
    ret;
}
"""


def _fake_cubin(ptx=FAKE_PTX, *, duplicate_ptx_section=False):
    """Build the small ELF64 subset the extractor reads."""
    section_names = [b".shstrtab", b".nv_debug_ptx_txt"]
    if duplicate_ptx_section:
        section_names.append(b".nv_debug_ptx_txt")

    names = b"\x00"
    name_offsets = []
    for name in section_names:
        name_offsets.append(len(names))
        names += name + b"\x00"

    ptx_data = (
        b"\x00" * 8 + b"\x00".join(line.encode() for line in ptx.splitlines()) + b"\x00"
    )
    payloads = [names] + [ptx_data] * (len(section_names) - 1)
    offsets = []
    cursor = 64
    for payload in payloads:
        offsets.append(cursor)
        cursor += len(payload)
    section_table_offset = cursor

    ident = b"\x7fELF\x02\x01\x01" + b"\x00" * 9
    header = struct.pack(
        "<16sHHIQQQIHHHHHH",
        ident,
        2,
        190,
        1,
        0,
        0,
        section_table_offset,
        0,
        64,
        0,
        0,
        64,
        len(section_names) + 1,
        1,
    )
    null_header = b"\x00" * 64
    section_headers = [
        struct.pack(
            "<IIQQQQIIQQ",
            name_offset,
            3 if index == 0 else 1,
            0,
            0,
            offsets[index],
            len(payloads[index]),
            0,
            0,
            1,
            0,
        )
        for index, name_offset in enumerate(name_offsets)
    ]
    return header + b"".join(payloads) + null_header + b"".join(section_headers)


def _validate(signature, arity=(1, 1), constants=None, ndim=1):
    from torch_tensorrt.kernels._cutile import validate_cutile_config

    values = {} if constants is None else constants
    return validate_cutile_config("ns::op", signature, values, arity, ndim)


def _identity_meta(x: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(x)


# ---- PTX extraction and ABI rewriting ----


def test_extract_ptx_from_cubin_preserves_complete_entry():
    from torch_tensorrt.kernels._cutile import extract_ptx_from_cubin

    ptx = extract_ptx_from_cubin(_fake_cubin())
    assert ptx is not None
    assert ptx.startswith(".version 9.3")
    assert "ret;" in ptx  # the vector-register braces did not end it early
    assert "trailing-elf-junk" not in ptx


@pytest.mark.parametrize(
    "blob",
    [
        b"not an elf",
        b"\x7fELF" + b"\x00" * 200,
        _fake_cubin(duplicate_ptx_section=True),
    ],
)
def test_extract_ptx_from_cubin_rejects_invalid_or_ambiguous_elf(blob):
    from torch_tensorrt.kernels._cutile import extract_ptx_from_cubin

    assert extract_ptx_from_cubin(blob) is None


def test_extract_ptx_preserves_module_helpers_around_entry():
    from torch_tensorrt.kernels._cutile import extract_ptx_from_cubin

    ptx = FAKE_PTX.replace(
        ".visible .entry", ".func helper_before() { ret; }\n.visible .entry"
    )
    ptx += ".func helper_after() { ret; }\n"
    extracted = extract_ptx_from_cubin(_fake_cubin(ptx))
    assert extracted is not None
    assert ".func helper_before" in extracted
    assert ".func helper_after" in extracted


def test_parse_and_reorder_single_entry():
    from torch_tensorrt.kernels._cutile import parse_entry, reorder_entry_params

    match, name, params = parse_entry(FAKE_PTX)
    assert match is not None and name == "relu_kernel" and len(params) == 6

    reordered = reorder_entry_params(FAKE_PTX, (0, 1, 2, 4, 5, 3))
    _, _, params = parse_entry(reordered)
    assert [param.split()[-1] for param in params] == [
        "relu_kernel_param_0",
        "relu_kernel_param_1",
        "relu_kernel_param_2",
        "relu_kernel_param_4",
        "relu_kernel_param_5",
        "relu_kernel_param_3",
    ]
    assert "mov.b64 {%r1, %r2}, %rd0;" in reordered


def test_entry_parser_ignores_comments_and_accepts_unqualified_entry():
    from torch_tensorrt.kernels._cutile import parse_entry

    ptx = "/* .visible .entry fake(.param .u64 x) */\n" + FAKE_PTX.replace(
        ".visible .entry", ".entry"
    )
    match, name, params = parse_entry(ptx)
    assert match is not None
    assert name == "relu_kernel"
    assert len(params) == 6

    _, name, _ = parse_entry(FAKE_PTX.replace("relu_kernel", "%relu_kernel"))
    assert name == "%relu_kernel"


def test_entry_rewrite_rejects_missing_ambiguous_or_wrong_arity():
    from torch_tensorrt.kernels._cutile import parse_entry, reorder_entry_params

    assert parse_entry(FAKE_PTX + FAKE_PTX) == (None, "", [])
    with pytest.raises(RuntimeError, match="reorder expects"):
        reorder_entry_params(FAKE_PTX, (0, 1, 2))
    with pytest.raises(RuntimeError, match="no '.entry'"):
        reorder_entry_params("// no entry", ())


def test_ptx_metadata_helpers():
    from torch_tensorrt.kernels._cutile import (
        cap_ptx_version,
        parse_ptx_version,
        parse_reqntid,
        set_ptx_version,
    )

    assert parse_reqntid(FAKE_PTX) == (128, 1, 1)
    assert parse_reqntid(FAKE_PTX.replace(".reqntid 128, 1, 1", ".reqntid 8, 4")) == (
        8,
        4,
        1,
    )
    assert parse_reqntid(
        FAKE_PTX.replace(".reqntid 128, 1, 1", ".reqntid 0x20, 0x4, 1")
    ) == (32, 4, 1)
    assert parse_reqntid(FAKE_PTX.replace(".reqntid 128, 1, 1", ".reqntid 032")) == (
        26,
        1,
        1,
    )
    assert parse_reqntid(
        FAKE_PTX.replace(".reqntid 128, 1, 1", ".reqntid 0b100000")
    ) == (32, 1, 1)
    assert parse_reqntid("// none") is None
    assert parse_reqntid("// .reqntid 7\n" + FAKE_PTX) == (128, 1, 1)
    assert parse_reqntid("/* .reqntid 7 */\n" + FAKE_PTX) == (128, 1, 1)
    assert parse_ptx_version(FAKE_PTX) == 93
    assert parse_ptx_version(set_ptx_version(FAKE_PTX, 90)) == 90
    assert parse_ptx_version(cap_ptx_version(FAKE_PTX, 90)) == 90
    assert parse_ptx_version(cap_ptx_version(FAKE_PTX, 95)) == 93


@pytest.mark.parametrize(
    "directive",
    [".maxntid", ".explicitcluster", ".reqnctapercluster", ".maxclusterrank"],
)
def test_unsupported_launch_directives_fail_closed(directive):
    from torch_tensorrt.kernels._cutile import unsupported_launch_directive

    assert unsupported_launch_directive(f"// {directive}\n{FAKE_PTX}") is None
    ptx = FAKE_PTX.replace(".reqntid", f"{directive} 1\n.reqntid")
    assert unsupported_launch_directive(ptx) == directive


def test_driver_verification_fails_closed(monkeypatch):
    from cuda.bindings import driver as cuda

    from torch_tensorrt.kernels import _cutile

    monkeypatch.setattr(
        _cutile,
        "_load_ptx",
        lambda ptx, kernel_name: cuda.CUresult.CUDA_ERROR_INVALID_PTX,
    )
    with pytest.raises(RuntimeError, match="could not load PTX ISA 9.3"):
        _cutile.verify_driver_accepts_ptx("ns::op", "k", FAKE_PTX)

    monkeypatch.setattr(
        _cutile,
        "_load_ptx",
        lambda ptx, kernel_name: cuda.CUresult.CUDA_SUCCESS,
    )
    assert _cutile.verify_driver_accepts_ptx("ns::op", "k", FAKE_PTX) is None

    def _broken(_ptx, _kernel_name):
        raise RuntimeError("driver unavailable")

    monkeypatch.setattr(_cutile, "_load_ptx", _broken)
    with pytest.raises(RuntimeError, match="refusing to embed unchecked code"):
        _cutile.verify_driver_accepts_ptx("ns::op", "k", FAKE_PTX)


def test_driver_verification_applies_to_native_arch_override(monkeypatch):
    from torch_tensorrt.kernels import _cutile

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(_cutile, "_default_arch", lambda: "sm_100")
    assert _cutile._should_verify_ptx(None)
    assert _cutile._should_verify_ptx("sm_100")
    assert not _cutile._should_verify_ptx("sm_90")


# ---- The ABI permutation, and the extra arguments that must match it ----


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
    from torch_tensorrt.kernels._cutile import cutile_param_order

    assert cutile_param_order(_validate(signature, arity, ndim=ndim)) == expected


class _FakeShapeExpr(list):
    def numel(self):
        total = 1
        for dim in self:
            total *= dim
        return total


class _FakeDesc:
    def __init__(self, *shape):
        self.shape_expr = _FakeShapeExpr(shape)


class _FakeSymIntExpr:
    def __init__(self, value=None):
        self.value = None if value is None else int(value)

    def __mul__(self, other):
        assert self.value is not None
        return _FakeSymInt32(self.value * int(other))

    def __int__(self):
        assert self.value is not None
        return self.value

    @property
    def is_constant(self):
        return self.value is not None

    def constant_value(self):
        assert self.value is not None
        return self.value


class _FakeSymInt32(_FakeSymIntExpr):
    pass


class _FakeSymIntExprs(list):
    def __init__(self, count):
        super().__init__([None] * count)


class _FakeTrtp:
    """Stand-in for tensorrt.plugin: the real SymInt32 only does arithmetic
    inside a live plugin's expression builder."""

    SymInt32 = _FakeSymInt32

    SymIntExprs = _FakeSymIntExprs

    class KernelLaunchParams:
        pass


@pytest.fixture
def stub_trtp(monkeypatch):
    from torch_tensorrt.kernels import _cutile

    monkeypatch.setattr(_cutile, "_trtp", lambda: _FakeTrtp)
    return _cutile


def test_extra_args_match_the_permutation(stub_trtp):
    """Extents and strides must fill the slots the permutation routes them to.

    Rank 1 is a flattened view, so its extent is the element count whatever the
    tensor's shape; rank 2 maps dimension for dimension with row-major strides.
    Inputs come before outputs, matching cutile_param_order.
    """
    layout = _validate(SIG_2IN_1OUT, arity=(2, 1))
    extra = stub_trtp.build_extra_args(
        [_FakeDesc(2, 4), _FakeDesc(8)], [_FakeDesc(8)], layout
    )
    assert [int(v) for v in extra] == [8, 1, 8, 1, 8, 1]

    rank2 = _validate(SIG_1IN_1OUT, ndim=2)
    values = stub_trtp._extents_and_strides(_FakeDesc(4, 256), rank2.inputs[0])
    assert [int(v) for v in values] == [4, 256, 256, 1]


def test_extra_args_reject_a_tensor_count_mismatch(stub_trtp):
    """Zipping would truncate and quietly emit too few extras."""
    layout = _validate(SIG_2IN_1OUT, arity=(2, 1))
    with pytest.raises(RuntimeError, match="1 input tensor.*describes 2"):
        stub_trtp.build_extra_args([_FakeDesc(8)], [_FakeDesc(8)], layout)


# ---- Registration-time validation ----


@pytest.mark.parametrize(
    "kwargs, message",
    [
        (dict(signature=SIG_1IN_1OUT, arity=(2, 1)), "2 tensor input"),
        (dict(signature={}, arity=(0, 0)), "signature is empty"),
        (dict(signature={"x": "weird", "out": "fp32"}), "unknown element type"),
        (
            dict(signature=SIG_1IN_1OUT, constants={"x": 128}),
            "both signature and constants",
        ),
        (
            dict(signature=SIG_1IN_1OUT, constants={"tile_size": "128"}),
            "bool, int, or float",
        ),
        (dict(signature=SIG_1IN_1OUT, ndim=0), "rank >= 1"),
        (dict(signature=SIG_1IN_1OUT, ndim=True), "must be an integer"),
        (dict(signature=[("x", "fp32")]), "must be mappings"),
    ],
)
def test_invalid_configurations_are_rejected(kwargs, message):
    with pytest.raises(ValueError, match=message):
        _validate(**kwargs)


def test_bool_int_and_float_constants_are_supported():
    layout = _validate(
        SIG_1IN_1OUT,
        constants={"flag": True, "count": 2, "scale": 0.5},
    )
    assert [param.name for param in layout.arrays] == ["x", "out"]


def test_tensor_schema_is_inferred_and_runtime_scalars_are_rejected():
    from torch_tensorrt.kernels._cutile import analyze_tensor_schema

    info = analyze_tensor_schema("ns::op", _identity_meta, None)
    assert info.schema == "(Tensor x) -> Tensor"
    assert info.input_names == ("x",)
    assert (info.num_inputs, info.num_outputs) == (1, 1)

    def _scalar_meta(x: torch.Tensor, alpha: float) -> torch.Tensor:
        return torch.empty_like(x)

    with pytest.raises(ValueError, match="runtime scalar.*alpha: float"):
        analyze_tensor_schema("ns::op", _scalar_meta, None)


def test_tensor_schema_fails_closed_on_incomplete_or_unsupported_types():
    from torch_tensorrt.kernels._cutile import analyze_tensor_schema

    def _untyped(x):
        return x

    with pytest.raises(ValueError, match="complete schema"):
        analyze_tensor_schema("ns::op", _untyped, None)
    with pytest.raises(ValueError, match="Tensor inputs only"):
        analyze_tensor_schema("ns::op", _identity_meta, "(Tensor? x) -> Tensor")
    with pytest.raises(ValueError, match="one or more Tensors"):
        analyze_tensor_schema("ns::op", _identity_meta, "(Tensor x) -> int")
    with pytest.raises(ValueError, match="keyword-only"):
        analyze_tensor_schema("ns::op", _identity_meta, "(*, Tensor x) -> Tensor")
    with pytest.raises(ValueError, match="aliased, mutable, or defaulted"):
        analyze_tensor_schema("ns::op", _identity_meta, "(Tensor(a!) x) -> Tensor(a!)")
    with pytest.raises(ValueError, match="aliased, mutable, or defaulted"):
        analyze_tensor_schema("ns::op", _identity_meta, "(Tensor x=None) -> Tensor")
    with pytest.raises(ValueError, match="qualified pair"):
        analyze_tensor_schema("unqualified", _identity_meta, None)

    def _reserved(outputs: torch.Tensor) -> torch.Tensor:
        return torch.empty_like(outputs)

    with pytest.raises(ValueError, match="reserved parameter"):
        analyze_tensor_schema("ns::op", _reserved, None)
    with pytest.raises(ValueError, match="meta_fn parameters"):
        analyze_tensor_schema("ns::op", _identity_meta, "(Tensor other) -> Tensor")
    with pytest.raises(ValueError, match="schema declares 2 outputs"):
        analyze_tensor_schema(
            "ns::op", _identity_meta, "(Tensor x) -> (Tensor, Tensor)"
        )


def test_kernel_parameter_names_and_order_are_exact():
    from torch_tensorrt.kernels._cutile import validate_kernel_parameters

    def _good(x, out, tile_size):
        pass

    def _swapped(out, x, tile_size):
        pass

    layout = _validate(SIG_1IN_1OUT)
    validate_kernel_parameters(
        "ns::op", SimpleNamespace(_pyfunc=_good), layout, {"tile_size": 128}
    )
    with pytest.raises(ValueError, match="kernel declares.*out.*x"):
        validate_kernel_parameters(
            "ns::op", SimpleNamespace(_pyfunc=_swapped), layout, {"tile_size": 128}
        )


@pytest.mark.parametrize(
    "value",
    [0, -1, True, 2**31, 1.5, (1, 65536), (1, 1, 65536), (1, 2, 3, 4)],
)
def test_invalid_launch_grid_is_rejected(stub_trtp, value):
    with pytest.raises(ValueError, match="grid"):
        stub_trtp.validate_launch_grid("ns::op", value)


def test_raw_trt_style_constant_launch_dimension_is_read(stub_trtp):
    value = SimpleNamespace(is_constant=lambda: True, get_constant_value=lambda: 65536)
    assert stub_trtp._launch_dim_constant(value) == 65536


def test_base_symbolic_integer_expression_is_a_valid_grid(stub_trtp):
    symbolic = _FakeSymIntExpr()
    assert stub_trtp.validate_launch_grid("ns::op", symbolic) == (symbolic,)


def test_block_dimensions_are_validated_and_reqntid_is_authoritative():
    from torch_tensorrt.kernels._cutile import resolve_block_dims

    assert resolve_block_dims("ns::op", "k", (128, 1, 1), None) == (128, 1, 1)
    assert resolve_block_dims("ns::op", "k", None, (16, 8)) == (16, 8, 1)
    assert resolve_block_dims("ns::op", "k", None, range(2, 4)) == (2, 3, 1)
    with pytest.raises(ValueError, match="must match"):
        resolve_block_dims("ns::op", "k", (128, 1, 1), 256)
    with pytest.raises(ValueError, match="1024 threads"):
        resolve_block_dims("ns::op", "k", None, (64, 32))


def test_custom_aot_launch_enforces_compiled_abi(stub_trtp):
    layout = _validate(SIG_1IN_1OUT)

    def _custom(
        _inputs,
        _outputs,
        _tactic,
        block=(128, 1, 1),
        grid=(2, 1, 1),
        shared_mem=0,
        extras=4,
    ):
        launch = _FakeTrtp.KernelLaunchParams()
        launch.grid_x, launch.grid_y, launch.grid_z = grid
        launch.block_x, launch.block_y, launch.block_z = block
        launch.shared_mem = shared_mem
        extra_args = _FakeSymIntExprs(extras)
        extra_args[:] = [_FakeSymInt32(0)] * extras
        return launch, extra_args

    checked = stub_trtp.make_checked_aot_fn(
        "ns::op", "kernel", layout, (128, 1, 1), _custom
    )
    launch, extra_args = checked([], [], 0)
    assert launch.block_x == 128
    assert len(extra_args) == 4

    bad_block = stub_trtp.make_checked_aot_fn(
        "ns::op",
        "kernel",
        layout,
        (128, 1, 1),
        lambda i, o, t: _custom(i, o, t, block=(64, 1, 1)),
    )
    with pytest.raises(ValueError, match=r"launches block .*\.reqntid"):
        bad_block([], [], 0)

    bad_grid = stub_trtp.make_checked_aot_fn(
        "ns::op",
        "kernel",
        layout,
        (128, 1, 1),
        lambda i, o, t: _custom(i, o, t, grid=(_FakeSymInt32(0), 1, 1)),
    )
    with pytest.raises(ValueError, match="grid dimension 0"):
        bad_grid([], [], 0)

    bad_shared_mem = stub_trtp.make_checked_aot_fn(
        "ns::op",
        "kernel",
        layout,
        (128, 1, 1),
        lambda i, o, t: _custom(i, o, t, shared_mem=_FakeSymInt32(-1)),
    )
    with pytest.raises(ValueError, match="shared_mem.*must be 0"):
        bad_shared_mem([], [], 0)

    too_much_shared_mem = stub_trtp.make_checked_aot_fn(
        "ns::op",
        "kernel",
        layout,
        (128, 1, 1),
        lambda i, o, t: _custom(i, o, t, shared_mem=_FakeSymInt32(2**30)),
    )
    with pytest.raises(ValueError, match="shared_mem.*must be 0"):
        too_much_shared_mem([], [], 0)

    bad_extras = stub_trtp.make_checked_aot_fn(
        "ns::op",
        "kernel",
        layout,
        (128, 1, 1),
        lambda i, o, t: _custom(i, o, t, extras=3),
    )
    with pytest.raises(ValueError, match="3 extra argument.*requires 4"):
        bad_extras([], [], 0)

    bad_extra_container = stub_trtp.make_checked_aot_fn(
        "ns::op",
        "kernel",
        layout,
        (128, 1, 1),
        lambda i, o, t: (_custom(i, o, t)[0], [_FakeSymInt32(0)] * 4),
    )
    with pytest.raises(ValueError, match="return TensorRT SymIntExprs"):
        bad_extra_container([], [], 0)

    bad_extra_type = stub_trtp.make_checked_aot_fn(
        "ns::op",
        "kernel",
        layout,
        (128, 1, 1),
        lambda i, o, t: (_custom(i, o, t)[0], _FakeSymIntExprs(4)),
    )
    with pytest.raises(ValueError, match="only TensorRT SymInt32"):
        bad_extra_type([], [], 0)


def test_custom_aot_launch_requires_the_structured_result(stub_trtp):
    checked = stub_trtp.make_checked_aot_fn(
        "ns::op", "kernel", _validate(SIG_1IN_1OUT), None, lambda *args: object()
    )
    with pytest.raises(ValueError, match="must return"):
        checked([], [], 0)


# ---- The dtype gate ----


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
            output = [
                torch.empty(2, dtype=d) if d is not None else None for d in out_dtype
            ]
        else:
            output = torch.empty(2, dtype=out_dtype) if out_dtype is not None else None
        self.meta = {"val": output}


def test_dtype_gate_declines_mismatched_inputs():
    """fp16 into an fp32-compiled kernel would otherwise reinterpret its bytes."""
    from torch_tensorrt.kernels._cutile import make_dtype_capability_validator

    validate = make_dtype_capability_validator("ns::op", _validate(SIG_1IN_1OUT))
    assert validate(_FakeNode([torch.float32], torch.float32), None) is True
    assert validate(_FakeNode([torch.float16], torch.float16), None) is False


@pytest.mark.parametrize(
    "node",
    [
        _FakeNode([], torch.float32),
        _FakeNode([None], torch.float32),
        _FakeNode([torch.float32, torch.float32], torch.float32),
        _FakeNode([torch.float32], None),
        _FakeNode([torch.float32], [torch.float32, torch.float32]),
        SimpleNamespace(args=_FakeNode([torch.float32], None).args),
    ],
)
def test_dtype_gate_rejects_missing_or_malformed_metadata(node):
    from torch_tensorrt.kernels._cutile import make_dtype_capability_validator

    validate = make_dtype_capability_validator("ns::op", _validate(SIG_1IN_1OUT))
    assert validate(node, None) is False


def test_dtype_gate_checks_output_and_composes_user_validator():
    from torch_tensorrt.kernels._cutile import make_dtype_capability_validator

    mixed = _validate({"x": "fp32", "out": "fp16"})
    validate = make_dtype_capability_validator("ns::op", mixed)
    assert validate(_FakeNode([torch.float32], torch.float32), None) is False

    validate = make_dtype_capability_validator(
        "ns::op", _validate(SIG_1IN_1OUT), lambda node, settings: False
    )
    assert validate(_FakeNode([torch.float32], torch.float32), None) is False


@skip_no_qdp
@pytest.mark.parametrize(
    "input_dtype, output_dtype",
    [(torch.float32, torch.float16), (torch.float16, torch.float16)],
)
def test_generated_descriptor_preserves_meta_dtypes(input_dtype, output_dtype):
    """QDP descriptors must model input dtype and allocate the meta output dtype."""
    import tensorrt as trt
    import tensorrt.plugin as trtp
    from tensorrt.plugin._lib import QDP_REGISTRY

    from torch_tensorrt.dynamo.conversion.plugins._generate_plugin import (
        _generate_plugin,
    )
    from torch_tensorrt.kernels._register import _register_pytorch_op

    suffix = f"{str(input_dtype).split('.')[-1]}_{str(output_dtype).split('.')[-1]}"
    op_name = f"ttk_test::cutile_desc_{suffix}"

    def _meta(x: torch.Tensor) -> torch.Tensor:
        return torch.empty_like(x, dtype=output_dtype)

    def _register():
        _register_pytorch_op(op_name, _meta, None)
        _generate_plugin(op_name)

    register_once(_register, key=op_name)
    shape = trtp.ShapeExprs(1)
    shape[0] = 16
    input_trt_dtype = trt.float16 if input_dtype == torch.float16 else trt.float32
    input_desc = trtp.TensorDesc(shape, dtype=input_trt_dtype)
    (output_desc,) = QDP_REGISTRY[op_name].register_func(input_desc)
    expected = trt.float16 if output_dtype == torch.float16 else trt.float32
    assert output_desc.dtype == expected


# ---- cutile_op plumbing with compilation and registration mocked ----


@pytest.fixture
def captured_registration(monkeypatch):
    from torch_tensorrt.kernels import _cutile, _register

    monkeypatch.setattr(
        _cutile,
        "compile_cutile_to_ptx",
        lambda *args, **kwargs: (b"// ptx", "relu_kernel", (128, 1, 1)),
    )
    captured = {}
    monkeypatch.setattr(
        _register,
        "register_cuda_python_plugin",
        lambda *args, **kwargs: captured.update(kwargs),
    )
    return captured


@skip_no_qdp
def test_cutile_op_delegates_to_ptx_op(captured_registration):
    ttk.cutile_op(
        "ttk_test::cutile_forward",
        kernel=object(),
        signature=SIG_1IN_1OUT,
        meta_fn=_identity_meta,
        grid=lambda inputs, outputs: 1,
        constants={"tile_size": 128},
    )

    captured = captured_registration
    assert captured["op_name"] == "ttk_test::cutile_forward"
    assert captured["precompiled_ptx"] == b"// ptx"
    assert captured["spec"].kernel_name == "relu_kernel"
    assert captured["schema"] == "(Tensor x) -> Tensor"
    assert callable(captured["capability_validator"])


@skip_no_qdp
def test_cutile_op_rejects_schema_to_kernel_input_reordering(captured_registration):
    def _meta(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.empty_like(a)

    with pytest.raises(ValueError, match="does not match the Torch schema input order"):
        ttk.cutile_op(
            "ttk_test::cutile_swapped_inputs",
            kernel=object(),
            signature={"b": "fp32", "a": "fp32", "out": "fp32"},
            meta_fn=_meta,
            grid=lambda inputs, outputs: 1,
        )


@skip_no_qdp
@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({}, "needs a callable grid"),
        ({"grid": 1}, "needs a callable grid"),
        ({"grid": lambda i, o: 1, "eager_fn": 1}, "eager_fn must be callable"),
        (
            {"grid": lambda i, o: 1, "capability_validator": 1},
            "capability_validator must be callable",
        ),
        ({"aot_fn": 1}, "aot_fn must be callable"),
        ({"grid": lambda i, o: 1, "aot_fn": lambda *args: None}, "both grid"),
        ({"aot_fn": lambda *args: None, "block_size": 32}, "would be ignored"),
    ],
)
def test_cutile_op_rejects_invalid_or_ignored_launch_options(
    captured_registration, kwargs, message
):
    with pytest.raises(ValueError, match=message):
        ttk.cutile_op(
            "ttk_test::cutile_invalid_launch",
            kernel=object(),
            signature=SIG_1IN_1OUT,
            meta_fn=_identity_meta,
            **kwargs,
        )


@skip_no_qdp
def test_cutile_op_rejects_existing_torch_op_before_compilation(
    captured_registration, monkeypatch
):
    from torch_tensorrt.kernels import _cutile, _register

    monkeypatch.setattr(_register, "_torch_op_already_registered", lambda _: True)
    monkeypatch.setattr(
        _cutile,
        "compile_cutile_to_ptx",
        lambda *args, **kwargs: pytest.fail("collision must be checked before compile"),
    )

    with pytest.raises(ValueError, match="already registered with PyTorch"):
        ttk.cutile_op(
            "ttk_test::cutile_existing_op",
            kernel=object(),
            signature=SIG_1IN_1OUT,
            meta_fn=_identity_meta,
            grid=lambda inputs, outputs: 1,
        )
    assert not captured_registration


@skip_no_qdp
def test_cutile_op_rechecks_collision_after_compilation(
    captured_registration, monkeypatch
):
    from torch_tensorrt.kernels import _register

    checks = iter((False, True))
    monkeypatch.setattr(
        _register, "_torch_op_already_registered", lambda _: next(checks)
    )

    with pytest.raises(ValueError, match="already registered with PyTorch"):
        ttk.cutile_op(
            "ttk_test::cutile_reentrant_op",
            kernel=object(),
            signature=SIG_1IN_1OUT,
            meta_fn=_identity_meta,
            grid=lambda inputs, outputs: 1,
        )
    assert not captured_registration


@skip_no_qdp
def test_derived_launch_validates_grid_and_block(captured_registration, stub_trtp):
    ttk.cutile_op(
        "ttk_test::cutile_bad_grid",
        kernel=object(),
        signature=SIG_1IN_1OUT,
        meta_fn=_identity_meta,
        grid=lambda inputs, outputs: (1, 2, 3, 4),
    )
    with pytest.raises(ValueError, match="4 dimension"):
        captured_registration["spec"].aot_fn([_FakeDesc(8)], [_FakeDesc(8)], 0)

    with pytest.raises(ValueError, match="must match"):
        ttk.cutile_op(
            "ttk_test::cutile_bad_block",
            kernel=object(),
            signature=SIG_1IN_1OUT,
            meta_fn=_identity_meta,
            grid=lambda inputs, outputs: 1,
            block_size=256,
        )


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

    @ct.kernel
    def _ttk_cast_kernel(x, out, tile_size: ct.Constant[int]):
        pid = ct.bid(0)
        tile = ct.load(x, index=(pid,), shape=(tile_size,))
        ct.store(out, index=(pid,), tile=tile.astype(ct.float16))

except ImportError:
    ct = None


def _register_add_one(op_name: str, with_eager: bool = True, grid=None) -> None:
    import tensorrt.plugin as trtp

    def _meta(x: torch.Tensor) -> torch.Tensor:
        return torch.empty_like(x)

    def _eager(x: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(x)
        flat_x, flat_out = x.contiguous().reshape(-1), out.reshape(-1)
        ct.launch(
            torch.cuda.current_stream().cuda_stream,
            (ct.cdiv(flat_x.numel(), TILE), 1, 1),
            _ttk_add_one_kernel,
            (flat_x, flat_out, TILE),
        )
        return out

    launch_grid = grid or (
        lambda inputs, outputs: (trtp.cdiv(inputs[0].shape_expr.numel(), TILE),)
    )
    register_once(
        lambda: ttk.cutile_op(
            op_name,
            kernel=_ttk_add_one_kernel,
            signature=SIG_1IN_1OUT,
            meta_fn=_meta,
            grid=launch_grid,
            constants={"tile_size": TILE},
            eager_fn=_eager if with_eager else None,
            supports_dynamic_shapes=True,
        ),
        key=op_name,
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
            meta_fn=_meta,
            grid=lambda inputs, outputs: (
                trtp.cdiv(inputs[0].shape_expr.numel(), TILE),
            ),
            constants={"tile_size": TILE},
            supports_dynamic_shapes=True,
        ),
        key=op_name,
    )


def _register_scale2d(op_name: str) -> None:
    import tensorrt.plugin as trtp

    def _meta(x: torch.Tensor) -> torch.Tensor:
        return torch.empty_like(x)

    register_once(
        lambda: ttk.cutile_op(
            op_name,
            kernel=_ttk_scale2d_kernel,
            signature={"x": "fp32", "out": "fp32"},
            meta_fn=_meta,
            grid=lambda inputs, outputs: (
                trtp.cdiv(inputs[0].shape_expr[0], 16),
                trtp.cdiv(inputs[0].shape_expr[1], 64),
            ),
            constants={"tile_m": 16, "tile_n": 64},
            ndim=2,
        ),
        key=op_name,
    )


def _register_cast(op_name: str) -> None:
    import tensorrt.plugin as trtp

    def _meta(x: torch.Tensor) -> torch.Tensor:
        return torch.empty_like(x, dtype=torch.float16)

    register_once(
        lambda: ttk.cutile_op(
            op_name,
            kernel=_ttk_cast_kernel,
            signature={"x": "fp32", "out": "fp16"},
            meta_fn=_meta,
            grid=lambda inputs, outputs: (
                trtp.cdiv(inputs[0].shape_expr.numel(), TILE),
            ),
            constants={"tile_size": TILE},
        ),
        key=op_name,
    )


@skip_no_cuda
@skip_no_qdp
@skip_no_cutile
class TestCuTileOpIntegration:
    def test_compiler_output_is_reordered_for_trt(self):
        from torch_tensorrt.kernels._cutile import (
            compile_cutile_to_ptx,
            parse_entry,
        )

        layout = _validate(SIG_1IN_1OUT, constants={"tile_size": TILE})
        ptx, kernel_name, reqntid = compile_cutile_to_ptx(
            "ns::op", _ttk_add_one_kernel, layout, {"tile_size": TILE}
        )
        _, entry_name, params = parse_entry(ptx.decode())
        assert entry_name == kernel_name
        assert reqntid == (128, 1, 1)
        assert [param.split("_param_")[-1] for param in params] == [
            "0",
            "1",
            "2",
            "4",
            "5",
            "3",
        ]

    def test_eager(self):
        _register_add_one("ttk_test::cutile_add_one_eager")
        x = torch.randn(1024, device="cuda")
        assert torch.allclose(
            torch.ops.ttk_test.cutile_add_one_eager(x), x + 1, atol=1e-4, rtol=1e-4
        )

    def test_runs_in_engine_without_an_eager_impl(self):
        """No eager_fn: falling back to PyTorch could not even run.

        Removing the fallback is what makes a passing result mean the cuTile
        kernel executed inside the engine -- with one present, a declined op
        returns the same numbers and the assertion proves nothing.
        """
        op = "ttk_test::cutile_add_one_trt_only"
        _register_add_one(op, with_eager=False)
        x = torch.randn(4, 256, device="cuda")
        trt = compile_op(op, [x])
        assert_ran_in_engine(trt, op)
        with torch.no_grad():
            assert torch.equal(trt(x), x + 1)

    def test_custom_aot_launch_is_checked_and_runs(self):
        import tensorrt.plugin as trtp

        op = "ttk_test::cutile_add_one_custom_aot"

        def _meta(x: torch.Tensor) -> torch.Tensor:
            return torch.empty_like(x)

        def _aot(inputs, outputs, _tactic):
            n = inputs[0].shape_expr.numel()
            launch = trtp.KernelLaunchParams()
            launch.grid_x = trtp.cdiv(n, TILE)
            launch.block_x = TILE
            launch.shared_mem = 0
            extras = trtp.SymIntExprs(4)
            extras[0] = trtp.SymInt32(n)
            extras[1] = trtp.SymInt32(1)
            extras[2] = trtp.SymInt32(outputs[0].shape_expr.numel())
            extras[3] = trtp.SymInt32(1)
            return launch, extras

        register_once(
            lambda: ttk.cutile_op(
                op,
                kernel=_ttk_add_one_kernel,
                signature=SIG_1IN_1OUT,
                meta_fn=_meta,
                aot_fn=_aot,
                constants={"tile_size": TILE},
            ),
            key=op,
        )
        x = torch.randn(1024, device="cuda")
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

    def test_rank2_extents_strides_and_grid(self):
        op = "ttk_test::cutile_scale2d"
        _register_scale2d(op)
        x = torch.randn(64, 256, device="cuda")
        trt = compile_op(op, [x])
        assert_ran_in_engine(trt, op)
        with torch.no_grad():
            assert torch.allclose(trt(x), x * 2, atol=1e-5, rtol=1e-5)

    def test_mixed_output_dtype_is_allocated_correctly(self):
        op = "ttk_test::cutile_fp32_to_fp16"
        _register_cast(op)
        x = torch.randn(4, 256, device="cuda", dtype=torch.float32)
        trt = compile_op(op, [x], enabled_precisions={torch.float32, torch.float16})
        assert_ran_in_engine(trt, op)
        with torch.no_grad():
            output = trt(x)
        assert output.dtype == torch.float16
        assert torch.allclose(output, x.to(torch.float16), atol=1e-3, rtol=1e-3)

    def test_dynamic_shapes(self):
        import tensorrt.plugin as trtp

        op = "ttk_test::cutile_add_one_dyn"
        _register_add_one(
            op,
            grid=lambda inputs, outputs: (
                trtp.SymInt32(inputs[0].shape_expr.numel()) // TILE,
            ),
        )
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
        op = "ttk_test::cutile_add_one_dtype"
        _register_add_one(op)
        x = torch.randn(4, 256, device="cuda", dtype=torch.float16)
        trt = compile_op(op, [x], enabled_precisions={torch.float16})
        # The mirror of assert_ran_in_engine: declined, so still in the graph.
        assert any(
            node.op == "call_function" and "cutile_add_one_dtype" in str(node.target)
            for node in trt.graph.nodes
        ), "the fp16 op was lowered to an fp32-compiled plugin"
        with torch.no_grad():
            assert torch.allclose(trt(x), x + 1, atol=1e-2, rtol=1e-2)
