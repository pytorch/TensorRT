"""Tests for triton_op (Triton kernel -> AOT QDP plugin path)."""

import subprocess
import sys
import textwrap
import types

import pytest
import tensorrt as trt
import torch

import torch_tensorrt
import torch_tensorrt.kernels as ttk

from .conftest import (
    register_once,
    skip_no_cuda,
    skip_no_qdp,
    skip_no_triton,
)
from .test_common import _FakeNode

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
        lambda *a, **k: _triton.CompiledTritonArtifact(
            b"// ptx bytes",
            "my_kernel_entry",
            4,
            0,
            types.SimpleNamespace(backend="cuda", arch=90, warp_size=32),
            90,
        ),
    )
    monkeypatch.setattr(_triton, "_device_arch", lambda device=None: 90)
    captured = {}
    monkeypatch.setattr(
        _register,
        "register_precompiled_qdp_plugin",
        lambda *a, **k: captured.update(k),
    )
    return captured


@pytest.fixture
def launch_int(request, monkeypatch):
    """Use real QDP wrappers with a minimal constant-expression builder."""
    import tensorrt.plugin as trtp

    def constant(value):
        return types.SimpleNamespace(
            is_constant=lambda: True, get_constant_value=lambda: value
        )

    monkeypatch.setattr(
        trtp.SymInt32.__bases__[0],
        "_exprBuilder",
        types.SimpleNamespace(constant=constant),
    )
    return int if request.param == "int" else getattr(trtp, request.param)


# ---- Triton version compatibility ----


@pytest.mark.parametrize("version", ["3.5.0", "3.5.0+git.abc123"])
def test_triton_version_guard_accepts_supported_releases(version):
    from torch_tensorrt.kernels._triton import _require_supported_triton_version

    _require_supported_triton_version(types.SimpleNamespace(__version__=version))


def test_triton_version_guard_rejects_unknown_versions():
    version = "development"
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

    def meta(x):
        return torch.empty_like(x)

    captured = captured_registration
    sig = {"x_ptr": "*fp32", "y_ptr": "*fp32"}
    ttk.triton_op(
        "ttk_test::triton_forward",
        kernel=_FakeKernel(sig),
        signature=sig,
        constexprs={},
        grid=lambda inputs, outputs: (1,),
        meta_fn=meta,
        schema="(Tensor x) -> Tensor",
        supports_dynamic_shapes=True,
    )

    assert captured["schema"] == "(Tensor x) -> Tensor"
    assert captured["op_name"] == "ttk_test::triton_forward"
    assert captured["ptx"] == b"// ptx bytes"
    assert captured["kernel_name"] == "my_kernel_entry"
    assert callable(captured["aot_fn"])
    assert captured["use_aot_if_available"] is True
    assert captured["supports_dynamic_shapes"] is True
    # A dtype capability validator is always installed, even with none passed.
    assert callable(captured["capability_validator"])


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
    assert "2 tensor input(s)" in message
    assert "(input_ptrs..., extra_scalars..., output_ptrs...)" in message


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
def test_constexprs_must_exactly_match_declaration():
    message = _sig_error(
        kernel=_FakeKernel(["x_ptr", "BLOCK", "y_ptr"], ["BLOCK"]),
        signature={"x_ptr": "*fp32", "y_ptr": "*fp32"},
        constexprs={},
    )
    assert "missing constexpr values" in message


@skip_no_qdp
def test_unsupported_scalar_signature_type_rejected():
    scalar_type = "i64"
    message = _sig_error(
        signature={"x_ptr": "*fp32", "value": scalar_type, "y_ptr": "*fp32"},
        extra_args_fn=lambda i, o: [1],
    )
    assert f"scalar type '{scalar_type}'" in message
    assert "SymInt32" in message


@skip_no_qdp
@pytest.mark.parametrize(
    "values, error, match",
    [
        pytest.param([], ValueError, "returned .* value", id="too-few"),
        pytest.param([1, 2], ValueError, "returned .* value", id="too-many"),
        pytest.param(1, TypeError, "must return an iterable", id="not-iterable"),
        pytest.param([True], TypeError, "must be an int or trtp.SymInt32", id="bool"),
        pytest.param([1.5], TypeError, "must be an int or trtp.SymInt32", id="float"),
    ],
)
def test_invalid_extra_args(captured_registration, values, error, match):
    ttk.triton_op(
        "ttk_test::triton_invalid_extras",
        kernel=_FakeKernel(["x_ptr", "n", "y_ptr"]),
        signature={"x_ptr": "*fp32", "n": "i32", "y_ptr": "*fp32"},
        constexprs={},
        grid=lambda i, o: (1,),
        meta_fn=_identity_meta,
        extra_args_fn=lambda i, o: values,
    )
    with pytest.raises(error, match=match):
        captured_registration["aot_fn"](
            [types.SimpleNamespace(dtype=trt.float32)],
            [types.SimpleNamespace(dtype=trt.float32)],
            0,
        )


@skip_no_qdp
@pytest.mark.parametrize(
    "launch_int, invalid_value",
    [("int", -(2**31) - 1), ("SymInt32", 2**31)],
    indirect=["launch_int"],
)
def test_extra_arg_value_must_fit_signed_i32(
    captured_registration, launch_int, invalid_value
):
    ttk.triton_op(
        "ttk_test::triton_scalar_range",
        kernel=_FakeKernel(["x_ptr", "n", "y_ptr"]),
        signature={"x_ptr": "*fp32", "n": "i32", "y_ptr": "*fp32"},
        constexprs={},
        grid=lambda i, o: (1,),
        meta_fn=_identity_meta,
        extra_args_fn=lambda i, o: [launch_int(invalid_value)],
    )

    with pytest.raises(ValueError, match="outside the signed i32 range"):
        captured_registration["aot_fn"](
            [types.SimpleNamespace(dtype=trt.float32)],
            [types.SimpleNamespace(dtype=trt.float32)],
            0,
        )


@skip_no_qdp
@pytest.mark.parametrize(
    "launch_int, value",
    [("ShapeExpr", -(2**31)), ("SymInt32", 2**31 - 1)],
    indirect=["launch_int"],
)
def test_constant_extra_args_preserve_valid_values(launch_int, value):
    from torch_tensorrt.kernels._triton import SignatureParam, make_symint32_args

    args = make_symint32_args(
        "ttk_test::scalar_bounds",
        (SignatureParam("n", False, torch.int32),),
        [launch_int(value)],
    )
    assert args[0]._expr.get_constant_value() == value


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


def test_malformed_pointer_spelling_rejected():
    pointer_type = "*fp32:16"
    from torch_tensorrt.kernels._triton import analyze_signature

    with pytest.raises(ValueError, match="malformed pointer type"):
        analyze_signature({"x_ptr": pointer_type, "y_ptr": "*fp32"}, (1, 1))


def test_untested_pointer_dtype_rejected():
    pointer_type = "fp64"
    from torch_tensorrt.kernels._triton import analyze_signature

    with pytest.raises(ValueError, match="unsupported pointer element type"):
        analyze_signature({"x_ptr": "*fp32", "y_ptr": f"*{pointer_type}"}, (1, 1))


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
        runtime = types.SimpleNamespace(
            driver=types.SimpleNamespace(
                active=types.SimpleNamespace(
                    get_current_target=lambda: types.SimpleNamespace(
                        backend="cuda", arch=90, warp_size=32
                    )
                )
            )
        )

        class compiler:
            ASTSource = staticmethod(lambda **kw: object())

        @staticmethod
        def compile(src, options=None, target=None):
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

    artifact = _triton.compile_triton_to_ptx(object(), CAPPING_SIG, {})

    assert requested == expected
    assert artifact.ptx == compiled_results[-1].asm["ptx"].encode("utf-8")
    assert artifact.kernel_name == compiled_results[-1].metadata.name


@pytest.mark.parametrize(
    "metadata, match",
    [
        ({"global_scratch_size": 128}, "scratch memory"),
        ({"num_ctas": 2}, "num_ctas=2"),
        ({"warp_size": 64}, "warp_size=64"),
        ({"launch_cooperative_grid": True}, "launch_cooperative_grid=True"),
        ({"launch_pdl": True}, "launch_pdl=True"),
        ({"tmem_size": 16}, "tmem_size=16"),
        ({"tensordesc_meta": [object()]}, "tensordesc_meta is non-empty"),
        ({"num_warps": 0}, "invalid num_warps"),
        ({"shared": -1}, "invalid shared memory"),
        ({"shared": 2**31}, "invalid shared memory"),
        ({"num_warps": True}, "must be an integer"),
        ({"shared": 1.5}, "must be an integer"),
    ],
)
def test_unsupported_launch_metadata(monkeypatch, metadata, match):
    from torch_tensorrt.kernels import _triton

    monkeypatch.setattr(_triton, "_driver_max_ptx_version", lambda: None)
    monkeypatch.setattr(_triton, "_triton_default_ptx_version", lambda: None)
    _stub_triton(monkeypatch, [], **metadata)
    with pytest.raises(RuntimeError, match=match):
        _triton.compile_triton_to_ptx(object(), CAPPING_SIG, {})


def test_missing_launch_metadata_fails_closed(monkeypatch):
    from torch_tensorrt.kernels import _triton

    monkeypatch.setattr(_triton, "_driver_max_ptx_version", lambda: None)
    _stub_triton(monkeypatch, [], omitted_metadata=("global_scratch_size",))
    with pytest.raises(RuntimeError, match="missing.*global_scratch_size"):
        _triton.compile_triton_to_ptx(object(), CAPPING_SIG, {})


@skip_no_qdp
@pytest.mark.parametrize(
    "grid, error, match",
    [
        pytest.param((), ValueError, "dimension", id="empty"),
        pytest.param((1, 2, 3, 4), ValueError, "dimension", id="too-many-axes"),
        pytest.param(
            (True,), TypeError, "must be an int or TensorRT symbolic", id="bool"
        ),
        pytest.param(
            (1.5,), TypeError, "must be an int or TensorRT symbolic", id="float"
        ),
    ],
)
def test_invalid_launch_grid(captured_registration, grid, error, match):
    ttk.triton_op(
        "ttk_test::triton_invalid_grid",
        kernel=_FakeKernel(["x_ptr", "y_ptr"]),
        signature={"x_ptr": "*fp32", "y_ptr": "*fp32"},
        constexprs={},
        grid=lambda i, o: grid,
        meta_fn=_identity_meta,
    )
    with pytest.raises(error, match=match):
        captured_registration["aot_fn"](
            [types.SimpleNamespace(dtype=trt.float32)],
            [types.SimpleNamespace(dtype=trt.float32)],
            0,
        )


@skip_no_qdp
@pytest.mark.parametrize(
    "launch_int, axis, dimension",
    [
        ("int", 0, 0),
        ("ShapeExpr", 0, 2**31),
        ("SymInt32", 1, 65536),
        ("ShapeExpr", 2, 65536),
    ],
    indirect=["launch_int"],
)
def test_grid_dimension_range_is_validated(
    captured_registration, launch_int, axis, dimension
):
    grid = [1, 1, 1]
    grid[axis] = launch_int(dimension)
    ttk.triton_op(
        "ttk_test::triton_grid_range",
        kernel=_FakeKernel(["x_ptr", "y_ptr"]),
        signature={"x_ptr": "*fp32", "y_ptr": "*fp32"},
        constexprs={},
        grid=lambda i, o: grid,
        meta_fn=_identity_meta,
    )

    with pytest.raises(ValueError, match=rf"grid dimension {axis} must be between"):
        captured_registration["aot_fn"](
            [types.SimpleNamespace(dtype=trt.float32)],
            [types.SimpleNamespace(dtype=trt.float32)],
            0,
        )


@skip_no_qdp
@pytest.mark.parametrize(
    "launch_int, values",
    [
        ("SymInt32", (2**31 - 1, 65535, 65535)),
    ],
    indirect=["launch_int"],
)
def test_constant_grid_bounds(launch_int, values):
    from torch_tensorrt.kernels._triton import validate_launch_grid

    grid = tuple(launch_int(value) for value in values)
    result = validate_launch_grid("ttk_test::grid_bounds", grid)
    assert all(actual is expected for actual, expected in zip(result, grid))


@skip_no_qdp
@pytest.mark.parametrize("wrapper, fake", [("SymInt32", False), ("ShapeExpr", True)])
def test_dynamic_launch_expressions_are_preserved(wrapper, fake):
    import tensorrt.plugin as trtp

    from torch_tensorrt.kernels._triton import (
        SignatureParam,
        make_symint32_args,
        validate_launch_grid,
    )

    value = getattr(trtp, wrapper)()
    if not fake:
        value._int_expr = types.SimpleNamespace(is_constant=lambda: False)
    assert validate_launch_grid("ttk_test::dynamic_grid", value)[0] is value
    args = make_symint32_args(
        "ttk_test::dynamic_scalar",
        (SignatureParam("n", False, torch.int32),),
        [value],
    )
    assert args[0] is value


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

    register_once(op_name, _register)


@skip_no_cuda
@skip_no_qdp
@skip_no_triton
class TestTritonOpIntegration:

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
        assert any(
            node.op == "call_function" and "triton_add_one_dtype" in str(node.target)
            for node in trt.graph.nodes
        ), "dtype-mismatched Triton op should remain a PyTorch fallback"
        with torch.no_grad():
            # The plugin is declined, so this runs in PyTorch — and is correct.
            assert torch.allclose(trt(x), x + 1, atol=1e-2, rtol=1e-2)

    def test_serialized_engine_runs_without_triton_in_fresh_process(self, tmp_path):
        """The AOT engine embeds PTX and needs no Triton/Python callback at runtime."""
        op_name = "ttk_test::triton_add_one_serialized"
        _register_add_one(op_name)

        class M(torch.nn.Module):
            def forward(self, x):
                return torch.ops.ttk_test.triton_add_one_serialized(x)

        x = torch.arange(1024, device="cuda", dtype=torch.float32).reshape(4, 256)
        torch.testing.assert_close(
            torch.ops.ttk_test.triton_add_one_serialized(x), x + 1
        )
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


@pytest.mark.parametrize(
    "override",
    [
        {"op_name": "invalid"},
        {"eager_fn": 123},
        {"capability_validator": 123},
        {"num_warps": True},
        {"num_warps": 3},
        {"num_stages": 0},
    ],
)
def test_invalid_registration_never_compiles(monkeypatch, override):
    from torch_tensorrt.kernels import _triton

    def unexpected_compile(*args, **kwargs):
        pytest.fail("Invalid registration reached the compiler")

    monkeypatch.setattr(_triton, "compile_triton_to_ptx", unexpected_compile)
    kwargs = dict(
        op_name="ttk_test::preflight",
        kernel=_FakeKernel(["x", "y"]),
        signature={"x": "*fp32", "y": "*fp32"},
        constexprs={},
        grid=lambda i, o: 1,
        meta_fn=_identity_meta,
    )
    kwargs.update(override)
    with pytest.raises(ValueError):
        ttk.triton_op(**kwargs)


def test_duplicate_registration_never_compiles(monkeypatch):
    from torch_tensorrt.kernels import _triton

    lib = torch.library.Library("ttk_preflight", "FRAGMENT")
    try:
        lib.define("duplicate(Tensor x) -> Tensor")
        monkeypatch.setattr(
            _triton, "compile_triton_to_ptx", lambda *a, **k: pytest.fail("compiled")
        )
        with pytest.raises(ValueError, match="already registered"):
            ttk.triton_op(
                "ttk_preflight::duplicate",
                kernel=_FakeKernel(["x", "y"]),
                signature={"x": "*fp32", "y": "*fp32"},
                constexprs={},
                grid=lambda i, o: 1,
                meta_fn=_identity_meta,
            )

    finally:
        lib._destroy()


@pytest.mark.parametrize("takes_arch", [False, True])
def test_ptxas_version_discovery_supports_both_signatures(monkeypatch, takes_arch):
    compiler = pytest.importorskip("triton.backends.nvidia.compiler")
    from torch_tensorrt.kernels import _triton

    seen = []

    def legacy():
        seen.append(None)
        return types.SimpleNamespace(version="13.0")

    def current(arch):
        seen.append(arch)
        return types.SimpleNamespace(version="13.0")

    monkeypatch.setattr(compiler, "get_ptxas", current if takes_arch else legacy)
    monkeypatch.setattr(_triton, "_device_arch", lambda device=None: 90)
    assert _triton._triton_default_ptx_version() == 90
    assert seen == ([90] if takes_arch else [None])


if triton is not None:

    @triton.jit
    def _ttk_add_reduce(x, y, cols, added, reduced, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        col = tl.arange(0, BLOCK)
        offset = row * cols + col
        values = tl.load(x + offset, col < cols, other=0).to(tl.float32)
        values += tl.load(y + offset, col < cols, other=0).to(tl.float32)
        tl.store(added + offset, values, col < cols)
        tl.store(reduced + row, tl.sum(values, 0))


@skip_no_cuda
@skip_no_qdp
@skip_no_triton
def test_multi_input_output_reduction_dynamic():
    dtype, spelling = torch.bfloat16, "bf16"
    import tensorrt.plugin as trtp

    from torch_tensorrt.kernels import _triton

    def meta(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.empty_like(x), x.new_empty((x.shape[0],), dtype=torch.float32)

    signature = {
        "x": f"*{spelling}",
        "y": f"*{spelling}",
        "cols": "i32",
        "added": f"*{spelling}",
        "reduced": "*fp32",
    }
    artifact = _triton.compile_triton_to_ptx(
        _ttk_add_reduce, signature, {"BLOCK": 512}, num_warps=8
    )
    assert artifact.shared_mem > 0
    assert artifact.num_warps == 8
    name = f"ttk_test::add_reduce_{spelling}"
    ttk.triton_op(
        name,
        _ttk_add_reduce,
        signature,
        {"BLOCK": 512},
        grid=lambda i, o: i[0].shape_expr[0],
        meta_fn=meta,
        extra_args_fn=lambda i, o: [trtp.SymInt32(i[0].shape_expr[1])],
        num_warps=8,
    )
    op = getattr(torch.ops.ttk_test, f"add_reduce_{spelling}")

    class Model(torch.nn.Module):
        def forward(self, x, y):
            return op(x, y)

    inputs = [
        torch_tensorrt.Input(
            min_shape=(1, 129), opt_shape=(3, 257), max_shape=(5, 511), dtype=dtype
        )
        for _ in range(2)
    ]
    compiled = torch_tensorrt.compile(
        Model(),
        inputs=inputs,
        min_block_size=1,
        require_full_compilation=True,
        enabled_precisions={dtype, torch.float32},
    )
    for shape in [(1, 129), (3, 257), (5, 511)]:
        x, y = [torch.randn(shape, device="cuda", dtype=dtype) for _ in range(2)]
        added, reduced = compiled(x, y)
        expected = x.float() + y.float()
        torch.testing.assert_close(added, expected.to(dtype))
        torch.testing.assert_close(reduced, expected.sum(-1), atol=1e-4, rtol=1e-4)


def test_conversion_and_aot_reject_target_mismatch(captured_registration, monkeypatch):
    from torch_tensorrt.kernels import _triton

    ttk.triton_op(
        "ttk_test::aot_target",
        kernel=_FakeKernel(["x", "y"]),
        signature={"x": "*fp32", "y": "*fp32"},
        constexprs={},
        grid=lambda i, o: 1,
        meta_fn=_identity_meta,
    )
    seen = []

    def arch(device=None):
        seen.append(device)
        return 80

    monkeypatch.setattr(_triton, "_device_arch", arch)
    with pytest.raises(RuntimeError, match="sm_90.*sm_80"):
        captured_registration["capability_validator"](
            _FakeNode([torch.float32], torch.float32),
            types.SimpleNamespace(device=types.SimpleNamespace(gpu_id=2)),
        )
    assert seen == [2]
    with pytest.raises(RuntimeError, match="sm_90.*sm_80"):
        captured_registration["aot_fn"](
            [types.SimpleNamespace(dtype=trt.float32)],
            [types.SimpleNamespace(dtype=trt.float32)],
            0,
        )
