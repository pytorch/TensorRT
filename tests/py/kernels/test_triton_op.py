"""Tests for triton_op (Triton kernel -> AOT QDP plugin path)."""

import contextlib
import os
import re
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

# Triton's switch for source line information in the PTX it emits.
_LINE_INFO = "TRITON_DISABLE_LINE_INFO"

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
    from torch_tensorrt.kernels._common import SignatureParam
    from torch_tensorrt.kernels._triton import make_symint32_args

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
            # Triton reads this setting while it compiles.
            self.disable_line_info = os.environ.get(_LINE_INFO)

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


# ---- No-GPU: plugin PTX without line information ----


class _CompilationKnobs:
    """Stands in for ``triton.knobs.compilation``.

    Like Triton's, ``scope()`` restores on exit every knob and environment
    variable it covers, including those other threads changed meanwhile.
    """

    _ENVIRONMENT = (_LINE_INFO, "TRITON_ALWAYS_COMPILE")

    def __init__(self):
        self.listener = None

    @contextlib.contextmanager
    def scope(self):
        knobs = dict(self.__dict__)
        environment = {name: os.environ.get(name) for name in self._ENVIRONMENT}
        try:
            yield
        finally:
            self.__dict__.clear()
            self.__dict__.update(knobs)
            for name, value in environment.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value


def _triton_with(compilation):
    return types.SimpleNamespace(knobs=types.SimpleNamespace(compilation=compilation))


def test_plugin_compiles_every_attempt_without_line_info(monkeypatch):
    from torch_tensorrt.kernels import _triton

    monkeypatch.delenv(_LINE_INFO, raising=False)
    monkeypatch.setattr(_triton, "_driver_max_ptx_version", lambda: 91)
    monkeypatch.setattr(_triton, "_triton_default_ptx_version", lambda: None)
    compiled = _stub_triton(monkeypatch, [])
    _triton.compile_triton_to_ptx(object(), CAPPING_SIG, {})
    # The first compile emits ISA 9.3, so a second one targets the driver's.
    assert [result.disable_line_info for result in compiled] == ["1", "1"]
    assert _LINE_INFO not in os.environ


def test_line_info_scope_leaves_settings_that_other_threads_change(monkeypatch):
    """Triton's scope() would undo every compilation setting changed meanwhile."""
    from torch_tensorrt.kernels import _triton

    monkeypatch.delenv(_LINE_INFO, raising=False)
    monkeypatch.delenv("TRITON_ALWAYS_COMPILE", raising=False)
    compilation = _CompilationKnobs()
    listener = object()
    with _triton._without_line_info(_triton_with(compilation)):
        assert os.environ[_LINE_INFO] == "1"
        compilation.listener = listener
        monkeypatch.setenv("TRITON_ALWAYS_COMPILE", "1")
    assert _LINE_INFO not in os.environ
    assert compilation.listener is listener
    assert os.environ["TRITON_ALWAYS_COMPILE"] == "1"


@pytest.mark.parametrize("setting", ["environment", "knob"])
def test_explicit_line_info_settings_stay_in_charge(monkeypatch, setting):
    from torch_tensorrt.kernels import _triton

    compilation = _CompilationKnobs()
    if setting == "environment":
        monkeypatch.setenv(_LINE_INFO, "0")
    else:
        # A knob set while Triton does not propagate knobs to the environment.
        monkeypatch.delenv(_LINE_INFO, raising=False)
        compilation.disable_line_info = False
    expected = os.environ.get(_LINE_INFO), dict(vars(compilation))
    with _triton._without_line_info(_triton_with(compilation)):
        assert (os.environ.get(_LINE_INFO), vars(compilation)) == expected
    assert (os.environ.get(_LINE_INFO), vars(compilation)) == expected


def test_line_info_scope_keeps_a_value_set_while_it_was_open(monkeypatch):
    from torch_tensorrt.kernels import _triton

    monkeypatch.delenv(_LINE_INFO, raising=False)
    with _triton._without_line_info(types.SimpleNamespace()):
        monkeypatch.setenv(_LINE_INFO, "0")
    assert os.environ[_LINE_INFO] == "0"


@skip_no_triton
def test_line_info_scope_reaches_tritons_compilation_setting(monkeypatch):
    import triton

    from torch_tensorrt.kernels import _triton

    compilation = getattr(getattr(triton, "knobs", None), "compilation", None)
    if compilation is None or "disable_line_info" in vars(compilation):
        pytest.skip("Triton without knobs, or with an explicit line info knob")
    monkeypatch.delenv(_LINE_INFO, raising=False)
    with _triton._without_line_info(triton):
        assert compilation.disable_line_info is True
    assert compilation.disable_line_info is False


# ---- No-GPU: plugin shared memory and i32 specializations ----

_PLUGIN_SHARED_MEMORY = 48 * 1024
# An extra that a dynamic shape leaves symbolic until TensorRT enqueues.
_SYMBOLIC = types.SimpleNamespace(is_constant=False)


def _artifact(shared_mem=0, kernel_name="my_kernel", num_warps=4, ptx=b"// ptx"):
    from torch_tensorrt.kernels import _triton

    target = types.SimpleNamespace(backend="cuda", arch=90, warp_size=32)
    return _triton.CompiledTritonArtifact(
        ptx, kernel_name, num_warps, shared_mem, target, 90
    )


@pytest.mark.parametrize(
    "shared_mem, limit, fits",
    [
        (_PLUGIN_SHARED_MEMORY, _PLUGIN_SHARED_MEMORY, True),
        (_PLUGIN_SHARED_MEMORY + 1, _PLUGIN_SHARED_MEMORY, False),
        (64 * 1024, 99 * 1024, True),
    ],
)
def test_plugin_shared_memory_is_checked_against_a_plugin_launch(
    monkeypatch, shared_mem, limit, fits
):
    from torch_tensorrt.kernels import _triton

    monkeypatch.setattr(
        _triton, "plugin_shared_memory_limit", lambda device=None: limit
    )
    if fits:
        _triton.check_plugin_shared_memory("ns::op", _artifact(shared_mem))
        return
    with pytest.raises(
        ValueError, match=f"needs {shared_mem} bytes .* at most {limit}"
    ):
        _triton.check_plugin_shared_memory("ns::op", _artifact(shared_mem))


@skip_no_qdp
def test_triton_op_rejects_a_kernel_whose_shared_memory_no_plugin_launch_provides(
    monkeypatch,
):
    """Such a plugin registered and built, then failed on every enqueue."""
    from torch_tensorrt.kernels import _register, _triton

    monkeypatch.setattr(
        _triton, "compile_triton_to_ptx", lambda *a, **k: _artifact(64 * 1024)
    )
    monkeypatch.setattr(
        _triton, "plugin_shared_memory_limit", lambda device=None: _PLUGIN_SHARED_MEMORY
    )
    monkeypatch.setattr(
        _register,
        "register_precompiled_qdp_plugin",
        lambda *a, **k: pytest.fail("registered a plugin that cannot launch"),
    )
    with pytest.raises(ValueError, match="smaller tiles or num_stages=1"):
        ttk.triton_op(
            "ttk_test::shared_memory_overflow",
            kernel=_FakeKernel(["x", "y"]),
            signature={"x": "*fp32", "y": "*fp32"},
            constexprs={},
            grid=lambda i, o: 1,
            meta_fn=_identity_meta,
        )


@pytest.mark.parametrize(
    "extras, hinted",
    [
        ((1024, 64), ("n", "m")),
        ((1024, 1000), ("n",)),
        ((1000, 8), ()),
        ((0, 16), ("n", "m")),
        # A dynamic dimension keeps an extra symbolic: never specialized.
        ((_SYMBOLIC, 1024), ("m",)),
        ((_SYMBOLIC, _SYMBOLIC), ()),
    ],
)
def test_only_constant_i32_extras_divisible_by_16_select_a_specialization(
    extras, hinted
):
    from torch_tensorrt.kernels import _triton

    layout = _triton.analyze_signature(
        {"x": "*fp32", "n": "i32", "m": "i32", "y": "*fp32"}, (1, 1)
    )
    launch = types.SimpleNamespace(block_x=128, shared_mem=0)
    requested = []

    def compile_specialized(names):
        requested.append(names)
        return _artifact(shared_mem=1024, num_warps=8, ptx=b"// specialized")

    specialized = _triton.specialize_on_constant_extras(
        lambda inputs, outputs, tactic: (launch, list(extras)),
        layout,
        compile_specialized,
    )
    result = specialized([], [], 0)
    assert requested == ([hinted] if hinted else [])
    if hinted:
        assert result == (launch, list(extras), ("my_kernel", "// specialized"))
        assert (launch.block_x, launch.shared_mem) == (256, 1024)
    else:
        assert result == (launch, list(extras))
        assert (launch.block_x, launch.shared_mem) == (128, 0)


def test_unusable_specializations_keep_the_registered_ptx(monkeypatch):
    from torch_tensorrt.kernels import _triton

    compiled = {
        ("n",): _artifact(ptx=b"// n"),
        ("m",): _artifact(kernel_name="renamed"),
        ("n", "m"): _artifact(shared_mem=64 * 1024),
    }
    calls = []

    def compile_triton_to_ptx(kernel, signature, constexprs, **options):
        calls.append(options["divisible_by_16"])
        if options["divisible_by_16"] not in compiled:
            raise RuntimeError("Triton rejected the hint")
        return compiled[options["divisible_by_16"]]

    monkeypatch.setattr(_triton, "compile_triton_to_ptx", compile_triton_to_ptx)
    monkeypatch.setattr(
        _triton, "plugin_shared_memory_limit", lambda device=None: _PLUGIN_SHARED_MEMORY
    )
    compile_specialized = _triton.specialization_compiler(
        "ns::op",
        object(),
        {},
        {},
        _artifact(),
        num_warps=4,
        num_stages=None,
        pointer_alignment=None,
    )
    assert compile_specialized(("n",)).ptx == b"// n"
    assert compile_specialized(("n",)).ptx == b"// n"
    # A renamed entry, too much shared memory, and a failed compile.
    assert compile_specialized(("m",)) is None
    assert compile_specialized(("n", "m")) is None
    assert compile_specialized(("k",)) is None
    assert compile_specialized(("k",)) is None
    assert calls == [("n",), ("m",), ("n", "m"), ("k",)]


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

    from torch_tensorrt.kernels._common import SignatureParam
    from torch_tensorrt.kernels._triton import (
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


_PTX_LINE_INFO = re.compile(r"(?m)^\s*\.(?:loc|file)\s")


@skip_no_cuda
@skip_no_triton
def test_plugin_ptx_carries_no_line_information(monkeypatch):
    """TensorRT reloads plugin PTX on every enqueue; line info only adds size."""
    from torch_tensorrt.kernels import _triton

    monkeypatch.delenv(_LINE_INFO, raising=False)
    signature = {"x_ptr": "*fp32", "n": "i32", "y_ptr": "*fp32"}
    constexprs = {"BLOCK": 256}
    artifact = _triton.compile_triton_to_ptx(_ttk_add_one_kernel, signature, constexprs)
    assert not _PTX_LINE_INFO.search(artifact.ptx.decode())
    # Triton's own compilation of the same kernel, as a JIT launch makes it.
    source = triton.compiler.ASTSource(
        fn=_ttk_add_one_kernel, signature=signature, constexprs=constexprs
    )
    assert _PTX_LINE_INFO.search(triton.compile(source).asm["ptx"])
    assert _LINE_INFO not in os.environ


if triton is not None:

    @triton.jit
    def _ttk_row_add_one(x, cols, y, BLOCK: tl.constexpr):
        col = tl.arange(0, BLOCK)
        offset = tl.program_id(0) * cols + col
        tl.store(y + offset, tl.load(x + offset, col < cols) + 1, col < cols)


@skip_no_cuda
@skip_no_qdp
@skip_no_triton
@pytest.mark.parametrize("static", [True, False], ids=["static", "dynamic"])
def test_only_an_extra_the_engine_fixes_selects_a_divisible_specialization(
    monkeypatch, static
):
    """TensorRT asks for the launch once, while it builds the engine, with a
    symbolic expression for every dimension the input profile lets vary. Such
    an extra keeps the registered PTX, even when the optimization shape makes
    it a multiple of 16, so no other value can reach a kernel that assumes one.
    """
    import uuid

    import tensorrt.plugin as trtp

    from torch_tensorrt.kernels import _triton

    # A fresh op: each registration caches the specializations it compiled.
    name = f"row_add_one_{uuid.uuid4().hex[:8]}"
    ttk.triton_op(
        f"ttk_test::{name}",
        kernel=_ttk_row_add_one,
        signature={"x": "*fp32", "cols": "i32", "y": "*fp32"},
        constexprs={"BLOCK": 512},
        grid=lambda i, o: (i[0].shape_expr[0],),
        meta_fn=_identity_meta,
        extra_args_fn=lambda i, o: [trtp.SymInt32(i[0].shape_expr[1])],
        pointer_alignment=16,
        supports_dynamic_shapes=True,
    )
    hinted = []
    compile_triton_to_ptx = _triton.compile_triton_to_ptx

    def record(*args, **kwargs):
        hinted.append(kwargs["divisible_by_16"])
        return compile_triton_to_ptx(*args, **kwargs)

    monkeypatch.setattr(_triton, "compile_triton_to_ptx", record)
    op = getattr(torch.ops.ttk_test, name)

    class Model(torch.nn.Module):
        def forward(self, x):
            return op(x)

    if static:
        inputs, shapes = [torch.randn(4, 256, device="cuda")], [(4, 256)]
    else:
        inputs = [
            torch_tensorrt.Input(
                min_shape=(2, 200), opt_shape=(4, 256), max_shape=(8, 300)
            )
        ]
        shapes = [(2, 200), (4, 256), (5, 257), (8, 300)]
    compiled = torch_tensorrt.compile(
        Model().cuda().eval(),
        ir="dynamo",
        inputs=inputs,
        min_block_size=1,
        require_full_compilation=True,
    )
    assert hinted == ([("cols",)] if static else [])
    for shape in shapes:
        x = torch.randn(shape, device="cuda")
        torch.testing.assert_close(compiled(x), x + 1)


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


# ---- AOT plugin code: alignment hints, specialization, shared memory, line info ----


class _HintedKernel:
    arg_names = ["x", "n", "y", "BLOCK"]


@pytest.mark.parametrize(
    "pointer_alignment, hinted", [(None, False), (8, False), (16, True), (256, True)]
)
def test_divisibility_hints_follow_pointer_alignment_and_named_i32_arguments(
    pointer_alignment, hinted
):
    from torch_tensorrt.kernels import _triton

    signature = {"x": "*fp16", "n": "i32", "y": "*fp16"}
    kernel = _HintedKernel()
    pointers = {(0,): [["tt.divisibility", 16]], (2,): [["tt.divisibility", 16]]}
    # Without a 16-byte promise, a plugin may get views at odd offsets, and
    # Triton specializes no smaller pointer alignment.
    attrs = _triton._divisibility_attrs(kernel, signature, pointer_alignment, ())
    assert attrs == (pointers if hinted else {})
    assert _triton._divisibility_attrs(
        kernel, signature, pointer_alignment, ("n",)
    ) == {**(pointers if hinted else {}), (1,): [["tt.divisibility", 16]]}
    with pytest.raises(ValueError, match="not i32 signature entries"):
        _triton._divisibility_attrs(kernel, signature, pointer_alignment, ("x",))


@skip_no_qdp
@pytest.mark.parametrize("pointer_alignment", [0, 24, 512, True, 16.0])
def test_triton_op_rejects_an_invalid_pointer_alignment(
    captured_registration, monkeypatch, pointer_alignment
):
    from torch_tensorrt.kernels import _triton

    monkeypatch.setattr(
        _triton,
        "compile_triton_to_ptx",
        lambda *a, **k: pytest.fail("compiled with an invalid pointer_alignment"),
    )
    with pytest.raises(
        ValueError,
        match=r"triton_op 'ttk_test::misaligned' pointer_alignment must be a power "
        r"of two between 1 and 256 bytes",
    ):
        ttk.triton_op(
            "ttk_test::misaligned",
            kernel=_FakeKernel(["x_ptr", "y_ptr"]),
            signature={"x_ptr": "*fp32", "y_ptr": "*fp32"},
            constexprs={},
            grid=lambda i, o: (1,),
            meta_fn=_identity_meta,
            pointer_alignment=pointer_alignment,
        )
    assert captured_registration == {}


def test_aot_impl_embeds_a_specialization_chosen_at_build_time():
    trtp = pytest.importorskip("tensorrt.plugin")
    from torch_tensorrt.kernels import _register

    schema = types.SimpleNamespace(tensor_arg_names=("x",), num_outputs=1)
    launch = trtp.KernelLaunchParams()
    impl = _register._make_aot_impl(
        schema, "registered ptx", "entry", lambda i, o, t: (launch, None, ("e2", "p2"))
    )
    name, ptx, params, extras = impl(object(), (object(),), 0)
    assert (name, ptx, params) == ("e2", "p2", launch)


def test_ttgir_allocations_are_summarized_largest_first():
    from torch_tensorrt.kernels import _triton

    ttgir = (
        "%1 = ttg.local_alloc : () -> !ttg.memdesc<64x128xf32, #shared, #smem, mutable>\n"
        "%2 = ttg.local_alloc %0 : (tensor<64x128xf32>) -> !ttg.memdesc<64x128xf32, #s>\n"
        "%3 = ttg.local_alloc : () -> !ttg.memdesc<2x32x64xbf16, #shared, #smem>\n"
        "%4 = ttg.local_alloc : () -> !ttg.memdesc<16xi1, #shared, #smem>\n"
    )
    compiled = types.SimpleNamespace(asm={"ttgir": ttgir})
    assert _triton.shared_memory_allocations(compiled) == (
        "2 x 64x128xf32 (32768 B each)",
        "1 x 2x32x64xbf16 (8192 B)",
        "1 x 16xi1 (16 B)",
    )
    assert _triton.shared_memory_allocations(types.SimpleNamespace(asm={})) == ()


_SCALED_ROWS, _SCALED_COLS = 64, 2048

if triton is not None:

    @triton.jit
    def _ttk_scale_static(x_ptr, y_ptr, N: tl.constexpr, BLOCK: tl.constexpr):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < N
        x = tl.load(x_ptr + offsets, mask=mask).to(tl.float32)
        tl.store(y_ptr + offsets, (x * 2.0 + 1.0).to(tl.bfloat16), mask=mask)

    @triton.jit
    def _ttk_scale_runtime(x_ptr, n, y_ptr, BLOCK: tl.constexpr):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < n
        x = tl.load(x_ptr + offsets, mask=mask).to(tl.float32)
        tl.store(y_ptr + offsets, (x * 2.0 + 1.0).to(tl.bfloat16), mask=mask)


_STATIC_SCALE = (
    {"x_ptr": "*bf16", "y_ptr": "*bf16"},
    {"N": _SCALED_ROWS * _SCALED_COLS, "BLOCK": 1024},
)
_RUNTIME_SCALE = ({"x_ptr": "*bf16", "n": "i32", "y_ptr": "*bf16"}, {"BLOCK": 1024})


def _vector_ops(artifact):
    return len(re.findall(r"\b(?:ld|st)\.global\.v4", artifact.ptx.decode()))


@skip_no_cuda
@skip_no_triton
def test_pointer_alignment_of_16_with_static_dims_vectorizes_the_plugin_ptx():
    from torch_tensorrt.kernels import _triton

    element = _triton.compile_triton_to_ptx(_ttk_scale_static, *_STATIC_SCALE)
    aligned = _triton.compile_triton_to_ptx(
        _ttk_scale_static, *_STATIC_SCALE, pointer_alignment=16
    )
    assert _vector_ops(element) == 0
    assert _vector_ops(aligned) > 0


@skip_no_cuda
@skip_no_qdp
@skip_no_triton
@pytest.mark.parametrize(
    "pointer_alignment, vectorized", [(None, False), (8, False), (16, True)]
)
def test_triton_op_registers_ptx_compiled_for_its_pointer_alignment(
    monkeypatch, pointer_alignment, vectorized
):
    from torch_tensorrt.kernels import _register

    registered = {}
    monkeypatch.setattr(
        _register,
        "register_precompiled_qdp_plugin",
        lambda *a, **k: registered.update(k),
    )
    ttk.triton_op(
        f"ttk_test::scale_aligned_to_{pointer_alignment}",
        kernel=_ttk_scale_static,
        signature=_STATIC_SCALE[0],
        constexprs=_STATIC_SCALE[1],
        grid=lambda i, o: (_SCALED_ROWS * _SCALED_COLS // 1024,),
        meta_fn=_identity_meta,
        pointer_alignment=pointer_alignment,
    )
    vector_ops = _vector_ops(types.SimpleNamespace(ptx=registered["ptx"]))
    assert (vector_ops > 0) is vectorized


@skip_no_cuda
@skip_no_triton
def test_build_time_constant_extras_are_hinted():
    from torch_tensorrt.kernels import _triton

    unknown = _triton.compile_triton_to_ptx(
        _ttk_scale_runtime, *_RUNTIME_SCALE, pointer_alignment=16
    )
    known = _triton.compile_triton_to_ptx(
        _ttk_scale_runtime,
        *_RUNTIME_SCALE,
        pointer_alignment=16,
        divisible_by_16=("n",),
    )
    assert _vector_ops(unknown) == 0 and _vector_ops(known) > 0


@skip_no_cuda
@skip_no_triton
def test_plugin_ptx_has_no_line_info_unless_the_environment_asks(monkeypatch):
    from torch_tensorrt.kernels import _triton

    def directives(artifact):
        return len(re.findall(r"^\s*\.loc\b", artifact.ptx.decode(), re.M))

    monkeypatch.delenv("TRITON_DISABLE_LINE_INFO", raising=False)
    assert (
        directives(_triton.compile_triton_to_ptx(_ttk_scale_static, *_STATIC_SCALE))
        == 0
    )
    assert "TRITON_DISABLE_LINE_INFO" not in os.environ
    monkeypatch.setenv("TRITON_DISABLE_LINE_INFO", "0")
    assert (
        directives(_triton.compile_triton_to_ptx(_ttk_scale_static, *_STATIC_SCALE)) > 0
    )


@skip_no_cuda
@skip_no_qdp
@skip_no_triton
@pytest.mark.parametrize("pointer_alignment", [None, 16])
def test_triton_op_engine_is_correct_and_specialized(monkeypatch, pointer_alignment):
    import tensorrt.plugin as trtp

    from torch_tensorrt.kernels import _triton

    compiled_names = []
    alignments = set()
    real = _triton.compile_triton_to_ptx

    def recording(*args, **kwargs):
        compiled_names.append(tuple(kwargs.get("divisible_by_16", ())))
        alignments.add(kwargs.get("pointer_alignment"))
        return real(*args, **kwargs)

    def meta(x: torch.Tensor) -> torch.Tensor:
        return torch.empty_like(x)

    monkeypatch.setattr(_triton, "compile_triton_to_ptx", recording)
    name = f"aligned_scale_{pointer_alignment}"
    ttk.triton_op(
        f"ttk_test::{name}",
        kernel=_ttk_scale_runtime,
        signature=_RUNTIME_SCALE[0],
        constexprs=_RUNTIME_SCALE[1],
        grid=lambda i, o: (trtp.cdiv(i[0].shape_expr.numel(), 1024),),
        meta_fn=meta,
        extra_args_fn=lambda i, o: [trtp.SymInt32(i[0].shape_expr.numel())],
        num_warps=4,
        pointer_alignment=pointer_alignment,
    )
    op = getattr(torch.ops.ttk_test, name)

    class Model(torch.nn.Module):
        def forward(self, x):
            return op(x)

    x = torch.randn(_SCALED_ROWS, _SCALED_COLS, device="cuda", dtype=torch.bfloat16)
    compiled = torch_tensorrt.compile(
        Model().eval().cuda(),
        ir="dynamo",
        inputs=[x],
        min_block_size=1,
        require_full_compilation=True,
        offload_module_to_cpu=False,
    )
    with torch.no_grad():
        actual = compiled(x)
    torch.testing.assert_close(actual, (x.float() * 2.0 + 1.0).to(torch.bfloat16))
    # The static engine fixes n = 131072, a multiple of 16, at build time.
    assert ("n",) in compiled_names
    # The registered PTX and its build-time specialization share the promise.
    assert alignments == {pointer_alignment}


@skip_no_cuda
@skip_no_qdp
@skip_no_triton
def test_triton_op_rejects_more_shared_memory_than_a_plugin_launch_has():
    from torch_tensorrt.kernels import _triton

    @triton.jit
    def big_tiles(a_ptr, b_ptr, c_ptr, BLOCK: tl.constexpr):
        rows = tl.arange(0, BLOCK)
        a = tl.load(a_ptr + rows[:, None] * BLOCK + rows[None, :])
        b = tl.load(b_ptr + rows[:, None] * BLOCK + rows[None, :])
        c = tl.dot(a, b)
        tl.store(c_ptr + rows[:, None] * BLOCK + rows[None, :], c.to(tl.float16))

    signature = {"a_ptr": "*fp16", "b_ptr": "*fp16", "c_ptr": "*fp16"}
    artifact = _triton.compile_triton_to_ptx(
        big_tiles, signature, {"BLOCK": 128}, num_warps=8, num_stages=1
    )
    if artifact.shared_mem <= _triton.plugin_shared_memory_limit():
        pytest.skip(f"this GPU stages these tiles in {artifact.shared_mem} bytes")

    def meta(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.empty_like(a)

    with pytest.raises(
        ValueError, match="bytes of shared memory, but a TensorRT plugin"
    ):
        ttk.triton_op(
            "ttk_test::oversized_tiles",
            kernel=big_tiles,
            signature=signature,
            constexprs={"BLOCK": 128},
            grid=lambda i, o: (1,),
            meta_fn=meta,
            num_warps=8,
            num_stages=1,
        )
