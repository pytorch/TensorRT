"""Tests for cuda_kernel_op's override path and shared internal helpers.

The override path is what replaces the old ``custom_cuda_kernel_op`` /
``custom_plugin`` / ``cuda_python`` entry points: pass ``meta_fn`` /
``eager_fn`` / ``aot_fn`` / ``schema`` as keyword arguments to
:func:`torch_tensorrt.kernels.cuda_kernel_op` and the matching
``KernelSpec`` fields become optional.
"""

import pytest
import torch

import torch_tensorrt
import torch_tensorrt.kernels as ttk
from torch_tensorrt.kernels._register import _infer_schema

from .conftest import (
    SIGMOID_SRC,
    make_eager_sigmoid,
    make_sigmoid_aot,
    skip_no_cuda,
    skip_no_qdp,
)

# ---- No-GPU: schema inference (small defs — needs real __annotations__) ----


def test_schema_single_tensor():
    def meta(x: torch.Tensor) -> torch.Tensor:
        return torch.empty_like(x)

    s = _infer_schema(meta)
    assert "Tensor x" in s and "-> Tensor" in s


def test_schema_two_tensors():
    def meta(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.empty_like(a)

    s = _infer_schema(meta)
    assert "Tensor a" in s and "Tensor b" in s


def test_schema_mixed_scalar():
    def meta(x: torch.Tensor, scale: float) -> torch.Tensor:
        return torch.empty_like(x)

    s = _infer_schema(meta)
    assert "Tensor x" in s and "float scale" in s


# ---- No-GPU: override-path plumbing ----


def test_overrides_forward_to_registrar(monkeypatch):
    """Override kwargs land on register_cuda_python_plugin with the right values."""
    from torch_tensorrt.kernels import _derive, _register

    captured = {}
    monkeypatch.setattr(
        _register,
        "register_cuda_python_plugin",
        lambda *a, **k: captured.update(k),
    )
    # Skip the real NVRTC compile — we're testing wiring, not codegen.
    monkeypatch.setattr(
        _derive,
        "_compile_kernel",
        lambda spec: (b"// fake ptx", None, None),
    )

    def _meta(x: torch.Tensor) -> torch.Tensor:
        return torch.empty_like(x)

    ttk.cuda_kernel_op(
        "ttk_test::override_forward",
        ttk.KernelSpec(
            kernel_source="// s",
            kernel_name="k",
            inputs=[ttk.InputDecl("x")],
        ),
        meta_fn=_meta,
        eager_fn=lambda x: x,
        aot_fn=lambda *a: None,
        schema="(Tensor x) -> Tensor",
        supports_dynamic_shapes=True,
    )

    assert captured["op_name"] == "ttk_test::override_forward"
    assert captured["schema"] == "(Tensor x) -> Tensor"
    assert captured["supports_dynamic_shapes"] is True
    assert captured["precompiled_ptx"] == b"// fake ptx"
    # User-supplied aot_fn always takes the AOT path.
    assert captured["use_aot_if_available"] is True
    assert captured["register_torch_op"] is True


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({}, "meta_fn is not provided"),
        ({"meta_fn": lambda x: x}, "eager_fn or aot_fn is not provided"),
    ],
)
def test_override_missing_required_dsl_field(kwargs, match):
    """Omitting both the DSL field and its override raises at validation time."""
    spec = ttk.KernelSpec(
        kernel_source="// s",
        kernel_name="k",
        inputs=[ttk.InputDecl("x")],
        # outputs / geometry deliberately absent
    )
    with pytest.raises(ValueError, match=match):
        ttk.cuda_kernel_op("ttk_test::missing", spec, **kwargs)


def test_precompiled_ptx_skips_nvrtc(monkeypatch):
    """register_cuda_python_plugin(precompiled_ptx=...) must not call compile_to_ptx."""
    from torch_tensorrt.kernels import _nvrtc, _register
    from torch_tensorrt.kernels._cuda_python_spec import CudaPythonSpec

    def _fail(*a, **k):
        raise AssertionError(
            "compile_to_ptx must NOT run when precompiled_ptx is provided"
        )

    monkeypatch.setattr(_nvrtc, "compile_to_ptx", _fail)
    for name in (
        "_register_pytorch_op",
        "_register_aot_impl",
        "custom_op",
    ):
        monkeypatch.setattr(_register, name, lambda *a, **k: None)

    spec = CudaPythonSpec(
        kernel_source="// ignored",
        kernel_name="k",
        aot_fn=lambda *a: None,
        eager_fn=lambda x: x,
    )

    def _meta(x: torch.Tensor) -> torch.Tensor:
        return torch.empty_like(x)

    _register.register_cuda_python_plugin(
        op_name="ttk_test::ptx_reused",
        spec=spec,
        meta_fn=_meta,
        precompiled_ptx=b"fake-ptx",
    )


def test_register_pytorch_op_partial_failure_is_atomic(monkeypatch):
    """A failure inside _register_pytorch_op must not leave the op half-registered.

    Without atomic teardown, ``lib.define`` from the first attempt would leave
    ``torch.ops.<ns>.<name>`` populated, ``_torch_op_already_registered`` would
    short-circuit on retry, and the user would silently keep a broken state
    with no CUDA / fake impl.
    """
    from torch_tensorrt.kernels import _register

    op_name = "ttk_test::partial_recovery"

    def _meta(x: torch.Tensor) -> torch.Tensor:
        return torch.empty_like(x)

    real_register_fake = torch.library.register_fake
    call_count = {"n": 0}

    def _flaky_register_fake(op):
        def _inner(fn):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise RuntimeError("simulated upstream failure")
            return real_register_fake(op)(fn)

        return _inner

    monkeypatch.setattr(torch.library, "register_fake", _flaky_register_fake)

    # First attempt: simulated failure must propagate.
    with pytest.raises(RuntimeError, match="simulated upstream failure"):
        _register._register_pytorch_op(op_name, _meta, eager_fn=None)

    # After failure, the op must look un-registered so a retry can recover.
    assert not _register._torch_op_already_registered(op_name)

    # Second attempt: the rigged register_fake lets this one through.
    _register._register_pytorch_op(op_name, _meta, eager_fn=None)
    assert _register._torch_op_already_registered(op_name)


# ---- GPU: NVRTC compilation ----


@skip_no_cuda
@skip_no_qdp
class TestNVRTC:
    def test_compiles_to_ptx(self):
        from torch_tensorrt.kernels._nvrtc import compile_to_ptx

        ptx, _, _ = compile_to_ptx(
            SIGMOID_SRC, "ttk_test_sigmoid", ["/usr/local/cuda/include"]
        )
        assert isinstance(ptx, bytes) and b"ttk_test_sigmoid" in ptx

    def test_invalid_source_raises(self):
        from torch_tensorrt.kernels._nvrtc import compile_to_ptx

        with pytest.raises(Exception):
            compile_to_ptx(
                "this is not valid CUDA !!!###", "bad", ["/usr/local/cuda/include"]
            )

    def test_arch_override_respected(self):
        from torch_tensorrt.kernels._nvrtc import compile_to_ptx

        arch = f"sm_{torch.cuda.get_device_capability()[0]}0"
        ptx, _, _ = compile_to_ptx(
            SIGMOID_SRC,
            "ttk_test_sigmoid",
            ["/usr/local/cuda/include"],
            arch_override=arch,
        )
        assert isinstance(ptx, bytes)


# ---- GPU: integration — register via override path, exercise eager + TRT ----


def _register_sigmoid_via_overrides(op_name: str) -> None:
    def _meta(x: torch.Tensor) -> torch.Tensor:
        return torch.empty_like(x)

    ttk.cuda_kernel_op(
        op_name,
        ttk.KernelSpec(
            kernel_source=SIGMOID_SRC,
            kernel_name="ttk_test_sigmoid",
            inputs=[ttk.InputDecl("x")],
        ),
        meta_fn=_meta,
        eager_fn=make_eager_sigmoid(),
        aot_fn=make_sigmoid_aot(),
        supports_dynamic_shapes=True,
    )


@skip_no_cuda
@skip_no_qdp
class TestOverrideIntegration:
    def test_register_and_eager(self):
        try:
            _register_sigmoid_via_overrides("ttk_test::sigmoid_eager")
        except Exception:
            pass
        x = torch.randn(1024, device="cuda")
        assert torch.allclose(
            torch.ops.ttk_test.sigmoid_eager(x), torch.sigmoid(x), atol=1e-4, rtol=1e-4
        )

    def test_trt_compile_dynamic_shapes(self):
        try:
            _register_sigmoid_via_overrides("ttk_test::sigmoid_dyn")
        except Exception:
            pass

        class M(torch.nn.Module):
            def forward(self, x):
                return torch.ops.ttk_test.sigmoid_dyn(x)

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
                assert torch.allclose(trt(x), torch.sigmoid(x), atol=1e-2, rtol=1e-2)


@skip_no_cuda
@skip_no_qdp
def test_schema_override_integration():
    """End-to-end: schema= overrides the inferred schema at real registration."""
    src = """
    extern "C" __global__ void schema_ov_noop(
            const float* x, int n, float alpha, float* y) {}
    """

    def _meta(x, alpha):  # no hints — only schema= makes ``float alpha`` land
        return torch.empty_like(x)

    ttk.cuda_kernel_op(
        "ttk_test::schema_ov",
        ttk.KernelSpec(
            kernel_source=src,
            kernel_name="schema_ov_noop",
            inputs=[ttk.InputDecl("x"), ttk.ScalarInput("alpha", float)],
        ),
        meta_fn=_meta,
        eager_fn=lambda x, alpha: alpha * x,  # reference impl — kernel is a no-op
        aot_fn=make_sigmoid_aot(),
        schema="(Tensor x, float alpha) -> Tensor",
        supports_dynamic_shapes=True,
    )

    schemas = [
        str(s) for s in torch._C._jit_get_schemas_for_operator("ttk_test::schema_ov")
    ]
    assert any("float alpha" in s for s in schemas)

    x = torch.randn(32, device="cuda")
    assert torch.allclose(torch.ops.ttk_test.schema_ov(x, 2.5), 2.5 * x, atol=1e-5)
