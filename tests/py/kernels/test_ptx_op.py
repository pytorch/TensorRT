"""Tests for ptx_op (pre-compiled PTX registration path)."""

import torch

import torch_tensorrt
import torch_tensorrt.kernels as ttk

from .conftest import (
    SIGMOID_SRC,
    make_eager_sigmoid,
    make_sigmoid_aot,
    skip_no_cuda,
    skip_no_qdp,
)

# ---- No-GPU: API plumbing ----


def test_ptx_op_forwards_precompiled_ptx(monkeypatch):
    """ptx_op must pass ``ptx`` through as ``precompiled_ptx=`` to the registrar."""
    from torch_tensorrt.kernels import _register

    captured = {}
    monkeypatch.setattr(
        _register,
        "register_cuda_python_plugin",
        lambda *a, **k: captured.update(k),
    )

    def _meta(x: torch.Tensor) -> torch.Tensor:
        return torch.empty_like(x)

    ttk.ptx_op(
        op_name="ttk_test::ptx_forward",
        ptx=b"// fake PTX bytes",
        kernel_name="k",
        meta_fn=_meta,
        eager_fn=lambda x: x,
        aot_fn=lambda *a: None,
        supports_dynamic_shapes=True,
    )

    assert captured["op_name"] == "ttk_test::ptx_forward"
    assert captured["precompiled_ptx"] == b"// fake PTX bytes"
    assert captured["supports_dynamic_shapes"] is True
    assert captured["register_torch_op"] is True


def test_ptx_op_kernel_name_lands_on_spec(monkeypatch):
    """The ``kernel_name`` argument must reach the CudaPythonSpec."""
    from torch_tensorrt.kernels import _register

    captured = {}
    monkeypatch.setattr(
        _register,
        "register_cuda_python_plugin",
        lambda *a, **k: captured.update(k),
    )

    def _meta(x: torch.Tensor) -> torch.Tensor:
        return torch.empty_like(x)

    ttk.ptx_op(
        op_name="ttk_test::ptx_named",
        ptx=b"// ptx",
        kernel_name="my_entrypoint",
        meta_fn=_meta,
        eager_fn=lambda x: x,
        aot_fn=lambda *a: None,
    )

    assert captured["spec"].kernel_name == "my_entrypoint"
    assert captured["spec"].kernel_source == ""  # ptx_op leaves source empty


# ---- GPU: integration — compile PTX once, register via ptx_op, exercise eager + TRT ----


def _make_sigmoid_ptx() -> bytes:
    """Compile SIGMOID_SRC to PTX bytes once for ptx_op tests."""
    from torch_tensorrt.kernels._nvrtc import compile_to_ptx

    ptx, _device, _kernel = compile_to_ptx(
        SIGMOID_SRC, "ttk_test_sigmoid", ["/usr/local/cuda/include"]
    )
    return ptx


def _register_sigmoid_via_ptx(op_name: str) -> None:
    ptx = _make_sigmoid_ptx()

    def _meta(x: torch.Tensor) -> torch.Tensor:
        return torch.empty_like(x)

    ttk.ptx_op(
        op_name=op_name,
        ptx=ptx,
        kernel_name="ttk_test_sigmoid",
        meta_fn=_meta,
        eager_fn=make_eager_sigmoid(),
        aot_fn=make_sigmoid_aot(),
        supports_dynamic_shapes=True,
    )


@skip_no_cuda
@skip_no_qdp
class TestPtxOpIntegration:
    def test_register_and_eager(self):
        try:
            _register_sigmoid_via_ptx("ttk_test::sigmoid_ptx_eager")
        except Exception:
            pass
        x = torch.randn(1024, device="cuda")
        assert torch.allclose(
            torch.ops.ttk_test.sigmoid_ptx_eager(x),
            torch.sigmoid(x),
            atol=1e-4,
            rtol=1e-4,
        )

    def test_trt_compile_dynamic_shapes(self):
        try:
            _register_sigmoid_via_ptx("ttk_test::sigmoid_ptx_dyn")
        except Exception:
            pass

        class M(torch.nn.Module):
            def forward(self, x):
                return torch.ops.ttk_test.sigmoid_ptx_dyn(x)

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
