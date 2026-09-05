"""Tests for ptx_op (pre-compiled PTX registration path)."""

import torch

import torch_tensorrt
import torch_tensorrt.kernels as ttk

from .conftest import (
    SIGMOID_SRC,
    make_eager_sigmoid,
    make_sigmoid_aot,
    register_once,
    skip_no_cuda,
    skip_no_nvrtc,
    skip_no_qdp,
)

# ---- No-GPU: API plumbing ----


@skip_no_qdp
def test_ptx_op_forwards_precompiled_ptx(monkeypatch):
    """ptx_op must pass PTX through to the common precompiled registrar."""
    from torch_tensorrt.kernels import _register

    captured = {}
    monkeypatch.setattr(
        _register,
        "register_precompiled_qdp_plugin",
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
    assert captured["ptx"] == b"// fake PTX bytes"
    assert captured["supports_dynamic_shapes"] is True


@skip_no_qdp
def test_ptx_op_forwards_kernel_name(monkeypatch):
    """The entry-point name must reach the common precompiled registrar."""
    from torch_tensorrt.kernels import _register

    captured = {}
    monkeypatch.setattr(
        _register,
        "register_precompiled_qdp_plugin",
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

    assert captured["kernel_name"] == "my_entrypoint"


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
@skip_no_nvrtc
class TestPtxOpIntegration:
    def test_register_and_eager(self):
        register_once(
            "ttk_test::sigmoid_ptx_eager",
            lambda: _register_sigmoid_via_ptx("ttk_test::sigmoid_ptx_eager"),
        )
        x = torch.randn(1024, device="cuda")
        assert torch.allclose(
            torch.ops.ttk_test.sigmoid_ptx_eager(x),
            torch.sigmoid(x),
            atol=1e-4,
            rtol=1e-4,
        )

    def test_trt_compile_dynamic_shapes(self):
        register_once(
            "ttk_test::sigmoid_ptx_dyn",
            lambda: _register_sigmoid_via_ptx("ttk_test::sigmoid_ptx_dyn"),
        )

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
