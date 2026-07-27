"""Tests for triton_op (Triton kernel -> AOT QDP plugin path)."""

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


# ---- No-GPU, no-Triton: PTX post-processing helpers ----


def test_parse_ptx_version():
    from torch_tensorrt.kernels._triton import _parse_ptx_version

    assert _parse_ptx_version(FAKE_PTX) == (9, 3)
    assert _parse_ptx_version("// no version here") is None


def test_ptx_version_to_int():
    from torch_tensorrt.kernels._triton import _ptx_version_to_int

    assert _ptx_version_to_int(9, 1) == 91
    assert _ptx_version_to_int(8, 8) == 88


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

    stripped = _strip_trailing_scratch_params(FAKE_PTX, "my_kernel", keep=3)
    _, params = _parse_entry_params(stripped, "my_kernel")
    assert len(params) == 3
    assert "my_kernel_param_3" not in stripped
    assert "my_kernel_param_4" not in stripped
    # The kept params and body are untouched.
    assert "my_kernel_param_2" in stripped
    assert "ret;" in stripped


def test_strip_trailing_scratch_params_noop_when_nothing_to_strip():
    from torch_tensorrt.kernels._triton import _strip_trailing_scratch_params

    assert _strip_trailing_scratch_params(FAKE_PTX, "my_kernel", keep=5) == FAKE_PTX
    # Unknown kernel name -> left untouched.
    assert _strip_trailing_scratch_params(FAKE_PTX, "other", keep=1) == FAKE_PTX


# ---- No-GPU: triton_op plumbing (compile + register mocked out) ----


@skip_no_qdp
def test_triton_op_forwards_to_registrar(monkeypatch):
    """triton_op must compile then forward the PTX + TritonSpec to the registrar."""
    from torch_tensorrt.kernels import _register, _triton
    from torch_tensorrt.kernels._triton_spec import TritonSpec

    monkeypatch.setattr(
        _triton,
        "compile_triton_to_ptx",
        lambda *a, **k: (b"// ptx bytes", "my_kernel_entry", 4, 0),
    )
    captured = {}
    monkeypatch.setattr(
        _register,
        "register_qdp_plugin",
        lambda *a, **k: captured.update(k),
    )

    def _meta(x: torch.Tensor) -> torch.Tensor:
        return torch.empty_like(x)

    ttk.triton_op(
        "ttk_test::triton_forward",
        kernel=object(),
        signature={"x_ptr": "*fp32"},
        constexprs={},
        grid=lambda inputs, outputs: (1,),
        meta_fn=_meta,
        supports_dynamic_shapes=True,
    )

    assert captured["op_name"] == "ttk_test::triton_forward"
    assert captured["precompiled_ptx"] == b"// ptx bytes"
    # triton_op builds a TritonSpec, not a CudaPythonSpec.
    assert isinstance(captured["spec"], TritonSpec)
    assert captured["spec"].kernel_name == "my_kernel_entry"
    assert captured["spec"].signature == {"x_ptr": "*fp32"}
    assert captured["use_aot_if_available"] is True
    assert captured["supports_dynamic_shapes"] is True


@skip_no_qdp
def test_triton_op_aot_override_used(monkeypatch):
    """A user-supplied aot_fn overrides the derived one."""
    from torch_tensorrt.kernels import _register, _triton

    monkeypatch.setattr(
        _triton,
        "compile_triton_to_ptx",
        lambda *a, **k: (b"// ptx", "k", 4, 0),
    )
    captured = {}
    monkeypatch.setattr(
        _register,
        "register_qdp_plugin",
        lambda *a, **k: captured.update(k),
    )

    sentinel = object()

    def _meta(x: torch.Tensor) -> torch.Tensor:
        return torch.empty_like(x)

    ttk.triton_op(
        "ttk_test::triton_override",
        kernel=object(),
        signature={"x_ptr": "*fp32"},
        constexprs={},
        grid=lambda i, o: (1,),
        meta_fn=_meta,
        aot_fn=lambda *a: sentinel,
    )

    assert captured["spec"].aot_fn is not None
    # The override is forwarded verbatim (not wrapped in the derived launcher).
    assert captured["spec"].aot_fn("i", "o", 0) is sentinel


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
