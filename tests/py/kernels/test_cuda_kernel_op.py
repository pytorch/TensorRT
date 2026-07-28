"""End-to-end tests for torch_tensorrt.kernels.cuda_kernel_op."""

import pytest
import torch

import torch_tensorrt
import torch_tensorrt.kernels as ttk
from torch_tensorrt.kernels._derive import _torch_output_shape_dtype
from torch_tensorrt.kernels._validation import _validate_spec

from .conftest import (
    ADD_SRC,
    RELU_FLAT_SRC,
    RELU_ND_SRC,
    ROW_SUM_SRC,
    SCALE_SRC,
    SIN_COS_SRC,
    register_once,
    skip_no_cuda,
    skip_no_qdp,
)

# ---- No-GPU: validation & shape inference ----


def _base_spec_kwargs(**overrides):
    kwargs = dict(
        kernel_source="// no-op",
        kernel_name="k",
        inputs=[ttk.InputDecl("x")],
        outputs=[ttk.OutputDecl("y", shape=ttk.SameAs(0))],
        extras=[],
        geometry=ttk.Elementwise(block=(256,), layout="flat"),
    )
    kwargs.update(overrides)
    return kwargs


@pytest.mark.parametrize(
    "overrides, match",
    [
        ({"extras": [ttk.Numel("y")]}, "references unknown tensor input"),
        (
            {"outputs": [ttk.OutputDecl("y", shape=ttk.ReduceDims(5, (-1,)))]},
            "input_idx=5",
        ),
        (
            {"outputs": [ttk.OutputDecl("y", shape=ttk.SameAs("nope"))]},
            "input_idx='nope'",
        ),
        (
            {
                "outputs": [
                    ttk.OutputDecl("y", shape=ttk.SameAs(0), dtype_from="not_an_input")
                ]
            },
            "dtype_from=",
        ),
        ({"geometry": ttk.Elementwise(block=(16, 16), layout="flat")}, "layout='flat'"),
        ({"inputs": [ttk.InputDecl("x"), ttk.InputDecl("x")]}, "duplicate names"),
        ({"geometry": ttk.Reduction(reduce_dims=(), block_size=256)}, "reduce_dims"),
    ],
)
def test_validate_spec_error_paths(overrides, match):
    spec = ttk.KernelSpec(**_base_spec_kwargs(**overrides))
    with pytest.raises(ValueError, match=match):
        _validate_spec(spec)


@pytest.mark.parametrize(
    "shape_rel, want",
    [
        (ttk.SameAs(0), (2, 3, 4)),
        (ttk.SameAs("x"), (2, 3, 4)),
        (ttk.ReduceDims(0, (-1,)), (2, 3)),
        (ttk.ReduceDims("x", (-1,)), (2, 3)),
        (ttk.ReduceDims(0, (1,), keepdim=True), (2, 1, 4)),
        (ttk.ReduceDims("x", (1,), keepdim=True), (2, 1, 4)),
    ],
)
def test_shape_inference(shape_rel, want):
    x = torch.empty(2, 3, 4)
    shape, _ = _torch_output_shape_dtype(
        ttk.OutputDecl("y", shape=shape_rel), [x], [ttk.InputDecl("x")]
    )
    assert shape == want


def test_shape_inference_name_with_scalar_input():
    """``SameAs(name)`` ignores ``ScalarInput`` entries and resolves into the
    tensor-only input list, even when the name appears after a scalar.
    """
    a = torch.empty(2, 5)
    b = torch.empty(7, 3)
    decls = [
        ttk.InputDecl("a"),
        ttk.ScalarInput("alpha", float),
        ttk.InputDecl("b"),
    ]
    # b is index 1 in the tensor-only list, but referenceable by name.
    shape, _ = _torch_output_shape_dtype(
        ttk.OutputDecl("y", shape=ttk.SameAs("b")),
        [a, b],
        [d for d in decls if isinstance(d, ttk.InputDecl)],
    )
    assert shape == (7, 3)


# ---- GPU: one geometry per class, eager + TRT compile ----


def _register_relu_flat():
    ttk.cuda_kernel_op(
        "ttk_kp::relu_flat",
        ttk.KernelSpec(
            kernel_source=RELU_FLAT_SRC,
            kernel_name="ttk_kp_relu_flat",
            inputs=[ttk.InputDecl("x")],
            outputs=[ttk.OutputDecl("y", shape=ttk.SameAs(0))],
            extras=[ttk.Numel("x")],
            geometry=ttk.Elementwise(block=(256,), layout="flat"),
        ),
    )


@skip_no_cuda
@skip_no_qdp
class TestFlat:
    def test_eager_any_rank(self):
        register_once(_register_relu_flat)
        x = torch.randn(2, 3, 5, 7, device="cuda")
        assert torch.allclose(torch.ops.ttk_kp.relu_flat(x), torch.relu(x), atol=1e-5)

    def test_trt_compile(self):
        register_once(_register_relu_flat)

        class M(torch.nn.Module):
            def forward(self, x):
                return torch.ops.ttk_kp.relu_flat(x)

        x = torch.randn(1, 4, 8, 8, device="cuda")
        trt = torch_tensorrt.compile(
            M().cuda().eval(),
            inputs=[x],
            enabled_precisions={torch.float32},
            min_block_size=1,
        )
        with torch.no_grad():
            assert torch.allclose(trt(x), torch.relu(x), atol=1e-2, rtol=1e-2)


def _register_relu_nd():
    ttk.cuda_kernel_op(
        "ttk_kp::relu_nd",
        ttk.KernelSpec(
            kernel_source=RELU_ND_SRC,
            kernel_name="ttk_kp_relu_nd",
            inputs=[ttk.InputDecl("x")],
            outputs=[ttk.OutputDecl("y", shape=ttk.SameAs(0))],
            extras=[ttk.DimSize("x", 0), ttk.DimSize("x", 1)],
            geometry=ttk.Elementwise(block=(16, 16), layout="nd"),
        ),
    )


@skip_no_cuda
@skip_no_qdp
class TestND:
    def test_eager_2d(self):
        register_once(_register_relu_nd)
        x = torch.randn(33, 47, device="cuda")
        assert torch.allclose(torch.ops.ttk_kp.relu_nd(x), torch.relu(x), atol=1e-5)

    def test_trt_compile(self):
        register_once(_register_relu_nd)

        class M(torch.nn.Module):
            def forward(self, x):
                return torch.ops.ttk_kp.relu_nd(x)

        x = torch.randn(33, 47, device="cuda")
        trt = torch_tensorrt.compile(
            M().cuda().eval(),
            inputs=[x],
            enabled_precisions={torch.float32},
            min_block_size=1,
        )
        with torch.no_grad():
            assert torch.allclose(trt(x), torch.relu(x), atol=1e-2, rtol=1e-2)

    def test_nd_block_mismatch_raises(self):
        # Register a 1-D kernel with a 2-D ND geometry — must raise at launch.
        src_1d = """
        extern "C" __global__ void ttk_kp_relu_bad_nd(
                const float* x, int n, float* y) {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < n) y[i] = x[i] > 0.f ? x[i] : 0.f;
        }
        """
        ttk.cuda_kernel_op(
            "ttk_kp::relu_bad_nd",
            ttk.KernelSpec(
                kernel_source=src_1d,
                kernel_name="ttk_kp_relu_bad_nd",
                inputs=[ttk.InputDecl("x")],
                outputs=[ttk.OutputDecl("y", shape=ttk.SameAs(0))],
                extras=[ttk.Numel("x")],
                geometry=ttk.Elementwise(block=(16, 16), layout="nd"),
            ),
        )
        x = torch.randn(32, device="cuda")
        with pytest.raises(ValueError, match="ndim >= 2"):
            torch.ops.ttk_kp.relu_bad_nd(x)


def _register_row_sum(op_name: str = "ttk_kp::row_sum", *, keepdim: bool = False):
    name = op_name.split("::")[-1]
    ttk.cuda_kernel_op(
        op_name,
        ttk.KernelSpec(
            kernel_source=ROW_SUM_SRC.replace("ttk_kp_row_sum", f"ttk_kp_{name}"),
            kernel_name=f"ttk_kp_{name}",
            inputs=[ttk.InputDecl("x")],
            outputs=[
                ttk.OutputDecl("y", shape=ttk.ReduceDims(0, (-1,), keepdim=keepdim))
            ],
            extras=[ttk.DimSize("x", -1)],
            geometry=ttk.Reduction(reduce_dims=(-1,), block_size=256),
        ),
    )


@skip_no_cuda
@skip_no_qdp
class TestReduction:
    def test_eager_any_rank(self):
        register_once(_register_row_sum)
        for shape in [(4, 128), (2, 3, 64)]:
            x = torch.randn(*shape, device="cuda")
            assert torch.allclose(
                torch.ops.ttk_kp.row_sum(x), x.sum(-1), atol=1e-3, rtol=1e-3
            )

    def test_trt_compile(self):
        register_once(_register_row_sum)

        class M(torch.nn.Module):
            def forward(self, x):
                return torch.ops.ttk_kp.row_sum(x)

        x = torch.randn(2, 128, device="cuda")
        trt = torch_tensorrt.compile(
            M().cuda().eval(),
            inputs=[x],
            enabled_precisions={torch.float32},
            min_block_size=1,
        )
        with torch.no_grad():
            assert torch.allclose(trt(x), x.sum(-1), atol=1e-2, rtol=1e-2)

    def test_keepdim_shape(self):
        register_once(
            lambda: _register_row_sum("ttk_kp::row_sum_keepdim", keepdim=True)
        )
        x = torch.randn(4, 64, device="cuda")
        out = torch.ops.ttk_kp.row_sum_keepdim(x)
        assert out.shape == (4, 1)
        assert torch.allclose(out, x.sum(-1, keepdim=True), atol=1e-3, rtol=1e-3)


# ---- GPU: multi-input, scalar input, custom geometry ----


@skip_no_cuda
@skip_no_qdp
def test_multi_input_add():
    ttk.cuda_kernel_op(
        "ttk_kp::add",
        ttk.KernelSpec(
            kernel_source=ADD_SRC,
            kernel_name="ttk_kp_add",
            inputs=[ttk.InputDecl("a"), ttk.InputDecl("b")],
            outputs=[ttk.OutputDecl("c", shape=ttk.SameAs(0))],
            extras=[ttk.Numel("a")],
            geometry=ttk.Elementwise(block=(256,), layout="flat"),
        ),
    )
    a = torch.randn(256, device="cuda")
    b = torch.randn(256, device="cuda")
    assert torch.allclose(torch.ops.ttk_kp.add(a, b), a + b, atol=1e-5)


def _register_scale():
    ttk.cuda_kernel_op(
        "ttk_kp::scale",
        ttk.KernelSpec(
            kernel_source=SCALE_SRC,
            kernel_name="ttk_kp_scale",
            inputs=[ttk.InputDecl("x"), ttk.ScalarInput("alpha", float)],
            outputs=[ttk.OutputDecl("y", shape=ttk.SameAs(0))],
            extras=[ttk.Numel("x")],
            geometry=ttk.Elementwise(block=(256,), layout="flat"),
        ),
        supports_dynamic_shapes=True,
    )


@skip_no_cuda
@skip_no_qdp
class TestScalarInput:
    def test_schema_has_float(self):
        register_once(_register_scale)
        schemas = torch._C._jit_get_schemas_for_operator("ttk_kp::scale")
        assert any("float alpha" in str(s) for s in schemas)

    def test_eager_run(self):
        register_once(_register_scale)
        x = torch.randn(256, device="cuda")
        assert torch.allclose(torch.ops.ttk_kp.scale(x, 2.5), 2.5 * x, atol=1e-5)

    def test_trt_compile(self):
        register_once(_register_scale)

        class M(torch.nn.Module):
            def forward(self, x):
                return torch.ops.ttk_kp.scale(x, 3.0)

        x = torch.randn(128, device="cuda")
        trt = torch_tensorrt.compile(
            M().cuda().eval(),
            inputs=[x],
            enabled_precisions={torch.float32},
            min_block_size=1,
        )
        with torch.no_grad():
            assert torch.allclose(trt(x), 3.0 * x, atol=1e-2, rtol=1e-2)


@skip_no_cuda
@skip_no_qdp
def test_custom_geometry_aot_path_only():
    """Custom geometry routes to user's aot_fn; eager must refuse."""
    import tensorrt.plugin as trtp

    captured = {}

    def _aot(inputs, outputs, tactic):
        captured["called"] = True
        n = inputs[0].shape_expr.numel()
        p = trtp.KernelLaunchParams()
        p.grid_x, p.block_x, p.shared_mem = trtp.cdiv(n, 256), 256, 0
        extra = trtp.SymIntExprs(1)
        extra[0] = trtp.SymInt32(n)
        return p, extra

    src = """
    extern "C" __global__ void ttk_kp_cust_relu(
            const float* x, int n, float* y) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) y[i] = x[i] > 0.f ? x[i] : 0.f;
    }
    """
    ttk.cuda_kernel_op(
        "ttk_kp::custom_geo_relu",
        ttk.KernelSpec(
            kernel_source=src,
            kernel_name="ttk_kp_cust_relu",
            inputs=[ttk.InputDecl("x")],
            outputs=[ttk.OutputDecl("y", shape=ttk.SameAs(0))],
            extras=[ttk.Numel("x")],
            geometry=ttk.Custom(fn=_aot),
        ),
    )

    x = torch.randn(512, device="cuda")
    with pytest.raises(RuntimeError, match="Custom geometry has no eager"):
        torch.ops.ttk_kp.custom_geo_relu(x)

    class M(torch.nn.Module):
        def forward(self, x):
            return torch.ops.ttk_kp.custom_geo_relu(x)

    trt = torch_tensorrt.compile(
        M().cuda().eval(),
        inputs=[x],
        enabled_precisions={torch.float32},
        min_block_size=1,
    )
    with torch.no_grad():
        assert torch.allclose(trt(x), torch.relu(x), atol=1e-2, rtol=1e-2)
    assert captured.get("called") is True


# ---- GPU: multi-output kernel (single launch emits multiple tensors) ----


def _register_sin_cos(op_name: str = "ttk_kp::sin_cos") -> None:
    ttk.cuda_kernel_op(
        op_name,
        ttk.KernelSpec(
            kernel_source=SIN_COS_SRC,
            kernel_name="ttk_kp_sin_cos",
            inputs=[ttk.InputDecl("x")],
            outputs=[
                ttk.OutputDecl("s", shape=ttk.SameAs("x")),
                ttk.OutputDecl("c", shape=ttk.SameAs("x")),
            ],
            extras=[ttk.Numel("x")],
            geometry=ttk.Elementwise(block=(256,), layout="flat"),
        ),
        supports_dynamic_shapes=True,
    )


@skip_no_cuda
@skip_no_qdp
class TestMultiOutput:
    def test_schema_returns_tuple(self):
        register_once(_register_sin_cos)
        schemas = torch._C._jit_get_schemas_for_operator("ttk_kp::sin_cos")
        # Two-tensor return is rendered as ``(Tensor, Tensor)`` in the schema.
        assert any("(Tensor, Tensor)" in str(s) for s in schemas)

    def test_eager_unpacks_to_two_tensors(self):
        register_once(_register_sin_cos)
        x = torch.randn(7, 31, device="cuda", dtype=torch.float32)
        s, c = torch.ops.ttk_kp.sin_cos(x)
        assert torch.allclose(s, torch.sin(x), atol=1e-3, rtol=1e-3)
        assert torch.allclose(c, torch.cos(x), atol=1e-3, rtol=1e-3)
        # Outputs are independent allocations, not views of one buffer.
        assert s.data_ptr() != c.data_ptr()

    def test_trt_compile(self):
        register_once(_register_sin_cos)

        class M(torch.nn.Module):
            def forward(self, x):
                return torch.ops.ttk_kp.sin_cos(x)

        x = torch.randn(4, 64, device="cuda", dtype=torch.float32)
        trt = torch_tensorrt.compile(
            M().cuda().eval(),
            inputs=[x],
            enabled_precisions={torch.float32},
            min_block_size=1,
        )
        with torch.no_grad():
            s, c = trt(x)
            assert torch.allclose(s, torch.sin(x), atol=1e-2, rtol=1e-2)
            assert torch.allclose(c, torch.cos(x), atol=1e-2, rtol=1e-2)
