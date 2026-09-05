"""Tests for cutile_op (cuTile kernel -> AOT QDP plugin path).

Scoped to what actually protects this feature. Its failure mode is silent: the
cuTile kernel ABI groups each array as ``(ptr, extents..., strides...)`` while
TensorRT launches with ``(input_ptrs..., extra_args..., output_ptrs...)``, and a
mismatch does not raise -- the kernel reads whatever landed in each slot and
returns plausible-looking numbers. So the permutation, the extra arguments that
must line up with it, and the dtype gate get direct tests; the rest is covered
end to end.
"""

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


def _validate(signature, arity=(1, 1), constants=None, ndim=1, **kwargs):
    from torch_tensorrt.kernels._cutile import validate_cutile_config

    return validate_cutile_config(
        "ns::op", signature, constants or {}, arity, ndim, **kwargs
    )


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


class _FakeSymInt32(int):
    def __mul__(self, other):
        return _FakeSymInt32(int(self) * int(other))


class _FakeTrtp:
    """Stand-in for tensorrt.plugin: the real SymInt32 only does arithmetic
    inside a live plugin's expression builder."""

    SymInt32 = _FakeSymInt32

    @staticmethod
    def SymIntExprs(count):
        return [None] * count


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
        # The signature must describe exactly the op's tensors...
        (dict(signature=SIG_1IN_1OUT, arity=(2, 1)), "2 tensor input"),
        # ...name a dtype cuTile can be compiled for...
        (dict(signature={"x": "weird", "out": "fp32"}), "unknown element type"),
        # ...and keep arrays and ct.Constant parameters apart.
        (
            dict(signature=SIG_1IN_1OUT, constants={"tile_size": 1.5}),
            "only int values",
        ),
        # Exactly one of grid= / aot_fn= builds the launch.
        (dict(signature=SIG_1IN_1OUT, has_grid=False), "needs a grid="),
        # An unreadable arity must ask for schema=, not guess the split.
        (dict(signature=SIG_1IN_1OUT, arity=None), "Pass schema="),
    ],
)
def test_invalid_configurations_are_rejected(kwargs, message):
    with pytest.raises(ValueError, match=message):
        _validate(**kwargs)


# ---- The dtype gate ----


class _FakeNode:
    """Minimal stand-in for the torch.fx.Node a capability validator receives."""

    def __init__(self, arg_dtypes, out_dtype):
        self.args = [
            type("_Arg", (), {"meta": {"val": torch.empty(2, dtype=d)}})()
            for d in arg_dtypes
        ]
        self.meta = {"val": torch.empty(2, dtype=out_dtype)}


def test_dtype_gate_declines_mismatched_inputs():
    """fp16 into an fp32-compiled kernel would otherwise reinterpret its bytes."""
    from torch_tensorrt.kernels._cutile import make_dtype_capability_validator

    validate = make_dtype_capability_validator("ns::op", _validate(SIG_1IN_1OUT))
    assert validate(_FakeNode([torch.float32], torch.float32), None) is True
    assert validate(_FakeNode([torch.float16], torch.float16), None) is False


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

except ImportError:
    ct = None


def _register_add_one(op_name: str, with_eager: bool = True) -> None:
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

    register_once(
        lambda: ttk.cutile_op(
            op_name,
            kernel=_ttk_add_one_kernel,
            signature=SIG_1IN_1OUT,
            meta_fn=_meta,
            grid=lambda inputs, outputs: (
                trtp.cdiv(inputs[0].shape_expr.numel(), TILE),
            ),
            constants={"tile_size": TILE},
            eager_fn=_eager if with_eager else None,
            supports_dynamic_shapes=True,
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
            signature=SIG_2IN_1OUT,
            meta_fn=_meta,
            grid=lambda inputs, outputs: (
                trtp.cdiv(inputs[0].shape_expr.numel(), TILE),
            ),
            constants={"tile_size": TILE},
            supports_dynamic_shapes=True,
        )
    )


@skip_no_cuda
@skip_no_qdp
@skip_no_cutile
class TestCuTileOpIntegration:
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

    def test_dynamic_shapes(self):
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
