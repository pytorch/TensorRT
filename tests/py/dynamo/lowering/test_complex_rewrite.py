"""Comprehensive numerical-equivalence tests for complex_graph_detection lowering pass.

Each test verifies:
    lowered_gm(view_as_real(z)) ≡ original_model(z)   (numerically)

The lowering pass rewrites complex-dtype ops to real arithmetic on a [..., 2]
layout (trailing dim encodes real/imag).  After lowering, all inputs and outputs
are in that real layout; the test harness converts back to complex before
comparison.

Organisation
------------
 1. Infrastructure helpers
 2. Elementwise arithmetic  (mul / div / add / sub variants)
 3. Complex-specific ops    (real, imag, conj, abs, angle, polar)
 4. Transcendental functions (exp, log, pow, sin/cos/tan …)
 5. Shape manipulation      (permute, reshape/view, flatten, squeeze/unsqueeze,
                              cat, stack, select, slice, split, chunk, expand,
                              transpose, t, clone, narrow, roll, flip)
 6. Matrix multiplication   (mm, bmm, matmul)
 7. Elementwise-safe pass-through verification
 8. Reduction ops           (sum / mean — positive dims pass, negative = xfail)
 9. Creation-op bugs        (ones_like → xfail, zeros_like → pass, full_like → xfail)
10. Chain / composition tests

xfail tests document known bugs or missing handlers.  They are expected to fail.
If they start passing a handler was fixed — remove the xfail marker.
"""

from __future__ import annotations

from typing import Any, Tuple

import pytest
import torch
import torch.nn as nn

from torch_tensorrt.dynamo._settings import CompilationSettings
from torch_tensorrt.dynamo.lowering.passes.complex_graph_rewrite import (
    complex_graph_detection,
)

# ---------------------------------------------------------------------------
# 1. Infrastructure
# ---------------------------------------------------------------------------

_RTOL = 1e-4
_ATOL = 1e-4


def _export_and_lower(
    model: nn.Module, example_inputs: Tuple[Any, ...]
) -> torch.fx.GraphModule:
    """Export *model* and apply the complex_graph_detection lowering pass."""
    with torch.no_grad():
        exp = torch.export.export(model.eval(), example_inputs)
    gm = exp.module()
    complex_graph_detection(gm, CompilationSettings())
    return gm


def _real_inputs(inputs: Tuple[Any, ...]) -> Tuple[Any, ...]:
    """Convert complex tensors to [..., 2] real layout."""
    return tuple(
        (
            torch.view_as_real(x).contiguous()
            if isinstance(x, torch.Tensor) and x.is_complex()
            else x
        )
        for x in inputs
    )


def _to_complex_if_needed(out: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Reinterpret *out* as complex if *ref* is complex and *out* has trailing dim 2."""
    if ref.is_complex() and not out.is_complex() and out.shape[-1] == 2:
        return torch.view_as_complex(out.contiguous())
    return out


def _assert_close(ref: torch.Tensor, got: torch.Tensor, tag: str) -> None:
    if ref.dtype == torch.bool:
        assert torch.equal(got, ref), f"{tag}: bool tensor mismatch"
        return
    if ref.is_complex():
        assert got.is_complex(), f"{tag}: expected complex output, got {got.dtype}"
        assert ref.shape == got.shape, f"{tag}: shape {got.shape} != {ref.shape}"
        torch.testing.assert_close(
            got.real.float(), ref.real.float(), rtol=_RTOL, atol=_ATOL
        )
        torch.testing.assert_close(
            got.imag.float(), ref.imag.float(), rtol=_RTOL, atol=_ATOL
        )
    else:
        assert not got.is_complex(), f"{tag}: expected real output, got {got.dtype}"
        assert ref.shape == got.shape, f"{tag}: shape {got.shape} != {ref.shape}"
        torch.testing.assert_close(got.float(), ref.float(), rtol=_RTOL, atol=_ATOL)


def _check_op(model: nn.Module, inputs: Tuple[Any, ...], tag: str) -> None:
    """Full pipeline: run model → export+lower → compare."""
    with torch.no_grad():
        ref = model(*inputs)

    gm = _export_and_lower(model, inputs)
    raw = gm(*_real_inputs(inputs))

    if isinstance(raw, (list, tuple)):
        ref_list = list(ref) if isinstance(ref, (list, tuple)) else [ref]
        for i, (r, o) in enumerate(zip(ref_list, raw)):
            got = _to_complex_if_needed(o, r)
            _assert_close(r, got, f"{tag}[{i}]")
    else:
        got = _to_complex_if_needed(raw, ref)
        _assert_close(ref, got, tag)


# Convenience: 2-D complex inputs used by most tests
def _z(rows: int = 3, cols: int = 4) -> torch.Tensor:
    return torch.randn(rows, cols, dtype=torch.complex64)


def _z3d(b: int = 2, m: int = 3, n: int = 4) -> torch.Tensor:
    return torch.randn(b, m, n, dtype=torch.complex64)


# ===========================================================================
# 2. Elementwise arithmetic
# ===========================================================================


@pytest.mark.unit
def test_mul_complex_complex():
    class M(nn.Module):
        def forward(self, x, y):
            return x * y

    z1, z2 = _z(), _z()
    _check_op(M(), (z1, z2), "mul_cc")


@pytest.mark.unit
def test_mul_complex_real():
    """Complex × real tensor — only the complex part gets the mul handler."""

    class M(nn.Module):
        def forward(self, z, r):
            return z * r  # z complex, r real — result is complex

    z = _z()
    r = torch.randn(3, 4)
    _check_op(M(), (z, r), "mul_cr")


@pytest.mark.unit
def test_mul_scalar():
    """z * scalar — both re/im scaled equally (elementwise-safe)."""

    class M(nn.Module):
        def forward(self, z):
            return z * 3.0

    _check_op(M(), (_z(),), "mul_scalar")


@pytest.mark.unit
def test_div_complex_complex():
    class M(nn.Module):
        def forward(self, x, y):
            return x / y

    _check_op(M(), (_z(), _z() + 0.1), "div_cc")


@pytest.mark.unit
def test_div_complex_scalar():
    class M(nn.Module):
        def forward(self, z):
            return z / 2.0

    _check_op(M(), (_z(),), "div_cscalar")


@pytest.mark.unit
def test_div_scalar_complex():
    """scalar / complex — s/(a+bi) = (sa - sbi)/(a²+b²)."""

    class M(nn.Module):
        def forward(self, z):
            return 4.0 / (z + 0.1)

    _check_op(M(), (_z(),), "div_scalar_c")


@pytest.mark.unit
def test_add_tensor():
    """z1 + z2 — both complex; elementwise-safe (component-wise)."""

    class M(nn.Module):
        def forward(self, x, y):
            return x + y

    _check_op(M(), (_z(), _z()), "add_tensor")


@pytest.mark.unit
def test_sub_tensor():
    class M(nn.Module):
        def forward(self, x, y):
            return x - y

    _check_op(M(), (_z(), _z()), "sub_tensor")


@pytest.mark.unit
def test_add_scalar():
    """(a+bi) + s = (a+s) + bi — scalar added to real part only."""

    class M(nn.Module):
        def forward(self, z):
            return z + 2.5

    _check_op(M(), (_z(),), "add_scalar")


@pytest.mark.unit
def test_sub_scalar():
    class M(nn.Module):
        def forward(self, z):
            return z - 1.0

    _check_op(M(), (_z(),), "sub_scalar")


@pytest.mark.unit
def test_neg():
    """Negation is elementwise-safe (flips sign of both re/im)."""

    class M(nn.Module):
        def forward(self, z):
            return -z

    _check_op(M(), (_z(),), "neg")


# ===========================================================================
# 3. Complex-specific ops
# ===========================================================================


@pytest.mark.unit
def test_real():
    """z.real → real tensor (select re component)."""

    class M(nn.Module):
        def forward(self, z):
            return z.real

    _check_op(M(), (_z(3, 5),), "real")  # shape (3,5) so last dim≠2


@pytest.mark.unit
def test_imag():
    class M(nn.Module):
        def forward(self, z):
            return z.imag

    _check_op(M(), (_z(3, 5),), "imag")


@pytest.mark.unit
def test_conj():
    """conj(a+bi) = a - bi."""

    class M(nn.Module):
        def forward(self, z):
            return torch.conj(z)

    _check_op(M(), (_z(),), "conj")


@pytest.mark.unit
def test_abs():
    """|a+bi| = sqrt(a²+b²) — real output."""

    class M(nn.Module):
        def forward(self, z):
            return torch.abs(z)

    _check_op(M(), (_z(3, 5),), "abs")


@pytest.mark.unit
def test_angle():
    """angle(a+bi) = atan2(b, a) — real output."""

    class M(nn.Module):
        def forward(self, z):
            return torch.angle(z)

    _check_op(M(), (_z(3, 5),), "angle")


@pytest.mark.unit
def test_polar():
    """polar(r, theta) = r*cos(theta) + i*r*sin(theta)."""

    class M(nn.Module):
        def forward(self, r, theta):
            return torch.polar(r, theta)

    r = torch.rand(3, 4) + 0.1
    theta = torch.randn(3, 4)
    _check_op(M(), (r, theta), "polar")


# ===========================================================================
# 4. Transcendental functions
# ===========================================================================


@pytest.mark.unit
def test_exp():
    """exp(a+bi) = e^a*(cos(b) + i*sin(b))."""

    class M(nn.Module):
        def forward(self, z):
            return torch.exp(z)

    _check_op(M(), (_z(),), "exp")


@pytest.mark.unit
def test_log():
    class M(nn.Module):
        def forward(self, z):
            return torch.log(z)

    _check_op(M(), (_z() + 0.1,), "log")


@pytest.mark.unit
def test_log2():
    class M(nn.Module):
        def forward(self, z):
            return torch.log2(z)

    _check_op(M(), (_z() + 0.1,), "log2")


@pytest.mark.unit
def test_log10():
    class M(nn.Module):
        def forward(self, z):
            return torch.log10(z)

    _check_op(M(), (_z() + 0.1,), "log10")


@pytest.mark.unit
def test_log1p():
    class M(nn.Module):
        def forward(self, z):
            return torch.log1p(z)

    _check_op(M(), (_z() + 0.1,), "log1p")


@pytest.mark.unit
def test_expm1():
    class M(nn.Module):
        def forward(self, z):
            return torch.expm1(z)

    # Use small values to keep numbers from overflowing
    z = torch.randn(3, 4, dtype=torch.complex64) * 0.3
    _check_op(M(), (z,), "expm1")


@pytest.mark.unit
def test_sqrt():
    class M(nn.Module):
        def forward(self, z):
            return torch.sqrt(z)

    _check_op(M(), (_z(),), "sqrt")


@pytest.mark.unit
def test_pow_scalar():
    """z**n via polar form."""

    class M(nn.Module):
        def forward(self, z):
            return z**2.0

    _check_op(M(), (_z() + 0.1,), "pow_scalar")


@pytest.mark.unit
def test_pow_tensor():
    """z1**z2 = exp(z2 * log(z1))."""

    class M(nn.Module):
        def forward(self, x, y):
            return x**y

    z1 = _z() + 0.5
    z2 = torch.randn(3, 4, dtype=torch.complex64) * 0.3
    _check_op(M(), (z1, z2), "pow_tensor")


@pytest.mark.unit
def test_sin():
    class M(nn.Module):
        def forward(self, z):
            return torch.sin(z)

    z = torch.randn(3, 4, dtype=torch.complex64) * 0.5
    _check_op(M(), (z,), "sin")


@pytest.mark.unit
def test_cos():
    class M(nn.Module):
        def forward(self, z):
            return torch.cos(z)

    z = torch.randn(3, 4, dtype=torch.complex64) * 0.5
    _check_op(M(), (z,), "cos")


@pytest.mark.unit
def test_tan():
    class M(nn.Module):
        def forward(self, z):
            return torch.tan(z)

    z = torch.randn(3, 4, dtype=torch.complex64) * 0.3
    _check_op(M(), (z,), "tan")


@pytest.mark.unit
def test_sinh():
    class M(nn.Module):
        def forward(self, z):
            return torch.sinh(z)

    z = torch.randn(3, 4, dtype=torch.complex64) * 0.5
    _check_op(M(), (z,), "sinh")


@pytest.mark.unit
def test_cosh():
    class M(nn.Module):
        def forward(self, z):
            return torch.cosh(z)

    z = torch.randn(3, 4, dtype=torch.complex64) * 0.5
    _check_op(M(), (z,), "cosh")


@pytest.mark.unit
def test_tanh():
    class M(nn.Module):
        def forward(self, z):
            return torch.tanh(z)

    z = torch.randn(3, 4, dtype=torch.complex64) * 0.5
    _check_op(M(), (z,), "tanh")


@pytest.mark.unit
def test_asin():
    class M(nn.Module):
        def forward(self, z):
            return torch.asin(z)

    z = torch.randn(3, 4, dtype=torch.complex64) * 0.5
    _check_op(M(), (z,), "asin")


@pytest.mark.unit
def test_acos():
    class M(nn.Module):
        def forward(self, z):
            return torch.acos(z)

    z = torch.randn(3, 4, dtype=torch.complex64) * 0.5
    _check_op(M(), (z,), "acos")


@pytest.mark.unit
def test_atan():
    class M(nn.Module):
        def forward(self, z):
            return torch.atan(z)

    z = torch.randn(3, 4, dtype=torch.complex64) * 0.5
    _check_op(M(), (z,), "atan")


@pytest.mark.unit
def test_asinh():
    class M(nn.Module):
        def forward(self, z):
            return torch.asinh(z)

    z = torch.randn(3, 4, dtype=torch.complex64) * 0.5
    _check_op(M(), (z,), "asinh")


@pytest.mark.unit
def test_acosh():
    class M(nn.Module):
        def forward(self, z):
            return torch.acosh(z)

    # acosh needs |z| > 1 to avoid NaN
    z = torch.randn(3, 4, dtype=torch.complex64) + 2.0
    _check_op(M(), (z,), "acosh")


@pytest.mark.unit
def test_atanh():
    class M(nn.Module):
        def forward(self, z):
            return torch.atanh(z)

    z = torch.randn(3, 4, dtype=torch.complex64) * 0.3
    _check_op(M(), (z,), "atanh")


@pytest.mark.unit
def test_isnan():
    """isnan/isinf: boolean output, checks re|im."""

    class M(nn.Module):
        def forward(self, z):
            return torch.isnan(z)

    _check_op(M(), (_z(3, 5),), "isnan")


@pytest.mark.unit
def test_isinf():
    class M(nn.Module):
        def forward(self, z):
            return torch.isinf(z)

    _check_op(M(), (_z(3, 5),), "isinf")


# ===========================================================================
# 5. Shape manipulation
# ===========================================================================


@pytest.mark.unit
def test_view_as_real_complex_bypass():
    """view_as_real → view_as_complex is a round-trip no-op after lowering."""

    class M(nn.Module):
        def forward(self, z):
            r = torch.view_as_real(z)
            return torch.view_as_complex(r)

    _check_op(M(), (_z(),), "var_vac_bypass")


@pytest.mark.unit
def test_permute():
    class M(nn.Module):
        def forward(self, z):
            return z.permute(1, 0)

    _check_op(M(), (_z(),), "permute_2d")


@pytest.mark.unit
def test_permute_3d():
    class M(nn.Module):
        def forward(self, z):
            return z.permute(0, 2, 1)

    _check_op(M(), (_z3d(),), "permute_3d")


@pytest.mark.unit
def test_reshape():
    class M(nn.Module):
        def forward(self, z):
            return z.reshape(12)

    _check_op(M(), (_z(),), "reshape")


@pytest.mark.unit
def test_reshape_batch():
    class M(nn.Module):
        def forward(self, z):
            return z.reshape(2, 6)

    _check_op(M(), (torch.randn(3, 4, dtype=torch.complex64),), "reshape_batch")


@pytest.mark.unit
def test_view():
    class M(nn.Module):
        def forward(self, z):
            return z.view(12)

    _check_op(M(), (torch.randn(3, 4, dtype=torch.complex64).contiguous(),), "view")


@pytest.mark.unit
def test_flatten_all():
    class M(nn.Module):
        def forward(self, z):
            return z.flatten()

    _check_op(M(), (_z3d(),), "flatten_all")


@pytest.mark.unit
def test_flatten_partial():
    class M(nn.Module):
        def forward(self, z):
            return z.flatten(1, -1)

    _check_op(M(), (_z3d(),), "flatten_partial")


@pytest.mark.unit
def test_flatten_start_neg():
    class M(nn.Module):
        def forward(self, z):
            return z.flatten(-2, -1)

    _check_op(M(), (_z3d(),), "flatten_neg_dims")


@pytest.mark.unit
def test_unsqueeze_pos():
    class M(nn.Module):
        def forward(self, z):
            return z.unsqueeze(0)

    _check_op(M(), (_z(),), "unsqueeze_pos")


@pytest.mark.unit
def test_unsqueeze_neg():
    class M(nn.Module):
        def forward(self, z):
            return z.unsqueeze(-1)

    _check_op(M(), (_z(),), "unsqueeze_neg")


@pytest.mark.unit
def test_unsqueeze_mid_neg():
    class M(nn.Module):
        def forward(self, z):
            return z.unsqueeze(-2)

    _check_op(M(), (_z3d(),), "unsqueeze_mid_neg")


@pytest.mark.unit
def test_squeeze_pos():
    class M(nn.Module):
        def forward(self, z):
            return z.squeeze(0)

    _check_op(M(), (torch.randn(1, 4, dtype=torch.complex64),), "squeeze_pos")


@pytest.mark.unit
def test_squeeze_neg():
    class M(nn.Module):
        def forward(self, z):
            return z.squeeze(-2)

    _check_op(M(), (torch.randn(3, 1, 4, dtype=torch.complex64),), "squeeze_neg")


@pytest.mark.unit
def test_squeeze_last_dim():
    """squeeze(dim=-1) removes the last *complex* dim (not real/imag encoding)."""

    class M(nn.Module):
        def forward(self, z):
            return z.squeeze(-1)

    _check_op(M(), (torch.randn(3, 1, dtype=torch.complex64),), "squeeze_last")


@pytest.mark.unit
def test_cat_dim0():
    class M(nn.Module):
        def forward(self, x, y):
            return torch.cat([x, y], dim=0)

    _check_op(M(), (_z(2, 4), _z(3, 4)), "cat_dim0")


@pytest.mark.unit
def test_cat_dim1():
    class M(nn.Module):
        def forward(self, x, y):
            return torch.cat([x, y], dim=1)

    _check_op(M(), (_z(3, 2), _z(3, 3)), "cat_dim1")


@pytest.mark.unit
def test_cat_neg_dim():
    """cat(tensors, dim=-1) on complex — must concat the last *complex* dim."""

    class M(nn.Module):
        def forward(self, x, y):
            return torch.cat([x, y], dim=-1)

    _check_op(M(), (_z(3, 2), _z(3, 3)), "cat_neg_dim")


@pytest.mark.unit
def test_stack_dim0():
    class M(nn.Module):
        def forward(self, x, y):
            return torch.stack([x, y], dim=0)

    _check_op(M(), (_z(), _z()), "stack_dim0")


@pytest.mark.unit
def test_stack_neg_dim():
    """stack(tensors, dim=-1) — new dim must land before real/imag encoding."""

    class M(nn.Module):
        def forward(self, x, y):
            return torch.stack([x, y], dim=-1)

    _check_op(M(), (_z(), _z()), "stack_neg_dim")


@pytest.mark.unit
def test_select_pos():
    class M(nn.Module):
        def forward(self, z):
            return z[1]

    _check_op(M(), (_z(),), "select_pos")


@pytest.mark.unit
def test_select_neg_dim():
    """select along the last complex dim (dim=-1)."""

    class M(nn.Module):
        def forward(self, z):
            return z.select(-1, 2)

    _check_op(M(), (_z(),), "select_neg_dim")


@pytest.mark.unit
def test_slice_pos():
    class M(nn.Module):
        def forward(self, z):
            return z[1:]

    _check_op(M(), (_z(),), "slice_pos")


@pytest.mark.unit
def test_slice_neg_dim():
    class M(nn.Module):
        def forward(self, z):
            return z[..., 1:3]

    _check_op(M(), (torch.randn(3, 6, dtype=torch.complex64),), "slice_neg_dim")


@pytest.mark.unit
def test_split():
    class M(nn.Module):
        def forward(self, z):
            a, b = z.split(2, dim=0)
            return a + b

    _check_op(M(), (torch.randn(4, 3, dtype=torch.complex64),), "split_pos")


@pytest.mark.unit
def test_split_neg_dim():
    class M(nn.Module):
        def forward(self, z):
            a, b = z.split(2, dim=-1)
            return a + b

    _check_op(M(), (torch.randn(3, 4, dtype=torch.complex64),), "split_neg")


@pytest.mark.unit
def test_chunk():
    class M(nn.Module):
        def forward(self, z):
            a, b = z.chunk(2, dim=0)
            return a * b

    _check_op(M(), (torch.randn(4, 3, dtype=torch.complex64),), "chunk_pos")


@pytest.mark.unit
def test_transpose_2d():
    class M(nn.Module):
        def forward(self, z):
            return z.transpose(0, 1)

    _check_op(M(), (_z(),), "transpose_2d")


@pytest.mark.unit
def test_transpose_neg():
    class M(nn.Module):
        def forward(self, z):
            return z.transpose(-2, -1)

    _check_op(M(), (_z3d(),), "transpose_neg")


@pytest.mark.unit
def test_t_default():
    """t.default (2D transpose) is elementwise-safe."""

    class M(nn.Module):
        def forward(self, z):
            return z.t()

    _check_op(M(), (_z(),), "t_default")


@pytest.mark.unit
def test_expand():
    class M(nn.Module):
        def forward(self, z):
            return z.expand(3, 4)

    _check_op(M(), (torch.randn(1, 4, dtype=torch.complex64),), "expand")


@pytest.mark.unit
def test_narrow_pos():
    """narrow along a non-negative dim — pass-through is correct."""

    class M(nn.Module):
        def forward(self, z):
            return z.narrow(0, 1, 2)

    _check_op(M(), (_z(),), "narrow_pos")


@pytest.mark.unit
def test_narrow_neg_dim():
    class M(nn.Module):
        def forward(self, z):
            return z.narrow(-1, 1, 2)

    _check_op(M(), (torch.randn(3, 5, dtype=torch.complex64),), "narrow_neg_dim")


@pytest.mark.unit
def test_roll_pos():
    """roll along a positive dim — pass-through is correct."""

    class M(nn.Module):
        def forward(self, z):
            return z.roll(1, 0)

    _check_op(M(), (_z(),), "roll_pos")


@pytest.mark.unit
def test_roll_neg_dim():
    class M(nn.Module):
        def forward(self, z):
            return z.roll(1, -1)

    _check_op(M(), (_z(),), "roll_neg_dim")


@pytest.mark.unit
def test_flip_pos():
    """flip along a positive dim — pass-through is correct."""

    class M(nn.Module):
        def forward(self, z):
            return z.flip([0])

    _check_op(M(), (_z(),), "flip_pos")


@pytest.mark.unit
def test_flip_neg_dim():
    class M(nn.Module):
        def forward(self, z):
            return z.flip([-1])

    _check_op(M(), (_z(),), "flip_neg_dim")


@pytest.mark.unit
def test_repeat():
    class M(nn.Module):
        def forward(self, z):
            return z.repeat(2, 1)

    _check_op(M(), (_z(),), "repeat")


# ===========================================================================
# 6. Matrix multiplication
# ===========================================================================


@pytest.mark.unit
def test_mm():
    class M(nn.Module):
        def forward(self, x, y):
            return torch.mm(x, y)

    x = torch.randn(3, 4, dtype=torch.complex64)
    y = torch.randn(4, 5, dtype=torch.complex64)
    _check_op(M(), (x, y), "mm")


@pytest.mark.unit
def test_bmm():
    class M(nn.Module):
        def forward(self, x, y):
            return torch.bmm(x, y)

    x = torch.randn(2, 3, 4, dtype=torch.complex64)
    y = torch.randn(2, 4, 5, dtype=torch.complex64)
    _check_op(M(), (x, y), "bmm")


@pytest.mark.unit
def test_matmul_2d():
    class M(nn.Module):
        def forward(self, x, y):
            return x @ y

    x = torch.randn(3, 4, dtype=torch.complex64)
    y = torch.randn(4, 5, dtype=torch.complex64)
    _check_op(M(), (x, y), "matmul_2d")


@pytest.mark.unit
def test_matmul_3d():
    class M(nn.Module):
        def forward(self, x, y):
            return x @ y

    x = torch.randn(2, 3, 4, dtype=torch.complex64)
    y = torch.randn(2, 4, 5, dtype=torch.complex64)
    _check_op(M(), (x, y), "matmul_3d")


@pytest.mark.unit
def test_mm_self_multiply():
    """mm(z, z) — self-multiplication should use the same node twice correctly."""

    class M(nn.Module):
        def forward(self, z):
            return torch.mm(z, z.t())

    z = torch.randn(4, 4, dtype=torch.complex64)
    _check_op(M(), (z,), "mm_self")


# ===========================================================================
# 7. Elementwise-safe pass-through verification
# ===========================================================================


@pytest.mark.unit
def test_clone():
    class M(nn.Module):
        def forward(self, z):
            return z.clone()

    _check_op(M(), (_z(),), "clone")


@pytest.mark.unit
def test_zeros_like():
    """zeros_like(z) → 0+0i (correct — all zeros in [..., 2] layout)."""

    class M(nn.Module):
        def forward(self, z):
            return torch.zeros_like(z)

    _check_op(M(), (_z(),), "zeros_like")


@pytest.mark.unit
def test_mul_scalar_elementwise():
    """mul.Scalar is elementwise-safe: scales both re and im."""

    class M(nn.Module):
        def forward(self, z):
            return torch.ops.aten.mul.Scalar(z, 2.5)

    _check_op(M(), (_z(),), "mul_scalar_aten")


@pytest.mark.unit
def test_div_scalar_elementwise():
    class M(nn.Module):
        def forward(self, z):
            return z / 4.0

    _check_op(M(), (_z(),), "div_scalar_elementwise")


# ===========================================================================
# 8. Reduction ops
#    — positive dims: pass-through gives correct results
#    — negative dims: no handler yet → xfail
# ===========================================================================


@pytest.mark.unit
def test_sum_pos_dim():
    """sum(z, dim=0) — positive dim, pass-through is correct."""

    class M(nn.Module):
        def forward(self, z):
            return z.sum(dim=0)

    _check_op(M(), (_z(),), "sum_pos")


@pytest.mark.unit
def test_sum_pos_dim_keepdim():
    class M(nn.Module):
        def forward(self, z):
            return z.sum(dim=1, keepdim=True)

    _check_op(M(), (_z3d(),), "sum_pos_keepdim")


@pytest.mark.unit
def test_sum_neg_dim():
    class M(nn.Module):
        def forward(self, z):
            return z.sum(dim=-1)

    _check_op(M(), (_z(),), "sum_neg")


@pytest.mark.unit
def test_mean_pos_dim():
    class M(nn.Module):
        def forward(self, z):
            return z.mean(dim=0)

    _check_op(M(), (_z(),), "mean_pos")


@pytest.mark.unit
def test_mean_neg_dim():
    class M(nn.Module):
        def forward(self, z):
            return z.mean(dim=-1)

    _check_op(M(), (_z(),), "mean_neg")


# ===========================================================================
# 9. Creation-op bugs (xfail = documented known failures)
# ===========================================================================


@pytest.mark.unit
def test_ones_like_bug():
    """ones_like(z) should give 1+0i everywhere, not 1+1i."""

    class M(nn.Module):
        def forward(self, z):
            return torch.ones_like(z)

    _check_op(M(), (_z(),), "ones_like")


@pytest.mark.unit
def test_full_like_bug():
    """full_like(z, 3.0) should give 3+0i everywhere."""

    class M(nn.Module):
        def forward(self, z):
            return torch.full_like(z, 3.0)

    _check_op(M(), (_z(),), "full_like")


# ===========================================================================
# 10. Chain / composition tests
# ===========================================================================


@pytest.mark.unit
def test_mul_then_exp():
    class M(nn.Module):
        def forward(self, x, y):
            return torch.exp(x * y)

    z = torch.randn(3, 4, dtype=torch.complex64) * 0.3
    _check_op(M(), (z, z.clone()), "mul_then_exp")


@pytest.mark.unit
def test_reshape_then_mul():
    class M(nn.Module):
        def forward(self, x, y):
            return x.reshape(12) * y

    x = _z()
    y = torch.randn(12, dtype=torch.complex64)
    _check_op(M(), (x, y), "reshape_then_mul")


@pytest.mark.unit
def test_mm_then_reshape():
    class M(nn.Module):
        def forward(self, x, y):
            return (x @ y).reshape(15)

    x = torch.randn(3, 4, dtype=torch.complex64)
    y = torch.randn(4, 5, dtype=torch.complex64)
    _check_op(M(), (x, y), "mm_then_reshape")


@pytest.mark.unit
def test_cat_then_exp():
    class M(nn.Module):
        def forward(self, x, y):
            return torch.exp(torch.cat([x, y], dim=0))

    z = torch.randn(2, 4, dtype=torch.complex64) * 0.3
    _check_op(M(), (z, z.clone()), "cat_then_exp")


@pytest.mark.unit
def test_unsqueeze_squeeze_round_trip():
    class M(nn.Module):
        def forward(self, z):
            return z.unsqueeze(1).squeeze(1)

    _check_op(M(), (_z(),), "unsqueeze_squeeze_rt")


@pytest.mark.unit
def test_permute_mul():
    class M(nn.Module):
        def forward(self, x, y):
            return x.permute(1, 0) * y.permute(1, 0)

    _check_op(M(), (_z(), _z()), "permute_mul")


@pytest.mark.unit
def test_transpose_then_mm():
    class M(nn.Module):
        def forward(self, x, y):
            return x @ y.transpose(-2, -1)

    x = torch.randn(3, 4, dtype=torch.complex64)
    y = torch.randn(5, 4, dtype=torch.complex64)
    _check_op(M(), (x, y), "transpose_mm")


@pytest.mark.unit
def test_rope_style_pattern():
    """RoPE-like pattern: split → mul with freqs → cat."""

    class M(nn.Module):
        def forward(self, q, freqs):
            # q: [B, T, D] complex, freqs: [T, D] complex
            return q * freqs.unsqueeze(0)

    q = _z3d(2, 8, 4)
    freqs = _z(8, 4)
    _check_op(M(), (q, freqs), "rope_style")


@pytest.mark.unit
def test_multiop_chain():
    """sin(exp(z) + conj(z)) — exercises several handlers in sequence."""

    class M(nn.Module):
        def forward(self, z):
            return torch.sin(torch.exp(z * 0.1) + torch.conj(z))

    z = torch.randn(3, 4, dtype=torch.complex64) * 0.2
    _check_op(M(), (z,), "multiop_chain")


@pytest.mark.unit
def test_abs_then_mul():
    """abs(z) is real; multiplying by a real scalar stays real."""

    class M(nn.Module):
        def forward(self, z):
            return torch.abs(z) * 2.0

    _check_op(M(), (_z(3, 5),), "abs_then_mul")


@pytest.mark.unit
def test_split_then_mul_then_cat():
    """split → element-wise mul → cat."""

    class M(nn.Module):
        def forward(self, z):
            a, b = z.split(2, dim=1)  # [3,2] each
            return torch.cat([a * b, b * a], dim=1)

    _check_op(M(), (torch.randn(3, 4, dtype=torch.complex64),), "split_mul_cat")
