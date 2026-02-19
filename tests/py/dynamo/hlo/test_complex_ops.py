"""
Numerical accuracy stress tests for complex tensor decomposition in torch-tensorrt.

The complex_graph_detection lowering pass rewrites complex-dtype ops to equivalent
real-arithmetic ops before TRT compilation. These tests verify correctness across:

  - I/O boundaries: complex inputs, complex outputs, mixed real/complex I/O
  - Internal subgraphs: complex ops entirely within a TRT block
  - Operator coverage: mul, add, sub, abs, angle, conj, real/imag extraction,
    gather/scatter (select, slice, index), reshape/view, cat/stack, where,
    unsqueeze/squeeze, expand/broadcast, type casting
  - Chains: multiple sequential complex ops
  - Multiple complex tensors interacting in one graph
  - Dynamic shapes: batch and seq_len as symbolic dims

All tests compare PyTorch (CPU/CUDA reference) vs TRT compiled output via
cosine similarity > COSINE_THRESHOLD on both real and imaginary parts.
"""

import pytest
import torch
import torch.nn as nn
import torch_tensorrt as torchtrt
from torch.export import Dim
from torch_tensorrt.dynamo.utils import COSINE_THRESHOLD, cosine_similarity


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_freqs(seq: int, dim: int, theta: float = 10000.0) -> torch.Tensor:
    """Complex unit-magnitude frequency tensor, shape (seq, dim//2)."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(seq, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs).cuda()


def _cossim_complex(py_out: torch.Tensor, trt_out: torch.Tensor, tag: str) -> None:
    """Assert cosine similarity on real and imaginary parts separately."""
    assert trt_out.is_complex(), f"{tag}: expected complex output, got {trt_out.dtype}"
    assert trt_out.shape == py_out.shape, f"{tag}: shape mismatch {trt_out.shape} vs {py_out.shape}"
    r = cosine_similarity(py_out.real.contiguous(), trt_out.real.contiguous())
    i = cosine_similarity(py_out.imag.contiguous(), trt_out.imag.contiguous())
    assert r > COSINE_THRESHOLD, f"{tag}: real part cosine sim {r:.4f} < {COSINE_THRESHOLD}"
    assert i > COSINE_THRESHOLD, f"{tag}: imag part cosine sim {i:.4f} < {COSINE_THRESHOLD}"


def _cossim_real(py_out: torch.Tensor, trt_out: torch.Tensor, tag: str) -> None:
    """Assert cosine similarity on a real-valued output."""
    assert not trt_out.is_complex(), f"{tag}: expected real output, got {trt_out.dtype}"
    s = cosine_similarity(py_out.contiguous(), trt_out.contiguous())
    assert s > COSINE_THRESHOLD, f"{tag}: cosine sim {s:.4f} < {COSINE_THRESHOLD}"


_COMPILE = dict(ir="dynamo", min_block_size=1, pass_through_build_failures=True)


# ===========================================================================
# 1. I/O boundary tests
# ===========================================================================

class ComplexInputRealOutput(nn.Module):
    """Complex input → real output (magnitude)."""
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: complex, output: real magnitude
        r = torch.view_as_real(z)
        real = r[..., 0]
        imag = r[..., 1]
        return torch.sqrt(real * real + imag * imag)


@pytest.mark.unit
def test_complex_input_real_output():
    model = ComplexInputRealOutput().eval().cuda()
    z = _make_freqs(8, 64)         # (8, 32) complex64
    inputs = (z,)
    trt_model = torchtrt.compile(model, inputs=inputs, **_COMPILE)
    py_out = model(*inputs)
    trt_out = trt_model(*inputs)
    _cossim_real(py_out, trt_out, "complex_input_real_output")
    torch._dynamo.reset()


class RealInputComplexOutput(nn.Module):
    """Real input → complex output (no view_as_real at graph output)."""
    def forward(self, x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        # x: real (B, S, H, D), freqs: complex (S, D//2)
        xc = torch.view_as_complex(x.reshape(*x.shape[:-1], -1, 2))
        return xc * freqs[None, :, None, :]  # complex output


@pytest.mark.unit
def test_real_input_complex_output():
    model = RealInputComplexOutput().eval().cuda()
    x = torch.randn(2, 8, 4, 64).cuda()
    freqs = _make_freqs(8, 64)
    inputs = (x, freqs)
    trt_model = torchtrt.compile(model, inputs=inputs, **_COMPILE)
    py_out = model(*inputs)
    trt_out = trt_model(*inputs)
    _cossim_complex(py_out, trt_out, "real_input_complex_output")
    torch._dynamo.reset()


class ComplexInputComplexOutput(nn.Module):
    """Complex input × complex input → complex output."""
    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a * b


@pytest.mark.unit
def test_complex_input_complex_output():
    model = ComplexInputComplexOutput().eval().cuda()
    a = _make_freqs(8, 64)
    b = _make_freqs(8, 64)
    inputs = (a, b)
    trt_model = torchtrt.compile(model, inputs=inputs, **_COMPILE)
    py_out = model(*inputs)
    trt_out = trt_model(*inputs)
    _cossim_complex(py_out, trt_out, "complex_input_complex_output")
    torch._dynamo.reset()


class MixedRealComplexInputRealOutput(nn.Module):
    """One real input, one complex input, real output."""
    def forward(self, x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        xc = torch.view_as_complex(x.reshape(*x.shape[:-1], -1, 2))
        prod = xc * freqs[None, :, None, :]
        return torch.view_as_real(prod).flatten(3)


@pytest.mark.unit
def test_mixed_real_complex_input_real_output():
    model = MixedRealComplexInputRealOutput().eval().cuda()
    x = torch.randn(2, 8, 4, 64).cuda()
    freqs = _make_freqs(8, 64)
    inputs = (x, freqs)
    trt_model = torchtrt.compile(model, inputs=inputs, **_COMPILE)
    py_out = model(*inputs)
    trt_out = trt_model(*inputs)
    _cossim_real(py_out, trt_out, "mixed_io_real_output")
    torch._dynamo.reset()


# ===========================================================================
# 2. Operator coverage
# ===========================================================================

class ComplexAdd(nn.Module):
    """Complex addition: (a+bi) + (c+di) = (a+c) + (b+d)i."""
    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a + b


@pytest.mark.unit
def test_complex_add_output():
    model = ComplexAdd().eval().cuda()
    a = _make_freqs(8, 64)
    b = _make_freqs(8, 64)
    inputs = (a, b)
    trt_model = torchtrt.compile(model, inputs=inputs, **_COMPILE)
    py_out = model(*inputs)
    trt_out = trt_model(*inputs)
    _cossim_complex(py_out, trt_out, "complex_add")
    torch._dynamo.reset()


class ComplexSub(nn.Module):
    """Complex subtraction."""
    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a - b


@pytest.mark.unit
def test_complex_sub_output():
    model = ComplexSub().eval().cuda()
    a = _make_freqs(8, 64)
    b = _make_freqs(8, 64)
    inputs = (a, b)
    trt_model = torchtrt.compile(model, inputs=inputs, **_COMPILE)
    py_out = model(*inputs)
    trt_out = trt_model(*inputs)
    _cossim_complex(py_out, trt_out, "complex_sub")
    torch._dynamo.reset()


class ComplexMulChain(nn.Module):
    """Chain of two complex multiplications: (a * b) * c."""
    def forward(self, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        return (a * b) * c


@pytest.mark.unit
def test_complex_mul_chain():
    model = ComplexMulChain().eval().cuda()
    a = _make_freqs(8, 64)
    b = _make_freqs(8, 64)
    c = _make_freqs(8, 64)
    inputs = (a, b, c)
    trt_model = torchtrt.compile(model, inputs=inputs, **_COMPILE)
    py_out = model(*inputs)
    trt_out = trt_model(*inputs)
    _cossim_complex(py_out, trt_out, "complex_mul_chain")
    torch._dynamo.reset()


class ComplexDiv(nn.Module):
    """Complex division: (a+bi)/(c+di) = ((ac+bd) + (bc-ad)i) / (c²+d²)."""
    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a / b


@pytest.mark.unit
def test_complex_div():
    model = ComplexDiv().eval().cuda()
    a = _make_freqs(8, 64)
    b = _make_freqs(8, 64, theta=500.0)  # different theta → different angles → non-trivial imaginary
    inputs = (a, b)
    trt_model = torchtrt.compile(model, inputs=inputs, **_COMPILE)
    py_out = model(*inputs)
    trt_out = trt_model(*inputs)
    _cossim_complex(py_out, trt_out, "complex_div")
    torch._dynamo.reset()


class ComplexScalarMul(nn.Module):
    """Scale a complex tensor by a real scalar."""
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # Scale by 2.0 — real * complex
        r = torch.view_as_real(z)
        scaled = r * 2.0
        return torch.view_as_complex(scaled)


@pytest.mark.unit
def test_complex_scalar_mul_output():
    model = ComplexScalarMul().eval().cuda()
    z = _make_freqs(8, 64)
    inputs = (z,)
    trt_model = torchtrt.compile(model, inputs=inputs, **_COMPILE)
    py_out = model(*inputs)
    trt_out = trt_model(*inputs)
    _cossim_complex(py_out, trt_out, "complex_scalar_mul")
    torch._dynamo.reset()


class ComplexAbs(nn.Module):
    """Complex magnitude: |z| = sqrt(re^2 + im^2) — real output."""
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        r = torch.view_as_real(z)
        return (r * r).sum(-1).sqrt()


@pytest.mark.unit
def test_complex_abs():
    model = ComplexAbs().eval().cuda()
    z = _make_freqs(8, 64)
    inputs = (z,)
    trt_model = torchtrt.compile(model, inputs=inputs, **_COMPILE)
    py_out = model(*inputs)
    trt_out = trt_model(*inputs)
    _cossim_real(py_out, trt_out, "complex_abs")
    torch._dynamo.reset()


class ComplexAbsNative(nn.Module):
    """torch.abs on a complex tensor — exercises the aten.abs.default rewrite."""
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return torch.abs(z)


@pytest.mark.unit
def test_complex_abs_native():
    model = ComplexAbsNative().eval().cuda()
    z = torch.polar(2 * torch.ones(8, 32), torch.randn(8, 32)).cuda()
    inputs = (z,)
    trt_model = torchtrt.compile(model, inputs=inputs, **_COMPILE)
    py_out = model(*inputs)
    trt_out = trt_model(*inputs)
    _cossim_real(py_out, trt_out, "complex_abs_native")
    torch._dynamo.reset()


class ComplexExp(nn.Module):
    """torch.exp on a complex tensor: exp(a+bi) = e^a*(cos b + i sin b)."""
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return torch.exp(z)


@pytest.mark.unit
def test_complex_exp():
    model = ComplexExp().eval().cuda()
    # small magnitudes to keep exp from overflowing
    z = torch.polar(0.1 * torch.ones(8, 32), torch.randn(8, 32)).cuda()
    inputs = (z,)
    trt_model = torchtrt.compile(model, inputs=inputs, **_COMPILE)
    py_out = model(*inputs)
    trt_out = trt_model(*inputs)
    _cossim_complex(py_out, trt_out, "complex_exp")
    torch._dynamo.reset()


class ComplexLog(nn.Module):
    """torch.log on a complex tensor: log(a+bi) = log|z| + i*angle(z)."""
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return torch.log(z)


@pytest.mark.unit
def test_complex_log():
    model = ComplexLog().eval().cuda()
    z = torch.polar(2 * torch.ones(8, 32), torch.randn(8, 32)).cuda()
    inputs = (z,)
    trt_model = torchtrt.compile(model, inputs=inputs, **_COMPILE)
    py_out = model(*inputs)
    trt_out = trt_model(*inputs)
    _cossim_complex(py_out, trt_out, "complex_log")
    torch._dynamo.reset()


class ComplexPow(nn.Module):
    """z**n via polar form: r^n * (cos(nθ) + i*sin(nθ))."""
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return z ** 3


@pytest.mark.unit
def test_complex_pow():
    model = ComplexPow().eval().cuda()
    z = torch.polar(torch.ones(8, 32), torch.randn(8, 32)).cuda()
    inputs = (z,)
    trt_model = torchtrt.compile(model, inputs=inputs, **_COMPILE)
    py_out = model(*inputs)
    trt_out = trt_model(*inputs)
    _cossim_complex(py_out, trt_out, "complex_pow")
    torch._dynamo.reset()


class ComplexSqrt(nn.Module):
    """torch.sqrt on a complex tensor: z**0.5."""
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(z)


@pytest.mark.unit
def test_complex_sqrt():
    model = ComplexSqrt().eval().cuda()
    z = torch.polar(4 * torch.ones(8, 32), torch.randn(8, 32)).cuda()
    inputs = (z,)
    trt_model = torchtrt.compile(model, inputs=inputs, **_COMPILE)
    py_out = model(*inputs)
    trt_out = trt_model(*inputs)
    _cossim_complex(py_out, trt_out, "complex_sqrt")
    torch._dynamo.reset()


class ComplexConj(nn.Module):
    """torch.conj on a complex tensor — exercises the _conj rewrite."""
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return torch.conj(z)


@pytest.mark.unit
def test_complex_conj():
    model = ComplexConj().eval().cuda()
    z = _make_freqs(8, 64)
    inputs = (z,)
    trt_model = torchtrt.compile(model, inputs=inputs, **_COMPILE)
    py_out = model(*inputs).resolve_conj()
    trt_out = trt_model(*inputs)
    _cossim_complex(py_out, trt_out, "complex_conj")
    torch._dynamo.reset()


class ComplexConjMul(nn.Module):
    """z * conj(z) = |z|^2 — real-valued result returned as complex."""
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        r = torch.view_as_real(z)
        re, im = r[..., 0], r[..., 1]
        # conj(z) has same real, negated imag
        real_part = re * re + im * im   # ac - b(-d) = ac + bd when c=a, d=-b
        imag_part = torch.zeros_like(real_part)
        return torch.view_as_complex(
            torch.stack([real_part, imag_part], dim=-1)
        )


@pytest.mark.unit
def test_complex_conj_mul():
    model = ComplexConjMul().eval().cuda()
    z = _make_freqs(8, 64)
    inputs = (z,)
    trt_model = torchtrt.compile(model, inputs=inputs, **_COMPILE)
    py_out = model(*inputs)
    trt_out = trt_model(*inputs)
    _cossim_complex(py_out, trt_out, "complex_conj_mul")
    torch._dynamo.reset()


# ===========================================================================
# 3. Gather/scatter: select, slice, index
# ===========================================================================

class ComplexSelect(nn.Module):
    """Select a slice along a dimension from a complex tensor."""
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (S, D) complex — select first half of S, then mul
        half = z[:4, :]           # slice along seq dim
        return half * z[4:, :]    # element-wise complex mul, real output via view_as_real
        # returns complex — covered by complex output test


@pytest.mark.unit
def test_complex_select_and_mul():
    model = ComplexSelect().eval().cuda()
    z = _make_freqs(8, 64)   # (8, 32) complex64
    inputs = (z,)
    trt_model = torchtrt.compile(model, inputs=inputs, **_COMPILE)
    py_out = model(*inputs)
    trt_out = trt_model(*inputs)
    _cossim_complex(py_out, trt_out, "complex_select_mul")
    torch._dynamo.reset()


class ComplexSlice(nn.Module):
    """Slice two halves of a complex tensor and multiply them."""
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (S, D) complex — split into first and second half along D
        half_d = z.shape[-1] // 2
        a = z[:, :half_d]   # (S, D//2) complex
        b = z[:, half_d:]   # (S, D//2) complex
        return a * b         # complex output


@pytest.mark.unit
def test_complex_slice_and_mul():
    model = ComplexSlice().eval().cuda()
    z = _make_freqs(8, 64)   # (8, 32) complex64
    inputs = (z,)
    trt_model = torchtrt.compile(model, inputs=inputs, **_COMPILE)
    py_out = model(*inputs)
    trt_out = trt_model(*inputs)
    _cossim_complex(py_out, trt_out, "complex_slice_mul")
    torch._dynamo.reset()


# ===========================================================================
# 4. Shape manipulation: reshape, unsqueeze, squeeze, expand, flatten
# ===========================================================================

class ComplexReshapeAndMul(nn.Module):
    """Reshape a complex tensor, then multiply."""
    def forward(self, x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        # x: real (B, S, H*D), freqs: complex (S, D//2)
        B, S, HD = x.shape
        H = 4
        D = HD // H
        xr = x.view(B, S, H, D)
        xc = torch.view_as_complex(xr.reshape(B, S, H, -1, 2))  # (B,S,H,D//2) complex
        return torch.view_as_real(xc * freqs[None, :, None, :]).flatten(3)


@pytest.mark.unit
def test_complex_reshape_and_mul():
    model = ComplexReshapeAndMul().eval().cuda()
    x = torch.randn(2, 8, 64).cuda()
    freqs = _make_freqs(8, 16)   # (8, 8) complex, head_dim=16 -> D//2=8
    inputs = (x, freqs)
    trt_model = torchtrt.compile(model, inputs=inputs, **_COMPILE)
    py_out = model(*inputs)
    trt_out = trt_model(*inputs)
    _cossim_real(py_out, trt_out, "complex_reshape_mul")
    torch._dynamo.reset()


class ComplexUnsqueezeExpand(nn.Module):
    """Unsqueeze and expand a complex tensor before multiplication."""
    def forward(self, z: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        # z: (S, D) complex, freqs: (D,) complex
        # unsqueeze freqs to broadcast over S
        return z * freqs.unsqueeze(0)  # (S,D) complex output


@pytest.mark.unit
def test_complex_unsqueeze_expand():
    model = ComplexUnsqueezeExpand().eval().cuda()
    z = _make_freqs(8, 64)      # (8, 32)
    freqs = _make_freqs(1, 64).squeeze(0)  # (32,)
    inputs = (z, freqs)
    trt_model = torchtrt.compile(model, inputs=inputs, **_COMPILE)
    py_out = model(*inputs)
    trt_out = trt_model(*inputs)
    _cossim_complex(py_out, trt_out, "complex_unsqueeze_expand")
    torch._dynamo.reset()


# ===========================================================================
# 5. Concatenation and stacking
# ===========================================================================

class ComplexCat(nn.Module):
    """Concatenate two complex tensors along the sequence dimension."""
    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.cat([a, b], dim=0)  # (2S, D) complex output


@pytest.mark.unit
def test_complex_cat():
    model = ComplexCat().eval().cuda()
    a = _make_freqs(4, 64)
    b = _make_freqs(4, 64)
    inputs = (a, b)
    trt_model = torchtrt.compile(model, inputs=inputs, **_COMPILE)
    py_out = model(*inputs)
    trt_out = trt_model(*inputs)
    _cossim_complex(py_out, trt_out, "complex_cat")
    torch._dynamo.reset()


class ComplexCatThenMul(nn.Module):
    """Concatenate two complex tensors then multiply by a third."""
    def forward(self, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        ab = torch.cat([a, b], dim=0)  # (2S, D)
        return ab * c                  # complex output


@pytest.mark.unit
def test_complex_cat_then_mul():
    model = ComplexCatThenMul().eval().cuda()
    a = _make_freqs(4, 64)
    b = _make_freqs(4, 64)
    c = _make_freqs(8, 64)
    inputs = (a, b, c)
    trt_model = torchtrt.compile(model, inputs=inputs, **_COMPILE)
    py_out = model(*inputs)
    trt_out = trt_model(*inputs)
    _cossim_complex(py_out, trt_out, "complex_cat_then_mul")
    torch._dynamo.reset()


class ComplexStackRealView(nn.Module):
    """Stack real-view representations of two complex tensors, then multiply.

    Tests that the rewriter correctly handles complex ops on stacked real tensors:
    view_as_real(a) and view_as_real(b) are stacked, then used to form two
    independent complex multiplications.
    """
    def forward(self, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        # a, b: (S, D) complex, c: (S, D) complex
        # Multiply each independently and add — tests multiple complex paths
        return torch.view_as_real(a * c).flatten(-2) + torch.view_as_real(b * c).flatten(-2)


@pytest.mark.unit
def test_complex_stack_real_view():
    model = ComplexStackRealView().eval().cuda()
    a = _make_freqs(8, 64)
    b = _make_freqs(8, 64)
    c = _make_freqs(8, 64)
    inputs = (a, b, c)
    trt_model = torchtrt.compile(model, inputs=inputs, **_COMPILE)
    py_out = model(*inputs)
    trt_out = trt_model(*inputs)
    _cossim_real(py_out, trt_out, "complex_stack_real_view")
    torch._dynamo.reset()


# ===========================================================================
# 6. Where / masked selection
# ===========================================================================

class ComplexWhere(nn.Module):
    """Conditional selection between two complex tensors."""
    def forward(self, mask: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # Operate on real/imag separately — where doesn't support complex natively
        ar = torch.view_as_real(a)
        br = torch.view_as_real(b)
        m = mask.unsqueeze(-1)       # broadcast over last (2,) dim
        out = torch.where(m, ar, br)
        return torch.view_as_complex(out.contiguous())


@pytest.mark.unit
def test_complex_where():
    model = ComplexWhere().eval().cuda()
    a = _make_freqs(8, 64)
    b = _make_freqs(8, 64)
    mask = (torch.randn(8, 32) > 0).cuda()
    inputs = (mask, a, b)
    trt_model = torchtrt.compile(model, inputs=inputs, **_COMPILE)
    py_out = model(*inputs)
    trt_out = trt_model(*inputs)
    _cossim_complex(py_out, trt_out, "complex_where")
    torch._dynamo.reset()


# ===========================================================================
# 7. Multiple complex subgraphs in one model
# ===========================================================================

class DualComplexPath(nn.Module):
    """Two independent complex multiplications merged at the output.

    freqs is passed already broadcast-ready (same shape as the complex view of x/y)
    so no indexing/unsqueeze is needed on the complex tensor.
    """
    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        freqs: torch.Tensor,
    ) -> torch.Tensor:
        # Path A: x rotated by freqs
        xa = torch.view_as_complex(x.reshape(*x.shape[:-1], -1, 2))
        out_a = torch.view_as_real(xa * freqs).flatten(3)
        # Path B: y rotated by same freqs
        xb = torch.view_as_complex(y.reshape(*y.shape[:-1], -1, 2))
        out_b = torch.view_as_real(xb * freqs).flatten(3)
        return out_a + out_b   # real output


@pytest.mark.unit
def test_dual_complex_path():
    model = DualComplexPath().eval().cuda()
    x = torch.randn(2, 8, 4, 64).cuda()
    y = torch.randn(2, 8, 4, 64).cuda()
    # freqs must match the complex view shape (2,8,4,32) — broadcast via register_buffer
    freqs = _make_freqs(8, 64).unsqueeze(0).unsqueeze(2).expand(2, 8, 4, 32).contiguous()
    inputs = (x, y, freqs)
    trt_model = torchtrt.compile(model, inputs=inputs, **_COMPILE)
    py_out = model(*inputs)
    trt_out = trt_model(*inputs)
    _cossim_real(py_out, trt_out, "dual_complex_path")
    torch._dynamo.reset()


# ===========================================================================
# 8. Complex ops interleaved with real ops
# ===========================================================================

class ComplexSandwich(nn.Module):
    """Real → complex → real → linear → complex → real sandwich.

    Uses a buffer for freqs so the complex tensor is a get_attr (not placeholder),
    which the rewriter handles via stacked real tensor.
    """
    def __init__(self, freqs: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("freqs", freqs)
        self.linear = nn.Linear(64, 64, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # real → complex rotation using buffer freqs (get_attr path)
        xc = torch.view_as_complex(x.reshape(*x.shape[:-1], -1, 2))
        rotated = torch.view_as_real(xc * self.freqs).flatten(3)  # (B,S,H,D) real
        # real linear
        out = self.linear(rotated)
        # another complex rotation
        outc = torch.view_as_complex(out.reshape(*out.shape[:-1], -1, 2))
        return torch.view_as_real(outc * self.freqs).flatten(3)


@pytest.mark.unit
def test_complex_sandwich():
    freqs = _make_freqs(8, 64).unsqueeze(0).unsqueeze(2).expand(2, 8, 4, 32).contiguous()
    model = ComplexSandwich(freqs).eval().cuda()
    x = torch.randn(2, 8, 4, 64).cuda()
    inputs = (x,)
    trt_model = torchtrt.compile(model, inputs=inputs, **_COMPILE)
    py_out = model(*inputs)
    trt_out = trt_model(*inputs)
    _cossim_real(py_out, trt_out, "complex_sandwich")
    torch._dynamo.reset()


# ===========================================================================
# 9. Complex nn.Parameter (get_attr path)
# ===========================================================================

class ComplexParamMul(nn.Module):
    """Complex weight stored as nn.Parameter — exercises the get_attr rewrite path."""
    def __init__(self, freqs: torch.Tensor) -> None:
        super().__init__()
        # nn.Parameter, not register_buffer — still a get_attr node in the exported graph
        self.freqs = nn.Parameter(freqs, requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xc = torch.view_as_complex(x.reshape(*x.shape[:-1], -1, 2))
        return torch.view_as_real(xc * self.freqs).flatten(3)


@pytest.mark.unit
def test_complex_param_get_attr():
    freqs = _make_freqs(8, 64).unsqueeze(0).unsqueeze(2).expand(2, 8, 4, 32).contiguous()
    model = ComplexParamMul(freqs).eval().cuda()
    x = torch.randn(2, 8, 4, 64).cuda()
    inputs = (x,)
    trt_model = torchtrt.compile(model, inputs=inputs, **_COMPILE)
    py_out = model(*inputs)
    trt_out = trt_model(*inputs)
    _cossim_real(py_out, trt_out, "complex_param_get_attr")
    torch._dynamo.reset()


# ===========================================================================
# 10. Dynamic shapes
# ===========================================================================

class ComplexMulDynamic(nn.Module):
    """Complex RoPE with dynamic seq_len."""
    def forward(self, x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        xc = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        return torch.view_as_real(xc * freqs[None, :, None, :]).flatten(3)


@pytest.mark.unit
def test_complex_mul_dynamic_seqlen():
    """Dynamic seq_len: x has shape (B, seq, H, D), freqs has shape (seq, D//2)."""
    model = ComplexMulDynamic().eval().cuda()
    x = torch.randn(2, 8, 4, 64).cuda()
    freqs = _make_freqs(8, 64)   # (8, 32)
    inputs = (x, freqs)

    # x dim-1 and freqs dim-0 are both the seq dimension — share the same Dim
    seq = Dim("seq", min=2, max=64)
    dynamic_shapes = ({1: seq}, {0: seq})
    ep = torch.export.export(model, inputs, dynamic_shapes=dynamic_shapes)
    trt_model = torchtrt.dynamo.compile(
        ep, inputs=inputs, min_block_size=1, pass_through_build_failures=True
    )
    py_out = model(*inputs)
    trt_out = trt_model(*inputs)
    _cossim_real(py_out, trt_out, "complex_mul_dynamic_seqlen")
    torch._dynamo.reset()


class ComplexOutputDynamic(nn.Module):
    """Complex output with dynamic batch."""
    def forward(self, x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        xc = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        return xc * freqs[None, :, None, :]  # complex output


@pytest.mark.unit
def test_complex_output_dynamic_batch():
    model = ComplexOutputDynamic().eval().cuda()
    x = torch.randn(2, 8, 4, 64).cuda()
    freqs = _make_freqs(8, 64)
    inputs = (x, freqs)

    batch = Dim("batch", min=1, max=8)
    dynamic_shapes = ({0: batch}, {})
    ep = torch.export.export(model, inputs, dynamic_shapes=dynamic_shapes)
    trt_model = torchtrt.dynamo.compile(
        ep, inputs=inputs, min_block_size=1, pass_through_build_failures=True
    )
    py_out = model(*inputs)
    trt_out = trt_model(*inputs)
    _cossim_complex(py_out, trt_out, "complex_output_dynamic_batch")
    torch._dynamo.reset()


# ===========================================================================
# 11. Numerical precision: complex64 vs truncated complex128
# ===========================================================================

class Complex128Model(nn.Module):
    """Uses complex128 (double precision)."""
    def forward(self, z: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        return torch.view_as_real(z * w).flatten(-2)


@pytest.mark.unit
def test_complex128_truncated_to_float32():
    """complex128 with truncate_double=True should compile to float32 arithmetic."""
    model = Complex128Model().eval().cuda()
    z = torch.polar(
        torch.ones(8, 32, dtype=torch.float64),
        torch.randn(8, 32, dtype=torch.float64),
    ).cuda()
    w = torch.polar(
        torch.ones(8, 32, dtype=torch.float64),
        torch.randn(8, 32, dtype=torch.float64),
    ).cuda()
    inputs = (z, w)
    trt_model = torchtrt.compile(
        model,
        inputs=inputs,
        ir="dynamo",
        min_block_size=1,
        pass_through_build_failures=True,
        truncate_double=True,
    )
    py_out = model(*inputs).float()  # cast reference to float32 for comparison
    trt_out = trt_model(*inputs)
    _cossim_real(py_out, trt_out, "complex128_truncated")
    torch._dynamo.reset()


# ===========================================================================
# 12. End-to-end: full attention-style block with complex RoPE
# ===========================================================================

class AttentionWithComplexRoPE(nn.Module):
    """Multi-head self-attention with complex-number RoPE and real output."""
    def __init__(self, d_model: int = 64, n_heads: int = 4) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def _apply_rope(self, x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        B, S, H, D = x.shape
        xc = torch.view_as_complex(x.reshape(B, S, H, -1, 2))
        return torch.view_as_real(xc * freqs[None, :, None, :]).flatten(3)

    def forward(self, x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        B, S, C = x.shape
        H, D = self.n_heads, self.head_dim
        q = self.q_proj(x).view(B, S, H, D)
        k = self.k_proj(x).view(B, S, H, D)
        v = self.v_proj(x).view(B, S, H, D)
        q = self._apply_rope(q, freqs)
        k = self._apply_rope(k, freqs)
        # Scaled dot-product attention
        scale = D ** -0.5
        attn = torch.einsum("bshd,bthd->bhst", q, k) * scale
        attn = torch.softmax(attn, dim=-1)
        out = torch.einsum("bhst,bthd->bshd", attn, v).reshape(B, S, C)
        return self.out_proj(out)


@pytest.mark.unit
def test_attention_with_complex_rope_static():
    model = AttentionWithComplexRoPE(d_model=64, n_heads=4).eval().cuda()
    x = torch.randn(2, 8, 64).cuda()
    freqs = _make_freqs(8, 16)   # head_dim=16, D//2=8
    inputs = (x, freqs)
    trt_model = torchtrt.compile(model, inputs=inputs, **_COMPILE)
    py_out = model(*inputs)
    trt_out = trt_model(*inputs)
    _cossim_real(py_out, trt_out, "attention_with_complex_rope")
    torch._dynamo.reset()


# ===========================================================================
# 13. Elementwise-safe structural ops (clone, permute)
# ===========================================================================


class ComplexClone(nn.Module):
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return z.clone() * z


@pytest.mark.unit
def test_complex_clone():
    model = ComplexClone().eval().cuda()
    z = _make_freqs(8, 32)
    trt_model = torchtrt.compile(model, inputs=(z,), **_COMPILE)
    _cossim_complex(model(z), trt_model(z), "complex_clone")
    torch._dynamo.reset()


class ComplexPermute(nn.Module):
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # permute spatial dims only, then apply mul so complex subgraph is detected
        return z.permute(1, 0) * z.permute(1, 0)


@pytest.mark.unit
def test_complex_permute():
    model = ComplexPermute().eval().cuda()
    z = _make_freqs(8, 32)  # (8, 16) complex64
    trt_model = torchtrt.compile(model, inputs=(z,), **_COMPILE)
    _cossim_complex(model(z), trt_model(z), "complex_permute")
    torch._dynamo.reset()


# ===========================================================================
# 14. Extraction / construction ops
# ===========================================================================


class ComplexRealExtract(nn.Module):
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return z.real


@pytest.mark.unit
def test_complex_real():
    model = ComplexRealExtract().eval().cuda()
    z = _make_freqs(8, 32)
    trt_model = torchtrt.compile(model, inputs=(z,), **_COMPILE)
    py_out = model(z)
    trt_out = trt_model(z)
    _cossim_real(py_out, trt_out, "complex_real")
    torch._dynamo.reset()


class ComplexImagExtract(nn.Module):
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return z.imag


@pytest.mark.unit
def test_complex_imag():
    model = ComplexImagExtract().eval().cuda()
    z = _make_freqs(8, 32)
    trt_model = torchtrt.compile(model, inputs=(z,), **_COMPILE)
    py_out = model(z)
    trt_out = trt_model(z)
    _cossim_real(py_out, trt_out, "complex_imag")
    torch._dynamo.reset()


class ComplexAngle(nn.Module):
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return torch.angle(z)


@pytest.mark.unit
def test_complex_angle():
    model = ComplexAngle().eval().cuda()
    z = _make_freqs(8, 32)
    trt_model = torchtrt.compile(model, inputs=(z,), **_COMPILE)
    py_out = model(z)
    trt_out = trt_model(z)
    _cossim_real(py_out, trt_out, "complex_angle")
    torch._dynamo.reset()


class ComplexPolar(nn.Module):
    def forward(self, r: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        return torch.polar(r, theta)


@pytest.mark.unit
def test_complex_polar():
    r = torch.rand(8, 16, device="cuda") + 0.1
    theta = torch.rand(8, 16, device="cuda") * 2 * 3.14159
    model = ComplexPolar().eval().cuda()
    trt_model = torchtrt.compile(model, inputs=(r, theta), **_COMPILE)
    py_out = model(r, theta)
    trt_out = trt_model(r, theta)
    _cossim_complex(py_out, trt_out, "complex_polar")
    torch._dynamo.reset()


class ComplexReciprocal(nn.Module):
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return torch.reciprocal(z)


@pytest.mark.unit
def test_complex_reciprocal():
    model = ComplexReciprocal().eval().cuda()
    # Use non-unit magnitude to avoid trivial 1/z=conj(z) for |z|=1
    z = _make_freqs(8, 32) * 2.0
    trt_model = torchtrt.compile(model, inputs=(z,), **_COMPILE)
    _cossim_complex(model(z), trt_model(z), "complex_reciprocal")
    torch._dynamo.reset()


class ComplexRsqrt(nn.Module):
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return torch.rsqrt(z)


@pytest.mark.unit
def test_complex_rsqrt():
    model = ComplexRsqrt().eval().cuda()
    # Use polar form with r > 0 so rsqrt is well-defined
    r = torch.rand(8, 16, device="cuda") + 0.5
    theta = torch.rand(8, 16, device="cuda") * 3.14159
    z = torch.polar(r, theta)
    trt_model = torchtrt.compile(model, inputs=(z,), **_COMPILE)
    _cossim_complex(model(z), trt_model(z), "complex_rsqrt")
    torch._dynamo.reset()


class ComplexAddScalar(nn.Module):
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # Exercise add.Scalar: (a+bi)+2 = (a+2)+bi
        return torch.view_as_complex(torch.view_as_real(z).add(0.0)) + 2.0


@pytest.mark.unit
def test_complex_add_scalar():
    """add.Scalar: scalar adds to real part only — (a+2) + bi."""
    model = ComplexAddScalar().eval().cuda()
    z = _make_freqs(8, 32)
    trt_model = torchtrt.compile(model, inputs=(z,), **_COMPILE)
    _cossim_complex(model(z), trt_model(z), "complex_add_scalar")
    torch._dynamo.reset()


class ComplexSgn(nn.Module):
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return torch.sgn(z)


@pytest.mark.unit
def test_complex_sgn():
    """sgn(z) = z/|z|, sgn(0) = 0."""
    model = ComplexSgn().eval().cuda()
    r = torch.rand(8, 16, device="cuda") + 0.1
    theta = torch.rand(8, 16, device="cuda") * 2 * 3.14159
    # Include one zero entry
    r[0, 0] = 0.0
    theta[0, 0] = 0.0
    z = torch.polar(r, theta)
    trt_model = torchtrt.compile(model, inputs=(z,), **_COMPILE)
    _cossim_complex(model(z), trt_model(z), "complex_sgn")
    torch._dynamo.reset()


class ComplexLog2(nn.Module):
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return torch.log2(z)


@pytest.mark.unit
def test_complex_log2():
    model = ComplexLog2().eval().cuda()
    r = torch.rand(8, 16, device="cuda") + 0.5
    theta = torch.rand(8, 16, device="cuda") * 3.14159
    z = torch.polar(r, theta)
    trt_model = torchtrt.compile(model, inputs=(z,), **_COMPILE)
    _cossim_complex(model(z), trt_model(z), "complex_log2")
    torch._dynamo.reset()


class ComplexLog10(nn.Module):
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return torch.log10(z)


@pytest.mark.unit
def test_complex_log10():
    model = ComplexLog10().eval().cuda()
    r = torch.rand(8, 16, device="cuda") + 0.5
    theta = torch.rand(8, 16, device="cuda") * 3.14159
    z = torch.polar(r, theta)
    trt_model = torchtrt.compile(model, inputs=(z,), **_COMPILE)
    _cossim_complex(model(z), trt_model(z), "complex_log10")
    torch._dynamo.reset()


class ComplexLog1p(nn.Module):
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return torch.log1p(z)


@pytest.mark.unit
def test_complex_log1p():
    model = ComplexLog1p().eval().cuda()
    # |z| < 1 for numerical stability
    r = torch.rand(8, 16, device="cuda") * 0.5
    theta = torch.rand(8, 16, device="cuda") * 2 * 3.14159
    z = torch.polar(r, theta)
    trt_model = torchtrt.compile(model, inputs=(z,), **_COMPILE)
    _cossim_complex(model(z), trt_model(z), "complex_log1p")
    torch._dynamo.reset()


class ComplexExpm1(nn.Module):
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return torch.expm1(z)


@pytest.mark.unit
def test_complex_expm1():
    model = ComplexExpm1().eval().cuda()
    # Small magnitude to avoid exp overflow
    r = torch.rand(8, 16, device="cuda") * 0.3
    theta = torch.rand(8, 16, device="cuda") * 2 * 3.14159
    z = torch.polar(r, theta)
    trt_model = torchtrt.compile(model, inputs=(z,), **_COMPILE)
    _cossim_complex(model(z), trt_model(z), "complex_expm1")
    torch._dynamo.reset()


class ComplexIsnan(nn.Module):
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # Output bool → cast to float for cosine sim
        return torch.isnan(z).float()


@pytest.mark.unit
def test_complex_isnan():
    model = ComplexIsnan().eval().cuda()
    z = _make_freqs(8, 32)
    trt_model = torchtrt.compile(model, inputs=(z,), **_COMPILE)
    py_out = model(z)
    trt_out = trt_model(z)
    # All-zero output: check element-wise equality
    assert torch.allclose(py_out, trt_out), "complex_isnan: output mismatch"
    torch._dynamo.reset()


class ComplexIsinf(nn.Module):
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return torch.isinf(z).float()


@pytest.mark.unit
def test_complex_isinf():
    model = ComplexIsinf().eval().cuda()
    z = _make_freqs(8, 32)
    trt_model = torchtrt.compile(model, inputs=(z,), **_COMPILE)
    py_out = model(z)
    trt_out = trt_model(z)
    assert torch.allclose(py_out, trt_out), "complex_isinf: output mismatch"
    torch._dynamo.reset()


# ===========================================================================
# 15. Trigonometric ops
# ===========================================================================


class ComplexSin(nn.Module):
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return torch.sin(z)


@pytest.mark.unit
def test_complex_sin():
    model = ComplexSin().eval().cuda()
    r = torch.ones(8, 16, device="cuda") * 0.5
    theta = torch.linspace(0.1, 1.5, 16, device="cuda").unsqueeze(0).expand(8, -1)
    z = torch.polar(r, theta)
    trt_model = torchtrt.compile(model, inputs=(z,), **_COMPILE)
    _cossim_complex(model(z), trt_model(z), "complex_sin")
    torch._dynamo.reset()


class ComplexCos(nn.Module):
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return torch.cos(z)


@pytest.mark.unit
def test_complex_cos():
    model = ComplexCos().eval().cuda()
    r = torch.ones(8, 16, device="cuda") * 0.5
    theta = torch.linspace(0.1, 1.5, 16, device="cuda").unsqueeze(0).expand(8, -1)
    z = torch.polar(r, theta)
    trt_model = torchtrt.compile(model, inputs=(z,), **_COMPILE)
    _cossim_complex(model(z), trt_model(z), "complex_cos")
    torch._dynamo.reset()


class ComplexSinh(nn.Module):
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return torch.sinh(z)


@pytest.mark.unit
def test_complex_sinh():
    model = ComplexSinh().eval().cuda()
    # Small imaginary part to avoid cosh overflow
    r = torch.rand(8, 16, device="cuda") * 0.5
    theta = torch.rand(8, 16, device="cuda") * 0.5
    z = torch.polar(r, theta)
    trt_model = torchtrt.compile(model, inputs=(z,), **_COMPILE)
    _cossim_complex(model(z), trt_model(z), "complex_sinh")
    torch._dynamo.reset()


class ComplexCosh(nn.Module):
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return torch.cosh(z)


@pytest.mark.unit
def test_complex_cosh():
    model = ComplexCosh().eval().cuda()
    r = torch.rand(8, 16, device="cuda") * 0.5
    theta = torch.rand(8, 16, device="cuda") * 0.5
    z = torch.polar(r, theta)
    trt_model = torchtrt.compile(model, inputs=(z,), **_COMPILE)
    _cossim_complex(model(z), trt_model(z), "complex_cosh")
    torch._dynamo.reset()


class ComplexTan(nn.Module):
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return torch.tan(z)


@pytest.mark.unit
def test_complex_tan():
    model = ComplexTan().eval().cuda()
    # Avoid a = ±pi/4 where denom → 0
    r = torch.rand(8, 16, device="cuda") * 0.4
    theta = torch.rand(8, 16, device="cuda") * 0.3 + 0.5
    z = torch.polar(r, theta)
    trt_model = torchtrt.compile(model, inputs=(z,), **_COMPILE)
    _cossim_complex(model(z), trt_model(z), "complex_tan")
    torch._dynamo.reset()


class ComplexTanh(nn.Module):
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return torch.tanh(z)


@pytest.mark.unit
def test_complex_tanh():
    model = ComplexTanh().eval().cuda()
    r = torch.rand(8, 16, device="cuda") * 0.5
    theta = torch.rand(8, 16, device="cuda") * 0.5
    z = torch.polar(r, theta)
    trt_model = torchtrt.compile(model, inputs=(z,), **_COMPILE)
    _cossim_complex(model(z), trt_model(z), "complex_tanh")
    torch._dynamo.reset()


# ===========================================================================
# 16. Inverse trigonometric ops
# ===========================================================================


class ComplexAsinh(nn.Module):
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return torch.asinh(z)


@pytest.mark.unit
def test_complex_asinh():
    model = ComplexAsinh().eval().cuda()
    r = torch.rand(8, 16, device="cuda") * 0.8 + 0.1
    theta = torch.rand(8, 16, device="cuda") * 2 * 3.14159
    z = torch.polar(r, theta)
    trt_model = torchtrt.compile(model, inputs=(z,), **_COMPILE)
    _cossim_complex(model(z), trt_model(z), "complex_asinh")
    torch._dynamo.reset()


class ComplexAcosh(nn.Module):
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return torch.acosh(z)


@pytest.mark.unit
def test_complex_acosh():
    model = ComplexAcosh().eval().cuda()
    # |Re(z)| > 1 for non-trivial (non-purely-imaginary) result
    r = torch.rand(8, 16, device="cuda") * 0.5 + 1.5
    theta = torch.rand(8, 16, device="cuda") * 0.4
    z = torch.polar(r, theta)
    trt_model = torchtrt.compile(model, inputs=(z,), **_COMPILE)
    _cossim_complex(model(z), trt_model(z), "complex_acosh")
    torch._dynamo.reset()


class ComplexAtanh(nn.Module):
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return torch.atanh(z)


@pytest.mark.unit
def test_complex_atanh():
    model = ComplexAtanh().eval().cuda()
    # |z| < 1 to stay within principal domain
    r = torch.rand(8, 16, device="cuda") * 0.6 + 0.1
    theta = torch.rand(8, 16, device="cuda") * 2 * 3.14159
    z = torch.polar(r, theta)
    trt_model = torchtrt.compile(model, inputs=(z,), **_COMPILE)
    _cossim_complex(model(z), trt_model(z), "complex_atanh")
    torch._dynamo.reset()


class ComplexAsin(nn.Module):
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return torch.asin(z)


@pytest.mark.unit
def test_complex_asin():
    """asin(z) = -i*log(iz + sqrt(1-z²)).
    Tested with |z| < 1 to avoid branch-cut ambiguity on the real axis."""
    model = ComplexAsin().eval().cuda()
    r = torch.rand(8, 16, device="cuda") * 0.6 + 0.1
    theta = torch.rand(8, 16, device="cuda") * 2 * 3.14159
    z = torch.polar(r, theta)
    trt_model = torchtrt.compile(model, inputs=(z,), **_COMPILE)
    _cossim_complex(model(z), trt_model(z), "complex_asin")
    torch._dynamo.reset()


class ComplexAcos(nn.Module):
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return torch.acos(z)


@pytest.mark.unit
def test_complex_acos():
    """acos(z) = -i*log(z + i*sqrt(1-z²)).
    Tested with |z| < 1 to avoid branch-cut ambiguity on the real axis."""
    model = ComplexAcos().eval().cuda()
    r = torch.rand(8, 16, device="cuda") * 0.6 + 0.1
    theta = torch.rand(8, 16, device="cuda") * 2 * 3.14159
    z = torch.polar(r, theta)
    trt_model = torchtrt.compile(model, inputs=(z,), **_COMPILE)
    _cossim_complex(model(z), trt_model(z), "complex_acos")
    torch._dynamo.reset()


class ComplexAtan(nn.Module):
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return torch.atan(z)


@pytest.mark.unit
def test_complex_atan():
    """atan(z) = (i/2)*log((1-iz)/(1+iz)).
    Tested with |z| < 1."""
    model = ComplexAtan().eval().cuda()
    r = torch.rand(8, 16, device="cuda") * 0.6 + 0.1
    theta = torch.rand(8, 16, device="cuda") * 2 * 3.14159
    z = torch.polar(r, theta)
    trt_model = torchtrt.compile(model, inputs=(z,), **_COMPILE)
    _cossim_complex(model(z), trt_model(z), "complex_atan")
    torch._dynamo.reset()


# ===========================================================================
# 17. Complex-complex power (pow.Tensor_Tensor)
# ===========================================================================


class ComplexPowTensorTensor(nn.Module):
    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        return torch.pow(z1, z2)


@pytest.mark.unit
def test_complex_pow_tensor_tensor():
    """z1**z2 = exp(z2 * log(z1)), both complex."""
    model = ComplexPowTensorTensor().eval().cuda()
    # Use unit-magnitude base to keep values bounded
    r1 = torch.ones(8, 16, device="cuda")
    theta1 = torch.rand(8, 16, device="cuda") * 2 * 3.14159
    z1 = torch.polar(r1, theta1)
    # Small exponent magnitude to avoid overflow
    r2 = torch.rand(8, 16, device="cuda") * 0.3
    theta2 = torch.rand(8, 16, device="cuda") * 2 * 3.14159
    z2 = torch.polar(r2, theta2)
    trt_model = torchtrt.compile(model, inputs=(z1, z2), **_COMPILE)
    _cossim_complex(model(z1, z2), trt_model(z1, z2), "complex_pow_tensor_tensor")
    torch._dynamo.reset()


# ===========================================================================
# 18. Composite complex-only multi-op chains
# ===========================================================================


class ComplexLogExp(nn.Module):
    """exp(log(z)) ≈ z — round-trip through log and exp."""
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return torch.exp(torch.log(z))


@pytest.mark.unit
def test_complex_log_exp():
    """exp(log(z)) ≈ z: round-trip verifies log and exp rewrites compose correctly."""
    model = ComplexLogExp().eval().cuda()
    r = torch.rand(8, 16, device="cuda") * 0.8 + 0.2
    theta = torch.rand(8, 16, device="cuda") * 1.5
    z = torch.polar(r, theta)
    trt_model = torchtrt.compile(model, inputs=(z,), **_COMPILE)
    _cossim_complex(model(z), trt_model(z), "complex_log_exp")
    torch._dynamo.reset()


class ComplexMulAddSub(nn.Module):
    """(a*b)+c-d — four complex operands, two muls and add/sub."""
    def forward(self, a, b, c, d):
        return (a * b) + c - d


@pytest.mark.unit
def test_complex_mul_add_sub():
    """(a*b)+c-d with four complex inputs."""
    model = ComplexMulAddSub().eval().cuda()
    def _rc():
        return torch.polar(torch.rand(8, 16, device="cuda") + 0.2,
                           torch.rand(8, 16, device="cuda") * 2 * 3.14159)
    a, b, c, d = _rc(), _rc(), _rc(), _rc()
    trt_model = torchtrt.compile(model, inputs=(a, b, c, d), **_COMPILE)
    _cossim_complex(model(a, b, c, d), trt_model(a, b, c, d), "complex_mul_add_sub")
    torch._dynamo.reset()


class ComplexConjThenMul(nn.Module):
    """conj(a) * b."""
    def forward(self, a, b):
        return torch.conj(a) * b


@pytest.mark.unit
def test_complex_conj_then_mul():
    """conj(a)*b: conjugate followed by complex multiply."""
    model = ComplexConjThenMul().eval().cuda()
    def _rc():
        return torch.polar(torch.rand(8, 16, device="cuda") + 0.2,
                           torch.rand(8, 16, device="cuda") * 2 * 3.14159)
    a, b = _rc(), _rc()
    trt_model = torchtrt.compile(model, inputs=(a, b), **_COMPILE)
    _cossim_complex(model(a, b), trt_model(a, b), "complex_conj_then_mul")
    torch._dynamo.reset()


class ComplexAbsThenLog(nn.Module):
    """log(abs(z)) — chain ending in real output."""
    def forward(self, z):
        return torch.log(torch.abs(z))


@pytest.mark.unit
def test_complex_abs_then_log():
    """log(|z|): abs(complex) → log(real), result is real."""
    model = ComplexAbsThenLog().eval().cuda()
    r = torch.rand(8, 16, device="cuda") * 0.8 + 0.2
    z = torch.polar(r, torch.rand(8, 16, device="cuda") * 2 * 3.14159)
    trt_model = torchtrt.compile(model, inputs=(z,), **_COMPILE)
    py_out = model(z)
    trt_out = trt_model(z)
    _cossim_real(py_out, trt_out, "complex_abs_then_log")
    torch._dynamo.reset()


class ComplexSqrtThenMul(nn.Module):
    """sqrt(a) * sqrt(b) — two sqrt rewrites in one graph."""
    def forward(self, a, b):
        return torch.sqrt(a) * torch.sqrt(b)


@pytest.mark.unit
def test_complex_sqrt_then_mul():
    """sqrt(a)*sqrt(b) ≈ sqrt(a*b) — exercises two sqrt rewrites in one graph."""
    model = ComplexSqrtThenMul().eval().cuda()
    r = torch.rand(8, 16, device="cuda") + 0.5
    a = torch.polar(r, torch.rand(8, 16, device="cuda") * 3.14159)
    b = torch.polar(r, torch.rand(8, 16, device="cuda") * 3.14159)
    trt_model = torchtrt.compile(model, inputs=(a, b), **_COMPILE)
    _cossim_complex(model(a, b), trt_model(a, b), "complex_sqrt_then_mul")
    torch._dynamo.reset()


class ComplexPowThenAdd(nn.Module):
    """z**2 + z — polynomial evaluation via pow + add."""
    def forward(self, z):
        return z ** 2 + z


@pytest.mark.unit
def test_complex_pow_then_add():
    """z² + z — quadratic in z, exercises pow.Tensor_Scalar → add chain."""
    model = ComplexPowThenAdd().eval().cuda()
    z = torch.polar(
        torch.rand(8, 16, device="cuda") * 0.8 + 0.2,
        torch.rand(8, 16, device="cuda") * 2 * 3.14159,
    )
    trt_model = torchtrt.compile(model, inputs=(z,), **_COMPILE)
    _cossim_complex(model(z), trt_model(z), "complex_pow_then_add")
    torch._dynamo.reset()


class ComplexSinCosPythagorean(nn.Module):
    """sin(z)² + cos(z)² — Pythagorean identity over ℂ."""
    def forward(self, z):
        s = torch.sin(z)
        c = torch.cos(z)
        return s * s + c * c


@pytest.mark.unit
def test_complex_sin_cos_pythagorean():
    """sin²(z) + cos²(z): TRT vs PyTorch agree numerically."""
    model = ComplexSinCosPythagorean().eval().cuda()
    r = torch.rand(8, 16, device="cuda") * 0.4
    theta = torch.linspace(0.1, 1.2, 16, device="cuda").unsqueeze(0).expand(8, -1).contiguous()
    z = torch.polar(r, theta)
    trt_model = torchtrt.compile(model, inputs=(z,), **_COMPILE)
    py_out = model(z)
    trt_out = trt_model(z)
    _cossim_real(py_out.real.contiguous(), trt_out.real.contiguous(), "complex_sin_cos_pythagorean")
    torch._dynamo.reset()


class ComplexExpThenAbs(nn.Module):
    """|exp(z)| = exp(Re(z)) — chain: exp → abs, result is real."""
    def forward(self, z):
        return torch.abs(torch.exp(z))


@pytest.mark.unit
def test_complex_exp_then_abs():
    """|exp(z)| = exp(Re(z)): exercises exp rewrite feeding into abs rewrite."""
    model = ComplexExpThenAbs().eval().cuda()
    r = torch.rand(8, 16, device="cuda") * 0.3
    theta = torch.rand(8, 16, device="cuda") * 2 * 3.14159
    z = torch.polar(r, theta)
    trt_model = torchtrt.compile(model, inputs=(z,), **_COMPILE)
    py_out = model(z)
    trt_out = trt_model(z)
    _cossim_real(py_out, trt_out, "complex_exp_then_abs")
    torch._dynamo.reset()


class ComplexNormalize(nn.Module):
    """z / |z| — normalize to unit circle via abs + divide."""
    def forward(self, z):
        mag = torch.abs(z)
        # Avoid aten.complex.default — build complex divisor via view_as_complex.
        mag_c = torch.view_as_complex(
            torch.stack([mag, torch.zeros_like(mag)], dim=-1)
        )
        return z / mag_c


@pytest.mark.unit
def test_complex_normalize():
    """z/|z|: unit-normalize a complex tensor."""
    model = ComplexNormalize().eval().cuda()
    z = torch.polar(
        torch.rand(8, 16, device="cuda") * 0.8 + 0.2,
        torch.rand(8, 16, device="cuda") * 2 * 3.14159,
    )
    trt_model = torchtrt.compile(model, inputs=(z,), **_COMPILE)
    _cossim_complex(model(z), trt_model(z), "complex_normalize")
    torch._dynamo.reset()


# ===========================================================================
# 19. Complex + real interleaved computations
# ===========================================================================


class ComplexMulThenRealLinear(nn.Module):
    """Complex rotation followed by a real-valued linear projection (core RoPE pattern)."""
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(64, 32, bias=False)

    def forward(self, x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        xc = torch.view_as_complex(x.reshape(*x.shape[:-1], -1, 2))
        rotated = torch.view_as_real(xc * freqs).flatten(-2)
        return self.proj(rotated)


@pytest.mark.unit
def test_complex_mul_then_real_linear():
    """Complex RoPE rotation followed by a real linear layer."""
    model = ComplexMulThenRealLinear().eval().cuda()
    x = torch.randn(2, 8, 64, device="cuda")
    freqs = _make_freqs(8, 64)
    trt_model = torchtrt.compile(model, inputs=(x, freqs), **_COMPILE)
    py_out = model(x, freqs)
    trt_out = trt_model(x, freqs)
    _cossim_real(py_out, trt_out, "complex_mul_then_real_linear")
    torch._dynamo.reset()


class RealNormThenComplexMul(nn.Module):
    """LayerNorm on the real input, then rotate with complex freqs."""
    def __init__(self, d: int = 64) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        xc = torch.view_as_complex(x.reshape(*x.shape[:-1], -1, 2))
        return torch.view_as_real(xc * freqs).flatten(-2)


@pytest.mark.unit
def test_real_norm_then_complex_mul():
    """LayerNorm (real) → view_as_complex → complex mul → view_as_real."""
    model = RealNormThenComplexMul(d=64).eval().cuda()
    x = torch.randn(2, 8, 64, device="cuda")
    freqs = _make_freqs(8, 64)
    trt_model = torchtrt.compile(model, inputs=(x, freqs), **_COMPILE)
    py_out = model(x, freqs)
    trt_out = trt_model(x, freqs)
    _cossim_real(py_out, trt_out, "real_norm_then_complex_mul")
    torch._dynamo.reset()


class ComplexMulThenRealActivation(nn.Module):
    """Complex rotation → real view → GELU activation."""
    def forward(self, x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        xc = torch.view_as_complex(x.reshape(*x.shape[:-1], -1, 2))
        real_out = torch.view_as_real(xc * freqs).flatten(-2)
        return torch.nn.functional.gelu(real_out)


@pytest.mark.unit
def test_complex_mul_then_gelu():
    """Complex rotation followed by GELU on the real-valued output."""
    model = ComplexMulThenRealActivation().eval().cuda()
    x = torch.randn(2, 8, 64, device="cuda")
    freqs = _make_freqs(8, 64)
    trt_model = torchtrt.compile(model, inputs=(x, freqs), **_COMPILE)
    py_out = model(x, freqs)
    trt_out = trt_model(x, freqs)
    _cossim_real(py_out, trt_out, "complex_mul_then_gelu")
    torch._dynamo.reset()


class RealScaleThenComplexAddSub(nn.Module):
    """Scale two real tensors, pack as complex, do add and sub."""
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # x, y: real (B, D, 2) — pack as complex
        xa = x * self.scale
        ya = y * self.scale
        zx = torch.view_as_complex(xa)
        zy = torch.view_as_complex(ya)
        return torch.view_as_real(zx + zy - zx)


@pytest.mark.unit
def test_real_scale_then_complex_add_sub():
    """Real scale → pack as complex → add/sub → unpack."""
    model = RealScaleThenComplexAddSub().eval().cuda()
    x = torch.randn(4, 16, 2, device="cuda")
    y = torch.randn(4, 16, 2, device="cuda")
    trt_model = torchtrt.compile(model, inputs=(x, y), **_COMPILE)
    py_out = model(x, y)
    trt_out = trt_model(x, y)
    _cossim_real(py_out, trt_out, "real_scale_then_complex_add_sub")
    torch._dynamo.reset()


class ComplexMagPhaseRecompose(nn.Module):
    """Decompose into magnitude + phase, apply real ops to each, recompose."""
    def __init__(self) -> None:
        super().__init__()
        self.mag_scale = nn.Parameter(torch.ones(1))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        mag = torch.abs(z)
        phase = torch.angle(z)
        mag2 = mag * self.mag_scale.abs()
        phase2 = torch.clamp(phase, -1.5, 1.5)
        return torch.polar(mag2, phase2)


@pytest.mark.unit
def test_complex_mag_phase_recompose():
    """Decompose z → (|z|, angle) → scale+clip → polar recompose."""
    model = ComplexMagPhaseRecompose().eval().cuda()
    r = torch.rand(8, 16, device="cuda") + 0.3
    theta = torch.rand(8, 16, device="cuda") * 2.0 - 1.0
    z = torch.polar(r, theta)
    trt_model = torchtrt.compile(model, inputs=(z,), **_COMPILE)
    _cossim_complex(model(z), trt_model(z), "complex_mag_phase_recompose")
    torch._dynamo.reset()


class ComplexResidual(nn.Module):
    """Complex residual: z + exp(log(z))."""
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return z + torch.exp(torch.log(z))


@pytest.mark.unit
def test_complex_residual():
    """z + exp(log(z)) ≈ 2z — residual connection through complex ops."""
    model = ComplexResidual().eval().cuda()
    r = torch.rand(8, 16, device="cuda") + 0.5
    theta = torch.rand(8, 16, device="cuda") * 1.5
    z = torch.polar(r, theta)
    trt_model = torchtrt.compile(model, inputs=(z,), **_COMPILE)
    _cossim_complex(model(z), trt_model(z), "complex_residual")
    torch._dynamo.reset()


class ComplexGatedMul(nn.Module):
    """Real sigmoid gate applied to a complex tensor."""
    def __init__(self) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(32, 16, bias=False)

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.gate_proj(x))
        gate_c = torch.view_as_complex(
            torch.stack([gate, torch.zeros_like(gate)], dim=-1)
        )
        return z * gate_c


@pytest.mark.unit
def test_complex_gated_mul():
    """Real sigmoid gate × complex tensor — real and complex subgraphs in one model."""
    model = ComplexGatedMul().eval().cuda()
    x = torch.randn(4, 32, device="cuda")
    z = torch.polar(
        torch.rand(4, 16, device="cuda") + 0.3,
        torch.rand(4, 16, device="cuda") * 2 * 3.14159,
    )
    trt_model = torchtrt.compile(model, inputs=(x, z), **_COMPILE)
    _cossim_complex(model(x, z), trt_model(x, z), "complex_gated_mul")
    torch._dynamo.reset()


# ===========================================================================
# 20. Multi-layer and branching subgraph integration tests
# ===========================================================================


class MultiHeadRoPE(nn.Module):
    """Apply independent RoPE rotations to Q, K, V and compute attention logits."""

    def __init__(self, seq: int = 8, dim: int = 32) -> None:
        super().__init__()
        self.freq_q = nn.Parameter(_make_freqs(seq, dim).detach())
        self.freq_k = nn.Parameter(_make_freqs(seq, dim).detach())
        self.freq_v = nn.Parameter(_make_freqs(seq, dim).detach())

    def forward(self, q, k, v):
        def rope(x, freq):
            xc = torch.view_as_complex(x.reshape(*x.shape[:-1], -1, 2))
            return torch.view_as_real(xc * freq).flatten(-2)

        q_r = rope(q, self.freq_q)
        k_r = rope(k, self.freq_k)
        v_r = rope(v, self.freq_v)
        scores = torch.bmm(q_r, k_r.transpose(1, 2)) / (q_r.shape[-1] ** 0.5)
        return torch.bmm(scores, v_r)


@pytest.mark.unit
def test_multi_head_rope():
    """Q/K/V independently rotated by RoPE, then bmm attention — 3 complex subgraphs."""
    model = MultiHeadRoPE(seq=8, dim=32).eval().cuda()
    B, S, D = 2, 8, 32
    q = torch.randn(B, S, D, device="cuda")
    k = torch.randn(B, S, D, device="cuda")
    v = torch.randn(B, S, D, device="cuda")
    trt_model = torchtrt.compile(model, inputs=(q, k, v), **_COMPILE)
    _cossim_real(model(q, k, v), trt_model(q, k, v), "multi_head_rope")
    torch._dynamo.reset()


class ParallelComplexBranches(nn.Module):
    """One complex input forks into two independent rotation paths, then concat + project."""

    def __init__(self, dim: int = 16) -> None:
        super().__init__()
        self.freq_a = nn.Parameter(_make_freqs(8, dim * 2).detach())
        self.freq_b = nn.Parameter(_make_freqs(8, dim * 2).detach())
        self.proj = nn.Linear(dim * 4, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = torch.view_as_complex(x.reshape(*x.shape[:-1], -1, 2))
        real_a = torch.view_as_real(z * self.freq_a).flatten(-2)
        real_b = torch.view_as_real(z * self.freq_b).flatten(-2)
        return self.proj(torch.cat([real_a, real_b], dim=-1))


@pytest.mark.unit
def test_parallel_complex_branches():
    """One complex input forks into two rotation paths, concat, then project."""
    model = ParallelComplexBranches(dim=16).eval().cuda()
    x = torch.randn(2, 8, 32, device="cuda")
    trt_model = torchtrt.compile(model, inputs=(x,), **_COMPILE)
    _cossim_real(model(x), trt_model(x), "parallel_complex_branches")
    torch._dynamo.reset()


class TransformerLikeBlock(nn.Module):
    """One layer: RoPE rotation + real FFN with residual."""

    def __init__(self, d: int = 32) -> None:
        super().__init__()
        self.freq = nn.Parameter(_make_freqs(8, d).detach())
        self.norm = nn.LayerNorm(d)
        self.ff1 = nn.Linear(d, d * 2, bias=False)
        self.ff2 = nn.Linear(d * 2, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xc = torch.view_as_complex(x.reshape(*x.shape[:-1], -1, 2))
        rotated = torch.view_as_real(xc * self.freq).flatten(-2)
        h = self.norm(rotated)
        h = torch.nn.functional.gelu(self.ff1(h))
        h = self.ff2(h)
        return rotated + h


class StackedTransformerBlocks(nn.Module):
    """Two sequential transformer-like blocks, each with complex RoPE."""

    def __init__(self, d: int = 32) -> None:
        super().__init__()
        self.block1 = TransformerLikeBlock(d)
        self.block2 = TransformerLikeBlock(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block2(self.block1(x))


@pytest.mark.unit
def test_stacked_transformer_blocks():
    """Two stacked transformer-like blocks, each containing a complex RoPE sub-graph."""
    model = StackedTransformerBlocks(d=32).eval().cuda()
    x = torch.randn(2, 8, 32, device="cuda")
    trt_model = torchtrt.compile(model, inputs=(x,), **_COMPILE)
    _cossim_real(model(x), trt_model(x), "stacked_transformer_blocks")
    torch._dynamo.reset()


class FourComplexInputsMulAdd(nn.Module):
    """z1*z2 + z3*z4 — four distinct complex runtime inputs."""
    def forward(self, z1, z2, z3, z4):
        return z1 * z2 + z3 * z4


@pytest.mark.unit
def test_four_complex_inputs_mul_add():
    """z1*z2 + z3*z4 — four complex runtime inputs, two muls and one add."""
    model = FourComplexInputsMulAdd().eval().cuda()

    def _rc(shape):
        return torch.polar(
            torch.rand(*shape, device="cuda") * 0.8 + 0.2,
            torch.rand(*shape, device="cuda") * 2 * 3.14159,
        )

    z1, z2, z3, z4 = [_rc((4, 16)) for _ in range(4)]
    trt_model = torchtrt.compile(model, inputs=(z1, z2, z3, z4), **_COMPILE)
    _cossim_complex(model(z1, z2, z3, z4), trt_model(z1, z2, z3, z4), "four_complex_inputs_mul_add")
    torch._dynamo.reset()


class CrossAttentionComplexQ(nn.Module):
    """Cross-attention: complex-rotated queries, real key/value projections."""

    def __init__(self, d_q: int = 32, d_kv: int = 64) -> None:
        super().__init__()
        self.freq = nn.Parameter(_make_freqs(8, d_q).detach())
        self.norm_q = nn.LayerNorm(d_q)
        self.Wk = nn.Linear(d_kv, d_q, bias=False)
        self.Wv = nn.Linear(d_kv, d_q, bias=False)

    def forward(self, q_real, kv):
        qc = torch.view_as_complex(q_real.reshape(*q_real.shape[:-1], -1, 2))
        q = self.norm_q(torch.view_as_real(qc * self.freq).flatten(-2))
        k = self.Wk(kv)
        v = self.Wv(kv)
        scores = torch.bmm(q, k.transpose(1, 2)) / (q.shape[-1] ** 0.5)
        return torch.bmm(scores, v)


@pytest.mark.unit
def test_cross_attention_complex_q():
    """Cross-attention: complex-rotated query, real key/value projections."""
    model = CrossAttentionComplexQ(d_q=32, d_kv=64).eval().cuda()
    q_real = torch.randn(2, 8, 32, device="cuda")
    kv = torch.randn(2, 12, 64, device="cuda")
    trt_model = torchtrt.compile(model, inputs=(q_real, kv), **_COMPILE)
    _cossim_real(model(q_real, kv), trt_model(q_real, kv), "cross_attention_complex_q")
    torch._dynamo.reset()


class ComplexRotator(nn.Module):
    """Single complex rotation layer wrapping a learnable frequency buffer."""

    def __init__(self, seq: int = 8, dim: int = 32) -> None:
        super().__init__()
        self.freq = nn.Parameter(_make_freqs(seq, dim).detach())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xc = torch.view_as_complex(x.reshape(*x.shape[:-1], -1, 2))
        return torch.view_as_real(xc * self.freq).flatten(-2)


class NestedComplexRotators(nn.Module):
    """Two ComplexRotator sub-modules with a real LayerNorm between them."""

    def __init__(self, d: int = 32) -> None:
        super().__init__()
        self.rot1 = ComplexRotator(seq=8, dim=d)
        self.norm = nn.LayerNorm(d)
        self.rot2 = ComplexRotator(seq=8, dim=d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.rot1(x)
        x = self.norm(x)
        return self.rot2(x)


@pytest.mark.unit
def test_nested_complex_rotators():
    """Two nested ComplexRotator sub-modules with a real LayerNorm between them."""
    model = NestedComplexRotators(d=32).eval().cuda()
    x = torch.randn(2, 8, 32, device="cuda")
    trt_model = torchtrt.compile(model, inputs=(x,), **_COMPILE)
    _cossim_real(model(x), trt_model(x), "nested_complex_rotators")
    torch._dynamo.reset()


class ComplexNormThenProject(nn.Module):
    """abs(z) → LayerNorm → rescale z: real and complex subgraphs share an edge."""

    def __init__(self, dim: int = 16) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        mag = torch.abs(z)
        scale = self.norm(mag)
        scale_c = torch.view_as_complex(
            torch.stack([scale, torch.zeros_like(scale)], dim=-1)
        )
        return z * scale_c


@pytest.mark.unit
def test_complex_norm_then_project():
    """abs(z) → LayerNorm → rescale z: real and complex subgraphs share an edge."""
    model = ComplexNormThenProject(dim=16).eval().cuda()
    z = torch.polar(
        torch.rand(4, 16, device="cuda") * 0.8 + 0.2,
        torch.rand(4, 16, device="cuda") * 2 * 3.14159,
    )
    trt_model = torchtrt.compile(model, inputs=(z,), **_COMPILE)
    _cossim_complex(model(z), trt_model(z), "complex_norm_then_project")
    torch._dynamo.reset()


class ComplexRotateProject(nn.Module):
    """Two complex rotations separated by a real linear layer."""

    def __init__(self, d: int = 32) -> None:
        super().__init__()
        self.freq1 = nn.Parameter(_make_freqs(8, d).detach())
        self.freq2 = nn.Parameter(_make_freqs(8, d).detach())
        self.proj = nn.Linear(d, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xc = torch.view_as_complex(x.reshape(*x.shape[:-1], -1, 2))
        r1 = torch.view_as_real(xc * self.freq1).flatten(-2)
        r2 = self.proj(r1)
        xc2 = torch.view_as_complex(r2.reshape(*r2.shape[:-1], -1, 2))
        return torch.view_as_real(xc2 * self.freq2).flatten(-2)


@pytest.mark.unit
def test_complex_rotate_project():
    """Two complex rotations separated by a real linear layer."""
    model = ComplexRotateProject(d=32).eval().cuda()
    x = torch.randn(2, 8, 32, device="cuda")
    trt_model = torchtrt.compile(model, inputs=(x,), **_COMPILE)
    _cossim_real(model(x), trt_model(x), "complex_rotate_project")
    torch._dynamo.reset()
