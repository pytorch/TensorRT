"""Tests for complex tensor graph-break behavior in torch-tensorrt.

These tests verify that when a model contains complex tensor operations mixed with
ops that have no handler in the complex-lowering rewriter, the compiler:

  1. Wraps the unsupported op with ``view_as_complex`` / ``view_as_real`` so it
     receives genuine complex-dtype inputs and returns a real-layout output.
  2. TRT, which has no complex-dtype support, naturally graph-breaks around the
     wrapped cluster and runs it as a PyTorch fallback block.
  3. The lowerable complex ops on both sides compile to TRT via
     ``complex_graph_detection``.
  4. The overall model produces numerically correct results end-to-end.

Background
----------
``complex_graph_detection`` rewrites complex-dtype ATen ops to equivalent
real-arithmetic ops before TRT compilation.  When an op is *not* registered
with ``@_complex_unpacker`` and is not in ``_ELEMENTWISE_SAFE`` the rewriter
inserts ``view_as_complex`` before each complex-layout input and
``view_as_real`` after the output, preserving correct semantics and letting
TRT's lack of complex support create the graph break automatically.

``cumsum`` is used as the representative unsupported op: it has well-defined
PyTorch semantics on complex tensors but has no handler in the rewriter.
"""

import pytest
import torch
import torch.nn as nn
import torch_tensorrt as torchtrt
from torch_tensorrt.dynamo.lowering.passes.complex_graph_rewrite import (
    complex_graph_detection,
)
from torch_tensorrt.dynamo._settings import CompilationSettings
from torch_tensorrt.dynamo.utils import COSINE_THRESHOLD, cosine_similarity

try:
    from torch_tensorrt.dynamo.runtime import PythonTorchTensorRTModule

    _PYTHON_RUNTIME_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PYTHON_RUNTIME_AVAILABLE = False


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_freqs(seq: int, dim: int, theta: float = 10000.0) -> torch.Tensor:
    """Complex unit-magnitude frequency tensor on CUDA, shape ``(seq, dim//2)``."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(seq, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs).cuda()


def _cossim_real(py_out: torch.Tensor, trt_out: torch.Tensor, tag: str) -> None:
    """Assert cosine similarity > COSINE_THRESHOLD on a real-valued output."""
    assert not trt_out.is_complex(), f"{tag}: expected real output, got {trt_out.dtype}"
    s = cosine_similarity(py_out.contiguous(), trt_out.contiguous())
    assert s > COSINE_THRESHOLD, f"{tag}: cosine sim {s:.4f} < {COSINE_THRESHOLD}"


def _count_trt_modules(mod: torch.nn.Module) -> int:
    """Return the number of ``PythonTorchTensorRTModule`` submodules (-1 if unavailable)."""
    if not _PYTHON_RUNTIME_AVAILABLE:
        return -1
    return sum(
        1 for _, m in mod.named_modules() if isinstance(m, PythonTorchTensorRTModule)
    )


def _export_and_lower(model: nn.Module, inputs: tuple) -> torch.fx.GraphModule:
    """Export model and apply complex_graph_detection lowering pass."""
    with torch.no_grad():
        ep = torch.export.export(model.eval(), inputs)
    gm = ep.module()
    complex_graph_detection(gm, CompilationSettings())
    return gm


# ===========================================================================
# Test 1 — unsupported op gets view_as_complex/view_as_real wrapper
# ===========================================================================


class ComplexMulThenCumsum(nn.Module):
    """Complex mul (lowerable) followed by cumsum (no rewriter handler).

    After ``complex_graph_detection`` the rewriter cannot handle ``cumsum``.
    It inserts ``view_as_complex`` before cumsum's input and ``view_as_real``
    after its output so the op runs in PyTorch with correct complex semantics
    while TRT compiles the surrounding real-arithmetic blocks.
    """

    def forward(self, z: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        rotated = z * freqs  # complex mul — lowered to real arithmetic by rewriter
        accumulated = torch.cumsum(rotated, dim=0)  # no handler → PyTorch fallback
        return torch.view_as_real(accumulated).flatten(-2)


@pytest.mark.unit
def test_unsupported_op_gets_complexify_wrap() -> None:
    """The rewriter wraps cumsum with view_as_complex/view_as_real.

    Structural check (no TRT required):
      - After lowering, the graph contains ``view_as_complex`` immediately
        before ``cumsum`` and ``view_as_real`` immediately after.
      - The ``view_as_complex`` input is the real-layout output of the
        rewritten complex mul — confirming it is a float32 ``(..., 2)`` node.
      - The ``view_as_real`` output feeds the downstream flatten.
      - The PyTorch cumsum receives a complex-dtype tensor (correct semantics).
    """
    model = ComplexMulThenCumsum().eval().cuda()
    z = _make_freqs(8, 64)
    freqs = _make_freqs(8, 64)

    gm = _export_and_lower(model, (z, freqs))

    nodes_by_target: dict = {}
    for n in gm.graph.nodes:
        nodes_by_target.setdefault(n.target, []).append(n)

    # view_as_complex must be present (inserted by the fallback wrapper)
    assert (
        torch.ops.aten.view_as_complex.default in nodes_by_target
    ), "Expected view_as_complex to be inserted before cumsum, but it was not found"

    # cumsum must still be present (it was NOT removed)
    assert (
        torch.ops.aten.cumsum.default in nodes_by_target
    ), "cumsum should remain in the graph (runs as PyTorch fallback)"

    # The view_as_complex output feeds directly into cumsum
    vc_node = nodes_by_target[torch.ops.aten.view_as_complex.default][0]
    cumsum_node = nodes_by_target[torch.ops.aten.cumsum.default][0]
    assert (
        cumsum_node.args[0] is vc_node
    ), f"cumsum's first arg should be the view_as_complex node, got {cumsum_node.args[0]}"

    # The view_as_complex input is a real-layout (is_complex_layout) node
    vc_input = vc_node.args[0]
    assert isinstance(vc_input, torch.fx.Node), "view_as_complex input must be a Node"
    assert vc_input.meta.get(
        "is_complex_layout", False
    ), "view_as_complex input should be a real-layout complex node (is_complex_layout=True)"

    # view_as_real must follow cumsum
    assert (
        torch.ops.aten.view_as_real.default in nodes_by_target
    ), "Expected view_as_real to be inserted after cumsum, but it was not found"
    vr_node = nodes_by_target[torch.ops.aten.view_as_real.default][0]
    assert (
        vr_node.args[0] is cumsum_node
    ), f"view_as_real's arg should be the cumsum node, got {vr_node.args[0]}"

    # After metadata propagation, cumsum receives a complex-dtype tensor
    vc_val = vc_node.meta.get("val")
    if vc_val is not None:
        assert vc_val.dtype in (
            torch.complex64,
            torch.complex128,
        ), f"view_as_complex output should be complex, got {vc_val.dtype}"


# ===========================================================================
# Test 2 — lowerable ops TRT, unsupported op PyTorch (with complex input),
#           lowerable ops TRT again; end-to-end numerical correctness
# ===========================================================================


class ComplexTwoTRTBlocksAroundCumsum(nn.Module):
    """Two complex-rotation TRT blocks with cumsum (PyTorch) in between.

    Expected graph after ``complex_graph_detection``:

        [Block A — TRT]
            z_real, freqs_real  →  re/im arithmetic for z * freqs  →  rotated_real

        [PyTorch fallback — complex inputs]
            view_as_complex(rotated_real)  →  cumsum(complex)  →  view_as_real  →  acc_real

        [Block B — TRT]
            acc_real, freqs_real  →  re/im arithmetic for acc * freqs  →  result_real
            result_real  →  view_as_real substitute  →  flatten  →  output
    """

    def forward(self, z: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        # Block A: complex rotate — lowered to real arithmetic
        rotated = z * freqs

        # Unsupported complex op — rewriter inserts view_as_complex/view_as_real;
        # TRT graph-breaks here; cumsum runs in PyTorch on a complex tensor
        accumulated = torch.cumsum(rotated, dim=0)

        # Block B: second complex rotate — lowered to real arithmetic
        result = accumulated * freqs
        return torch.view_as_real(result).flatten(-2)


@pytest.mark.unit
def test_complex_partial_lowering_with_graph_break() -> None:
    """Lowerable complex ops compile to TRT; cumsum runs in PyTorch on complex input.

    Asserts:
      1. The compiled model is numerically correct (cosine sim > threshold).
      2. At least one ``PythonTorchTensorRTModule`` submodule exists — confirming
         the lowerable complex ops were compiled to TRT, not all relegated to
         PyTorch fallback.
      3. After lowering, cumsum receives a complex-dtype tensor (the
         view_as_complex wrapper was inserted correctly).
    """
    model = ComplexTwoTRTBlocksAroundCumsum().eval().cuda()
    z = _make_freqs(8, 64)
    freqs = _make_freqs(8, 64)
    inputs = (z, freqs)

    # Structural check: verify cumsum gets a complex input after lowering
    gm = _export_and_lower(model, inputs)
    for n in gm.graph.nodes:
        if n.target == torch.ops.aten.cumsum.default:
            vc_val = n.args[0].meta.get("val")
            if vc_val is not None:
                assert vc_val.dtype in (
                    torch.complex64,
                    torch.complex128,
                ), f"cumsum should receive a complex tensor, got {vc_val.dtype}"
            break

    # End-to-end: compile and verify numerical correctness
    ep = torch.export.export(model, inputs)
    trt_model = torchtrt.dynamo.compile(
        ep,
        inputs=inputs,
        min_block_size=1,
        pass_through_build_failures=True,
        use_python_runtime=True,
    )
    py_out = model(*inputs)
    trt_out = trt_model(*inputs)
    _cossim_real(py_out, trt_out, "complex_partial_lowering_with_graph_break")

    # Verify at least one TRT block was created for the lowerable complex ops
    n_trt = _count_trt_modules(trt_model)
    if n_trt >= 0:
        assert n_trt >= 1, (
            f"Expected at least one TRT submodule (lowerable complex ops should "
            f"compile to TRT) but found {n_trt}."
        )

    torch._dynamo.reset()
