# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

"""The float64 oracle, its fallbacks, and the rules that compare a candidate's
outputs with it."""

import logging
import math

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

from torch_tensorrt.dynamo.region_compilation._generation import _reference, _worker
from torch_tensorrt.dynamo.region_compilation._plugins import _TensorBoundary

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required"
)
CPU = torch.device("cpu")


def _boundary(shape, dtype=torch.float32, device="cpu"):
    return _TensorBoundary(tuple(shape), dtype, torch.device(device))


def _captured(function, *inputs):
    """An ATen graph saved and loaded as the controller does with a region."""
    graph = make_fx(function)(*inputs)
    return _reference.decode_reference(_reference.encode_reference(graph))


# A window long enough that a 1% scale error stands out of bfloat16 rounding:
# measured 2.8x eager's error, against 0.5x for a correct reordering.
WINDOW = 64


def _attention(k, v, q):
    """Sliding-window attention with bfloat16 matmuls, a finfo.min fill and
    .type_as(q), as a captured decoder layer computes it."""
    return _attention_in(k, v, q, torch.float32, None)


def _attention_in(k, v, q, softmax_dtype, fill):
    n = q.shape[2]
    i = torch.arange(n, device=q.device).view(n, 1)
    j = torch.arange(n, device=q.device).view(1, n)
    keep = (j <= i) & ((i - j) < WINDOW)
    w = torch.matmul(q, k.expand(-1, q.shape[1], -1, -1).transpose(2, 3)) * 0.125
    w = torch.where(keep, w, torch.finfo(q.dtype).min if fill is None else fill)
    p = torch.softmax(w, dim=-1, dtype=softmax_dtype).type_as(q)
    out = torch.matmul(p, v.expand(-1, q.shape[1], -1, -1))
    return out.transpose(1, 2).contiguous()


def _attention_inputs(seed=0, n=32, dtype=torch.bfloat16):
    torch.manual_seed(seed)
    k, v = (torch.randn(1, 1, n, 64, dtype=dtype) for _ in range(2))
    q = torch.randn(1, 2, n, 64, dtype=dtype)
    return k, v, q


def _exact_attention(k, v, q):
    fill = torch.finfo(q.dtype).min
    double = (value.double() for value in (k, v, q))
    return _attention_in(*double, torch.float64, fill)


def _reordered_attention(k, v, q, scale=0.125, window=WINDOW):
    """A correct kernel's numerics: float32 scores and probabilities."""
    n = q.shape[2]
    i = torch.arange(n).view(n, 1)
    j = torch.arange(n).view(1, n)
    keep = (j <= i) & ((i - j) < window)
    scores = q.float() @ k.float().expand(-1, q.shape[1], -1, -1).transpose(2, 3)
    p = torch.softmax(torch.where(keep, scores * scale, -math.inf), dim=-1)
    out = p @ v.float().expand(-1, q.shape[1], -1, -1)
    return out.transpose(1, 2).contiguous().to(q.dtype)


def _check(candidate, inputs, graph, specs, **kwargs):
    expected = _worker._expected(graph, inputs, specs, exact_reference=graph, **kwargs)
    _worker._check_outputs(
        _worker._outputs(candidate(*inputs)), expected, "case 'x'", rtol=None, atol=None
    )


# ---- The float64 reference ----


def test_float64_reference_promotes_every_floating_dtype_argument():
    def body(x, y):
        a = x.to(torch.bfloat16).to(torch.float32)  # _to_copy
        b = torch.full((4,), 2.0, dtype=torch.float32)  # factory
        c = torch.scalar_tensor(torch.finfo(torch.float16).min, dtype=torch.float16)
        d = torch.sum(y, dtype=torch.float32)  # reduction dtype
        e = torch.arange(4, dtype=torch.int64)  # integers keep their dtype
        return a + b + torch.where(e > 1, c.float(), d) + e.to(torch.float16)

    x = torch.randn(4)
    graph = _captured(body, x, torch.randn(4).half())
    exact = _worker._float64_reference(graph)
    dtypes = {
        value
        for node in exact.graph.nodes
        if node.op == "call_function"
        for value in (*node.args, *node.kwargs.values())
        if isinstance(value, torch.dtype)
    }
    assert torch.float64 in dtypes and torch.int64 in dtypes
    assert not dtypes & {torch.float16, torch.bfloat16, torch.float32}
    # The original graph keeps its dtypes.
    assert any(node.kwargs.get("dtype") == torch.bfloat16 for node in graph.graph.nodes)
    (out,) = _worker._outputs(exact(x.double(), torch.randn(4).double()))
    assert out.dtype == torch.float64


def test_float64_reference_evaluates_explicit_casts_without_rounding():
    inputs = _attention_inputs()
    graph = _captured(_attention, *inputs)
    exact = _worker._float64_reference(graph)
    (oracle,) = _worker._outputs(exact(*(value.double() for value in inputs)))
    wanted = _exact_attention(*inputs)
    # The body's own bfloat16 roundings are gone: the oracle is the exact value.
    torch.testing.assert_close(oracle, wanted, rtol=1e-12, atol=1e-12)
    eager = _attention(*inputs).double()
    assert (eager - wanted).norm() > 1e3 * (oracle - wanted).norm()


def test_float64_reference_moves_captured_devices():
    graph = _captured(lambda x: x + torch.ones(3, device="cpu"), torch.randn(3))
    node = next(node for node in graph.graph.nodes if "device" in node.kwargs)
    node.kwargs = {**node.kwargs, "device": torch.device("cuda", 0)}
    moved = _worker._float64_reference(graph, device=CPU)
    node = next(node for node in moved.graph.nodes if "device" in node.kwargs)
    assert node.kwargs["device"] == CPU


# ---- When the oracle cannot run ----


def test_bit_reinterpretation_is_refused_without_a_fallback():
    graph = _captured(lambda x: x.view(torch.int32) + 1, torch.randn(4))
    oracle = _worker._Oracle(graph)
    inputs = (torch.randn(4),)
    with pytest.raises(_worker._EvaluationError) as failure:
        oracle(inputs, _worker._outputs(graph(*inputs)))
    assert failure.value.stage == "setup"
    assert "reinterprets the bits" in str(failure.value)
    assert "set KernelGenerator rtol and atol" in str(failure.value)
    assert len(oracle.failures) == 1


def test_an_oracle_that_runs_out_of_memory_may_be_evaluated_again():
    def exhausted(x):
        raise torch.OutOfMemoryError("CUDA out of memory")

    with pytest.raises(_worker._EvaluationError) as failure:
        _worker._Oracle(exhausted)((torch.ones(2),), (torch.ones(2),))
    assert failure.value.stage == "setup" and failure.value.retryable is True


def test_half_precision_regions_fall_back_to_a_float32_oracle(monkeypatch):
    inputs = _attention_inputs()
    graph = _captured(_attention, *inputs)
    real = _worker._float64_reference

    def no_float64(reference, dtype=torch.float64, device=None):
        if dtype == torch.float64:
            raise NotImplementedError("no float64 kernel")
        return real(reference, dtype, device)

    monkeypatch.setattr(_worker, "_float64_reference", no_float64)
    oracle = _worker._Oracle(graph)
    expected = _worker._expected(
        graph,
        inputs,
        [_boundary((1, 32, 2, 64), torch.bfloat16)],
        exact_reference=oracle,
    )
    assert expected.basis == "float32 on cpu"
    assert expected.oracle[0].dtype == torch.float64
    # The float32 oracle is still far more exact than bfloat16 eager.
    wanted = _exact_attention(*inputs)
    eager_error = (expected.eager[0].double() - wanted).norm()
    assert (expected.oracle[0] - wanted).norm() < 1e-3 * eager_error
    # float32 regions get no float32 oracle.
    with pytest.raises(_worker._EvaluationError, match="float64 on cpu"):
        _worker._Oracle(_captured(torch.sin, torch.randn(3)))(
            (torch.randn(3),), (torch.randn(3),)
        )


@requires_cuda
def test_the_oracle_moves_to_the_cpu_when_the_device_lacks_memory(monkeypatch):
    inputs = tuple(value.cuda() for value in _attention_inputs())
    graph = _captured(_attention, *inputs)
    # No free device memory, and none cached by this process either.
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device=None: (0, 1 << 34))
    monkeypatch.setattr(
        torch.cuda, "memory_reserved", lambda device=None: torch.cuda.memory_allocated()
    )
    oracle = _worker._Oracle(graph)
    expected = _worker._expected(
        graph,
        inputs,
        [_boundary((1, 32, 2, 64), torch.bfloat16, "cuda:0")],
        exact_reference=oracle,
    )
    assert expected.basis == "float64 on cpu"
    assert expected.oracle[0].device == inputs[0].device
    assert "free" in oracle.failures[0]
    torch.testing.assert_close(
        expected.oracle[0].cpu(),
        _exact_attention(*(value.cpu() for value in inputs)),
        rtol=1e-12,
        atol=1e-12,
    )


def _sdpa_reference(**kwargs):
    """F.scaled_dot_product_attention as region capture leaves it; make_fx
    would trace PyTorch's fused flash variant instead."""
    graph = torch.fx.Graph()
    query, key, value = (graph.placeholder(name) for name in ("q", "k", "v"))
    attention = torch.ops.aten.scaled_dot_product_attention.default
    graph.output((graph.call_function(attention, (query, key, value), kwargs),))
    module = torch.fx.GraphModule(torch.nn.Module(), graph)
    return _reference.decode_reference(_reference.encode_reference(module))


def test_float64_attention_counts_the_score_matrices_eager_never_holds():
    # Eager PyTorch's fused attention kernels never hold the score matrix,
    # one element per query and key position of each query head, but the
    # float64 reference's math kernel holds several.
    query = torch.randn(2, 8, 128, 64, dtype=torch.float16)
    key = torch.randn(2, 2, 96, 64, dtype=torch.float16)
    graph = _sdpa_reference(enable_gqa=True)
    scores = 2 * 8 * 128 * 96 * 8
    assert (
        _worker._attention_bytes(graph, (query, key, key))
        == _worker._ATTENTION_SCORE_COPIES * scores
    )
    # Other operations hold in float64 what eager PyTorch's peak shows.
    assert _worker._attention_bytes(_captured(torch.sin, query), (query,)) == 0


@requires_cuda
def test_the_oracle_moves_to_the_cpu_when_attention_scores_would_not_fit(monkeypatch):
    torch.manual_seed(0)
    inputs = tuple(
        torch.randn(1, 4, 512, 32, device="cuda", dtype=torch.float16) for _ in range(3)
    )
    graph = _sdpa_reference(is_causal=True)
    eager = _worker._outputs(graph(*inputs))
    scores = 4 * 512 * 512 * 8
    exact, basis = _worker._Oracle(graph)(inputs, eager, eager_peak_bytes=1 << 20)
    assert basis == "float64 on cuda:0"
    # Room for 8 / itemsize times eager's peak, but not for the scores.
    monkeypatch.setattr(
        torch.cuda, "mem_get_info", lambda device=None: (2 * scores, 1 << 34)
    )
    monkeypatch.setattr(
        torch.cuda, "memory_reserved", lambda device=None: torch.cuda.memory_allocated()
    )
    oracle = _worker._Oracle(graph)
    moved, basis = oracle(inputs, eager, eager_peak_bytes=1 << 20)
    assert basis == "float64 on cpu"
    assert "float64 needs about" in oracle.failures[0]
    torch.testing.assert_close(moved, exact, rtol=1e-12, atol=1e-12)


# ---- Comparison rules ----


@pytest.mark.parametrize("seed", range(3))
def test_aggregate_bound_rejects_a_scale_error_the_elementwise_rule_admits(seed):
    inputs = _attention_inputs(seed, n=128)
    graph = _captured(_attention, *inputs)
    specs = [_boundary((1, 128, 2, 64), torch.bfloat16)]
    _check(_reordered_attention, inputs, graph, specs)
    wrong = lambda k, v, q: _reordered_attention(k, v, q, scale=0.125 * 1.01)
    expected = _worker._expected(graph, inputs, specs, exact_reference=graph)
    (actual,) = _worker._outputs(wrong(*inputs))
    eager, oracle = expected.eager[0], expected.oracle[0]
    problem = _worker._oracle_mismatch(actual, eager, oracle)
    assert problem is not None and "relative L2 norm" in problem
    assert "every element is within its allowance" in problem
    with pytest.raises(_worker._EvaluationError, match="a wrong scale"):
        _check(wrong, inputs, graph, specs)
    shifted = lambda k, v, q: _reordered_attention(k, v, q, window=WINDOW + 1)
    with pytest.raises(_worker._EvaluationError, match="exceeds the allowance"):
        _check(shifted, inputs, graph, specs)


def test_aggregate_floor_admits_float32_reordering_and_errors_below_atol():
    torch.manual_seed(0)
    unit = 2.0**-24
    # Large outputs: a differently ordered recurrence, several units of
    # roundoff off everywhere, against a correctly rounded eager result.
    exact = torch.randn(4096, dtype=torch.float64) * 1000
    eager = exact.float()
    reordered = (exact * (1 + 6 * unit)).float()
    assert _worker._oracle_mismatch(reordered, eager, exact) is None
    # Small outputs: 3x eager's error, but far below atol everywhere.
    small = torch.randn(4096, dtype=torch.float64) * 0.02
    noisy = (small * (1 + 2e-6 * torch.randn(4096, dtype=torch.float64))).float()
    worse = (small * (1 + 6e-6 * torch.randn(4096, dtype=torch.float64))).float()
    assert _worker._oracle_mismatch(worse, noisy, small) is None
    # The same relative error on large outputs is far above eager's own.
    noisy_large = (exact * (1 + 2e-7 * torch.randn(4096, dtype=torch.float64))).float()
    shifted = (exact * (1 + 2e-6)).float()
    assert _worker._oracle_mismatch(shifted, noisy_large, exact) is not None


def test_failure_reports_name_the_basis_and_eager_error():
    inputs = _attention_inputs()
    graph = _captured(_attention, *inputs)
    with pytest.raises(_worker._EvaluationError) as failure:
        _check(
            lambda k, v, q: _reordered_attention(k, v, q, window=4),
            inputs,
            graph,
            [_boundary((1, 32, 2, 64), torch.bfloat16)],
        )
    message = str(failure.value)
    assert "against the reference evaluated in float64 on cpu" in message
    assert "eager PyTorch errs by" in message and "relative L2 norm" in message


def test_injected_values_may_propagate_where_they_reach():
    eager = torch.tensor([0.25, 2.0, 3.0], dtype=torch.float32)
    oracle = eager.double()
    reached = torch.tensor([True, False, False])
    nan_first = torch.tensor([math.nan, 2.0, 3.0])
    assert _worker._oracle_mismatch(nan_first, eager, oracle) is not None
    assert (
        _worker._oracle_mismatch(nan_first, eager, oracle, propagated=reached) is None
    )
    # A finite value where the injected value reaches must still be accurate.
    garbage = torch.tensor([7.0, 2.0, 3.0])
    assert _worker._oracle_mismatch(garbage, eager, oracle, propagated=reached)
    # Elsewhere a non-finite value is contamination.
    leaked = torch.tensor([0.25, math.nan, 3.0])
    assert _worker._oracle_mismatch(leaked, eager, oracle, propagated=reached)


def test_propagation_marks_outputs_an_infinite_input_reaches_through_a_mask_fill():
    # exp(-inf) and the dtype-minimum fill turn an infinite input into finite
    # eager values; a NaN shows which outputs it reaches.
    inputs = _attention_inputs(n=96)
    inputs[0].reshape(-1)[0] = -math.inf  # k[0, 0, 0, 0]
    graph = _captured(_attention, *(value.clone() for value in inputs))
    expected = _worker._expected(
        graph,
        inputs,
        [_boundary((1, 96, 2, 64), torch.bfloat16)],
        exact_reference=graph,
    )
    marked = _worker._propagation(graph, inputs, 0, expected)
    (reached,) = marked.propagated
    # Queries within the window after key 0 are reached, the others are not.
    assert reached[0, :WINDOW].all() and not reached[0, WINDOW:].any()


def test_explicit_tolerances_leave_elements_where_eager_is_not_finite_free():
    reference = lambda x: (torch.log(x),)
    inputs = (torch.tensor([0.0, -1.0, 2.0]),)
    expected = _worker._expected(
        reference, inputs, [_boundary((3,))], exact_reference=None
    )
    assert expected.oracle is None
    # log(0) = -inf and log(-1) = NaN: any value passes there.
    actual = (torch.tensor([5.0, 5.0, math.log(2.0) + 1e-3]),)
    _worker._check_outputs(actual, expected, "case 'x'", rtol=1e-2, atol=1e-2)
    with pytest.raises(_worker._EvaluationError, match="Tensor-likes are not close"):
        _worker._check_outputs(actual, expected, "case 'x'", rtol=1e-5, atol=1e-5)


def test_a_reference_that_overflows_on_finite_inputs_is_reported(caplog):
    expected = _worker._Expected((torch.tensor([math.inf, 1.0]),), None)
    with caplog.at_level(logging.WARNING, logger=_worker.__name__):
        _worker._warn_nonfinite("random 1", (torch.ones(2),), expected)
        _worker._warn_nonfinite(
            "input 0[0] = inf", (torch.tensor([math.inf, 1.0]),), expected
        )
    assert caplog.text.count("non-finite output elements") == 1
    assert "'random 1'" in caplog.text


# ---- TF32 ----


def test_tf32_context_restores_the_global_flags():
    matmul = torch.backends.cuda.matmul.allow_tf32
    convolution = torch.backends.cudnn.allow_tf32
    with pytest.raises(RuntimeError, match="inside"):
        with _worker._tf32_matmuls():
            assert torch.backends.cuda.matmul.allow_tf32
            assert torch.backends.cudnn.allow_tf32
            raise RuntimeError("inside")
    assert torch.backends.cuda.matmul.allow_tf32 == matmul
    assert torch.backends.cudnn.allow_tf32 == convolution


def test_the_oracle_runs_with_tf32_off_and_restores_the_settings():
    seen = []

    def reference(x):
        if x.dtype == torch.float64:
            seen.append(
                (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
            )
        return x * 2

    before = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        oracle = _worker._Oracle(reference)
        exact, basis = oracle((torch.ones(3),), (torch.ones(3) * 2,))
        assert basis == "float64 on cpu" and seen == [(False, False)]
        assert torch.backends.cuda.matmul.allow_tf32 is True
        assert torch.backends.cudnn.allow_tf32 is True
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = before


def test_relaxed_eager_widens_the_basis():
    torch.manual_seed(0)
    exact = torch.randn(256, dtype=torch.float64)
    eager = exact.float()  # correctly rounded
    tf32_like = (exact * (1 + 4e-4 * torch.randn(256, dtype=torch.float64))).float()
    assert _worker._oracle_mismatch(tf32_like, eager, exact) is not None
    assert _worker._oracle_mismatch(tf32_like, eager, exact, tf32_like) is None
    wrong = (exact * 1.01).float()
    assert _worker._oracle_mismatch(wrong, eager, exact, tf32_like) is not None


@requires_cuda
def test_fp32_matmul_precision_high_admits_tf32_kernels_only():
    torch.manual_seed(0)
    a = torch.randn(256, 512, device="cuda")
    b = torch.randn(512, 256, device="cuda")
    graph = _captured(torch.mm, a, b)
    specs = [_boundary((256, 256), torch.float32, "cuda:0")]

    def tf32_mm(x, y):
        with _worker._tf32_matmuls():
            return torch.mm(x, y)

    for precision, candidate, passes in (
        ("highest", torch.mm, True),
        ("highest", tf32_mm, False),
        ("high", tf32_mm, True),
        ("high", lambda x, y: torch.mm(x, y) * 1.01, False),
    ):
        expected = _worker._expected(
            graph,
            (a, b),
            specs,
            exact_reference=graph,
            fp32_matmul_precision=precision,
        )
        assert (expected.relaxed is not None) == (precision == "high")
        assert not torch.backends.cuda.matmul.allow_tf32
        try:
            _worker._check_outputs(
                (candidate(a, b),), expected, "case 'x'", rtol=None, atol=None
            )
            passed = True
        except _worker._EvaluationError:
            passed = False
        assert passed == passes, (precision, candidate)


@requires_cuda
def test_fp32_matmul_precision_highest_holds_convolutions_to_ieee_float32():
    # PyTorch's default lets cuDNN convolutions use TF32, unlike matmuls, but
    # under "highest" eager PyTorch's error, which sets the allowance, must
    # still be IEEE float32's.
    torch.manual_seed(0)
    x = torch.randn(4, 64, 32, 32, device="cuda")
    w = torch.randn(64, 64, 3, 3, device="cuda") / 24
    graph = _captured(lambda x, w: torch.nn.functional.conv2d(x, w, padding=1), x, w)
    specs = [_boundary((4, 64, 32, 32), torch.float32, "cuda:0")]

    def im2col(tf32):
        """A convolution kernel that sums in another order than cuDNN."""

        def convolution(x, w):
            columns = torch.nn.functional.unfold(x, 3, padding=1)
            with _worker._tf32(tf32):
                return (w.reshape(64, -1) @ columns).reshape(4, 64, 32, 32)

        return convolution

    def wrong(x, w):
        return torch.nn.functional.conv2d(x, w, padding=1) * 1.01

    default = torch.backends.cudnn.allow_tf32
    torch.backends.cudnn.allow_tf32 = True
    try:
        for precision, candidate, passes in (
            ("highest", im2col(False), True),
            ("highest", im2col(True), False),
            ("high", im2col(True), True),
            ("high", wrong, False),
        ):
            expected = _worker._expected(
                graph,
                (x, w),
                specs,
                exact_reference=graph,
                fp32_matmul_precision=precision,
            )
            assert torch.backends.cudnn.allow_tf32
            try:
                _worker._check_outputs(
                    (candidate(x, w),), expected, "case 'x'", rtol=None, atol=None
                )
                passed = True
            except _worker._EvaluationError:
                passed = False
            assert passed == passes, (precision, candidate)
        # Explicit tolerances compare with eager PyTorch at PyTorch's defaults.
        expected = _worker._expected(graph, (x, w), specs, exact_reference=None)
        torch.testing.assert_close(
            expected.eager[0], torch.nn.functional.conv2d(x, w, padding=1)
        )
    finally:
        torch.backends.cudnn.allow_tf32 = default
