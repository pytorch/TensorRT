# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

"""Validation inputs keep each input's sampled domain, and catch the kernels
that pass on the model's own example values only.

Counterexamples run the evaluator's own case generator, expectations and checks
on the CPU, with a plain reference function that the float64 oracle evaluates
as given.
"""

import math

import pytest
import torch

from torch_tensorrt.dynamo.region_compilation._generation import _worker
from torch_tensorrt.dynamo.region_compilation._plugins import _TensorBoundary

CPU = torch.device("cpu")


def _spec(shape, dtype=torch.float32):
    return _TensorBoundary(tuple(shape), dtype, CPU)


def _wide(value):
    """float32 for low-precision tensors, float64 for the oracle's."""
    return value.to(torch.promote_types(value.dtype, torch.float32))


def _battery(specs, outputs, samples, reference, candidate, seed=11, num_tests=3):
    """The labels of the cases a candidate fails, with their messages."""
    injected, originals, robust, failures = {}, {}, set(), {}
    oracle = _worker._Oracle(reference)
    for label, inputs in _worker._input_cases(
        specs, num_tests, True, seed, samples, injected, originals, robust
    ):
        expected = _worker._case_expectation(
            reference,
            label,
            inputs,
            outputs,
            oracle=oracle,
            precision="highest",
            injected=injected,
            originals=originals,
            robustness=label in robust,
        )
        try:
            _worker._verify_case(
                candidate, inputs, expected, outputs, label, rtol=None, atol=None
            )
        except _worker._EvaluationError as error:
            failures[label] = str(error)
    return failures


def _cases(specs, samples, seed=3, num_tests=3):
    injected, originals = {}, {}
    cases = dict(
        _worker._input_cases(specs, num_tests, True, seed, samples, injected, originals)
    )
    return cases, injected, originals


# ---- Domains: how each input varies ----

# A sign-constrained decay, positions, a bfloat16 activation, an additive mask.
SPECS = [
    _spec((64,)),
    _spec((2,)),
    _spec((16, 8), torch.bfloat16),
    _spec((32,)),
]


def _samples():
    torch.manual_seed(0)
    decay = -torch.rand(64) * 15 - 1
    decay[5] = 0.0  # a zero must stay zero
    positions = torch.tensor([3.0, 40.0])
    activation = torch.randn(16, 8).bfloat16()
    mask = torch.where(torch.rand(32) > 0.5, 0.0, -math.inf)
    first = (decay, positions, activation, mask)
    second = (decay * 0.5, positions + 1, activation * 2, mask.flip(0))
    return [first, second]


def test_domains_record_the_sampled_range_of_each_input():
    samples = _samples()
    decay, positions, activation, mask = _worker._domains(samples, SPECS)
    assert decay.kind == "continuous" and not decay.integer
    assert decay.high == 0.0 and decay.low == pytest.approx(float(samples[0][0].min()))
    assert positions.kind == "integer" and positions.integer
    # An additive 0/-inf mask has one finite value, so it varies as a constant.
    assert mask.kind == "constant" and mask.integer
    assert activation.kind == "continuous"
    assert activation.low == float(samples[1][2].float().min())
    assert activation.high == float(samples[1][2].float().max())


def test_domains_name_how_each_input_varies():
    specs = [_spec((64,)), _spec((2,)), _spec((32,)), _spec((16,)), _spec((5,))]
    samples = [
        (
            -torch.rand(64) - 1,  # continuous, negative
            torch.tensor([0.0, 8191.0]),  # integer bounds
            (torch.arange(32) % 4 == 0).float(),  # 0/1 mask, a quarter ones
            torch.ones(16),  # constant
            torch.tensor([0.3, -0.2, 0.7, 0.1, 0.25]),  # small
        )
    ]
    kinds = [domain.kind for domain in _worker._domains(samples, specs)]
    assert kinds == ["continuous", "integer", "mask", "constant", "small"]
    mask = _worker._domains(samples, specs)[2]
    assert mask.density == pytest.approx(0.25) and mask.integer
    # Without a finite value, every case takes a sample's own values.
    (empty,) = _worker._domains([(torch.full((9,), math.nan),)], [_spec((9,))])
    assert empty == _worker._Domain(None, None, "copied", True)
    # An additive mask of two integer values varies as integers.
    (additive,) = _worker._domains(
        [(torch.where(torch.arange(9) % 2 == 0, 0.0, -1e4),)], [_spec((9,))]
    )
    assert additive.kind == "integer"
    # Two other values are no mask.
    (two,) = _worker._domains(
        [(torch.where(torch.arange(9) % 2 == 0, 0.25, 0.75),)], [_spec((9,))]
    )
    assert two.kind == "continuous"


def test_only_monotone_integer_inputs_are_sorted():
    specs = [_spec((2,)), _spec((2,)), _spec((16,))]
    sample = (
        torch.tensor([0.0, 64.0]),  # [start, end]
        torch.tensor([40.0, 3.0]),  # decreasing
        torch.cat([torch.zeros(10), torch.ones(6)]),  # left padding
    )
    domains = _worker._domains([sample], specs)
    assert [domain.monotone for domain in domains] == [True, False, True]
    for label, case in _worker._input_cases(specs, 4, False, 7, [sample]):
        if label.startswith("random"):
            assert case[0][0] <= case[0][1], label
            # Sorted 0/1 masks keep the padding contiguous.
            assert bool((case[2][1:] >= case[2][:-1]).all()), label


# ---- Cases derived from samples ----


def test_sample_derived_cases_keep_labels_count_and_domains():
    samples = _samples()
    before = [tuple(value.clone() for value in sample) for sample in samples]
    injected = {}
    cases = list(_worker._input_cases(SPECS, 3, True, 11, samples, injected))
    labels = [label for label, _ in cases]
    plain = [label for label, _ in _worker._input_cases(SPECS, 3, True, 11)]
    # The battery keeps its size and labels; the samples still follow it.
    assert labels == plain + ["sample 1", "sample 2"]
    assert injected == {
        f"input {i}[0] = {v}": i for i in range(4) for v in ("nan", "inf", "-inf")
    }
    domains = _worker._domains(samples, SPECS)
    for label, case in cases:
        if label.startswith("random"):
            # Random cases keep continuous inputs within their sampled range.
            for index in (0, 2):
                value, domain = case[index], domains[index]
                if injected.get(label) == index:
                    value = value.reshape(-1)[1:]
                finite = value[value.isfinite()].float()
                assert finite.min() >= domain.low and finite.max() <= domain.high, (
                    label,
                    index,
                )
        decay = case[0]
        if not label.startswith("input 0"):
            # Neither offset +100 nor constant 1 makes the decay positive.
            assert (decay <= 0).all(), label
    by_label = dict(cases)
    # The first random case keeps the first sample's positions and mask; the
    # others vary them: positions as integers within the sampled range.
    assert torch.equal(by_label["random 1"][1], samples[0][1])
    assert torch.equal(by_label["random 1"][3], samples[0][3])
    positions = by_label["random 2"][1]
    assert not torch.equal(positions, samples[1][1])
    assert (positions == positions.round()).all()
    assert positions.min() >= 3 and positions.max() <= 41
    # Continuous values vary but keep their sign and zeros.
    random1 = by_label["random 1"][0]
    assert not torch.equal(random1, samples[0][0])
    assert random1[5] == 0 and (random1 < 0).sum() == 63
    # Non-finite injection starts from a sample-derived case, whose mask keeps
    # the sample's infinities in place.
    nan_case = by_label["input 2[0] = nan"]
    assert nan_case[2].reshape(-1)[0].isnan()
    assert nan_case[2].reshape(-1)[1:].isfinite().all()
    assert any(
        torch.equal(nan_case[3].isinf(), sample[3].isinf()) for sample in samples
    )
    # The captured samples themselves are never modified.
    for sample, kept in zip(samples, before):
        for value, wanted in zip(sample, kept):
            assert torch.equal(value.nan_to_num(), wanted.nan_to_num())


def test_sample_derived_cases_are_seeded():
    samples = _samples()
    first = list(_worker._input_cases(SPECS, 2, True, 5, samples))
    again = list(_worker._input_cases(SPECS, 2, True, 5, samples))
    other = list(_worker._input_cases(SPECS, 2, True, 6, samples))
    for (label, case), (_, repeated) in zip(first, again):
        for value, wanted in zip(case, repeated):
            torch.testing.assert_close(value, wanted, rtol=0, atol=0, equal_nan=True)
    assert not torch.equal(first[0][1][0], other[0][1][0])


def test_sample_nonfinite_values_stay_in_place():
    values = torch.linspace(-2, 2, 16)
    values[3] = -math.inf
    cases = dict(_worker._input_cases([_spec((16,))], 2, False, 3, [(values,)]))
    for label in ("random 1", "scaled x100", "offset +100", "constant 0.0"):
        assert cases[label][0][3] == -math.inf, label
        assert cases[label][0].isfinite().sum() == 15, label


def test_degenerate_inputs_vary_in_every_random_case_but_the_first():
    specs = [_spec((64,)), _spec((2,)), _spec((256,)), _spec((16,)), _spec((1,))]
    sample = (
        torch.randn(64),
        torch.tensor([3.0, 40.0]),
        (torch.arange(256) % 4 == 0).float(),
        torch.ones(16),
        torch.tensor([0.0]),
    )
    cases, _, _ = _cases(specs, [sample])
    first = cases["random 1"]
    # The first random case computes what the model does: only the continuous
    # input varies.
    assert not torch.equal(first[0], sample[0])
    assert all(torch.equal(a, b) for a, b in zip(first[1:], sample[1:]))
    for label in ("random 2", "random 3"):
        positions, mask, ones, zero = cases[label][1:]
        # Random integers within the sampled range.
        assert not torch.equal(positions, sample[1])
        assert (positions == positions.round()).all()
        assert positions.min() >= 3 and positions.max() <= 40
        # A random 0/1 mask of about the sample's density.
        assert set(mask.unique().tolist()) <= {0.0, 1.0}
        assert not torch.equal(mask, sample[2])
        assert 0.1 < float(mask.mean()) < 0.4
        # Constants scale by exp(z / 2), keeping their sign; zeros become N(0, 1).
        assert (ones > 0).all() and ones.unique().numel() == 16
        assert zero.item() != 0
    # The raw sample is still checked as it is.
    assert all(torch.equal(a, b) for a, b in zip(cases["sample 1"], sample))


def test_magnitude_and_constant_cases_keep_signs_not_ranges():
    specs = [_spec((64,)), _spec((64,)), _spec((64,)), _spec((2,))]
    sample = (
        torch.rand(64) + 1,  # in [1, 2]
        -torch.rand(64) - 1,  # in [-2, -1]
        torch.randn(64),  # both signs
        torch.tensor([0.0, 8.0]),  # integer
    )
    cases, _, _ = _cases(specs, [sample])
    positive, negative, mixed, bounds = cases["scaled x100"]
    assert positive.min() > 50 and negative.max() < -50 and mixed.abs().max() > 50
    assert (bounds == bounds.round()).all() and bounds.max() <= 8
    positive, negative, mixed, bounds = cases["offset +100"]
    assert positive.min() > 100 and mixed.min() > 90
    # Offsets never change an input's sign; integer inputs are never offset.
    assert negative.max() < 0 and bounds.max() <= 8
    positive, negative, mixed, bounds = cases["offset -100"]
    assert negative.max() < -100 and positive.min() > 0 and mixed.max() < -90
    assert cases["scaled x0.001"][0].max() < 0.01
    # Constants only where the observed sign allows them.
    positive, negative, mixed, bounds = cases["constant -1.0"]
    assert (negative == -1).all() and (mixed == -1).all()
    assert positive.min() > 0 and bounds.min() >= 0
    positive, negative, mixed, bounds = cases["constant 1.0"]
    assert (positive == 1).all() and (bounds == 1).all() and negative.max() < 0
    assert all((value == 0).all() for value in cases["constant 0.0"])


def test_injected_cases_record_the_value_they_replace():
    specs = [_spec((8,)), _spec((1,))]
    sample = (torch.linspace(1, 2, 8), torch.tensor([0.5]))
    cases, injected, originals = _cases(specs, [sample])
    for index in range(2):
        for special in ("nan", "inf", "-inf"):
            label = f"input {index}[0] = {special}"
            assert injected[label] == index
            assert math.isfinite(originals[label])
            assert not math.isfinite(cases[label][index].reshape(-1)[0])


def test_without_samples_the_cases_are_unconstrained_and_strict():
    robust = set()
    cases = dict(_worker._input_cases(SPECS, 1, False, 3, robust=robust))
    # Standard normal values, including positive decays.
    assert (cases["random 1"][0] > 0).any()
    assert (cases["offset +100"][0] > 50).all()
    # Magnitude cases are no robustness cases: they are checked as strictly.
    assert "scaled x100" in cases and robust == set()


# ---- Counterexamples the cases must reject ----


def _rms(x, gamma):
    wide = _wide(x)
    scale = torch.rsqrt(wide.pow(2).mean(-1, keepdim=True) + 1e-6)
    return (wide * scale * _wide(gamma)).to(x.dtype)


def _rms_reordered(x, gamma):
    wide = _wide(x)
    inverse = 1 / torch.sqrt(wide.pow(2).sum(-1, keepdim=True) / x.shape[-1] + 1e-6)
    return (_wide(gamma) * (wide * inverse)).to(x.dtype)


def _rms_without_gamma(x, gamma):
    wide = _wide(x)
    return (wide * torch.rsqrt(wide.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype)


@pytest.mark.parametrize("seed", [11, 12])
def test_a_kernel_ignoring_a_weight_of_ones_is_rejected(seed):
    torch.manual_seed(0)
    specs = [_spec((16, 256), torch.bfloat16), _spec((256,), torch.bfloat16)]
    samples = [(torch.randn(16, 256).bfloat16(), torch.ones(256).bfloat16())]
    outputs = [specs[0]]
    assert _battery(specs, outputs, samples, _rms, _rms_reordered, seed) == {}
    failures = _battery(specs, outputs, samples, _rms, _rms_without_gamma, seed)
    # The first random case keeps the weight of ones; the others vary it.
    assert "random 1" not in failures and "sample 1" not in failures
    assert {"random 2", "random 3", "input 1[0] = nan"} <= set(failures)


def test_a_nan_in_an_input_must_reach_the_output_of_a_kernel_that_reads_it():
    # The weight of ones itself, with a NaN: every finite element of the
    # gamma-ignoring kernel is right, but no NaN comes out.
    torch.manual_seed(0)
    gamma = torch.ones(256).bfloat16()
    gamma[0] = math.nan
    inputs = (torch.randn(16, 256).bfloat16(), gamma)
    outputs = [_spec((16, 256), torch.bfloat16)]
    expected = _worker._case_expectation(
        _rms,
        "input 1[0] = nan",
        inputs,
        outputs,
        oracle=_worker._Oracle(_rms),
        precision="highest",
        injected={"input 1[0] = nan": 1},
        originals={"input 1[0] = nan": 1.0},
    )
    # Column 0 depends on the element and is NaN in eager PyTorch.
    assert expected.required[0][:, 0].all() and not expected.required[0][:, 1:].any()
    _worker._verify_case(
        _rms_reordered, inputs, expected, outputs, "nan", rtol=None, atol=None
    )
    with pytest.raises(_worker._EvaluationError) as failure:
        _worker._verify_case(
            _rms_without_gamma, inputs, expected, outputs, "nan", rtol=None, atol=None
        )
    assert failure.value.stage == "correctness"
    assert "Every output is finite for case 'nan'" in str(failure.value)
    # Explicit tolerances apply the same requirement.
    with pytest.raises(_worker._EvaluationError, match="Every output is finite"):
        _worker._verify_case(
            _rms_without_gamma, inputs, expected, outputs, "nan", rtol=1, atol=1
        )


def test_a_kernel_whose_clamp_drops_the_nan_passes_if_it_reads_the_input():
    # torch.clamp keeps a NaN; torch.fmax, like CUDA's fmaxf and Triton's
    # tl.maximum, drops it, so the injected NaN never reaches the output.
    torch.manual_seed(0)
    specs = [_spec((64,)), _spec((64,)), _spec((64,))]
    samples = [(torch.randn(64), -torch.rand(64) - 2, torch.randn(64))]

    def reference(dt, bias, x):
        return torch.clamp(torch.nn.functional.softplus(dt + bias), min=1e-3) * x

    def dropping(dt, bias, x):
        step = torch.nn.functional.softplus(dt + bias)
        return torch.fmax(step, torch.full_like(step, 1e-3)) * x

    def hard_coded(dt, bias, x):
        # Reads dt everywhere but at element 0, which keeps the example's value.
        dt = torch.cat([samples[0][0][:1], dt[1:]])
        return dropping(dt, bias, x)

    assert _battery(specs, [specs[0]], samples, reference, dropping) == {}
    failures = _battery(specs, [specs[0]], samples, reference, hard_coded)
    assert "stay the same when the element takes another finite value" in (
        failures["input 0[0] = nan"]
    )


def test_nan_requirement_ignores_outputs_that_eager_multiplies_by_zero():
    reference = lambda x, keep: (x * keep,)
    inputs = (torch.tensor([math.nan, 2.0, 3.0]), torch.tensor([0.0, 1.0, 1.0]))
    expected = _worker._expected(reference, inputs, [_spec((3,))], exact_reference=None)
    expected = _worker._propagation(reference, inputs, 0, expected, 1.0)
    # NaN * 0 is NaN in eager PyTorch, but the element never matters.
    assert expected.required is None
    expected = _worker._propagation(
        reference, inputs, 0, expected._replace(required=None), None
    )
    assert expected.required is None
    inputs = (inputs[0], torch.ones(3))
    expected = _worker._expected(reference, inputs, [_spec((3,))], exact_reference=None)
    expected = _worker._propagation(reference, inputs, 0, expected, 1.0)
    assert expected.required[0].tolist() == [True, False, False]
    # An infinity's propagation is not required.
    inputs = (torch.tensor([math.inf, 2.0, 3.0]), torch.ones(3))
    expected = _worker._expected(reference, inputs, [_spec((3,))], exact_reference=None)
    assert _worker._propagation(reference, inputs, 0, expected, 1.0).required is None


def test_a_kernel_hard_coding_a_scalar_input_is_rejected():
    torch.manual_seed(0)
    specs = [_spec((8, 64)), _spec((1,))]
    samples = [(torch.randn(8, 64), torch.tensor([0.5]))]
    outputs = [specs[0]]
    reference = lambda x, s: x * s
    assert _battery(specs, outputs, samples, reference, lambda x, s: s * x) == {}
    failures = _battery(specs, outputs, samples, reference, lambda x, s: x * 0.5)
    assert {"random 2", "random 3", "scaled x100"} <= set(failures)
    assert "Every output is finite" in failures["input 1[0] = nan"]


def test_a_softmax_without_max_subtraction_is_rejected():
    torch.manual_seed(0)
    specs = [_spec((8, 128))]
    samples = [(torch.randn(8, 128) * 4,)]  # logits within about [-16, 16]

    def stable(x):
        shifted = torch.exp(x - x.amax(-1, keepdim=True))
        return shifted / shifted.sum(-1, keepdim=True)

    def naive(x):
        exponentials = torch.exp(x)
        return exponentials / exponentials.sum(-1, keepdim=True)

    reference = lambda x: torch.softmax(x, -1)
    assert _battery(specs, specs, samples, reference, stable) == {}
    failures = _battery(specs, specs, samples, reference, naive)
    # Magnitude cases leave the sampled range, keeping each value's sign; as
    # robustness cases they still require finite outputs.
    assert {"scaled x100", "offset +100"} <= set(failures)
    assert failures["scaled x100"].startswith(
        "Output 0 is wrong for robustness case 'scaled x100'"
    )
    assert "where eager PyTorch is finite" in failures["scaled x100"]


CHUNKS, SCALE = 4, 0.25


def _kv_partial(q, kp, vp, bounds):
    """Split-KV decode attention over the slots in a [lo, hi) window."""
    b, hq, _, d = q.shape
    g, s = kp.shape[1], kp.shape[2]
    r, c = hq // g, s // CHUNKS
    qg = q.reshape(b, g, r, d)
    kc = kp.reshape(b, g, CHUNKS, c, d)
    vc = vp.reshape(b, g, CHUNKS, c, d)
    idx = torch.arange(s, dtype=bounds.dtype).view(1, 1, CHUNKS, 1, c)
    valid = (idx >= bounds[:, 0].view(b, 1, 1, 1, 1)) & (
        idx < bounds[:, 1].view(b, 1, 1, 1, 1)
    )
    scores = torch.einsum("bgrd,bgncd->bgnrc", qg, kc) * SCALE
    scores = torch.where(valid, scores, torch.full_like(idx, -math.inf))
    m = torch.amax(scores, dim=-1, keepdim=True)
    m = torch.where(m == -math.inf, torch.zeros_like(m), m)
    p = torch.exp(scores - m)
    total = p.sum(-1, keepdim=True)
    weighted = torch.einsum(
        "bgnrc,bgncd->bgnrd", p.type_as(qg), vc * valid.transpose(-1, -2)
    )
    return weighted / torch.clamp(total, min=1e-30), (m + torch.log(total)).squeeze(-1)


def _kv_masked(q, kp, vp, bounds, window=None):
    """A kernel-like version that never reads a slot outside the window.

    ``window`` maps the bounds to the window the kernel actually uses.
    """
    b, hq, _, d = q.shape
    g, s = kp.shape[1], kp.shape[2]
    r, c = hq // g, s // CHUNKS
    lo, hi = (bounds[0, 0], bounds[0, 1]) if window is None else window(bounds)
    queries = q.float().reshape(b, g, r, d)
    out = torch.zeros(b, g, CHUNKS, r, d)
    lse = torch.full((b, g, CHUNKS, r), -math.inf)
    for chunk in range(CHUNKS):
        idx = torch.arange(chunk * c, (chunk + 1) * c, dtype=torch.float32)
        slots = idx[(idx >= lo) & (idx < hi)].long()
        if not slots.numel():
            continue
        keys, values = kp[:, :, slots].float(), vp[:, :, slots].float()
        scores = torch.einsum("bgrd,bgcd->bgrc", queries, keys) * SCALE
        m = scores.amax(-1, keepdim=True)
        p = torch.exp(scores - m)
        total = p.sum(-1, keepdim=True)
        out[:, :, chunk] = torch.einsum("bgrc,bgcd->bgrd", p, values) / total
        lse[:, :, chunk] = (m + torch.log(total)).squeeze(-1)
    return out, lse


def _kv_ignoring_bounds(q, kp, vp, bounds):
    return _kv_masked(q, kp, vp, bounds, lambda _: (0.0, float(kp.shape[2])))


def _kv_tile_aligned(q, kp, vp, bounds):
    # Rounds the end down to the 8-slot tile, as a kernel without a tail mask.
    return _kv_masked(
        q, kp, vp, bounds, lambda b: (b[0, 0], torch.floor(b[0, 1] / 8) * 8)
    )


def _kv_problem():
    torch.manual_seed(0)
    q = torch.randn(1, 2, 1, 16).bfloat16()
    kp = torch.randn(1, 1, 64, 16).bfloat16()
    vp = torch.randn(1, 1, 64, 16).bfloat16()
    # The example window covers every slot and ends on a tile boundary.
    bounds = torch.tensor([[0.0, 64.0]])
    specs = [_spec(v.shape, v.dtype) for v in (q, kp, vp, bounds)]
    outputs = [_spec((1, 1, CHUNKS, 2, 16)), _spec((1, 1, CHUNKS, 2))]
    return specs, outputs, [(q, kp, vp, bounds)]


@pytest.mark.parametrize("seed", [11, 12, 13])
def test_a_kv_window_kernel_must_honour_varied_bounds(seed):
    specs, outputs, samples = _kv_problem()
    bounds = _worker._domains(samples, specs)[3]
    assert bounds.kind == "integer" and bounds.monotone
    # Random windows keep start <= end, as the samples do.
    cases = dict(_worker._input_cases(specs, 3, False, seed, samples))
    for label in ("random 2", "random 3"):
        start, end = cases[label][3].reshape(-1).tolist()
        assert 0 <= start <= end <= 64, (label, start, end)
    # A masked kernel never reads outside the window, where eager PyTorch
    # multiplies an injected NaN by zero: it passes every case.
    assert _battery(specs, outputs, samples, _kv_partial, _kv_masked, seed) == {}
    for wrong in (_kv_ignoring_bounds, _kv_tile_aligned):
        failures = _battery(specs, outputs, samples, _kv_partial, wrong, seed)
        # Both pass on the example window itself, and fail once it varies.
        assert "random 1" not in failures and "sample 1" not in failures
        assert failures, wrong.__name__


# ---- Robustness cases: the magnitude cases beyond the sampled range ----


def _shifted_exp(a, x):
    # Like exponentials of differences of long prefix sums: exact in real
    # arithmetic, but rounding the shift, which grows with the row's largest
    # decay, costs every element precision once inputs leave the sampled range.
    largest = a.abs().nan_to_num(nan=0.0, posinf=0.0, neginf=0.0).amax(-1)
    shift = 64 * largest[..., None]
    return torch.exp((a + shift) - shift) * x


def test_robustness_cases_tolerate_precision_lost_beyond_the_sampled_range():
    torch.manual_seed(0)
    specs = [_spec((8, 8)), _spec((8, 8))]
    samples = [(-torch.rand(8, 8) * 0.045 - 0.005, torch.randn(8, 8))]
    reference = lambda a, x: torch.exp(a) * x
    robust = set()
    cases = dict(_worker._input_cases(specs, 3, True, 11, samples, robust=robust))
    # The decay is negative, so only the -100 offset moves it.
    assert robust == {"scaled x100", "offset +100", "offset -100"}
    assert _battery(specs, [specs[0]], samples, reference, _shifted_exp) == {}
    # The in-range bounds would reject the same output.
    a, x = cases["scaled x100"]
    expected = _worker._expected(
        reference, (a, x), [specs[0]], exact_reference=reference
    )
    actual = _shifted_exp(a, x)
    assert _worker._oracle_mismatch(actual, expected.eager[0], expected.oracle[0])
    assert (
        _worker._robustness_mismatch(actual, expected.eager[0], expected.oracle[0])
        is None
    )


def test_robustness_cases_reject_gross_errors():
    torch.manual_seed(0)
    specs = [_spec((64, 1024))]
    samples = [(torch.randn(64, 1024),)]

    def reference(x):
        return x * 2

    def few_wrong(x):
        # Three elements off by a fifth of the largest output, only for large
        # inputs: about 0.6% of the output's norm, under the 1% bound, but
        # twice the 10% guard per element.
        y = x * 2
        if x.abs().max() > 50:
            y.reshape(-1)[:3] += 0.2 * y.abs().max()
        return y

    def scaled(x):
        return x * (2.04 if x.abs().max() > 50 else 2)

    failures = _battery(specs, specs, samples, reference, few_wrong)
    assert {"scaled x100", "offset +100", "offset -100"} <= set(failures)
    assert not {"random 1", "random 2", "sample 1"} & set(failures)
    assert "of the output's largest exact magnitude" in failures["scaled x100"]
    failures = _battery(specs, specs, samples, reference, scaled)
    assert {"scaled x100", "offset +100", "offset -100"} <= set(failures)


# ---- Inputs that leave the region's domain ----


def _log_shifted(x):
    return torch.log(x - 0.5)  # NaN wherever x < 0.5: 69% of N(0, 1) values


def _unchecked_cases(specs, samples):
    injected, originals, unchecked, labels = {}, {}, {}, []
    for label, inputs in _worker._input_cases(
        specs, 3, True, 5, samples, injected, originals
    ):
        labels.append(label)
        expected = _worker._case_expectation(
            _log_shifted,
            label,
            inputs,
            specs,
            oracle=_worker._Oracle(_log_shifted),
            precision="highest",
            injected=injected,
            originals=originals,
        )
        if label not in injected:
            problem = _worker._unchecked(expected)
            if problem is not None:
                unchecked[label] = problem
    return unchecked, labels


def test_unconstrained_inputs_that_leave_most_outputs_unchecked_are_inconclusive(
    tmp_path,
):
    specs = [_spec((256,))]
    unchecked, labels = _unchecked_cases(specs, [])
    assert {"random 1", "random 2", "random 3"} <= set(unchecked)
    with pytest.raises(_worker._EvaluationError) as failure:
        _worker._require_conclusive(unchecked, labels, sampled=False)
    assert failure.value.stage == "setup" and failure.value.retryable is False
    assert "validation_inputs" in str(failure.value)
    report = _worker._error_report(failure.value, tmp_path, "search")
    assert report["stage"] == "setup" and report["retryable"] is False
    # Inputs derived from samples in the region's domain check every element.
    unchecked, labels = _unchecked_cases(specs, [(torch.rand(256) + 1,)])
    assert not any(label.startswith("random") for label in unchecked)
    _worker._require_conclusive(unchecked, labels, sampled=True)
    # Samples are the model's own values: even mostly non-finite outputs stand.
    _worker._require_conclusive({"random 1": "x"}, ["random 1"], sampled=True)
