# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Output comparison with explicit numerical and pytree semantics."""

from __future__ import annotations

import math
from typing import Any, Iterable

import torch
from torch.utils import _pytree
from torch_tensorrt.reporting import (
    Failure,
    FailureCategory,
    OutputComparison,
    ReportStatus,
    Stage,
    ValidationReport,
    ValidationSample,
)

from ._execution import (
    _device,
    _invoke,
    _isolated_module,
    _prepare,
    _preserve_rng,
    _reject_module_input_aliases,
)
from ._inputs import InputSample, VerificationError, _fingerprint


def _finite_tolerance(value: float, name: str, *, positive: bool = False) -> float:
    """Validate and normalize one numerical comparison setting."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise VerificationError(f"{name} must be a finite number")
    value = float(value)
    if not math.isfinite(value) or value < 0 or (positive and value == 0):
        raise VerificationError(
            f"{name} must be {'positive' if positive else 'nonnegative'} and finite"
        )
    return value


def _safe_stat(value: float) -> float | None:
    """Replace nonfinite statistics with the report-safe value None."""
    return value if math.isfinite(value) else None


def _tensor_comparison(
    reference: torch.Tensor,
    actual: torch.Tensor,
    *,
    sample_id: str,
    path: str,
    atol: float,
    rtol: float,
    relative_floor: float,
) -> OutputComparison:
    """Compare one tensor pair and collect exact or tolerance-based metrics."""
    values: dict[str, Any] = dict(
        sample_id=sample_id,
        path=path,
        passed=False,
        reference_shape=list(reference.shape),
        actual_shape=list(actual.shape),
        reference_dtype=str(reference.dtype),
        actual_dtype=str(actual.dtype),
        numel=reference.numel(),
    )
    if reference.shape != actual.shape:
        values["reason"] = "shape_mismatch"
        return OutputComparison(**values)
    if reference.dtype != actual.dtype:
        values["reason"] = "dtype_mismatch"
        return OutputComparison(**values)
    if reference.device.type not in {"cpu", "cuda"} or actual.device.type not in {
        "cpu",
        "cuda",
    }:
        values["reason"] = "unsupported_tensor_device"
        return OutputComparison(**values)
    if (
        reference.layout != torch.strided
        or actual.layout != torch.strided
        or reference.is_quantized
        or actual.is_quantized
    ):
        values["reason"] = "unsupported_tensor_layout"
        return OutputComparison(**values)
    # Empty tensors pass only after dtype/shape/layout agreement; metrics are absent.
    if reference.numel() == 0:
        values["passed"] = True
        values["reason"] = "empty_tensor"
        return OutputComparison(**values)
    ref = reference.detach().resolve_conj().resolve_neg().cpu().reshape(-1)
    got = actual.detach().resolve_conj().resolve_neg().cpu().reshape(-1)
    if not reference.is_floating_point() and not reference.is_complex():
        # Python integers avoid both int64 subtraction overflow and FP64 rounding.
        pairs = list(zip(got.tolist(), ref.tolist()))
        errors = [float(abs(int(a) - int(b))) for a, b in pairs]
        relative = [
            error / max(abs(int(b)), relative_floor)
            for error, (_, b) in zip(errors, pairs)
        ]
        values["passed"] = all(a == b for a, b in pairs)
        values["mismatch_count"] = sum(a != b for a, b in pairs)
        values["max_abs_error"] = _safe_stat(max(errors))
        values["mean_abs_error"] = _safe_stat(
            math.fsum(error / len(errors) for error in errors)
        )
        values["max_relative_error"] = _safe_stat(max(relative))
        values["mean_relative_error"] = _safe_stat(
            math.fsum(error / len(relative) for error in relative)
        )
    else:
        values["reference_nan_count"] = int(torch.isnan(ref).sum().item())
        values["actual_nan_count"] = int(torch.isnan(got).sum().item())
        values["reference_inf_count"] = int(torch.isinf(ref).sum().item())
        values["actual_inf_count"] = int(torch.isinf(got).sum().item())
        if any(
            (
                values["reference_nan_count"],
                values["actual_nan_count"],
                values["reference_inf_count"],
                values["actual_inf_count"],
            )
        ):
            values["mismatch_count"] = int(
                ((~torch.isfinite(ref)) | (~torch.isfinite(got))).sum().item()
            )
            values["reason"] = "nonfinite_output"
            return OutputComparison(**values)
        dtype = torch.complex128 if reference.is_complex() else torch.float64
        ref, got = ref.to(dtype=dtype), got.to(dtype=dtype)
        absolute = (got - ref).abs()
        magnitude = ref.abs()
        relative = absolute / magnitude.clamp_min(relative_floor)
        tolerance = atol + rtol * magnitude
        close = absolute <= tolerance
        overflow = (
            ~torch.isfinite(absolute)
            | ~torch.isfinite(magnitude)
            | ~torch.isfinite(tolerance)
        )
        if bool(overflow.any().item()):
            # Finite float64/complex128 values can overflow in subtraction,
            # magnitude, or tolerance. Normalize only those elements so an
            # ``inf <= inf`` comparison never decides whether they pass.
            if reference.is_complex():
                scale = torch.maximum(
                    torch.maximum(ref.real.abs(), ref.imag.abs()),
                    torch.maximum(got.real.abs(), got.imag.abs()),
                )
            else:
                scale = torch.maximum(ref.abs(), got.abs())
            scale = torch.where(scale == 0, 1.0, scale)
            normalized_difference = (got / scale - ref / scale).abs()
            normalized_reference = (ref / scale).abs()
            normalized_limit = atol / scale + rtol * normalized_reference
            close = torch.where(
                overflow, normalized_difference <= normalized_limit, close
            )
            normalized_relative = normalized_difference / torch.maximum(
                normalized_reference, relative_floor / scale
            )
            relative = torch.where(overflow, normalized_relative, relative)
        values["passed"] = bool(torch.all(close).item())
        values["mismatch_count"] = int((~close).sum().item())
        values["max_abs_error"] = _safe_stat(float(absolute.max().item()))
        # Dividing first avoids overflow in a sum of finite large errors.
        values["mean_abs_error"] = _safe_stat(
            float((absolute / absolute.numel()).sum().item())
        )
        values["max_relative_error"] = _safe_stat(float(relative.max().item()))
        values["mean_relative_error"] = _safe_stat(
            float((relative / relative.numel()).sum().item())
        )
    if not values["passed"]:
        values["reason"] = (
            "tolerance_exceeded"
            if reference.is_floating_point() or reference.is_complex()
            else "exact_value_mismatch"
        )
    return OutputComparison(**values)


def _compare_outputs(
    reference: Any,
    actual: Any,
    sample_id: str,
    atol: float,
    rtol: float,
    relative_floor: float,
) -> list[OutputComparison]:
    """Compare matching leaves from two output pytrees."""
    reference_paths, reference_spec = _pytree.tree_flatten_with_path(reference)
    actual_paths, actual_spec = _pytree.tree_flatten_with_path(actual)
    if reference_spec != actual_spec:
        return [
            OutputComparison(
                sample_id=sample_id,
                path="output",
                passed=False,
                reason="pytree_structure_mismatch",
            )
        ]
    results: list[OutputComparison] = []
    for (key, ref), (_, got) in zip(reference_paths, actual_paths):
        path = "output" + _pytree.keystr(key)
        if isinstance(ref, torch.Tensor) and isinstance(got, torch.Tensor):
            results.append(
                _tensor_comparison(
                    ref,
                    got,
                    sample_id=sample_id,
                    path=path,
                    atol=atol,
                    rtol=rtol,
                    relative_floor=relative_floor,
                )
            )
            continue
        values = dict(
            sample_id=sample_id,
            path=path,
            passed=False,
            reference_dtype=type(ref).__name__,
            actual_dtype=type(got).__name__,
            numel=1,
            mismatch_count=1,
        )
        if type(ref) is not type(got):
            values["reason"] = "leaf_type_mismatch"
        elif ref is None or isinstance(ref, (bool, int, str)):
            values["passed"] = ref == got
            values["mismatch_count"] = 0 if values["passed"] else 1
            values["reason"] = "" if values["passed"] else "exact_value_mismatch"
        elif isinstance(ref, (float, complex)):
            dtype = torch.complex128 if isinstance(ref, complex) else torch.float64
            comparison = _tensor_comparison(
                torch.tensor(ref, dtype=dtype),
                torch.tensor(got, dtype=dtype),
                sample_id=sample_id,
                path=path,
                atol=atol,
                rtol=rtol,
                relative_floor=relative_floor,
            )
            values = dict(comparison.to_dict())
            values["reference_dtype"] = values["actual_dtype"] = type(ref).__name__
        else:
            values["reason"] = "unsupported_output_leaf"
        results.append(OutputComparison(**values))
    return results


def validate(
    reference: torch.nn.Module,
    compiled: torch.nn.Module,
    samples: Iterable[InputSample],
    *,
    atol: float = 1e-5,
    rtol: float = 1e-5,
    relative_floor: float = 1e-12,
    seed: int = 0,
    job_id: str | None = None,
) -> ValidationReport:
    """Compare every output leaf under the declared stateless eval protocol.

    Integer/bool leaves require exact equality. Floating and complex leaves use
    ``abs(actual-reference) <= atol + rtol * abs(reference)``; NaN/Inf always fail,
    even when both outputs contain the same nonfinite value. Relative statistics
    use ``max(abs(reference), relative_floor)``. Empty tensor metrics are None.
    Each model/sample call starts with ``(seed + sample_index) % 2**64``. Exported
    GraphModules retain their frozen capture mode. This validates only the supplied
    values and profile points, never an entire shape range.
    Mutation checks cover inputs and registered parameters/buffers; mutations of
    ordinary Python attributes or unregistered tensor attributes are unsupported.
    """
    atol = _finite_tolerance(atol, "atol")
    rtol = _finite_tolerance(rtol, "rtol")
    relative_floor = _finite_tolerance(relative_floor, "relative_floor", positive=True)
    try:
        samples = list(samples)
    except TypeError as error:
        raise VerificationError(
            "samples must be a nonempty iterable of InputSample objects"
        ) from error
    if not samples or not all(isinstance(sample, InputSample) for sample in samples):
        raise VerificationError(
            "samples must be a nonempty iterable of InputSample objects"
        )
    ids = [sample.sample_id or str(index) for index, sample in enumerate(samples)]
    if len(set(ids)) != len(ids):
        raise VerificationError("InputSample IDs must be unique within validation")
    evidence: list[ValidationSample] = []
    outputs: list[OutputComparison] = []
    profile_points: list[str] = []
    passed_samples = 0
    with _preserve_rng(seed):
        for sample in samples:
            _reject_module_input_aliases(reference, sample)
            _reject_module_input_aliases(compiled, sample)
        reference_copy = _isolated_module(reference)
        compiled_copy = _isolated_module(compiled)
        for index, sample in enumerate(samples):
            sample_id = ids[index]
            evidence.append(
                ValidationSample(
                    sample_id=sample_id,
                    label=sample.label,
                    profile=sample.profile,
                    input_fingerprint=_fingerprint(sample),
                )
            )
            device = _device(sample, reference_copy, compiled_copy)
            prepared, guard = _prepare(reference_copy, sample, (seed + index) % 2**64)
            expected = _invoke(reference_copy, prepared)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            guard.check()
            prepared, guard = _prepare(compiled_copy, sample, (seed + index) % 2**64)
            actual = _invoke(compiled_copy, prepared)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            guard.check()
            comparisons = _compare_outputs(
                expected, actual, sample_id, atol, rtol, relative_floor
            )
            outputs.extend(comparisons)
            if all(comparison.passed for comparison in comparisons):
                passed_samples += 1
            if sample.profile is not None and sample.profile not in profile_points:
                profile_points.append(sample.profile)
    passed = passed_samples == len(samples)
    failure = (
        None
        if passed
        else Failure(
            stage=Stage.VALIDATION,
            category=FailureCategory.VALIDATION_ERROR,
            message="Compiled outputs failed the declared validation rule",
            suggested_actions=["Inspect failed output comparisons and tested samples"],
        )
    )
    return ValidationReport(
        job_id=job_id or "",
        seed=seed,
        atol=atol,
        rtol=rtol,
        relative_floor=relative_floor,
        samples=evidence,
        outputs=outputs,
        profile_points=profile_points,
        sample_count=len(samples),
        passed_samples=passed_samples,
        passed=passed,
        status=ReportStatus.SUCCEEDED if passed else ReportStatus.FAILED,
        failure=failure,
    )
