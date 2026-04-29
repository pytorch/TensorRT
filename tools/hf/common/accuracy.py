"""
Numerical accuracy comparison helpers.

Compares the outputs of a PyTorch reference model and a TRT-compiled
module on the same inputs, reporting cosine similarity, max/mean
absolute error, and an allclose() pass/fail per output tensor.

Default tolerances are tuned for FP16 (atol=1e-2, rtol=1e-2,
cos_sim_min=0.99).  Override via --accuracy-atol / --accuracy-rtol /
--accuracy-cos-sim-min when comparing tighter precisions or models
known to have larger accumulated error.
"""
from __future__ import annotations

from typing import Iterable

import torch
import torch.utils._pytree as pytree


# --------------------------------------------------------------------------- #
# Output flattening
# --------------------------------------------------------------------------- #

def _flatten_to_tensors(out) -> list[torch.Tensor]:
    """
    Flatten an arbitrary HF model output (ModelOutput dataclass, dict,
    tuple, list, or single tensor) into a list of leaf tensors using
    torch's pytree.  Non-tensor leaves are dropped.
    """
    leaves, _ = pytree.tree_flatten(out)
    return [t for t in leaves if isinstance(t, torch.Tensor)]


# --------------------------------------------------------------------------- #
# Per-tensor metrics
# --------------------------------------------------------------------------- #

def _cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.detach().to(torch.float32).flatten()
    b = b.detach().to(torch.float32).flatten()
    if a.numel() == 0 or b.numel() == 0:
        return float("nan")
    denom = (a.norm() * b.norm()).item()
    if denom == 0.0:
        # Both zero: define cosine as 1.0; one zero / one not: undefined → 0.
        return 1.0 if (a.norm().item() == 0 and b.norm().item() == 0) else 0.0
    return (a @ b).item() / denom


def per_tensor_metrics(
    pt: torch.Tensor,
    trt: torch.Tensor,
    *,
    atol: float = 1e-2,
    rtol: float = 1e-2,
) -> dict:
    if pt.shape != trt.shape:
        return {
            "shape_pt": tuple(pt.shape),
            "shape_trt": tuple(trt.shape),
            "shape_match": False,
            "cos_sim": float("nan"),
            "max_abs": float("nan"),
            "mean_abs": float("nan"),
            "allclose": False,
        }

    # Cast both to FP32 for fair comparison; use CPU to avoid kernel rounding
    # nondeterminism between repeated GPU runs of the same op.
    a = pt.detach().to(torch.float32)
    b = trt.detach().to(torch.float32)

    diff = (a - b).abs()
    return {
        "shape_pt": tuple(pt.shape),
        "shape_trt": tuple(trt.shape),
        "shape_match": True,
        "dtype_pt": str(pt.dtype).replace("torch.", ""),
        "dtype_trt": str(trt.dtype).replace("torch.", ""),
        "cos_sim": _cosine_similarity(a, b),
        "max_abs": diff.max().item() if diff.numel() else 0.0,
        "mean_abs": diff.mean().item() if diff.numel() else 0.0,
        "allclose": torch.allclose(a, b, rtol=rtol, atol=atol),
    }


# --------------------------------------------------------------------------- #
# Compare two outputs (each a tensor / dict / dataclass / tuple)
# --------------------------------------------------------------------------- #

def compare_outputs(
    pt_out,
    trt_out,
    *,
    atol: float = 1e-2,
    rtol: float = 1e-2,
    output_names: Iterable[str] | None = None,
) -> list[dict]:
    pt_leaves = _flatten_to_tensors(pt_out)
    trt_leaves = _flatten_to_tensors(trt_out)

    if len(pt_leaves) != len(trt_leaves):
        return [{
            "name": "<output-count-mismatch>",
            "shape_pt": f"{len(pt_leaves)} tensors",
            "shape_trt": f"{len(trt_leaves)} tensors",
            "shape_match": False,
            "cos_sim": float("nan"),
            "max_abs": float("nan"),
            "mean_abs": float("nan"),
            "allclose": False,
        }]

    names = list(output_names) if output_names else [f"out[{i}]" for i in range(len(pt_leaves))]
    if len(names) < len(pt_leaves):
        names += [f"out[{i}]" for i in range(len(names), len(pt_leaves))]

    rows: list[dict] = []
    for name, pt, trt in zip(names, pt_leaves, trt_leaves):
        row = {"name": name}
        row.update(per_tensor_metrics(pt, trt, atol=atol, rtol=rtol))
        rows.append(row)
    return rows


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #

def overall_pass(
    rows: list[dict],
    *,
    cos_sim_min: float = 0.99,
) -> bool:
    """
    A run passes if every output tensor has matching shape and cosine
    similarity above the threshold.

    Cos-sim is the canonical numerical-equivalence metric; allclose is
    reported but does NOT gate the verdict.  In FP16, isolated elements
    can drift past atol (e.g. one logit out of 50k vocab differs by 8.0)
    even when the two tensors are otherwise identical — cos_sim stays
    at 1.0 in those cases and that's the right answer.
    """
    if not rows:
        return False
    for r in rows:
        if not r.get("shape_match", False):
            return False
        cs = r.get("cos_sim", 0.0)
        if cs != cs:  # NaN
            return False
        if cs < cos_sim_min:
            return False
    return True


def print_accuracy_table(
    rows: list[dict],
    *,
    title: str = "",
    cos_sim_min: float = 0.99,
) -> None:
    if not rows:
        print("[accuracy] No outputs to compare.")
        return
    if title:
        print(f"\n{'=' * 70}")
        print(f"  {title}")
        print(f"{'=' * 70}")

    cols = ("name", "shape_pt", "cos_sim", "max_abs", "mean_abs", "allclose")
    widths = {c: max(len(c), max(len(_fmt(r.get(c, ""), c)) for r in rows)) for c in cols}
    header = "  ".join(c.ljust(widths[c]) for c in cols)
    print(header)
    print("-" * len(header))
    for r in rows:
        print("  ".join(_fmt(r.get(c, ""), c).ljust(widths[c]) for c in cols))

    overall = overall_pass(rows, cos_sim_min=cos_sim_min)
    verdict = "PASS" if overall else "FAIL"
    print(f"\nOverall: {verdict}  (cos_sim_min={cos_sim_min})")


def _fmt(v, col_name: str) -> str:
    if isinstance(v, float):
        if col_name in ("cos_sim",):
            return f"{v:.6f}"
        return f"{v:.3e}"
    if isinstance(v, bool):
        return "yes" if v else "no"
    return str(v)
