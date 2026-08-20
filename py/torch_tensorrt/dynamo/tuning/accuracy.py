"""Accuracy loss metrics matching trtexec Global Performance Tuner validators."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch_tensorrt.dynamo.utils import cosine_similarity

TensorTree = Union[torch.Tensor, Sequence["TensorTree"], Dict[str, "TensorTree"]]


def _as_float_tensor(t: torch.Tensor) -> torch.Tensor:
    return t.detach().float().flatten()


def loss_l0(
    actual: torch.Tensor,
    reference: torch.Tensor,
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> float:
    """Fraction of elements outside ``atol + rtol * abs(ref)`` (PyTorch allclose)."""
    a = _as_float_tensor(actual)
    b = _as_float_tensor(reference)
    if a.numel() == 0:
        raise ValueError("Cannot compute L0 accuracy on empty tensors")
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch for L0: {tuple(a.shape)} vs {tuple(b.shape)}")
    outside = torch.abs(a - b) > (atol + rtol * torch.abs(b))
    return float(outside.float().mean().item())


def loss_l1(actual: torch.Tensor, reference: torch.Tensor) -> float:
    a = _as_float_tensor(actual)
    b = _as_float_tensor(reference)
    if a.numel() == 0:
        raise ValueError("Cannot compute L1 accuracy on empty tensors")
    return float(torch.mean(torch.abs(a - b)).item())


def loss_l2(actual: torch.Tensor, reference: torch.Tensor) -> float:
    a = _as_float_tensor(actual)
    b = _as_float_tensor(reference)
    if a.numel() == 0:
        raise ValueError("Cannot compute L2 accuracy on empty tensors")
    return float(torch.mean((a - b) ** 2).item())


def loss_linf(actual: torch.Tensor, reference: torch.Tensor) -> float:
    a = _as_float_tensor(actual)
    b = _as_float_tensor(reference)
    if a.numel() == 0:
        raise ValueError("Cannot compute LInf accuracy on empty tensors")
    return float(torch.max(torch.abs(a - b)).item())


def loss_cos(actual: torch.Tensor, reference: torch.Tensor) -> float:
    """``1 - cosine_similarity`` (lower is better; 0 = perfect match)."""
    return 1.0 - float(cosine_similarity(reference, actual))


def compute_tensor_loss(
    actual: torch.Tensor,
    reference: torch.Tensor,
    algorithm: str = "l0",
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> float:
    algo = algorithm.lower()
    if algo == "l0":
        return loss_l0(actual, reference, atol=atol, rtol=rtol)
    if algo == "l1":
        return loss_l1(actual, reference)
    if algo == "l2":
        return loss_l2(actual, reference)
    if algo in {"linf", "linfinity"}:
        return loss_linf(actual, reference)
    if algo in {"cos", "cosine"}:
        return loss_cos(actual, reference)
    raise ValueError(
        f"Unknown accuracy_algorithm={algorithm}; "
        "expected one of 'l0', 'l1', 'l2', 'linf', 'cos'."
    )


def _flatten_named_tensors(
    tree: TensorTree, prefix: str = "output"
) -> List[Tuple[str, torch.Tensor]]:
    """Flatten a nested tree of tensors into a list of (name, tensor) tuples."""
    if isinstance(tree, torch.Tensor):
        return [(prefix, tree)]
    if isinstance(tree, dict):
        out: List[Tuple[str, torch.Tensor]] = []
        for k, v in tree.items():
            out.extend(_flatten_named_tensors(v, f"{prefix}.{k}"))
        return out
    if isinstance(tree, (list, tuple)):
        seq_out: List[Tuple[str, torch.Tensor]] = []
        for i, v in enumerate(tree):
            seq_out.extend(_flatten_named_tensors(v, f"{prefix}.{i}"))
        return seq_out
    raise TypeError(f"Unsupported output type for accuracy: {type(tree)}")


def compute_output_losses(
    actual: TensorTree,
    reference: TensorTree,
    algorithm: str = "l0",
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> Dict[str, float]:
    """Compute per-tensor accuracy loss for nested outputs."""
    actual_flat = _flatten_named_tensors(actual)
    reference_flat = _flatten_named_tensors(reference)
    if len(actual_flat) != len(reference_flat):
        raise ValueError(
            f"Output arity mismatch: actual={len(actual_flat)} ref={len(reference_flat)}"
        )
    losses: Dict[str, float] = {}
    for (aname, at), (rname, rt) in zip(actual_flat, reference_flat):
        name = aname if aname == rname else f"{aname}/{rname}"
        losses[name] = compute_tensor_loss(
            at, rt, algorithm=algorithm, atol=atol, rtol=rtol
        )
    return losses


def accuracy_failed(
    losses: Dict[str, float],
    threshold: Optional[float],
) -> bool:
    """Return True if any tensor accuracy loss exceeds the threshold."""
    if threshold is None:
        return False
    return any(v > threshold for v in losses.values())
