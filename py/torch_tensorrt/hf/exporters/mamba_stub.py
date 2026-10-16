"""Import stubs so Hub ``modeling_nemotron_h.py`` can load without ``mamba-ssm``.

Copied from Edge-LLM ``nemotron_h_patch``: only the sys.modules stubs, not the
ONNX ``from_config`` / dense-MoE forward replacements.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from typing import Any, Dict

import torch
import torch.nn.functional as F


def _rms_norm_ref(
    x,
    weight,
    bias,
    z=None,
    eps=1e-6,
    group_size=None,
    norm_before_gate=True,
    upcast=True,
):
    dtype = x.dtype
    weight = weight.float()
    bias = bias.float() if bias is not None else None
    if upcast:
        x = x.float()
        z = z.float() if z is not None else z
    if z is not None and not norm_before_gate:
        x = x * F.silu(z)
    if group_size is None:
        rstd = 1 / torch.sqrt((x.square()).mean(dim=-1, keepdim=True) + eps)
        out = (x * rstd * weight) + bias if bias is not None else (x * rstd * weight)
    else:
        *lead, last = x.shape
        x_group = x.reshape(*lead, last // group_size, group_size)
        rstd = 1 / torch.sqrt((x_group.square()).mean(dim=-1, keepdim=True) + eps)
        out = (x_group * rstd).reshape(*lead, last) * weight
        if bias is not None:
            out = out + bias
    if z is not None and norm_before_gate:
        out *= F.silu(z)
    return out.to(dtype)


def _stub_layernorm_gated() -> None:
    name = "mamba_ssm.ops.triton.layernorm_gated"
    stub = types.ModuleType(name)
    stub.rmsnorm_fn = _rms_norm_ref
    sys.modules[name] = stub


def _stub_if_broken(pkg_name: str, sentinel_attrs: Dict[str, Any]) -> None:
    try:
        spec = importlib.util.find_spec(pkg_name)
    except (ValueError, ImportError):
        spec = None
    if spec is None:
        return
    stub = types.ModuleType(pkg_name)
    stub.__spec__ = importlib.util.spec_from_loader(pkg_name, loader=None)
    for attr, value in sentinel_attrs.items():
        setattr(stub, attr, value)
    sys.modules[pkg_name] = stub


def apply() -> None:
    _stub_layernorm_gated()
    _stub_if_broken(
        "causal_conv1d",
        {"causal_conv1d_fn": None, "causal_conv1d_update": None},
    )
    _stub_if_broken(
        "mamba_ssm.ops.triton.selective_state_update",
        {"selective_state_update": None},
    )
    _stub_if_broken(
        "mamba_ssm.ops.triton.ssd_combined",
        {
            "mamba_chunk_scan_combined": None,
            "mamba_split_conv1d_scan_combined": None,
        },
    )
