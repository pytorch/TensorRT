"""Host/device helpers used by the example scripts and attention patches."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch

_THOR_CUDA_LIB = Path("/usr/local/cuda-13.0/thor/targets/aarch64-linux/lib")


def configure_thor_pytorch() -> None:
    """Use PyTorch fallbacks for ops whose pip CUDA wheels mismatch DriveOS Thor."""
    on_thor = os.environ.get("TRT_VLA_THOR", "auto")
    if on_thor == "auto":
        on_thor = "1" if _THOR_CUDA_LIB.is_dir() else "0"
    if on_thor == "1":
        torch.backends.cudnn.enabled = False


def force_hf_attention(module: Any, attn: str, use_cache: bool | None = False) -> None:
    """Force HuggingFace attention implementation on a module tree."""
    for m in module.modules():
        cfg = getattr(m, "config", None)
        if cfg is None:
            continue
        if hasattr(cfg, "_attn_implementation"):
            cfg._attn_implementation = attn
        if hasattr(cfg, "attn_implementation"):
            cfg.attn_implementation = attn
        if use_cache is not None and hasattr(cfg, "use_cache"):
            cfg.use_cache = use_cache

    cfg = getattr(module, "config", None)
    if cfg is None:
        return
    for name in ("vision_config", "text_config"):
        sub_cfg = getattr(cfg, name, None)
        if sub_cfg is None:
            continue
        if hasattr(sub_cfg, "_attn_implementation"):
            sub_cfg._attn_implementation = attn
        if hasattr(sub_cfg, "attn_implementation"):
            sub_cfg.attn_implementation = attn
        if use_cache is not None and hasattr(sub_cfg, "use_cache"):
            sub_cfg.use_cache = use_cache
