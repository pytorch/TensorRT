from __future__ import annotations

import argparse

import torch

from ...config import EdgeConfig
from ...utils import force_hf_attention
from .helpers import (
    alpamayo_language,
    alpamayo_visual,
)

DEFAULT_CHECKPOINT = "nvidia/Alpamayo-1.5-10B"


def prepare_export(
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
):
    """Load a base or ModelOpt-quantized Alpamayo checkpoint."""
    try:
        import modelopt.torch.opt as mto
    except ImportError as exc:
        raise ImportError(
            "Alpamayo export requires NVIDIA ModelOpt. "
            "Install the nvidia-modelopt package."
        ) from exc

    # Must be enabled before ``from_pretrained`` restores modelopt_state.pth.
    mto.enable_huggingface_checkpointing()

    from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5

    model = (
        Alpamayo1_5.from_pretrained(
            args.checkpoint or DEFAULT_CHECKPOINT,
            dtype=dtype,
            attn_implementation="eager",
        )
        .to(device=device, dtype=dtype)
        .eval()
    )

    force_hf_attention(alpamayo_visual(model), "eager")
    force_hf_attention(alpamayo_language(model), "eager")
    force_hf_attention(model.expert, "eager")

    export_config = EdgeConfig(
        model_type="alpamayo",
        engine_dir=args.engine_dir or "/tmp/alpamayo_edge_exporter",
        max_seq_len=args.max_seq_len or 4096,
    )
    return (
        model,
        {
            "device": device,
            "dtype": dtype,
            "clip_id": getattr(args, "clip_id", None),
            "t0_us": getattr(args, "t0_us", 5_100_000),
        },
        export_config,
        "velocity",
    )
