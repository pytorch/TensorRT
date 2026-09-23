from __future__ import annotations

import argparse
from pathlib import Path

import torch

from ...config import EdgeConfig
from ...quantization import (
    is_modelopt_fp8_checkpoint,
    load_modelopt_fp8_model,
)
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
    from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5

    checkpoint = args.checkpoint or DEFAULT_CHECKPOINT
    if Path(checkpoint).is_dir() and is_modelopt_fp8_checkpoint(checkpoint):
        from accelerate import init_empty_weights
        from alpamayo1_5.config import Alpamayo1_5Config

        class ExportAlpamayo1_5(Alpamayo1_5):
            """Adapt Alpamayo's Transformers 4.x tie_weights override."""

            def tie_weights(self, *args, **kwargs):
                del args, kwargs
                return super().tie_weights()

        config = Alpamayo1_5Config.from_pretrained(checkpoint)
        config.attn_implementation = "eager"
        with init_empty_weights():
            model = ExportAlpamayo1_5(config)
        stats = load_modelopt_fp8_model(
            model,
            checkpoint,
            device=device,
            dtype=dtype,
        )
        print(
            "Loaded compressed ModelOpt checkpoint directly: "
            f"{stats['fp8_linears']} FP8 linears; "
            f"{stats['unsupported_attention_quantizers']} "
            "attention-core quantizer tensors use FP16 plugin execution."
        )
        # Move non-persistent rotary/frequency buffers created by the model
        # constructor. Do not pass dtype: compressed FP8 weights must stay FP8.
        model.to(device=device).eval()
    else:
        try:
            import modelopt.torch.opt as mto
        except ImportError as exc:
            raise ImportError(
                "Alpamayo export requires NVIDIA ModelOpt. "
                "Install the nvidia-modelopt package."
            ) from exc

        # Enable restoration for ordinary fake-quant ModelOpt checkpoints.
        mto.enable_huggingface_checkpointing()
        model = (
            Alpamayo1_5.from_pretrained(
                checkpoint,
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
