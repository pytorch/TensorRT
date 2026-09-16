from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ...config import EdgeConfig
from .mamba_stub import apply as apply_mamba_stub

DEFAULT_CHECKPOINT = "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16"


def prepare_export(
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
):
    checkpoint = args.checkpoint or DEFAULT_CHECKPOINT
    apply_mamba_stub()
    model = (
        AutoModelForCausalLM.from_pretrained(
            checkpoint,
            trust_remote_code=True,
            dtype=dtype,
        )
        .to(device=device, dtype=dtype)
        .eval()
    )
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    encoded = tokenizer(args.prompt, return_tensors="pt")
    sample_inputs = {name: tensor.to(device) for name, tensor in encoded.items()}
    config = EdgeConfig(
        model_type="nemotron_h",
        engine_dir=args.engine_dir or "/tmp/nemotron_edge_exporter",
        max_seq_len=args.max_seq_len or 128,
    )
    return model, sample_inputs, config, "logits"
