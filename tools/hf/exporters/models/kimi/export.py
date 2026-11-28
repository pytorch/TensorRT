from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ...config import EdgeConfig

DEFAULT_CHECKPOINT = "inference-optimization/Kimi-K3-0.40B"


def prepare_export(
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
):
    checkpoint = args.checkpoint or DEFAULT_CHECKPOINT
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
        model_type="kimi",
        engine_dir=args.engine_dir or "/tmp/kimi",
        max_seq_len=args.max_seq_len or 1024,
    )
    return model, sample_inputs, config, "logits"
