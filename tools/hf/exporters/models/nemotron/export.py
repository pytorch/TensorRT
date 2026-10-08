from __future__ import annotations

import argparse

import torch
import transformers
from packaging.version import Version
from transformers import AutoModelForCausalLM, AutoTokenizer

from ...config import EdgeConfig

DEFAULT_CHECKPOINT = "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16"
MIN_TRANSFORMERS = "5.3.0"


def prepare_export(
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
):
    checkpoint = args.checkpoint or DEFAULT_CHECKPOINT
    # Native transformers Nemotron-H, not the checkpoint's remote code: the Hub
    # file's non-CUDA Mamba2 fallback expands B/C to heads with ``repeat`` instead
    # of ``repeat_interleave``, so the eager reference is wrong for n_groups > 1.
    # Native support (with the correct expansion) starts at transformers 5.3.0.
    if Version(transformers.__version__) < Version(MIN_TRANSFORMERS):
        raise RuntimeError(
            f"Nemotron-H export needs transformers>={MIN_TRANSFORMERS} for the native "
            f"model (found {transformers.__version__}); the checkpoint's remote code "
            "gives a wrong eager reference without mamba-ssm."
        )
    model = (
        AutoModelForCausalLM.from_pretrained(
            checkpoint, trust_remote_code=False, dtype=dtype
        )
        .to(device=device, dtype=dtype)
        .eval()
    )
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
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
