#!/usr/bin/env python3
"""EdgeExporter on kimi-k3 model."""

from __future__ import annotations

import torch
import torch_tensorrt
from exporters import EdgeConfig, EdgeExporter
from exporters.measure import print_bench
from exporters.plugin.plugin_utils import load_plugins_for_trt
from transformers import AutoModelForCausalLM, AutoTokenizer

checkpoint = "inference-optimization/Kimi-K3-0.40B"


def load_policy(device: torch.device, dtype: torch.dtype):
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
    # if pad token is not set, set it to eos token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def main() -> None:
    load_plugins_for_trt()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16

    model, tokenizer = load_policy(device, dtype)
    encoded = tokenizer("Hello, how are you?", return_tensors="pt")
    sample_inputs = {
        "input_ids": encoded["input_ids"].to(device),
        "attention_mask": encoded["attention_mask"].to(device),
    }

    exporter = EdgeExporter()
    config = EdgeConfig(
        model_type="kimi",
        engine_dir="/tmp/kimi",
        max_seq_len=1024,
    )

    program = exporter.export(model, sample_inputs, config=config)

    print("engines:", exporter.engines)

    with torch.no_grad():
        out = program.module()(**exporter.sample)

    logits = out[0] if isinstance(out, (tuple, list)) else out
    print("logits", tuple(logits.shape), "mean", float(logits.float().mean()))
    print_bench(exporter.bench)


if __name__ == "__main__":
    main()
