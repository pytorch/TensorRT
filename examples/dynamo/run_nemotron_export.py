#!/usr/bin/env python3
"""Smoke EdgeExporter on Nemotron-H (one language engine: attn + mamba + MoE).

Pass the HF causal LM. Collation is tokenizer → input_ids; the spec embeds and
pads to max_seq_len. apply_mamba_stub() must run before from_pretrained.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]  # TensorRT/
_TRT_PY = _REPO_ROOT / "py"
_TEST = Path("/home/micwilliams/workspace/Test")
_NEMOTRON = _TEST / "nemotron"

for p in (_NEMOTRON, _TEST):
    s = str(p)
    while s in sys.path:
        sys.path.remove(s)
    sys.path.insert(0, s)

import torch  # noqa: E402
import torch_tensorrt  # noqa: E402

_src_pkg = str(_TRT_PY / "torch_tensorrt")
if _src_pkg not in list(torch_tensorrt.__path__):
    torch_tensorrt.__path__.append(_src_pkg)

from mamba_stub import apply as apply_mamba_stub
from torch_tensorrt.hf.exporters import EdgeConfig, EdgeExporter
from transformers import AutoModelForCausalLM, AutoTokenizer
from trt.plugin.plugin_utils import load_plugins_for_trt
from trt.utils import configure_thor_pytorch


def load_nemotron(checkpoint: str, device: torch.device, dtype: torch.dtype):
    apply_mamba_stub()
    model = (
        AutoModelForCausalLM.from_pretrained(
            checkpoint,
            trust_remote_code=True,
            torch_dtype=dtype,
        )
        .to(device=device, dtype=dtype)
        .eval()
    )
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--compile", action="store_true", help="Build TRT engines (default: dryrun)"
    )
    parser.add_argument("--engine-dir", default="/tmp/nemotron_edge_exporter")
    parser.add_argument(
        "--checkpoint",
        default="nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16",
    )
    parser.add_argument("--prompt", default="Hello.")
    parser.add_argument("--max-seq-len", type=int, default=128)
    args = parser.parse_args()

    configure_thor_pytorch()
    load_plugins_for_trt()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16

    model, tokenizer = load_nemotron(args.checkpoint, device, dtype)
    encoded = tokenizer(args.prompt, return_tensors="pt")
    sample_inputs = {
        "input_ids": encoded["input_ids"].to(device),
        "attention_mask": encoded["attention_mask"].to(device),
    }

    exporter = EdgeExporter()
    config = EdgeConfig(
        model_type="nemotron_h",
        engine_dir=args.engine_dir,
        max_seq_len=args.max_seq_len,
        dryrun=not args.compile,
        skip_runtime_export=False,
    )
    program = exporter.export(model, sample_inputs, config=config)

    print("engines:", exporter.engines)
    print("saved:", exporter.save_engines())
    print("runtime keys:", sorted(exporter.sample))

    with torch.no_grad():
        if hasattr(program, "module"):
            out = program.module()(**exporter.sample)
        else:
            out = program(**exporter.sample)

    logits = out[0] if isinstance(out, (tuple, list)) else out
    print("logits", tuple(logits.shape), "mean", float(logits.float().mean()))


if __name__ == "__main__":
    main()
