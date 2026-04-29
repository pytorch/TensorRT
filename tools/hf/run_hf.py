"""
run_hf.py – Compile and benchmark any HuggingFace model with Torch-TensorRT.

Single-process usage
--------------------
# Encoder (BERT):
  uv run python run_hf.py --model bert-base-uncased --benchmark

Multi-device (tensor-parallel) usage via torchtrtrun
----------------------------------------------------
# Single node, 2 GPUs (Llama-family LLM only — auto-activates TP):
  torchtrtrun --nproc_per_node=2 run_hf.py \\
      --model meta-llama/Llama-3.2-1B-Instruct --benchmark

# Two nodes, 1 GPU per node:
  # Node 0:
  torchtrtrun --nproc_per_node=1 --nnodes=2 --node_rank=0 \\
      --master_addr=10.0.0.1 --master_port=29500 \\
      run_hf.py --model meta-llama/Llama-3.1-8B-Instruct --benchmark
  # Node 1:
  torchtrtrun --nproc_per_node=1 --nnodes=2 --node_rank=1 \\
      --master_addr=10.0.0.1 --master_port=29500 \\
      run_hf.py --model meta-llama/Llama-3.1-8B-Instruct --benchmark

Under torchtrtrun the run automatically:
  - Initializes torch.distributed with NCCL (2-hour timeout for engine builds).
  - Pins the process to cuda:LOCAL_RANK.
  - Builds a 1-D TP device mesh and shards the model with Megatron-style
    column/row sharding (Llama-family layouts only for now).
  - Forces --mode compile (torch.export does not currently work with TP).
  - Wraps inference in distributed_context so NCCL is released cleanly.

Standard usage (single process)
--------------------------------

# LLM (Path A: cast in PyTorch):
  uv run python run_hf.py --model meta-llama/Llama-3.2-1B-Instruct \\
      --precision FP16 --isl 512 --num-tokens 128 --benchmark

# LLM (Path B: TRT autocast — model stays FP32):
  uv run python run_hf.py --model meta-llama/Llama-3.2-1B-Instruct \\
      --precision FP16 --autocast --isl 512 --benchmark

# Diffusion:
  uv run python run_hf.py --model stabilityai/stable-diffusion-2-1 \\
      --precision FP16 --image-size 512 --num-inference-steps 20 --benchmark

# Whisper ASR:
  uv run python run_hf.py --model openai/whisper-large-v3 --precision FP16 --benchmark

# Override family detection:
  uv run python run_hf.py --model some/model --task text-generation --benchmark

# VLM (LLaVA, PaliGemma, Qwen2-VL, SmolVLM, …):
  uv run python run_hf.py --model HuggingFaceTB/SmolVLM-256M-Instruct --benchmark

# VLA (OpenVLA, SpatialVLA — robotics; loads with trust_remote_code=True):
  uv run python run_hf.py --model openvla/openvla-7b --benchmark

# Object detection (DETR, RT-DETR):
  uv run python run_hf.py --model facebook/detr-resnet-50 --benchmark

# SAM (image encoder only):
  uv run python run_hf.py --model facebook/sam-vit-base --benchmark

# torch.compile fast path (no export, no serialization):
  uv run python run_hf.py --model bert-base-uncased --mode compile --benchmark

KV cache modes (LLM only)
-------------------------
  (default)            No KV cache: prefill throughput benchmark only.
  --cache static_v1    FX-graph lowering pass (tools/llm/static_cache_v1.py).
  --cache static_v2    FX-graph lowering pass (tools/llm/static_cache_v2.py).
  --cache hf_static    HF-native StaticCache via
                       TorchExportableModuleWithStaticCache (transformers ≥ 4.43).
                       Forward becomes (input_ids, cache_position) → logits.
                       Works with both --mode compile and --mode export.
                       Export mode uses strict=True (strict=False triggers a PyTorch
                       internal bug in run_decompositions; see issue #4162).

Precision handling
------------------
Torch-TensorRT runs with use_explicit_typing=True (strongly typed network),
so dtype must come from exactly one source:

  Path A (default) : PyTorch casts the model (model.to(fp16)).  Compile sets
                     use_fp32_acc=True for FP16 matmul accumulation.
  Path B (--autocast) : Model stays FP32; TRT compiler casts via
                        enable_autocast=True, autocast_low_precision_type=<dtype>.

`enabled_precisions` is deprecated under use_explicit_typing and is never set.
"""

from __future__ import annotations

import sys
from pathlib import Path

import tyro

# Make strategies/ and common/ importable as top-level packages.
_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE), str(_HERE / "strategies"), str(_HERE / "common")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from common.dist import (
    init_distributed,
    is_distributed,
    is_master,
    rank,
    world_size,
)
from detect import detect_family
from strategies.base import CLIArgs, RunConfig


def _build_strategy(family: str, cfg: RunConfig):
    if family == "llm":
        if is_distributed():
            from strategies.llm_tp import LLMTPStrategy

            return LLMTPStrategy(cfg)
        from strategies.llm import LLMStrategy

        return LLMStrategy(cfg)
    if family == "encoder":
        from strategies.encoder import EncoderStrategy

        return EncoderStrategy(cfg)
    if family == "seq2seq":
        from strategies.seq2seq import SeqToSeqStrategy

        return SeqToSeqStrategy(cfg)
    if family == "diffusion":
        from strategies.diffusion import DiffusionStrategy

        return DiffusionStrategy(cfg)
    if family == "audio":
        from strategies.audio import AudioStrategy

        return AudioStrategy(cfg)
    if family == "multimodal":
        from strategies.multimodal import MultimodalStrategy

        return MultimodalStrategy(cfg)
    if family == "video_diffusion":
        from strategies.video_diffusion import VideoDiffusionStrategy

        return VideoDiffusionStrategy(cfg)
    if family == "vlm":
        from strategies.vlm import VLMStrategy

        return VLMStrategy(cfg)
    if family == "vla":
        from strategies.vla import VLAStrategy

        return VLAStrategy(cfg)
    if family == "detection":
        from strategies.detection import DetectionStrategy

        return DetectionStrategy(cfg)
    raise ValueError(f"No strategy implemented for family '{family}'")


def main() -> None:
    args: CLIArgs = tyro.cli(CLIArgs)

    if not args.debug:
        import logging

        import torch_tensorrt

        torch_tensorrt.logging.set_level(logging.WARNING)

    # Initialize torch.distributed if launched under torchtrtrun
    # (WORLD_SIZE > 1).  No-op for single-process runs.
    init_distributed()

    if is_distributed() and is_master():
        print(f"[run_hf] Distributed mode: rank {rank()}/{world_size()}")

    if is_master():
        print(f"[run_hf] Detecting family for '{args.model}' ...")
    family = detect_family(args.model, task_override=args.task)
    if is_master():
        print(f"[run_hf] Family = {family}")
    if is_distributed() and family != "llm":
        raise NotImplementedError(
            f"Tensor-parallel support is currently only wired for the 'llm' "
            f"family.  Got '{family}'.  Run without torchtrtrun for "
            f"single-device compilation."
        )

    # CLIArgs subclasses RunConfig — pass it directly; strategies just ignore
    # the extra benchmark/generate/json_out fields.
    strategy = _build_strategy(family, args)

    strategy.load()
    strategy.compile()

    if args.generate:
        strategy.generate()

    if args.accuracy:
        strategy.accuracy()

    rows: list[dict] = []
    if args.benchmark:
        rows = strategy.benchmark()

    if args.json_out and rows:
        from common.metrics import dump_json

        dump_json(rows, args.json_out)


if __name__ == "__main__":
    main()
