"""
Tensor Parallel Mixtral (MoE) inference across two nodes with Torch-TensorRT.

Reads RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT from the environment.
Each node must have exactly one GPU (cuda:0 / LOCAL_RANK=0).

Attention weights (Q/K/V/O) are sharded with ColwiseParallel / RowwiseParallel.
MoE expert projections (w1/w3 = gate+up, w2 = down) are sharded the same way;
the sparse router (block_sparse_moe.gate) is left replicated on every rank.

Usage
-----
# Node 0 (spirit, 169.254.204.57) — run from /home/naren/tensorrt:
  RANK=0 WORLD_SIZE=2 MASTER_ADDR=169.254.204.57 MASTER_PORT=29500 LOCAL_RANK=0 \\
    uv run python tools/llm/tensor_parallel_mixtral_llm.py

# Node 1 (opportunity, 169.254.217.57):
  RANK=1 WORLD_SIZE=2 MASTER_ADDR=169.254.204.57 MASTER_PORT=29500 LOCAL_RANK=0 \\
    uv run python tools/llm/tensor_parallel_mixtral_llm.py

Optional args:
  --model   mistralai/Mixtral-8x7B-Instruct-v0.1  (default)
  --prompt  "Your prompt here"
  --precision FP16|BF16|FP32
  --num_tokens 128
  --debug
"""

import argparse
import datetime
import logging
import os
from contextlib import nullcontext

import torch
import torch.distributed as dist
import torch.distributed.tensor._dtensor_spec
import torch.utils._pytree
from torch.distributed.device_mesh import init_device_mesh

# DTensorSpec must be a pytree constant before torch.export traces a TP model.
torch.utils._pytree.register_constant(
    torch.distributed.tensor._dtensor_spec.DTensorSpec
)

# One GPU per node: LOCAL_RANK defaults to 0.
local_rank = int(os.environ.get("LOCAL_RANK", 0))
torch.cuda.set_device(local_rank)
DEVICE = torch.device(f"cuda:{local_rank}")

# Use a 2-hour timeout — TRT engine building for a large MoE model can take
# many minutes, which would otherwise trip the NCCL watchdog.
dist.init_process_group(backend="nccl", timeout=datetime.timedelta(hours=2))
rank = dist.get_rank()
world_size = dist.get_world_size()

import torch.distributed.checkpoint as dcp
import torch_tensorrt
from torch_tensorrt.distributed import setup_nccl_for_torch_tensorrt

setup_nccl_for_torch_tensorrt()

from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    parallelize_module,
)
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from utils import generate, record_stats, time_generate

logging.basicConfig(
    level=logging.INFO,
    format=f"[Rank {rank}] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)
logger.info(f"dist init OK  rank={rank}/{world_size}  device={DEVICE}")


def build_tp_plan(cfg):
    tp_plan = {}
    for i in range(cfg.num_hidden_layers):
        tp_plan.update(
            {
                f"model.layers.{i}.self_attn.q_proj": ColwiseParallel(),
                f"model.layers.{i}.self_attn.k_proj": ColwiseParallel(),
                f"model.layers.{i}.self_attn.v_proj": ColwiseParallel(),
                f"model.layers.{i}.self_attn.o_proj": RowwiseParallel(),
            }
        )
        for j in range(cfg.num_local_experts):
            tp_plan.update(
                {
                    f"model.layers.{i}.block_sparse_moe.experts.{j}.w1": ColwiseParallel(),
                    f"model.layers.{i}.block_sparse_moe.experts.{j}.w3": ColwiseParallel(),
                    f"model.layers.{i}.block_sparse_moe.experts.{j}.w2": RowwiseParallel(),
                }
            )
    return tp_plan


def get_model(args, device_mesh):
    dtype_map = {"FP16": torch.float16, "BF16": torch.bfloat16, "FP32": torch.float32}
    torch_dtype = dtype_map[args.precision]

    if args.sharded_checkpoint:
        # Fast path: load config only, initialize with random weights, then
        # apply sharding plan and overwrite with DCP shards.  Each rank reads
        # only its own ~47GB slice from the shared filesystem.
        logger.info(f"Loading config from {args.model} ...")
        cfg = AutoConfig.from_pretrained(args.model)
        with torch.no_grad():
            model = (
                AutoModelForCausalLM.from_config(cfg, attn_implementation="sdpa")
                .eval()
                .to(torch_dtype)
                .to(DEVICE)
            )
        parallelize_module(model, device_mesh, build_tp_plan(cfg))
        logger.info(f"Loading sharded weights from {args.sharded_checkpoint} ...")
        dcp.load({"model": model.state_dict()}, checkpoint_id=args.sharded_checkpoint)
        logger.info("Sharded checkpoint loaded.")
    else:
        logger.info(f"Loading {args.model} in {args.precision} ...")
        with torch.no_grad():
            model = (
                AutoModelForCausalLM.from_pretrained(
                    args.model,
                    use_cache=False,
                    attn_implementation="sdpa",
                    torch_dtype=torch_dtype,
                )
                .eval()
                .to(DEVICE)
            )
        cfg = model.config
        parallelize_module(model, device_mesh, build_tp_plan(cfg))

    cfg = model.config
    assert cfg.num_key_value_heads % world_size == 0, (
        f"num_key_value_heads ({cfg.num_key_value_heads}) not divisible by world_size ({world_size})"
    )
    assert cfg.num_attention_heads % world_size == 0, (
        f"num_attention_heads ({cfg.num_attention_heads}) not divisible by world_size ({world_size})"
    )

    # After column-sharding Q/K/V, each rank holds num_heads // world_size
    # heads. Patch these so HuggingFace attention reshapes correctly.
    for layer in model.model.layers:
        layer.self_attn.num_heads = cfg.num_attention_heads // world_size
        layer.self_attn.num_key_value_heads = cfg.num_key_value_heads // world_size

    logger.info("Model loaded and sharded across ranks.")
    return model


def compile_torchtrt(model, args):
    use_fp32_acc = args.precision == "FP16"
    use_explicit_typing = args.precision in ("FP16", "BF16")

    if args.precision == "FP16":
        enabled_precisions = {torch.float16}
    elif args.precision == "BF16":
        enabled_precisions = {torch.bfloat16}
    else:
        enabled_precisions = {torch.float32}

    with torch_tensorrt.logging.debug() if args.debug else nullcontext():
        trt_model = torch.compile(
            model,
            backend="torch_tensorrt",
            dynamic=True,
            options={
                "enabled_precisions": enabled_precisions,
                "use_explicit_typing": use_explicit_typing,
                "use_fp32_acc": use_fp32_acc,
                "device": DEVICE,
                "disable_tf32": True,
                "use_python_runtime": False,
                "use_distributed_mode_trace": True,
                "debug": args.debug,
                "min_block_size": 1,
                "assume_dynamic_shape_support": True,
            },
        )
    return trt_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Two-node Mixtral MoE TP inference with Torch-TensorRT"
    )
    parser.add_argument(
        "--model",
        default="mistralai/Mixtral-8x7B-Instruct-v0.1",
        help="HF model name or local path",
    )
    parser.add_argument(
        "--prompt", default="What is mixture of experts?", help="Input prompt"
    )
    parser.add_argument(
        "--precision",
        default="BF16",
        choices=["FP16", "BF16", "FP32"],
        help="Model precision (BF16 recommended for Mixtral)",
    )
    parser.add_argument("--num_tokens", type=int, default=128)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--sharded_checkpoint",
        type=str,
        default="",
        help="Path to DCP sharded checkpoint (e.g. /mnt/cluster-shared/mixtral_sharded). "
             "If set, skips HF weight download and loads only this rank's shard.",
    )
    args = parser.parse_args()

    device_mesh = init_device_mesh("cuda", (world_size,))

    with torch.inference_mode():
        model = get_model(args, device_mesh)

        tokenizer = AutoTokenizer.from_pretrained(args.model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        input_ids = tokenizer(args.prompt, return_tensors="pt")["input_ids"].to(DEVICE)
        max_len = input_ids.shape[1] + args.num_tokens

        logger.info("Compiling with Torch-TensorRT ...")
        trt_model = compile_torchtrt(model, args)

        # Explicitly warm up to trigger TRT engine building, then barrier so
        # both ranks finish building before the generation loop starts.
        # Without this, a slow build on one rank times out the other at the
        # next NCCL collective.
        logger.info("Warming up TRT model (triggering engine build)...")
        _position_ids = torch.arange(input_ids.shape[1]).unsqueeze(0).to(DEVICE)
        _ = trt_model(input_ids.clone(), position_ids=_position_ids)
        dist.barrier()
        logger.info("All ranks finished TRT compilation, starting inference...")

        trt_tokens = generate(
            trt_model,
            input_ids.clone(),
            max_len,
            tokenizer.eos_token_id,
            dynamic_seqlen_range=(1, max_len),
        )
        if rank == 0:
            print("\n===== TensorRT-TP (Mixtral) =====")
            print(tokenizer.decode(trt_tokens[0], skip_special_tokens=True))

    dist.destroy_process_group()
    logger.info("Done.")
