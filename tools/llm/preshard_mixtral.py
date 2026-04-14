"""
Pre-shard Mixtral weights for fast distributed loading.

Loads the full model once (per rank), applies the tensor-parallel sharding
plan, then saves each rank's local slice to a shared DCP checkpoint on
/mnt/cluster-shared.  Subsequent inference runs can load directly from the
checkpoint without touching the full HuggingFace weights.

Usage (run once across both nodes)
-----------------------------------
# Node 0 (spirit):
  RANK=0 WORLD_SIZE=2 MASTER_ADDR=169.254.204.57 MASTER_PORT=29500 LOCAL_RANK=0 \\
    uv run python tools/llm/preshard_mixtral.py

# Node 1 (opportunity):
  RANK=1 WORLD_SIZE=2 MASTER_ADDR=169.254.204.57 MASTER_PORT=29500 LOCAL_RANK=0 \\
    uv run python tools/llm/preshard_mixtral.py

Or with torchtrtrun:
  # spirit:
  torchtrtrun --nproc_per_node=1 --nnodes=2 --node_rank=0 \\
      --rdzv_endpoint=169.254.204.57:29500 tools/llm/preshard_mixtral.py

  # opportunity:
  torchtrtrun --nproc_per_node=1 --nnodes=2 --node_rank=1 \\
      --rdzv_endpoint=169.254.204.57:29500 tools/llm/preshard_mixtral.py
"""

import argparse
import datetime
import logging
import os

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.distributed.tensor._dtensor_spec
import torch.utils._pytree
from torch.distributed.device_mesh import init_device_mesh

torch.utils._pytree.register_constant(
    torch.distributed.tensor._dtensor_spec.DTensorSpec
)

local_rank = int(os.environ.get("LOCAL_RANK", 0))
torch.cuda.set_device(local_rank)
DEVICE = torch.device(f"cuda:{local_rank}")

dist.init_process_group(backend="nccl", timeout=datetime.timedelta(hours=2))
rank = dist.get_rank()
world_size = dist.get_world_size()

from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    parallelize_module,
)
from transformers import AutoModelForCausalLM

logging.basicConfig(
    level=logging.INFO,
    format=f"[Rank {rank}] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def build_tp_plan(cfg, num_experts, world_size):
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
        for j in range(num_experts):
            tp_plan.update(
                {
                    f"model.layers.{i}.block_sparse_moe.experts.{j}.w1": ColwiseParallel(),
                    f"model.layers.{i}.block_sparse_moe.experts.{j}.w3": ColwiseParallel(),
                    f"model.layers.{i}.block_sparse_moe.experts.{j}.w2": RowwiseParallel(),
                }
            )
    return tp_plan


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-shard Mixtral for TP inference")
    parser.add_argument(
        "--model",
        default="mistralai/Mixtral-8x7B-Instruct-v0.1",
        help="HF model name or local path",
    )
    parser.add_argument(
        "--precision",
        default="BF16",
        choices=["FP16", "BF16", "FP32"],
    )
    parser.add_argument(
        "--output_dir",
        default="/mnt/cluster-shared/mixtral_sharded",
        help="DCP checkpoint directory on shared filesystem",
    )
    args = parser.parse_args()

    dtype_map = {"FP16": torch.float16, "BF16": torch.bfloat16, "FP32": torch.float32}
    torch_dtype = dtype_map[args.precision]

    device_mesh = init_device_mesh("cuda", (world_size,))

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
    tp_plan = build_tp_plan(cfg, cfg.num_local_experts, world_size)
    parallelize_module(model, device_mesh, tp_plan)

    # Fix head counts after sharding
    for layer in model.model.layers:
        layer.self_attn.num_heads = cfg.num_attention_heads // world_size
        layer.self_attn.num_key_value_heads = cfg.num_key_value_heads // world_size

    logger.info(f"Saving sharded checkpoint to {args.output_dir} ...")
    dcp.save({"model": model.state_dict()}, checkpoint_id=args.output_dir)
    logger.info("Done. Each rank's shard saved to shared filesystem.")

    dist.destroy_process_group()
