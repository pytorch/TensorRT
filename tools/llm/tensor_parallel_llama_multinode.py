"""
Tensor Parallel Llama inference across two nodes with Torch-TensorRT (C++ runtime).

Reads RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT from the environment.
Each node must have exactly one GPU (cuda:0). LOCAL_RANK is used for
device selection and defaults to 0 when not set.

TP plan uses ColwiseParallel / RowwiseParallel (megatron-style column/row
linear sharding). Native TRT attention handles SDPA — no external converter.

Usage
-----
# Node 0 (rank 0) — run from /home/naren/tensorrt:
  RANK=0 WORLD_SIZE=2 MASTER_ADDR=<node0-ip> MASTER_PORT=29500 LOCAL_RANK=0 \\
    uv run python tools/llm/tensor_parallel_llama_multinode.py

# Node 1 (rank 1):
  RANK=1 WORLD_SIZE=2 MASTER_ADDR=<node0-ip> MASTER_PORT=29500 LOCAL_RANK=0 \\
    uv run python tools/llm/tensor_parallel_llama_multinode.py

Optional args:
  --model   meta-llama/Llama-3.2-1B-Instruct  (default)
  --prompt  "Your prompt here"
  --precision FP16|BF16|FP32
  --num_tokens 64
  --debug
"""

import argparse
import datetime
import logging
import os
import sys
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

# One GPU per node: use LOCAL_RANK (defaults to 0).
local_rank = int(os.environ.get("LOCAL_RANK", 0))
torch.cuda.set_device(local_rank)
DEVICE = torch.device(f"cuda:{local_rank}")

# Use a 2-hour timeout so TRT engine building does not trigger the NCCL watchdog.
dist.init_process_group(backend="nccl", timeout=datetime.timedelta(hours=2))
rank = dist.get_rank()
world_size = dist.get_world_size()

import torch_tensorrt
from torch_tensorrt.distributed import setup_nccl_for_torch_tensorrt

setup_nccl_for_torch_tensorrt()

from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    parallelize_module,
)
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import generate, record_stats, time_generate

logging.basicConfig(
    level=logging.INFO,
    format=f"[Rank {rank}] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)
logger.info(f"dist init OK  rank={rank}/{world_size}  device={DEVICE}")


def get_model(args, device_mesh):
    logger.info(f"Loading {args.model} ...")
    with torch.no_grad():
        model = (
            AutoModelForCausalLM.from_pretrained(
                args.model,
                use_cache=False,
                attn_implementation="sdpa",
            )
            .eval()
            .to(DEVICE)
        )

    # Native TRT attention handles SDPA correctly for TP models — no external
    # SDPA converter needed (and the custom converter's causal mask logic is
    # incorrect in the tensor-parallel setting).

    if args.precision == "FP16":
        model = model.to(torch.float16)
    elif args.precision == "BF16":
        model = model.to(torch.bfloat16)

    assert model.config.num_key_value_heads % world_size == 0, (
        f"num_key_value_heads ({model.config.num_key_value_heads}) not "
        f"divisible by world_size ({world_size})"
    )
    assert model.config.num_attention_heads % world_size == 0, (
        f"num_attention_heads ({model.config.num_attention_heads}) not "
        f"divisible by world_size ({world_size})"
    )

    # Megatron-style column/row parallel sharding via PyTorch DTensor.
    tp_plan = {}
    for i in range(model.config.num_hidden_layers):
        tp_plan.update(
            {
                f"model.layers.{i}.self_attn.q_proj": ColwiseParallel(),
                f"model.layers.{i}.self_attn.k_proj": ColwiseParallel(),
                f"model.layers.{i}.self_attn.v_proj": ColwiseParallel(),
                f"model.layers.{i}.self_attn.o_proj": RowwiseParallel(),
                f"model.layers.{i}.mlp.gate_proj": ColwiseParallel(),
                f"model.layers.{i}.mlp.up_proj": ColwiseParallel(),
                f"model.layers.{i}.mlp.down_proj": RowwiseParallel(),
            }
        )
    parallelize_module(model, device_mesh, tp_plan)

    # After column-sharding Q/K/V, each rank holds num_heads // world_size
    # heads. Patch these so HuggingFace attention reshapes correctly.
    for layer in model.model.layers:
        layer.self_attn.num_heads = model.config.num_attention_heads // world_size
        layer.self_attn.num_key_value_heads = (
            model.config.num_key_value_heads // world_size
        )

    logger.info("Model loaded and sharded across ranks.")
    return model


def compile_torchtrt(model, args):
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
                "use_explicit_typing": True,
                "use_fp32_acc": True,
                "device": DEVICE,
                "disable_tf32": True,
                "use_python_runtime": False,
                "debug": args.debug,
                "min_block_size": 1,
                "assume_dynamic_shape_support": True,
            },
        )
    return trt_model


if __name__ == "__main__":
    trt_model = None  # ensure names exist for the finally block
    model = None

    parser = argparse.ArgumentParser(
        description="Two-node Llama TP inference with Torch-TensorRT (C++ runtime)"
    )
    parser.add_argument(
        "--model", default="meta-llama/Llama-3.2-1B-Instruct", help="HF model name"
    )
    parser.add_argument(
        "--prompt", default="What is tensor parallelism?", help="Input prompt"
    )
    parser.add_argument(
        "--precision",
        default="FP16",
        choices=["FP16", "BF16", "FP32"],
        help="Model precision",
    )
    parser.add_argument("--num_tokens", type=int, default=64)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    device_mesh = init_device_mesh("cuda", (world_size,))

    with torch.inference_mode():
        model = get_model(args, device_mesh)

        tokenizer = AutoTokenizer.from_pretrained(args.model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        input_ids = tokenizer(args.prompt, return_tensors="pt")["input_ids"].to(DEVICE)
        max_len = input_ids.shape[1] + args.num_tokens

        logger.info("Running uncompiled PyTorch baseline ...")
        torch_tokens = generate(
            model, input_ids.clone(), max_len, tokenizer.eos_token_id
        )
        if rank == 0:
            print("\n===== PyTorch-TP (uncompiled) =====")
            print(tokenizer.decode(torch_tokens[0], skip_special_tokens=True))
            sys.stdout.flush()

        logger.info("Compiling with Torch-TensorRT (C++ runtime)...")
        trt_model = compile_torchtrt(model, args)

        # Use distributed_context to manage the NCCL lifecycle.  On __exit__
        # it calls release_nccl_comm() on all engines in the module, making
        # dist.destroy_process_group() safe without manual cleanup ordering.
        with torch_tensorrt.distributed.distributed_context(
            dist.group.WORLD, trt_model
        ) as trt_model:
            # Trigger TRT engine build explicitly and barrier so all ranks
            # finish compilation before the generation loop starts.
            logger.info("Warming up TRT model (triggering engine build)...")
            _warmup_ids = input_ids.clone()
            torch._dynamo.mark_dynamic(_warmup_ids, 1)
            _position_ids = torch.arange(_warmup_ids.shape[1]).unsqueeze(0).to(DEVICE)
            torch._dynamo.mark_dynamic(_position_ids, 1)
            _ = trt_model(_warmup_ids, position_ids=_position_ids)
            dist.barrier()
            logger.info("All ranks compiled. Starting TRT inference...")

            logger.info("Running TRT-compiled model ...")
            trt_tokens = generate(
                trt_model,
                input_ids.clone(),
                max_len,
                tokenizer.eos_token_id,
                dynamic_seqlen_range=(1, max_len),
            )
            if rank == 0:
                print("\n===== TensorRT-TP (C++ runtime) =====")
                print(tokenizer.decode(trt_tokens[0], skip_special_tokens=True))
                sys.stdout.flush()

    logger.info("Done.")
