"""
.. _run_llm_tp:

Tensor Parallel LLM inference with Torch-TensorRT
==================================================

This script extends run_llm.py to support Tensor Parallelism (TP) across multiple GPUs.
Weights in Attention (Q, K, V, O projections) and MLP (gate, up, down projections)
are sharded across ranks using PyTorch's parallelize_module API. AllReduce is inserted
automatically by RowwiseParallel at the output projection of each sub-block.
Torch-TensorRT lowers the resulting collective ops into TRT ncclwrapper via TRT-MD.

Usage
-----
.. code-block:: bash

   mpirun -n 2 python3 tensor_parallel_llama_llm.py \\
       --model meta-llama/Llama-3.2-1B-Instruct \\
       --prompt "What is parallel programming?" \\
       --model_precision FP16 --num_tokens 128
"""

import argparse
import logging
import os
from contextlib import nullcontext

# Distributed init must happen before importing torch_tensorrt.
# mpirun sets OMPI_COMM_WORLD_LOCAL_RANK / OMPI_COMM_WORLD_SIZE but NOT the
# RANK/WORLD_SIZE/MASTER_ADDR/MASTER_PORT vars that PyTorch's env:// rendezvous
# expects, so we translate them here.
import torch
import torch.distributed as dist
import torch.distributed.tensor._dtensor_spec
import torch.utils._pytree
from torch.distributed.device_mesh import init_device_mesh

# DTensorSpec appears in the graph during torch.export of a TP model.
# Register it as a pytree constant so the exporter treats it as a
# compile-time constant rather than a dynamic input.
torch.utils._pytree.register_constant(
    torch.distributed.tensor._dtensor_spec.DTensorSpec
)

_ompi_rank = int(os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", 0))
_ompi_size = int(os.environ.get("OMPI_COMM_WORLD_SIZE", 1))
os.environ.setdefault("RANK", str(_ompi_rank))
os.environ.setdefault("WORLD_SIZE", str(_ompi_size))
os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29501")

dist.init_process_group(backend="nccl")
rank = dist.get_rank()
world_size = dist.get_world_size()
local_rank = int(os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count()))
DEVICE = torch.device(f"cuda:{local_rank}")
torch.cuda.set_device(DEVICE)


def initialize_logger(
    rank, logger_file_name, file_level=logging.DEBUG, console_level=logging.INFO
):
    """Initialize rank-specific Torch-TensorRT logger with configurable handler levels."""
    logger = logging.getLogger("torch_tensorrt")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    # File handler
    fh = logging.FileHandler(logger_file_name + f"_{rank}.log", mode="w")
    fh.setLevel(file_level)
    fh.setFormatter(
        logging.Formatter(
            f"[Rank {rank}] %(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
    )
    logger.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(console_level)
    ch.setFormatter(logging.Formatter(f"[Rank {rank}] %(levelname)s: %(message)s"))
    logger.addHandler(ch)

    logger.propagate = False
    return logger


# Initialize logger for this rank
logger = initialize_logger(
    rank, "llm_tp_log_mod", file_level=logging.DEBUG, console_level=logging.INFO
)
logger.info(
    f"Initialized distributed environment: rank={rank}, world_size={world_size}, device={DEVICE}"
)

import torch_tensorrt
from torch_tensorrt.distributed import setup_nccl_for_torch_tensorrt

setup_nccl_for_torch_tensorrt()
from torch.distributed._tensor import Replicate, Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    parallelize_module,
)
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import generate, record_stats, time_generate


def get_model(args, device_mesh):
    with torch.no_grad():
        model = (
            AutoModelForCausalLM.from_pretrained(
                args.model,
                use_cache=False,
                attn_implementation="sdpa",
                ignore_mismatched_sizes=True,
            )
            .eval()
            .to(DEVICE)
        )
        # NOTE: The custom SDPA converter's dynamic causal mask computation is
        # incorrect for tensor-parallel models. TRT correctly handles the native
        # aten.scaled_dot_product_attention without the custom converter.
        # register_sdpa.enable_sdpa_converter(args.model, model.config)

    if args.model_precision == "FP16":
        model = model.to(torch.float16)
    elif args.model_precision == "BF16":
        model = model.to(torch.bfloat16)

    # Build TP plan: ColwiseParallel for first linear in each pair,
    # RowwiseParallel for second linear (inserts AllReduce when output_layouts=Replicate).
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

    # HuggingFace attention uses self.num_heads / self.num_key_value_heads to
    # reshape Q/K/V outputs. After weight sharding each rank only holds
    # num_heads // world_size columns, so these attributes must be updated or
    # the reshape will produce a shape mismatch error.
    for layer in model.model.layers:
        layer.self_attn.num_heads = model.config.num_attention_heads // world_size
        layer.self_attn.num_key_value_heads = (
            model.config.num_key_value_heads // world_size
        )

    return model


def compile_torchtrt(model, input_ids, args):
    use_fp32_acc = False
    use_explicit_typing = False
    if args.model_precision == "FP16":
        enabled_precisions = {torch.float32}
        use_fp32_acc = True
        use_explicit_typing = True
    elif args.model_precision == "BF16":
        enabled_precisions = {torch.bfloat16}
    else:
        enabled_precisions = {torch.float32}

    # torch.export does not support DTensor-parallelized models (sharding propagation
    # fails during run_decompositions). Use torch.compile with dynamic=True so that
    # torch._dynamo traces via aot_autograd (distributed mode is auto-detected)
    # and builds a single TRT engine with dynamic sequence-length profiles.
    with torch_tensorrt.logging.debug() if args.debug else nullcontext():
        trt_model = torch.compile(
            model,
            backend="torch_tensorrt",
            dynamic=True,
            options={
                "use_explicit_typing": True,
                "use_fp32_acc": use_fp32_acc,
                "device": DEVICE,
                "disable_tf32": True,
                "use_python_runtime": False,
                "debug": args.debug,
                "min_block_size": 1,
                "assume_dynamic_shape_support": True,
            },
        )

    return trt_model


def print_outputs(backend_name, gen_tokens, tokenizer):
    print(f"========= {backend_name} =========")
    print(
        f"{backend_name} model generated text: ",
        tokenizer.decode(gen_tokens[0], skip_special_tokens=True),
    )
    print("===================================")


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(
        description="Run tensor parallel LLM inference with Torch-TensorRT"
    )
    arg_parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.2-1B-Instruct",
        help="Name of LLM model",
    )
    arg_parser.add_argument(
        "--tokenizer",
        type=str,
        default="",
        help="Name of LLM model tokenizer",
    )
    arg_parser.add_argument(
        "--prompt",
        type=str,
        default="What is parallel programming?",
        help="Prompt",
    )
    arg_parser.add_argument(
        "--model_precision",
        type=str,
        default="FP16",
        help="Precision to use in the model. Options: FP16, BF16, FP32",
    )
    arg_parser.add_argument(
        "--num_tokens",
        type=int,
        default=128,
        help="Number of output tokens to generate",
    )
    arg_parser.add_argument(
        "--min_block_size",
        type=int,
        default=1,
        help="Minimum block size for TensorRT compilation",
    )
    arg_parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Enable benchmarking mode (default: False)",
    )
    arg_parser.add_argument(
        "--iterations",
        type=int,
        default=5,
        help="Number of benchmark iterations",
    )
    arg_parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for benchmarking",
    )
    arg_parser.add_argument(
        "--isl",
        type=int,
        default=2048,
        help="Input sequence length for benchmarking",
    )
    arg_parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging (default: False)",
    )
    args = arg_parser.parse_args()

    device_mesh = init_device_mesh("cuda", (world_size,))

    with torch.inference_mode():
        model = get_model(args, device_mesh)

        assert model.config.num_key_value_heads % world_size == 0, (
            f"num_key_value_heads ({model.config.num_key_value_heads}) must be "
            f"divisible by world_size ({world_size})."
        )

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or args.model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        if args.benchmark:
            input_ids = torch.randint(
                1, 10000, (args.batch_size, args.isl), dtype=torch.int64
            ).to(DEVICE)
        else:
            model_inputs = tokenizer(args.prompt, return_tensors="pt")
            input_ids = model_inputs["input_ids"].to(DEVICE)

        MAX_OUTPUT_SEQ_LENGTH = input_ids.shape[1] + args.num_tokens

        # Run uncompiled torch model first for comparison
        torch_gen_tokens = generate(
            model,
            input_ids.clone(),
            MAX_OUTPUT_SEQ_LENGTH,
            tokenizer.eos_token_id,
        )
        if rank == 0:
            print_outputs("Torch-TP (uncompiled)", torch_gen_tokens, tokenizer)

        trt_model = compile_torchtrt(model, input_ids, args)

        # Pass dynamic_seqlen_range so torch.compile traces once with a
        # dynamic seq-len dimension and reuses the same TRT engine for all
        # generation steps, avoiding per-step recompilation that would race
        # with distributed setup_nccl_comm() barriers across ranks.
        trt_gen_tokens = generate(
            trt_model,
            input_ids.clone(),
            MAX_OUTPUT_SEQ_LENGTH,
            tokenizer.eos_token_id,
            dynamic_seqlen_range=(1, MAX_OUTPUT_SEQ_LENGTH),
        )

        if args.benchmark:
            trt_timings = time_generate(
                generate,
                trt_model,
                input_ids.clone(),
                MAX_OUTPUT_SEQ_LENGTH,
                tokenizer.eos_token_id,
                iterations=args.iterations,
            )

        if rank == 0:
            if not args.benchmark:
                print_outputs("TensorRT-TP", trt_gen_tokens, tokenizer)
            else:
                trt_stats = record_stats(
                    "TensorRT-TP",
                    trt_timings,
                    args.model_precision,
                    batch_size=args.batch_size,
                    compile_time_s=None,
                )
                print("=========TensorRT-TP PERFORMANCE============")
                print(trt_stats)

    dist.destroy_process_group()
