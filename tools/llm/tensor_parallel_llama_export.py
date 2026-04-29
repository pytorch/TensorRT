"""
Tensor Parallel Llama: torch.export → save → load TRT engines across nodes.

Two modes:
  export  — torch.export → TRT AOT compile → save per-rank engines to disk
  load    — load per-rank engines from disk → run inference (no model needed)

**Known limitation**: torch.export does not yet support DTensor-parallelized
models (sharding propagation fails on symbolic reshapes).  This script
therefore exports the model *before* DTensor sharding and manually slices
the weights per-rank, injecting NCCL all-reduce ops via Torch-TensorRT's
distributed compilation path.



Usage
-----
# Export mode — export + compile + save engines (run on each node):
  torchtrtrun --nproc_per_node=1 --nnodes=2 --node_rank=0 \\
      --rdzv_endpoint=<node0-ip>:29500 \\
      tools/llm/tensor_parallel_llama_export.py --mode export --save_dir /tmp/llama_tp_engines

# Load mode — load saved engines + inference (no model download needed):
  torchtrtrun --nproc_per_node=1 --nnodes=2 --node_rank=0 \\
      --rdzv_endpoint=<node0-ip>:29500 \\
      tools/llm/tensor_parallel_llama_export.py --mode load --save_dir /tmp/llama_tp_engines
"""

import argparse
import datetime
import logging
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed.tensor._dtensor_spec
import torch.utils._pytree

# DTensorSpec must be a pytree constant before torch.export traces a TP model.
torch.utils._pytree.register_constant(
    torch.distributed.tensor._dtensor_spec.DTensorSpec
)


# One GPU per node: use LOCAL_RANK (defaults to 0).
local_rank = int(os.environ.get("LOCAL_RANK", 0))
torch.cuda.set_device(local_rank)
DEVICE = torch.device(f"cuda:{local_rank}")

# 2-hour timeout so TRT engine building doesn't trigger the NCCL watchdog.
dist.init_process_group(backend="nccl", timeout=datetime.timedelta(hours=2))
rank = dist.get_rank()
world_size = dist.get_world_size()

import torch_tensorrt
from torch_tensorrt.distributed import setup_nccl_for_torch_tensorrt

setup_nccl_for_torch_tensorrt()

from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format=f"[Rank {rank}] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)
logger.info(f"dist init OK  rank={rank}/{world_size}  device={DEVICE}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rank_path(save_dir, rank, world_size):
    return str(Path(save_dir) / f"llama_tp_rank{rank}_of_{world_size}.pt2")


def _extract_logits(outputs):
    """Get logits from HuggingFace output, tuple, or plain tensor."""
    if hasattr(outputs, "logits"):
        return outputs.logits
    if isinstance(outputs, (tuple, list)):
        return outputs[0]
    return outputs


def generate_greedy(model, input_ids, max_len, eos_token_id):
    """Greedy decode that works with any model output format."""
    seq = input_ids.clone()
    for _ in range(max_len - input_ids.shape[1]):
        position_ids = torch.arange(seq.shape[1]).unsqueeze(0).to(seq.device)
        outputs = model(seq, position_ids=position_ids)
        logits = _extract_logits(outputs)
        next_token = logits[:, -1, :].argmax(dim=-1)
        seq = torch.cat([seq, next_token[:, None]], dim=-1)
        if (next_token == eos_token_id).all():
            break
    return seq


# ---------------------------------------------------------------------------
# Manual weight slicing for export (no DTensor)
# ---------------------------------------------------------------------------


class _RowParallelLinear(torch.nn.Module):
    """Linear layer followed by NCCL all-reduce (row-parallel pattern).

    Replaces DTensor's RowwiseParallel for the export path.  The all-reduce
    uses ``_c10d_functional`` ops which torch.export traces correctly.
    """

    def __init__(self, linear: torch.nn.Linear, group_name: str):
        super().__init__()
        self.linear = linear
        self.group_name = group_name

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.linear(x)
        out = torch.ops._c10d_functional.all_reduce(out, "sum", self.group_name)
        out = torch.ops._c10d_functional.wait_tensor(out)
        return out


def get_exportable_model(args, rank, world_size):
    """Load model, slice weights per-rank, insert explicit all-reduce ops.

    Unlike DTensor-based sharding (which torch.export cannot trace), this
    manually slices weights and wraps row-parallel layers with explicit
    _c10d_functional.all_reduce ops that torch.export handles correctly.
    """
    logger.info(f"Loading {args.model} (manual shard for export) ...")
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

    # Get the default process group name for NCCL all-reduce.
    default_pg = dist.distributed_c10d._get_default_group()
    group_name = default_pg.group_name

    for layer in model.model.layers:
        attn = layer.self_attn
        mlp = layer.mlp

        # Column-parallel: slice output dim (dim 0)
        for proj in [attn.q_proj, attn.k_proj, attn.v_proj, mlp.gate_proj, mlp.up_proj]:
            w = proj.weight.data
            chunk = w.shape[0] // world_size
            proj.weight = torch.nn.Parameter(
                w[rank * chunk : (rank + 1) * chunk].contiguous()
            )
            if proj.bias is not None:
                b = proj.bias.data
                proj.bias = torch.nn.Parameter(
                    b[rank * chunk : (rank + 1) * chunk].contiguous()
                )

        # Row-parallel: slice input dim (dim 1) + wrap with all-reduce
        for attr in ["o_proj", "down_proj"]:
            proj = getattr(attn if attr == "o_proj" else mlp, attr)
            w = proj.weight.data
            chunk = w.shape[1] // world_size
            proj.weight = torch.nn.Parameter(
                w[:, rank * chunk : (rank + 1) * chunk].contiguous()
            )
            setattr(
                attn if attr == "o_proj" else mlp,
                attr,
                _RowParallelLinear(proj, group_name),
            )

    # Patch head counts for the sharded attention
    for layer in model.model.layers:
        layer.self_attn.num_heads = model.config.num_attention_heads // world_size
        layer.self_attn.num_key_value_heads = (
            model.config.num_key_value_heads // world_size
        )

    logger.info(f"Weights sliced + all-reduce inserted for rank {rank}/{world_size}.")
    return model


# ---------------------------------------------------------------------------
# export mode: torch.export → TRT AOT compile → save
# ---------------------------------------------------------------------------


def export_and_save(input_ids, args):
    """Export model (without DTensor), compile with TRT, save per-rank.

    Since torch.export cannot trace DTensor-parallelized models, we:
    1. Load the model, manually slice weights per-rank
    2. Wrap row-parallel layers with explicit _c10d_functional.all_reduce
    3. torch.export.export() (works: no DTensor, explicit NCCL ops)
    4. torch_tensorrt.dynamo.compile() with use_distributed_mode_trace=True
    5. torch_tensorrt.save() per-rank
    """
    model = get_exportable_model(args, rank, world_size)

    position_ids = torch.arange(input_ids.shape[1]).unsqueeze(0).to(DEVICE)
    max_seq = args.max_seq_len
    isl = input_ids.shape[1]

    logger.info("Exporting manually-sharded model with torch.export ...")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        seq_len = torch.export.Dim("seq_len", min=1, max=max_seq)
        try:
            ep = torch.export.export(
                model,
                args=(input_ids,),
                kwargs={"position_ids": position_ids},
                dynamic_shapes=({1: seq_len}, {1: seq_len}),
                strict=False,
            )
        except Exception:
            logger.warning("Dynamic export failed, trying with deferred asserts ...")
            ep = torch.export._trace._export(
                model,
                args=(input_ids,),
                kwargs={"position_ids": position_ids},
                dynamic_shapes=({1: seq_len}, {1: seq_len}),
                strict=False,
                prefer_deferred_runtime_asserts_over_guards=True,
            )
    logger.info("Export succeeded.")

    logger.info("Compiling exported program with TRT (AOT) ...")
    with (
        torch_tensorrt.logging.debug()
        if args.debug
        else torch.autocast("cuda", dtype=torch.float16)
    ):
        trt_model = torch_tensorrt.dynamo.compile(
            ep,
            inputs=[
                torch_tensorrt.Input(
                    min_shape=(1, 1),
                    opt_shape=(1, isl),
                    max_shape=(1, max_seq),
                    dtype=torch.int64,
                    name="input_ids",
                ),
            ],
            kwarg_inputs={
                "position_ids": torch_tensorrt.Input(
                    min_shape=(1, 1),
                    opt_shape=(1, isl),
                    max_shape=(1, max_seq),
                    dtype=torch.int64,
                    name="position_ids",
                ),
            },
            use_fp32_acc=True,
            device=DEVICE,
            disable_tf32=True,
            use_python_runtime=False,
            min_block_size=1,
            use_distributed_mode_trace=True,
            assume_dynamic_shape_support=True,
        )

    # Eagerly initialize the NCCL communicator so TRT's bind_nccl_comm()
    # finds a non-null ncclComm_t when the first execute_engine() call runs.
    # _c10d_functional.all_reduce (used by the ref model below) does not
    # trigger eager_connect_single_device, so getCommPtr() would still return
    # 0 without this call, causing TRT's getCommunicator() to assert.
    from torch_tensorrt.distributed._nccl_utils import initialize_nccl_comm

    initialize_nccl_comm()
    logger.info("NCCL communicator eagerly initialized for export verification")

    # Verify
    logger.info("Verifying compiled model ...")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        ref = _extract_logits(model(input_ids, position_ids=position_ids))
        trt = _extract_logits(trt_model(input_ids, position_ids=position_ids))
    logger.info(f"Max logit diff: {(ref.float() - trt.float()).abs().max().item():.6f}")

    # Save outside autocast — serialization doesn't need it and retrace=True
    # would fail (execute_engine has no AutocastCUDA kernel for torch.export).
    save_path = _rank_path(args.save_dir, rank, world_size)
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving TRT engine to {save_path} ...")
    torch_tensorrt.save(trt_model, save_path, retrace=False)

    dist.barrier()
    logger.info("All ranks saved.")

    del model
    torch.cuda.empty_cache()
    return trt_model


# ---------------------------------------------------------------------------
# load mode: load per-rank engines → inference
# ---------------------------------------------------------------------------


def load_and_run(input_ids, tokenizer, args):
    """Load saved per-rank TRT engine and run inference. Returns the engine so
    the caller can explicitly delete it before tearing down the process group."""
    # HuggingFace output types must be pytree-registered before torch.export.load
    # can deserialize the saved ExportedProgram's output spec.
    # Eagerly initialize PyTorch's NCCL communicator so TRT's
    # bind_nccl_comm() can extract the ncclComm_t on first engine execution.
    from torch_tensorrt.distributed._nccl_utils import initialize_nccl_comm
    from transformers.modeling_outputs import CausalLMOutputWithPast  # noqa: F401

    initialize_nccl_comm()
    logger.info("NCCL communicator eagerly initialized")

    save_path = _rank_path(args.save_dir, rank, world_size)
    logger.info(f"Loading TRT engine from {save_path} ...")
    loaded = torch_tensorrt.load(save_path)
    trt_model = loaded.module()
    logger.info("Engine loaded.")

    max_len = input_ids.shape[1] + args.num_tokens
    loaded_tokens = generate_greedy(
        trt_model,
        input_ids.clone(),
        max_len,
        tokenizer.eos_token_id,
    )

    if rank == 0:
        print("\n===== TensorRT-TP (loaded from disk) =====")
        print(tokenizer.decode(loaded_tokens[0], skip_special_tokens=True))
        sys.stdout.flush()

    return trt_model, loaded_tokens


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Llama TP: torch.export → save → load with TRT engines"
    )
    parser.add_argument("--model", default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--prompt", default="What is tensor parallelism?")
    parser.add_argument("--num_tokens", type=int, default=64)
    parser.add_argument("--max_seq_len", type=int, default=128)
    parser.add_argument(
        "--mode",
        required=True,
        choices=["export", "load"],
        help="export: AOT compile + save engines | load: load engines + infer",
    )
    parser.add_argument("--save_dir", default="/tmp/llama_tp_engines")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    input_ids = tokenizer(args.prompt, return_tensors="pt")["input_ids"].to(DEVICE)
    max_len = input_ids.shape[1] + args.num_tokens

    trt_model = None
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        if args.mode == "export":
            trt_model = export_and_save(input_ids.clone(), args)

            logger.info("Running freshly compiled model ...")
            trt_tokens = generate_greedy(
                trt_model,
                input_ids.clone(),
                max_len,
                tokenizer.eos_token_id,
            )
            if rank == 0:
                print("\n===== TensorRT-TP (freshly compiled) =====")
                print(tokenizer.decode(trt_tokens[0], skip_special_tokens=True))
                sys.stdout.flush()

        elif args.mode == "load":
            trt_model, _ = load_and_run(input_ids, tokenizer, args)

    # Delete the TRT engine before destroying the process group — the engine
    # holds a reference to the NCCL communicator and will segfault if NCCL is
    # torn down first.
    # del trt_model
    # torch.cuda.empty_cache()
    dist.destroy_process_group()
    logger.info("Done.")
    # Bypass Python GC — TRT/CUDA destructors can segfault during interpreter shutdown.
    os._exit(0)
