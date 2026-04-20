"""
Tensor Parallel Distributed Inference with Torch-TensorRT (torchrun)
=====================================================================

Same model as tensor_parallel_simple_example.py but launched with
torchrun / ``python -m torch_tensorrt.distributed.run`` instead of mpirun.



Usage
-----
.. code-block:: bash

   # Single-node, 2 GPUs
   torchrun --nproc_per_node=2 tensor_parallel_simple_example_torchrun.py

   # Two nodes, 1 GPU each — run on BOTH nodes simultaneously:
   #   Node 0 (spirit):
   RANK=0 WORLD_SIZE=2 MASTER_ADDR=<spirit_ip> MASTER_PORT=29500 LOCAL_RANK=0 \\
       uv run python tensor_parallel_simple_example_torchrun.py

   #   Node 1 (opportunity):
   RANK=1 WORLD_SIZE=2 MASTER_ADDR=<spirit_ip> MASTER_PORT=29500 LOCAL_RANK=0 \\
       uv run python tensor_parallel_simple_example_torchrun.py

   # Or via torchtrtrun (sets up NCCL library paths automatically):
   python -m torch_tensorrt.distributed.run --nproc_per_node=2 \\
       tensor_parallel_simple_example_torchrun.py

Optional args:
  --mode       jit_python | jit_cpp | export | load  (default: jit_python)
  --save-path  /tmp/tp_model.ep
  --precision  FP16 | BF16 | FP32  (default: FP16)
  --debug
"""

import argparse
import datetime
import logging
import os
from contextlib import nullcontext

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.utils._pytree
from torch.distributed.device_mesh import init_device_mesh
from torch_tensorrt.distributed import setup_nccl_for_torch_tensorrt

torch.utils._pytree.register_constant(
    torch.distributed.tensor._dtensor_spec.DTensorSpec
)

# One GPU per node; LOCAL_RANK defaults to 0 for plain env-var launch.
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

from torch.distributed._tensor import Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    parallelize_module,
)

logging.basicConfig(
    level=logging.INFO,
    format=f"[Rank {rank}] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)
logger.info(f"dist init OK  rank={rank}/{world_size}  device={DEVICE}")


class ToyModel(nn.Module):
    """MLP based model"""

    def __init__(self):
        super().__init__()
        self.in_proj = nn.Linear(10, 3200)
        self.relu = nn.ReLU()
        self.out_proj = nn.Linear(3200, 1600)
        self.in_proj2 = nn.Linear(1600, 500)
        self.out_proj2 = nn.Linear(500, 100)

    def forward(self, x):
        x = self.out_proj(self.relu(self.in_proj(x)))
        x = self.relu(x)
        x = self.out_proj2(self.relu(self.in_proj2(x)))
        return x


def get_model(device_mesh):
    assert (
        world_size % 2 == 0
    ), f"TP examples require an even number of GPUs, got {world_size}"
    model = ToyModel().to(DEVICE)
    parallelize_module(
        module=model,
        device_mesh=device_mesh,
        parallelize_plan={
            "in_proj": ColwiseParallel(input_layouts=Shard(0)),
            "out_proj": RowwiseParallel(output_layouts=Shard(0)),
            "in_proj2": ColwiseParallel(input_layouts=Shard(0)),
            "out_proj2": RowwiseParallel(output_layouts=Shard(0)),
        },
    )
    logger.info("Model built and sharded across ranks.")
    return model


def compile_torchtrt(model, args):
    model.eval()

    use_fp32_acc = False
    use_explicit_typing = False
    if args.precision == "FP16":
        enabled_precisions = {torch.float16}
        use_fp32_acc = True
        use_explicit_typing = True
    elif args.precision == "BF16":
        enabled_precisions = {torch.bfloat16}
        use_explicit_typing = True
    else:
        enabled_precisions = {torch.float32}
        use_explicit_typing = True

    use_python_runtime = args.mode == "jit_python"

    with torch_tensorrt.logging.debug() if args.debug else nullcontext():
        trt_model = torch.compile(
            model,
            backend="torch_tensorrt",
            dynamic=False,
            options={
                "enabled_precisions": enabled_precisions,
                "use_explicit_typing": use_explicit_typing,
                "use_fp32_acc": use_fp32_acc,
                "device": DEVICE,
                "disable_tf32": True,
                "use_python_runtime": use_python_runtime,
                "debug": args.debug,
                "min_block_size": 1,
            },
        )
    return trt_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tensor Parallel Simple Example (torchrun)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["jit_python", "jit_cpp", "export", "load"],
        default="jit_python",
    )
    parser.add_argument("--save-path", type=str, default="/tmp/tp_model.ep")
    parser.add_argument(
        "--precision",
        default="FP16",
        choices=["FP16", "BF16", "FP32"],
    )
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    device_mesh = init_device_mesh("cuda", (world_size,))

    with torch.inference_mode():
        model = get_model(device_mesh)

        torch.manual_seed(0)
        inp = torch.rand(20, 10, device=DEVICE)
        python_result = model(inp)

        if args.mode == "load":
            logger.info(f"Loading from {args.save_path}")
            loaded_program = torch_tensorrt.load(args.save_path)
            output = loaded_program.module()(inp)
            assert (python_result - output).std() < 0.01, "Result mismatch"
            logger.info("Load successful!")

        elif args.mode in ("jit_python", "jit_cpp"):
            trt_model = compile_torchtrt(model, args)

            # Warmup: trigger engine build on all ranks, then barrier so no
            # rank races ahead to the next NCCL collective before others finish.
            logger.info("Warming up (triggering TRT engine build)...")
            _ = trt_model(inp)
            dist.barrier()
            logger.info("All ranks compiled. Running inference...")

            with torch_tensorrt.distributed.distributed_context(
                dist.group.WORLD, trt_model
            ) as dist_model:
                output = dist_model(inp)

            assert (python_result - output).std() < 0.01, "Result mismatch"
            logger.info("JIT compile successful!")

        elif args.mode == "export":
            with torch.inference_mode():
                exported_program = torch.export.export(model, (inp,), strict=False)
            trt_model = torch_tensorrt.dynamo.compile(
                exported_program,
                inputs=[inp],
                use_explicit_typing=True,
                use_fp32_acc=True,
                device=DEVICE,
                disable_tf32=True,
                use_python_runtime=False,
                min_block_size=1,
                use_distributed_mode_trace=True,
                assume_dynamic_shape_support=True,
            )
            with torch.inference_mode():
                output = trt_model(inp)
            assert (python_result - output).std() < 0.01, "Result mismatch"
            save_path = torch_tensorrt.save(trt_model, args.save_path, inputs=[inp])
            logger.info(f"Saved to {save_path}")
            dist.barrier()

    dist.destroy_process_group()
    logger.info("Done!")
    # Bypass Python GC — TRT/CUDA destructors can segfault during
    # interpreter shutdown due to unpredictable destruction order.
    os._exit(0)
