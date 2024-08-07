import os
import time

import torch
import torch_tensorrt
from llama3_model import ModelArgs, ParallelTransformer
from torch.distributed._composable.fsdp import MixedPrecisionPolicy
from torch.distributed._composable.fsdp.fully_shard import fully_shard
from torch.distributed._tensor import Replicate, Shard
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
)
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    PrepareModuleInput,
    RowwiseParallel,
    SequenceParallel,
    parallelize_module,
)

# Taken and modified pytorch lightening
# https://lightning.ai/lightning-ai/studios/tensor-parallelism-supercharging-large-model-training-with-pytorch-lightning


tp_size = 8

# understand world topology
_rank = int(os.environ["RANK"])
_world_size = int(os.environ["WORLD_SIZE"])


tp_mesh = init_device_mesh(device_type="cuda", mesh_shape=(_world_size,))

model_args = ModelArgs(
    vocab_size=32000,
    dim=8192,
    n_layers=80,
    n_heads=64,
    rope_theta=500000.0,
    n_kv_heads=8,
    device="cuda",
)

with torch.no_grad():
    model = ParallelTransformer(model_args, tp_mesh)
    torch.manual_seed(0)
    inp = torch.randint(32000, (8, 256), device="cuda")
    python_result = model(inp)
    torch_tensorrt.runtime.set_multi_device_safe_mode(True)
    model = torch.compile(
        model,
        fullgraph=True,
        backend="torch_tensorrt",
        options={
            "truncate_long_and_double": True,
            "enabled_precisions": {torch.float32, torch.float16},
            "use_python_runtime": True,
            "workspace_size": 1 << 33,
        },
        dynamic=False,
    )
    for i in range(15):
        # seeding with dp_rank to ensure identical inputs for TP groups
        torch.manual_seed(i)
        start = time.time()
        output = model(inp)
        end = time.time()
        if i == 0:
            print(f"Compilation time is {end-start}")
            assert (
                python_result - output
            ).std() < 0.01, "Compilation result is not correct."
        elif _rank == 0:
            print(f"Inference time is {end-start}")
