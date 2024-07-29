import os
import sys
import time

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch_tensorrt

"""
This example copies some code from https://github.com/pytorch/examples/blob/main/distributed/tensor_parallelism/fsdp_tp_example.py
"""
from llama2_model import ModelArgs, Transformer
from torch.distributed._tensor import Replicate, Shard
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    PrepareModuleInput,
    RowwiseParallel,
    SequenceParallel,
    parallelize_module,
)

tp_size = 2

# understand world topology
_rank = int(os.environ["RANK"])
_world_size = int(os.environ["WORLD_SIZE"])


tp_mesh = init_device_mesh(device_type="cuda", mesh_shape=(_world_size,))

# create model and move it to GPU - init"cuda"_mesh has already mapped GPU ids.
simple_llama2_config = ModelArgs(dim=256, n_layers=2, n_heads=16, vocab_size=32000)

model = Transformer.from_model_args(simple_llama2_config).to("cuda")

# init model weights
model.init_weights()

# parallelize the first embedding and the last linear out projection
model = parallelize_module(
    model,
    tp_mesh,
    {
        "tok_embeddings": RowwiseParallel(
            input_layouts=Replicate(),
            output_layouts=Shard(1),
        ),
        "norm": SequenceParallel(),
        "output": ColwiseParallel(input_layouts=Shard(1), output_layouts=Replicate()),
    },
)

for layer_id, transformer_block in enumerate(model.layers):
    layer_tp_plan = {
        "attention_norm": SequenceParallel(),
        "attention": PrepareModuleInput(
            input_layouts=(Shard(1), None),
            desired_input_layouts=(Replicate(), None),
        ),
        "attention.wq": ColwiseParallel(),
        "attention.wk": ColwiseParallel(),
        "attention.wv": ColwiseParallel(),
        "attention.wo": RowwiseParallel(output_layouts=Shard(1)),
        "ffn_norm": SequenceParallel(),
        "feed_forward": PrepareModuleInput(
            input_layouts=(Shard(1),),
            desired_input_layouts=(Replicate(),),
        ),
        "feed_forward.w1": ColwiseParallel(),
        "feed_forward.w2": RowwiseParallel(output_layouts=Shard(1)),
        "feed_forward.w3": ColwiseParallel(),
    }

    # Adjust attention module to use the local number of heads
    attn_layer = transformer_block.attention
    attn_layer.n_heads = attn_layer.n_heads // tp_mesh.size()
    attn_layer.n_kv_heads = attn_layer.n_kv_heads // tp_mesh.size()

    # Custom parallelization plan for the model
    parallelize_module(
        module=transformer_block, device_mesh=tp_mesh, parallelize_plan=layer_tp_plan
    )

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
    },
    dynamic=False,
)

with torch.no_grad():
    for i in range(3):
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
