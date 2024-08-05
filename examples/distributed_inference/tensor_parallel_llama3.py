import torch
import torch_tensorrt
from llama3_model import Transformer, ModelArgs
from torch.distributed._composable.fsdp import MixedPrecisionPolicy
from torch.distributed._composable.fsdp.fully_shard import fully_shard
from torch.distributed._tensor import Replicate, Shard
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    PrepareModuleInput,
    RowwiseParallel,
    SequenceParallel,
    parallelize_module,
)
import time
from torch.distributed.device_mesh import init_device_mesh
import os

# Taken and modified pytorch lightening
# https://lightning.ai/lightning-ai/studios/tensor-parallelism-supercharging-large-model-training-with-pytorch-lightning
def parallelize(model: Transformer, tp_mesh: DeviceMesh) -> Transformer:
    """Apply parallelisms and activation checkpointing to the model.

    NOTE: The passed-in model preferably should be on meta device. Otherwise,
    the model must fit on GPU or CPU memory.

    """

    if tp_mesh.size() > 1:
        # 1. Parallelize the first embedding and the last linear proj layer
        # 2. Parallelize the root norm layer over the sequence dim
        # 3. Shard the first transformer block's inputs

        # Parallelize the first embedding and the last linear out projection
        plan = {
            "tok_embeddings": RowwiseParallel(input_layouts=Replicate(),
                                              output_layouts=Shard(1),),
            "output": ColwiseParallel(
                input_layouts=Shard(1),
            ),
            "norm": SequenceParallel(),
        }
        model = parallelize_module(model, tp_mesh, plan)

        # Parallelize each transformer block
        for transformer_block in model.layers.values():
            plan = {
                "attention": PrepareModuleInput(
                    input_layouts=(Shard(1), None),
                    desired_input_layouts=(Replicate(), None),
                ),
                "attention.wq": ColwiseParallel(),
                "attention.wk": ColwiseParallel(),
                "attention.wv": ColwiseParallel(),
                "attention.wo": RowwiseParallel(output_layouts=Shard(1)),
                "attention_norm": SequenceParallel(),
                "feed_forward": PrepareModuleInput(
                    input_layouts=(Shard(1),),
                    desired_input_layouts=(Replicate(),),
                ),
                "feed_forward.w1": ColwiseParallel(),
                "feed_forward.w2": RowwiseParallel(output_layouts=Shard(1)),
                "feed_forward.w3": ColwiseParallel(),
                "ffn_norm": SequenceParallel(),
            }

            # Adjust attention module to use the local number of heads
            attn_layer = transformer_block.attention
            attn_layer.n_heads = attn_layer.n_heads // tp_mesh.size()
            attn_layer.n_kv_heads = attn_layer.n_kv_heads // tp_mesh.size()

            # Apply the plan for the current transformer block
            parallelize_module(transformer_block, tp_mesh, plan)

    return model


tp_size = 4

# understand world topology
_rank = int(os.environ["RANK"])
_world_size = int(os.environ["WORLD_SIZE"])


tp_mesh = init_device_mesh(device_type="cuda", mesh_shape=(_world_size,))

model_args = ModelArgs(vocab_size=128256, dim=8192, n_layers=80, n_heads=64, rope_theta=500000.0, n_kv_heads=8)

# model_args = ModelArgs(vocab_size=32000, dim=2048, n_layers=8, n_heads=32)
model = Transformer(model_args).to("cuda")
model = parallelize(model, tp_mesh)
model.eval()
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
