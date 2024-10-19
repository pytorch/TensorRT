import ctypes
import logging
import os
import site
import sys
import time
from enum import IntEnum, IntFlag, auto
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import tensorrt as trt
import tensorrt_llm
import torch
import torch.distributed as dist
import torch.nn as nn
import torch_tensorrt
from torch.distributed._tensor import Shard
from torch.distributed._tensor.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    parallelize_module,
)
from torch.fx import GraphModule, Node
from torch.fx.node import Argument, Target
from torch_tensorrt.dynamo.conversion._ConversionContext import ConversionContext
from torch_tensorrt.dynamo.conversion._ConverterRegistry import (
    dynamo_tensorrt_converter,
)
from torch_tensorrt.dynamo.lowering.passes.fuse_distributed_ops import (
    custom_fused_all_gather_op,
    custom_fused_reduce_scatter_op,
)
from torch_tensorrt.dynamo.types import TRTTensor
from torch_tensorrt.fx.converters.converter_utils import set_layer_name


# This is required for env initialization since we use mpirun
def initialize(rank=0, world_size=1, port=29500):
    local_rank = int(
        os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", rank % torch.cuda.device_count())
    )
    world_size = int(os.environ.get("OMPI_COMM_WORLD_SIZE", world_size))

    # Set up environment variable to run with mpirun
    os.environ["RANK"] = str(local_rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)

    # Necessary to assign a device to each rank.
    torch.cuda.set_device(local_rank)

    # We use nccl backend
    dist.init_process_group("nccl")

    # set a manual seed for reproducibility
    torch.manual_seed(1111)

    return local_rank, world_size


initialize()
# create a device mesh based on the given world_size.
_world_size = int(os.environ["WORLD_SIZE"])

device_mesh = init_device_mesh(device_type="cuda", mesh_shape=(_world_size,))
_rank = device_mesh.get_rank()
device_id = _rank % torch.cuda.device_count()  # Ensure each rank gets a unique device
torch.cuda.set_device(device_id)


logger = logging.getLogger()
logger.setLevel(logging.INFO)
fh = logging.FileHandler(f"./tensor_parallel_simple_example_{_rank}.log", mode="w")
fh.setLevel(logging.INFO)
logger.addHandler(fh)


# TensorRT NCCL plugins
tensorrt_llm_lib_path = tensorrt_llm.__file__
plugin_lib_path = tensorrt_llm_lib_path + "/libs/libnvinfer_plugin_tensorrt_llm.so"
try:
    ctypes.CDLL(plugin_lib_path)
    logger.info(f"plugin loaded successfully")
except OSError as e:
    logger.info(f"unsuccessful load : {e}")
trt.init_libnvinfer_plugins(None, "")
# Iterate over all registered plugin creators
plugin_registry = trt.get_plugin_registry()
for plugin_creator in plugin_registry.plugin_creator_list:
    logger.info(
        f"Plugin Name: {plugin_creator.name}, Namespace: {plugin_creator.plugin_namespace}, Version: {plugin_creator.plugin_version}"
    )


# class for AllReduce
class AllReduceStrategy(IntEnum):
    """Warning: actual definition is in kernels/customAllReduceKernels.h.

    They must be kept in sync.
    """

    NCCL = 0
    ONESHOT = 1
    TWOSHOT = 2
    AUTO = 3


class AllReduceConfig(IntFlag):
    """Warning: actual definition is in kernels/customAllReduceKernels.h.

    They must be kept in sync
    """

    USE_MEMCPY = auto()
    PUSH_MODE = auto()


@dynamo_tensorrt_converter(custom_fused_all_gather_op)
def insert_nccl_gather_op(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    plug_inputs = [args[0]]
    allgather_plg_creator = trt.get_plugin_registry().get_plugin_creator(
        "AllGather", "1", "tensorrt_llm"
    )
    assert allgather_plg_creator is not None
    world_size = dist.get_world_size()
    group = list(range(world_size))
    group = trt.PluginField(
        "group", np.array(group, dtype=np.int32), trt.PluginFieldType.INT32
    )
    p_dtype = trt.float16
    pf_type = trt.PluginField(
        "type_id", np.array([int(p_dtype)], np.int32), trt.PluginFieldType.INT32
    )
    pfc = trt.PluginFieldCollection([group, pf_type])
    allgather = allgather_plg_creator.create_plugin("allgather", pfc)
    layer = ctx.net.add_plugin_v2(plug_inputs, allgather)
    set_layer_name(layer, target, name)
    return layer.get_output(0)


@dynamo_tensorrt_converter(custom_fused_reduce_scatter_op)
def insert_nccl_reduce_scatter_plugin(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    plug_inputs = [args[0]]
    allreduce_plg_creator = trt.get_plugin_registry().get_plugin_creator(
        "ReduceScatter", "1", "tensorrt_llm"
    )

    assert allreduce_plg_creator is not None

    counter = 0
    strategy = AllReduceStrategy.NCCL
    config = AllReduceConfig(0)

    world_size = dist.get_world_size()
    group = list(range(world_size))
    group = trt.PluginField(
        "group", np.array(group, dtype=np.int32), trt.PluginFieldType.INT32
    )

    p_dtype = trt.float16
    pf_dtype = trt.PluginField(
        "type_id", np.array([int(p_dtype)], np.int32), trt.PluginFieldType.INT32
    )
    pfc = [group, pf_dtype]
    p_strategy = trt.PluginField(
        "strategy", np.array([int(strategy)], np.int8), trt.PluginFieldType.INT8
    )
    pfc.append(p_strategy)
    p_config = trt.PluginField(
        "config", np.array([int(config)], np.int8), trt.PluginFieldType.INT8
    )
    pfc.append(p_config)
    p_counter = trt.PluginField(
        "counter", np.array([counter], np.int32), trt.PluginFieldType.INT32
    )
    pfc.append(p_counter)

    pfc = trt.PluginFieldCollection(pfc)
    ar_plug = allreduce_plg_creator.create_plugin("allreduce", pfc)

    layer = ctx.net.add_plugin_v2(plug_inputs, ar_plug)
    set_layer_name(layer, target, name)
    return layer.get_output(0)


"""
This example copies some code from https://github.com/pytorch/examples/blob/main/distributed/tensor_parallelism/tensor_parallel_example.py
"""


class ToyModel(nn.Module):
    """MLP based model"""

    def __init__(self):
        super(ToyModel, self).__init__()
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


print(f"Starting PyTorch TP example on rank {_rank}.")
assert (
    _world_size % 2 == 0
), f"TP examples require even number of GPUs, but got {_world_size} gpus"


# # create model and move it to GPU - init"cuda"_mesh has already mapped GPU ids.
tp_model = ToyModel().to("cuda")


# Custom parallelization plan for the model
tp_model = parallelize_module(
    module=tp_model,
    device_mesh=device_mesh,
    parallelize_plan={
        "in_proj": ColwiseParallel(input_layouts=Shard(0)),
        "out_proj": RowwiseParallel(output_layouts=Shard(0)),
        "in_proj2": ColwiseParallel(input_layouts=Shard(0)),
        "out_proj2": RowwiseParallel(output_layouts=Shard(0)),
    },
)
torch.manual_seed(0)
inp = torch.rand(20, 10, device="cuda")
python_result = tp_model(inp)


backend = "torch_tensorrt"
tp_model = torch.compile(
    tp_model,
    backend=backend,
    options={
        "truncate_long_and_double": True,
        "enabled_precisions": {torch.float32, torch.float16},
        "use_python_runtime": True,
        "min_block_size": 1,
    },
    dynamic=False,
)

for i in range(10):
    # For TP, input needs to be same across all TP ranks.
    # Setting the random seed is to mimic the behavior of dataloader.
    torch.manual_seed(i)
    inp = torch.rand(20, 10, device="cuda")
    start = time.time()
    output = tp_model(inp)
    end = time.time()
    if i == 0:
        print(f"Compilation time is {end-start}")
        assert (
            python_result - output
        ).std() < 0.01, "Compilation result is not correct."
    elif _rank == 0:
        print(f"Inference time is {end-start}")
