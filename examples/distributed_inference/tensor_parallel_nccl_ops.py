import ctypes
import logging
import os
import site
from enum import IntEnum, IntFlag, auto
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import tensorrt as trt
import tensorrt_llm
import torch
import torch.distributed as dist
import torch_tensorrt
from torch.distributed._tensor.device_mesh import init_device_mesh
from torch.fx import GraphModule, Node
from torch.fx.node import Argument, Target
from torch_tensorrt.dynamo.conversion._ConversionContext import ConversionContext
from torch_tensorrt.dynamo.conversion._ConverterRegistry import (
    dynamo_tensorrt_converter,
)
from torch_tensorrt.dynamo.lowering.passes.fuse_distributed_ops import (
    tensorrt_fused_nccl_all_gather_op,
    tensorrt_fused_nccl_reduce_scatter_op,
)
from torch_tensorrt.dynamo.types import TRTTensor
from torch_tensorrt.fx.converters.converter_utils import set_layer_name


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


def initialize_logger(rank, logger_file_name):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(logger_file_name + f"_{rank}.log", mode="w")
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)
    return logger


# This is required for env initialization since we use mpirun
def initialize_distributed_env(rank=0, world_size=1, port=29500):
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


def register_nccl_ops(logger_file_name):
    # Initialization
    initialize_distributed_env()
    # create a device mesh based on the given world_size.
    _world_size = int(os.environ["WORLD_SIZE"])

    device_mesh = init_device_mesh(device_type="cuda", mesh_shape=(_world_size,))
    _rank = device_mesh.get_rank()
    logger = initialize_logger(_rank, logger_file_name)
    device_id = (
        _rank % torch.cuda.device_count()
    )  # Ensure each rank gets a unique device
    torch.cuda.set_device(device_id)

    # TensorRT NCCL plugins
    # Iterate over all registered plugin creators
    plugin_registry = trt.get_plugin_registry()
    for plugin_creator in plugin_registry.plugin_creator_list:
        logger.info(
            f"Plugin Name: {plugin_creator.name}, Namespace: {plugin_creator.plugin_namespace}, Version: {plugin_creator.plugin_version}"
        )

    @dynamo_tensorrt_converter(tensorrt_fused_nccl_all_gather_op)
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
        _world_size = os.environ.get("WORLD_SIZE")
        if _world_size is not None:
            _world_size = int(_world_size)
        else:
            raise RuntimeError(
                f"The WORLD_SIZE env variable is not set in distributed environment"
            )
        group = list(range(_world_size))
        group = trt.PluginField(
            "group", np.array(group, dtype=np.int32), trt.PluginFieldType.INT32
        )
        p_dtype = trt.float32
        pf_type = trt.PluginField(
            "type_id", np.array([int(p_dtype)], np.int32), trt.PluginFieldType.INT32
        )
        pfc = trt.PluginFieldCollection([group, pf_type])
        allgather = allgather_plg_creator.create_plugin("allgather", pfc)
        layer = ctx.net.add_plugin_v2(plug_inputs, allgather)
        set_layer_name(layer, target, name)
        return layer.get_output(0)

    @dynamo_tensorrt_converter(tensorrt_fused_nccl_reduce_scatter_op)
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
        _world_size = os.environ.get("WORLD_SIZE")
        if _world_size is not None:
            _world_size = int(_world_size)
        else:
            raise RuntimeError(
                f"The WORLD_SIZE env variable is not set in distributed environment"
            )
        group = list(range(_world_size))
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

    return device_mesh, _world_size, _rank, logger
