import logging
import os
from enum import IntEnum, IntFlag, auto
from typing import Optional, Tuple, Union

import numpy as np
import tensorrt as trt
from torch.fx.node import Argument, Target
from torch_tensorrt._features import needs_native_collectives
from torch_tensorrt.dynamo._SourceIR import SourceIR
from torch_tensorrt.dynamo.conversion._ConversionContext import ConversionContext
from torch_tensorrt.dynamo.conversion.converter_utils import set_layer_name

logger = logging.getLogger(__name__)


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


def _get_distributed_rank_and_world_size() -> Tuple[int, int]:
    """Get rank and world_size for TRT collective layer construction.

    Prefers torch.distributed when initialized (reliable, unaffected by env var
    contamination from single-rank test setup).  Falls back to RANK/WORLD_SIZE
    env vars when dist is not initialized (e.g. AOT-export without a live PG).

    Returns:
        (rank, world_size) tuple.

    Raises:
        RuntimeError: If neither dist nor env vars provide world_size.
    """
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        dist_rank = dist.get_rank()
        dist_world_size = dist.get_world_size()

        env_world_size_str = os.environ.get("WORLD_SIZE")
        env_rank_str = os.environ.get("RANK")
        if env_world_size_str is not None and env_rank_str is not None:
            env_world_size = int(env_world_size_str)
            env_rank = int(env_rank_str)
            if env_world_size != dist_world_size or env_rank != dist_rank:
                raise RuntimeError(
                    f"RANK/WORLD_SIZE env vars ({env_rank}/{env_world_size}) conflict with "
                    f"torch.distributed ({dist_rank}/{dist_world_size}). "
                    f"Unset RANK and WORLD_SIZE or ensure they match the active process group."
                )

        return dist_rank, dist_world_size
    else:
        _world_size = os.environ.get("WORLD_SIZE")
        if _world_size is None:
            raise RuntimeError(
                "The WORLD_SIZE env variable is not set in distributed environment"
            )
        world_size = int(_world_size)
        rank = int(os.environ.get("RANK", 0))
        return rank, world_size



def _collective_group_ranks(group_name, world_size):
    """Global ranks of the collective's process group.

    The native ``add_dist_collective`` layer needs the set of ranks that participate in
    *this* collective. Resolving it from the op's ``group_name`` lets a collective target a
    process **subgroup** (e.g. context/sequence-parallel over one subgroup while tensor-parallel
    uses another -- a 2-D device mesh) instead of always the whole world. Falls back to the world
    group when the group cannot be resolved (single-program / group not created in this process).
    """
    import numpy as np
    if group_name:
        try:
            import torch.distributed as dist
            from torch.distributed.distributed_c10d import _resolve_process_group

            ranks = dist.get_process_group_ranks(_resolve_process_group(group_name))
            return np.array(sorted(ranks), dtype=np.int64)
        except Exception as e:  # noqa: BLE001
            logger.warning(
                f"Could not resolve process group '{group_name}' ({e}); using world group"
            )
    return np.arange(world_size, dtype=np.int64)


def _mark_collective_boundary(ctx, input_tensor, output_tensor):
    """Optionally mark the collective's tensors for debug so they are not fused away."""
    mode = os.environ.get("TRT_COLL_BOUNDARY", "").lower()
    if mode in ("input", "both"):
        ctx.net.mark_debug(input_tensor)
    if mode in ("output", "both"):
        ctx.net.mark_debug(output_tensor)


# Myelin fusion-boundary for native DistCollective. Myelin folds identity / same-dtype
# casts away, so only a genuine dtype *reformat* keeps a native collective out of a
# ForeignNode (see pytorch/TensorRT#4381). fp32 by default; set to None to disable, or
# trt.float16 to keep the communication volume down.
_COLL_BOUNDARY_DTYPE = trt.float32

_coll_boundary_orig_dtype: dict = {}


def _coll_boundary(ctx, t, name, where):
    """Insert a dtype reformat around a DistCollective so Myelin will not fuse it.

    ``where='in'``  casts the collective input to the boundary dtype (the collective then
    runs at that precision); ``where='out'`` casts the output back to the original dtype.
    A no-op when the boundary dtype is disabled or already matches.
    """
    if _COLL_BOUNDARY_DTYPE is None:
        return t
    if where == "in":
        _coll_boundary_orig_dtype[name] = t.dtype
        target = _COLL_BOUNDARY_DTYPE
    else:
        target = _coll_boundary_orig_dtype.get(name, t.dtype)
    if t.dtype == target:
        return t
    c = ctx.net.add_cast(t, target)
    c.name = f"{name}_myelin_bnd_{where}"
    o = c.get_output(0)
    try:
        o.dtype = target
    except Exception:  # noqa: BLE001
        pass
    return o


def nccl_all_gather(
    ctx: ConversionContext,
    target: Union[Target, str],
    source_ir: Optional[SourceIR],
    name: str,
    plug_inputs: Tuple[Argument, ...],
) -> trt.ITensor:
    rank, world_size = _get_distributed_rank_and_world_size()
    if world_size == 1:
        return plug_inputs[0]

    allgather_plg_creator = trt.get_plugin_registry().get_plugin_creator(
        "AllGather", "1", "tensorrt_llm"
    )
    assert allgather_plg_creator is not None
    logger.debug(
        f"Adding TRT-LLM NCCL gather: name={name}, rank={rank}, world_size={world_size}"
    )

    group = list(range(world_size))
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
    set_layer_name(layer, target, name, source_ir)
    return layer.get_output(0)


def nccl_all_reduce(
    ctx: ConversionContext,
    target: Union[Target, str],
    source_ir: Optional[SourceIR],
    name: str,
    plug_inputs: Tuple[Argument, ...],
) -> trt.ITensor:
    rank, world_size = _get_distributed_rank_and_world_size()
    if world_size == 1:
        return plug_inputs[0]

    allreduce_plg_creator = trt.get_plugin_registry().get_plugin_creator(
        "AllReduce", "1", "tensorrt_llm"
    )
    assert allreduce_plg_creator is not None

    counter = 0
    strategy = AllReduceStrategy.NCCL
    config = AllReduceConfig(0)
    logger.debug(
        f"Adding TRT-LLM NCCL all reduce: name={name}, rank={rank}, world_size={world_size}"
    )
    group = list(range(world_size))
    group = trt.PluginField(
        "group", np.array(group, dtype=np.int32), trt.PluginFieldType.INT32
    )

    p_dtype = trt.float32
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
    set_layer_name(layer, target, name, source_ir)
    return layer.get_output(0)


def nccl_reduce_scatter(
    ctx: ConversionContext,
    target: Union[Target, str],
    source_ir: Optional[SourceIR],
    name: str,
    plug_inputs: Tuple[Argument, ...],
) -> trt.ITensor:
    rank, world_size = _get_distributed_rank_and_world_size()
    if world_size == 1:
        return plug_inputs[0]

    allreduce_plg_creator = trt.get_plugin_registry().get_plugin_creator(
        "ReduceScatter", "1", "tensorrt_llm"
    )

    assert allreduce_plg_creator is not None

    counter = 0
    strategy = AllReduceStrategy.NCCL
    config = AllReduceConfig(0)
    logger.debug(
        f"Adding TRT-LLM NCCL reduce scatter: name={name}, rank={rank}, world_size={world_size}"
    )
    group = list(range(world_size))
    group = trt.PluginField(
        "group", np.array(group, dtype=np.int32), trt.PluginFieldType.INT32
    )

    p_dtype = trt.float32
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
    set_layer_name(layer, target, name, source_ir)
    return layer.get_output(0)


@needs_native_collectives
def nccl_all_gather_native(
    ctx: ConversionContext,
    target: Union[Target, str],
    source_ir: Optional[SourceIR],
    name: str,
    plug_inputs: Tuple[Argument, ...],
    group_name: Optional[str] = None,
) -> trt.ITensor:
    """
    Implement all_gather using native TensorRT DistCollective API.

    This operation gathers tensors from all ranks and concatenates them.
    Each rank contributes a tensor, and all ranks receive the concatenated result.

    Returns:
        Output tensor after all_gather operation

    Example:
        Input on rank 0: [1, 2]  shape=(2,)
        Input on rank 1: [3, 4]  shape=(2,)
        Output on all ranks: [1, 2, 3, 4]  shape=(4,)
    """
    rank, world_size = _get_distributed_rank_and_world_size()

    # TRT add_dist_collective crashes with world_size=1; all_gather of a single rank
    # is an identity op.
    if world_size == 1:
        return plug_inputs[0]
    logger.debug(
        f"Adding native all_gather: name={name}, rank={rank}, world_size={world_size}"
    )

    # Get the input tensor
    input_tensor = _coll_boundary(ctx, plug_inputs[0], name, "in")

    try:
        # Use native TensorRT DistCollective API for ALL_GATHER
        # For ALL_GATHER, the reduce operation and root rank parameters are ignored
        # The last parameter (group) can be None to include all ranks
        import numpy as np

        # Create array of all participating rank IDs [0, 1, 2, ..., world_size-1]
        groups = _collective_group_ranks(group_name, world_size)

        if os.environ.get("AG_VIA_ALLREDUCE") == "1":
            # Express all_gather as place-into-slot + ALL_REDUCE(SUM). The ALL_REDUCE output
            # shape == input shape (built explicitly via concat below), so Myelin needs no
            # build-time world_size for shape inference (unlike a native ALL_GATHER, which
            # requires myelinGraphSetWorldSize). Numerically identical to all_gather; cost is
            # len(groups)x the communication volume. Opt-in via env AG_VIA_ALLREDUCE=1.
            raw = plug_inputs[0]
            ng = int(len(groups))
            glist = [int(x) for x in groups.tolist()]
            slot = glist.index(rank) if rank in glist else 0
            ish = [int(x) for x in raw.shape]
            d0 = ish[0]

            def _z(nrows):
                sh = tuple([nrows] + ish[1:])
                z = ctx.net.add_constant(
                    sh, trt.Weights(np.zeros(int(np.prod(sh)), dtype=np.float32))
                ).get_output(0)
                if z.dtype != raw.dtype:
                    z = ctx.net.add_cast(z, raw.dtype).get_output(0)
                return z

            parts = []
            if slot > 0:
                parts.append(_z(slot * d0))
            parts.append(raw)
            if (ng - 1 - slot) > 0:
                parts.append(_z((ng - 1 - slot) * d0))
            cat = ctx.net.add_concatenation(parts)
            cat.axis = 0
            pfo = ctx.net.add_cast(cat.get_output(0), trt.float32).get_output(0)
            ar = ctx.net.add_dist_collective(
                pfo,
                trt.CollectiveOperation.ALL_REDUCE,
                trt.ReduceOperation.SUM,
                -1,
                groups,
            )
            ar.num_ranks = ng
            set_layer_name(ar, target, name, source_ir)
            return ctx.net.add_cast(ar.get_output(0), raw.dtype).get_output(0)

        logger.debug(
            f"Creating ALL_GATHER layer: groups={groups.tolist()}, groupSize={world_size}"
        )
        layer = ctx.net.add_dist_collective(
            input_tensor,
            trt.CollectiveOperation.ALL_GATHER,
            trt.ReduceOperation.NONE,  # Ignored for ALL_GATHER
            -1,  # Root rank - ignored for ALL_GATHER
            groups,  # None means all ranks participate (world_size ranks)
        )

        logger.debug(f"Successfully created native ALL_GATHER layer: {name}")
        logger.debug(
            f"Calling add_dist_collective: input_shape={input_tensor.shape}, "
            f"groups={groups.tolist()}, groupSize={len(groups)} (inferred from array)"
        )

        set_layer_name(layer, target, name, source_ir)

        # num_ranks must be set BEFORE get_output(0) so Myelin can infer the collective's
        # output shape (e.g. all_gather dim0 = in_dim0 * num_ranks) for shape inference.
        layer.num_ranks = len(groups)
        output = _coll_boundary(ctx, layer.get_output(0), name, "out")
        _mark_collective_boundary(ctx, input_tensor, output)

        return output

    except Exception as e:
        logger.error(f"Native ALL_GATHER failed: {e} (type: {type(e).__name__})")
        raise


@needs_native_collectives
def nccl_reduce_scatter_native(
    ctx: ConversionContext,
    target: Union[Target, str],
    source_ir: Optional[SourceIR],
    name: str,
    plug_inputs: Tuple[Argument, ...],
    reduce_op: str = "sum",
    group_name: Optional[str] = None,
) -> trt.ITensor:
    """
    Implement reduce_scatter using native TensorRT DistCollective API.

    This operation reduces tensors from all ranks and scatters the result.
    The input is split along dimension 0 and each rank receives one chunk
    after the reduction operation.

    Returns:
        Output tensor after reduce_scatter operation
        reduce_op: Reduction operation ("sum", "min", "max", "avg", "product")

    Example (with SUM reduction):
        Input on rank 0: [1, 2, 3, 4]  shape=(4,)
        Input on rank 1: [5, 6, 7, 8]  shape=(4,)
        Output on rank 0: [1+5, 2+6] = [6, 8]  shape=(2,)
        Output on rank 1: [3+7, 4+8] = [10, 12]  shape=(2,)
    """
    rank, world_size = _get_distributed_rank_and_world_size()
    logger.debug(
        f"Adding native reduce_scatter: name={name}, rank={rank}, world_size={world_size}, reduce_op={reduce_op}"
    )

    # TRT add_dist_collective crashes with world_size=1; reduce_scatter of a single rank
    # is an identity op.
    if world_size == 1:
        return plug_inputs[0]

    # Get the input tensor
    input_tensor = _coll_boundary(ctx, plug_inputs[0], name, "in")

    reduce_op_map = {
        "sum": trt.ReduceOperation.SUM,
        "min": trt.ReduceOperation.MIN,
        "max": trt.ReduceOperation.MAX,
        "avg": trt.ReduceOperation.AVG,
        "product": trt.ReduceOperation.PROD,
    }

    if reduce_op.lower() not in reduce_op_map:
        raise ValueError(
            f"Unsupported reduce operation: {reduce_op}. "
            f"Supported: {list(reduce_op_map.keys())}"
        )

    trt_reduce_op = reduce_op_map[reduce_op.lower()]

    try:
        groups = _collective_group_ranks(group_name, world_size)

        layer = ctx.net.add_dist_collective(
            input_tensor,
            trt.CollectiveOperation.REDUCE_SCATTER,
            trt_reduce_op,
            -1,
            groups,
        )

        set_layer_name(layer, target, name, source_ir)

        # num_ranks must be set BEFORE get_output(0) so Myelin can infer the collective's
        # output shape (e.g. all_gather dim0 = in_dim0 * num_ranks) for shape inference.
        layer.num_ranks = len(groups)
        output = _coll_boundary(ctx, layer.get_output(0), name, "out")
        _mark_collective_boundary(ctx, input_tensor, output)
        logger.debug(
            f"Successfully created native REDUCE_SCATTER layer: {name}, reduce_op={reduce_op}, groups={groups.tolist()}"
        )

        return output

    except Exception as e:
        logger.error(f"Native REDUCE_SCATTER failed: {e} (type: {type(e).__name__})")
        raise


@needs_native_collectives
def nccl_all_reduce_native(
    ctx: ConversionContext,
    target: Union[Target, str],
    source_ir: Optional[SourceIR],
    name: str,
    plug_inputs: Tuple[Argument, ...],
    reduce_op: str = "sum",
    group_name: Optional[str] = None,
) -> trt.ITensor:
    """
    Implement all_reduce using native TensorRT DistCollective API.

    This operation reduces tensors across all ranks in-place. Every rank
    receives the same reduced result.

    Returns:
        Output tensor after all_reduce operation

    Args:
        reduce_op: Reduction operation ("sum", "min", "max", "avg", "product")

    Example (with SUM reduction):
        Input on rank 0: [1, 2, 3, 4]  shape=(4,)
        Input on rank 1: [5, 6, 7, 8]  shape=(4,)
        Output on rank 0: [6, 8, 10, 12]  shape=(4,)
        Output on rank 1: [6, 8, 10, 12]  shape=(4,)
    """
    rank, world_size = _get_distributed_rank_and_world_size()

    # TRT add_dist_collective crashes with world_size=1; all_reduce of a single rank
    # is an identity op.
    if world_size == 1:
        return plug_inputs[0]

    logger.debug(
        f"Adding native all_reduce: name={name}, rank={rank}, world_size={world_size}, reduce_op={reduce_op}"
    )

    input_tensor = _coll_boundary(ctx, plug_inputs[0], name, "in")

    reduce_op_map = {
        "sum": trt.ReduceOperation.SUM,
        "min": trt.ReduceOperation.MIN,
        "max": trt.ReduceOperation.MAX,
        "avg": trt.ReduceOperation.AVG,
        "product": trt.ReduceOperation.PROD,
    }

    if reduce_op.lower() not in reduce_op_map:
        raise ValueError(
            f"Unsupported reduce operation: {reduce_op}. "
            f"Supported: {list(reduce_op_map.keys())}"
        )

    trt_reduce_op = reduce_op_map[reduce_op.lower()]

    try:
        # Create array of all participating rank IDs [0, 1, ..., world_size-1]
        # Passing None for groups can be treated as a no-op by TRT; use an explicit
        # rank array (same as ALL_GATHER) to ensure the reduction is performed.
        groups = _collective_group_ranks(group_name, world_size)

        layer = ctx.net.add_dist_collective(
            input_tensor,
            trt.CollectiveOperation.ALL_REDUCE,
            trt_reduce_op,
            -1,
            groups,
        )

        set_layer_name(layer, target, name, source_ir)

        # num_ranks must be set BEFORE get_output(0) so Myelin can infer the collective's
        # output shape (e.g. all_gather dim0 = in_dim0 * num_ranks) for shape inference.
        layer.num_ranks = len(groups)
        output = _coll_boundary(ctx, layer.get_output(0), name, "out")
        _mark_collective_boundary(ctx, input_tensor, output)
        logger.debug(
            f"Successfully created native ALL_REDUCE layer: {name}, reduce_op={reduce_op}, groups={groups.tolist()}"
        )

        return output

    except Exception as e:
        logger.error(f"Native ALL_REDUCE failed: {e} (type: {type(e).__name__})")
        raise


@needs_native_collectives
def nccl_all_to_all_native(
    ctx: ConversionContext,
    target: Union[Target, str],
    source_ir: Optional[SourceIR],
    name: str,
    plug_inputs: Tuple[Argument, ...],
    group_name: Optional[str] = None,
) -> trt.ITensor:
    """
    Implement all_to_all using native TensorRT DistCollective API.

    This operation sends a chunk of data from each rank. The i-th rank will receive the data
    at the i-th position of every other rank.

    Returns:
        Output tensor after all_to_all operation

    Example:
        Input on rank 0: [1, 2]  shape=(2,)
        Input on rank 1: [3, 4]  shape=(2,)
        Output on rank 0: [1, 3] shape=(2,)
        Output on rank 1: [2, 4] shape=(2,)
    """
    rank, world_size = _get_distributed_rank_and_world_size()

    # TRT add_dist_collective crashes with world_size=1; all_to_all of a single rank
    # is an identity op.
    if world_size == 1:
        return plug_inputs[0]
    logger.debug(
        f"Adding native all_to_all: name={name}, rank={rank}, world_size={world_size}"
    )

    # Get the input tensor
    input_tensor = _coll_boundary(ctx, plug_inputs[0], name, "in")

    try:
        # Use native TensorRT DistCollective API for ALL_TO_ALL
        # For ALL_TO_ALL, the reduce operation and root rank parameters are ignored
        # The last parameter (group) can be None to include all ranks
        import numpy as np

        # Create array of all participating rank IDs [0, 1, 2, ..., world_size-1]
        groups = _collective_group_ranks(group_name, world_size)

        logger.debug(
            f"Creating ALL_TO_ALL layer: groups={groups.tolist()}, groupSize={world_size}"
        )
        layer = ctx.net.add_dist_collective(
            input_tensor,
            trt.CollectiveOperation.ALL_TO_ALL,
            trt.ReduceOperation.NONE,  # Ignored for ALL_TO_ALL
            -1,  # Root rank - ignored for ALL_TO_ALL
            groups,  # None means all ranks participate (world_size ranks)
        )

        logger.debug(f"Successfully created native ALL_TO_ALL layer: {name}")
        logger.debug(
            f"Calling add_dist_collective: input_shape={input_tensor.shape}, "
            f"groups={groups.tolist()}, groupSize={len(groups)} (inferred from array)"
        )

        set_layer_name(layer, target, name, source_ir)

        # num_ranks must be set BEFORE get_output(0) so Myelin can infer the collective's
        # output shape (e.g. all_gather dim0 = in_dim0 * num_ranks) for shape inference.
        layer.num_ranks = len(groups)
        output = _coll_boundary(ctx, layer.get_output(0), name, "out")
        _mark_collective_boundary(ctx, input_tensor, output)

        return output

    except Exception as e:
        logger.error(f"Native ALL_TO_ALL failed: {e} (type: {type(e).__name__})")
        raise


@needs_native_collectives
def nccl_scatter_native(
    ctx: ConversionContext,
    target: Union[Target, str],
    source_ir: Optional[SourceIR],
    name: str,
    plug_inputs: Tuple[Argument, ...],
    root: int = 0,
    group_name: Optional[str] = None,
) -> trt.ITensor:
    """
    Implement scatter using native TensorRT DistCollective API.

    This operation has the root rank send a chunk of its data to every other rank.

    Returns:
        Output tensor after scatter operation

    Example:
        Input on rank 0: [1, 2]  shape=(2,)
        Input on rank 1: None  shape=(2,)
        Output on rank 0: [1] shape=(1,)
        Output on rank 1: [2] shape=(1,)
    """
    rank, world_size = _get_distributed_rank_and_world_size()

    # TRT add_dist_collective crashes with world_size=1; scatter of a single rank
    # is an identity op.
    if world_size == 1:
        return plug_inputs[0]
    logger.debug(
        f"Adding native scatter: name={name}, rank={rank}, world_size={world_size}"
    )

    # Get the input tensor
    input_tensor = _coll_boundary(ctx, plug_inputs[0], name, "in")

    try:
        # Use native TensorRT DistCollective API for SCATTER
        # For SCATTER, the reduce operation parameter is ignored
        # The last parameter (group) can be None to include all ranks
        import numpy as np

        # Create array of all participating rank IDs [0, 1, 2, ..., world_size-1]
        groups = _collective_group_ranks(group_name, world_size)

        logger.debug(
            f"Creating scatter layer: groups={groups.tolist()}, groupSize={world_size}"
        )
        layer = ctx.net.add_dist_collective(
            input_tensor,
            trt.CollectiveOperation.SCATTER,
            trt.ReduceOperation.NONE,  # Ignored for SCATTER
            root,
            groups,  # None means all ranks participate (world_size ranks)
        )

        logger.debug(f"Successfully created native SCATTER layer: {name}")
        logger.debug(
            f"Calling add_dist_collective: input_shape={input_tensor.shape}, "
            f"root={root}, groups={groups.tolist()}, groupSize={len(groups)} (inferred from array)"
        )

        set_layer_name(layer, target, name, source_ir)

        # num_ranks must be set BEFORE get_output(0) so Myelin can infer the collective's
        # output shape (e.g. all_gather dim0 = in_dim0 * num_ranks) for shape inference.
        layer.num_ranks = len(groups)
        output = _coll_boundary(ctx, layer.get_output(0), name, "out")
        _mark_collective_boundary(ctx, input_tensor, output)

        return output

    except Exception as e:
        logger.error(f"Native SCATTER failed: {e} (type: {type(e).__name__})")
        raise


@needs_native_collectives
def nccl_gather_native(
    ctx: ConversionContext,
    target: Union[Target, str],
    source_ir: Optional[SourceIR],
    name: str,
    plug_inputs: Tuple[Argument, ...],
    root: int = 0,
    group_name: Optional[str] = None,
) -> trt.ITensor:
    """
    Implement gather using native TensorRT DistCollective API.

    This operation has the root rank receive a chunk of data from every other rank

    Returns:
        Output tensor after gather operation

    Example:
        Input on rank 0: [1]  shape=(1,)
        Input on rank 1: [2]  shape=(1,)
        Output on rank 0: [1, 2] shape=(2,)
        Output on rank 1: [undefined, undefined] shape=(2,)
    """
    rank, world_size = _get_distributed_rank_and_world_size()

    # TRT add_dist_collective crashes with world_size=1; gather of a single rank
    # is an identity op.
    if world_size == 1:
        return plug_inputs[0]
    logger.debug(
        f"Adding native gather: name={name}, rank={rank}, world_size={world_size}"
    )

    # Get the input tensor
    input_tensor = _coll_boundary(ctx, plug_inputs[0], name, "in")

    try:
        # Use native TensorRT DistCollective API for GATHER
        # For GATHER, the reduce operation parameter is ignored
        # The last parameter (group) can be None to include all ranks
        import numpy as np

        # Create array of all participating rank IDs [0, 1, 2, ..., world_size-1]
        groups = _collective_group_ranks(group_name, world_size)

        logger.debug(
            f"Creating gather layer: groups={groups.tolist()}, groupSize={world_size}"
        )
        layer = ctx.net.add_dist_collective(
            input_tensor,
            trt.CollectiveOperation.GATHER,
            trt.ReduceOperation.NONE,  # Ignored for GATHER
            root,
            groups,  # None means all ranks participate (world_size ranks)
        )

        logger.debug(f"Successfully created native GATHER layer: {name}")
        logger.debug(
            f"Calling add_dist_collective: input_shape={input_tensor.shape}, "
            f"root={root}, groups={groups.tolist()}, groupSize={len(groups)} (inferred from array)"
        )

        set_layer_name(layer, target, name, source_ir)

        # num_ranks must be set BEFORE get_output(0) so Myelin can infer the collective's
        # output shape (e.g. all_gather dim0 = in_dim0 * num_ranks) for shape inference.
        layer.num_ranks = len(groups)
        output = _coll_boundary(ctx, layer.get_output(0), name, "out")
        _mark_collective_boundary(ctx, input_tensor, output)

        return output

    except Exception as e:
        logger.error(f"Native GATHER failed: {e} (type: {type(e).__name__})")
        raise
