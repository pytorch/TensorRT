# mypy: disallow-untyped-decorators=False

import logging

import torch
from torch_tensorrt.dynamo._settings import CompilationSettings

# dead-code elimination, linting, and recompilation for graph, in-place
from torch_tensorrt.dynamo.lowering.passes.pass_utils import (
    clean_up_graph_after_modifications,
)

logger = logging.getLogger(__name__)


# Registered as torch.library custom ops so the resulting OpOverloads are valid
# call_function targets for FX graphs that get wrapped in torch.export.ExportedProgram
# (torch._export.verifier._check_valid_op rejects plain Python functions).


@torch.library.custom_op("tensorrt::fused_nccl_all_gather", mutates_args=())
def _fused_nccl_all_gather_impl(
    inp: torch.Tensor, group_size: int, group_name: str
) -> torch.Tensor:
    return torch.ops._c10d_functional.wait_tensor.default(
        torch.ops._c10d_functional.all_gather_into_tensor.default(
            inp, group_size, group_name
        )
    )


@_fused_nccl_all_gather_impl.register_fake
def _(inp: torch.Tensor, group_size: int, group_name: str) -> torch.Tensor:
    out_shape = (inp.shape[0] * group_size,) + tuple(inp.shape[1:])
    return inp.new_empty(out_shape)


@torch.library.custom_op("tensorrt::fused_nccl_reduce_scatter", mutates_args=())
def _fused_nccl_reduce_scatter_impl(
    inp: torch.Tensor, reduce_op: str, group_size: int, group_name: str
) -> torch.Tensor:
    return torch.ops._c10d_functional.wait_tensor.default(
        torch.ops._c10d_functional.reduce_scatter_tensor.default(
            inp, reduce_op, group_size, group_name
        )
    )


@_fused_nccl_reduce_scatter_impl.register_fake
def _(
    inp: torch.Tensor, reduce_op: str, group_size: int, group_name: str
) -> torch.Tensor:
    out_shape = (inp.shape[0] // group_size,) + tuple(inp.shape[1:])
    return inp.new_empty(out_shape)


@torch.library.custom_op("tensorrt::fused_nccl_all_reduce", mutates_args=())
def _fused_nccl_all_reduce_impl(
    inp: torch.Tensor, reduce_op: str, group_name: str
) -> torch.Tensor:
    return torch.ops._c10d_functional.wait_tensor.default(
        torch.ops._c10d_functional.all_reduce.default(inp, reduce_op, group_name)
    )


@_fused_nccl_all_reduce_impl.register_fake
def _(inp: torch.Tensor, reduce_op: str, group_name: str) -> torch.Tensor:
    return torch.empty_like(inp)


@torch.library.custom_op("tensorrt::fused_nccl_all_to_all", mutates_args=())
def _fused_nccl_all_to_all_impl(
    inp: torch.Tensor,
    output_splits: list[int] | None,
    input_splits: list[int] | None,
    group_name: str,
) -> torch.Tensor:
    out_shape = inp.shape
    return inp.new_empty(out_shape)


@_fused_nccl_all_to_all_impl.register_fake
def _(
    inp: torch.Tensor,
    output_splits: list[int] | None,
    input_splits: list[int] | None,
    group_name: str,
) -> torch.Tensor:
    return torch.ops._c10d_functional.wait_tensor.default(
        torch.ops._c10d_functional.all_to_all_single.default(
            inp, output_splits, input_splits, group_name
        )
    )


@torch.library.custom_op("tensorrt::fused_nccl_scatter", mutates_args=())
def _fused_nccl_scatter_impl(
    inp: torch.Tensor, src: int, group_name: str
) -> torch.Tensor:
    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()

    out = torch.ops._c10d_functional.broadcast.default(inp, src, group_name)
    out = torch.ops._c10d_functional.wait_tensor.default(out)

    chunk = out.shape[0] // world_size
    return out[rank * chunk : (rank + 1) * chunk]


@_fused_nccl_scatter_impl.register_fake
def _(inp: torch.Tensor, src: int, group_name: str) -> torch.Tensor:
    world_size = torch.distributed.get_world_size()
    out_shape = (inp.shape[0] // world_size,) + tuple(inp.shape[1:])
    return inp.new_empty(out_shape)


@torch.library.custom_op("tensorrt::fused_nccl_gather", mutates_args=())
def _fused_nccl_gather_impl(
    inp: torch.Tensor, src: int, group_name: str
) -> torch.Tensor:

    # Perform all_gather
    world_size = torch.distributed.get_world_size()
    out = _fused_nccl_all_gather_impl(inp, world_size, group_name)

    # TRT leads to undefined data after gather on non-root ranks
    # so maintain that here for parity's sake
    rank = torch.distributed.get_rank()
    return out if rank == src else torch.empty_like(out)


@_fused_nccl_gather_impl.register_fake
def _(inp: torch.Tensor, src: int, group_name: str) -> torch.Tensor:
    world_size = torch.distributed.get_world_size()
    out_shape = (inp.shape[0] * world_size,) + tuple(inp.shape[1:])
    return inp.new_empty(out_shape)


# Public aliases — used as FX node targets in the fuse pass, as converter keys
# in custom_ops_converters.py, and in test equality checks. Each is the
# torch._ops.OpOverload created by the custom_op decoration above.
tensorrt_fused_nccl_all_gather_op = torch.ops.tensorrt.fused_nccl_all_gather.default
tensorrt_fused_nccl_reduce_scatter_op = (
    torch.ops.tensorrt.fused_nccl_reduce_scatter.default
)
tensorrt_fused_nccl_all_reduce_op = torch.ops.tensorrt.fused_nccl_all_reduce.default
tensorrt_fused_nccl_all_to_all_op = torch.ops.tensorrt.fused_nccl_all_to_all.default
tensorrt_fused_nccl_scatter_op = torch.ops.tensorrt.fused_nccl_scatter.default
tensorrt_fused_nccl_gather_op = torch.ops.tensorrt.fused_nccl_gather.default


def fuse_distributed_ops(
    gm: torch.fx.GraphModule, settings: CompilationSettings
) -> torch.fx.GraphModule:
    modified_graph = False
    for node in gm.graph.nodes:
        if (
            node.target
            in (
                torch.ops._c10d_functional.all_gather_into_tensor.default,
                torch.ops._c10d_functional.reduce_scatter_tensor.default,
                torch.ops._c10d_functional.all_reduce.default,
                torch.ops._c10d_functional.all_to_all_single.default,
            )
            and len(node.users) == 1
            and list(node.users)[0].target
            == torch.ops._c10d_functional.wait_tensor.default
        ):
            wait_tensor_node = list(node.users)[0]
            if node.target == torch.ops._c10d_functional.all_gather_into_tensor.default:
                with gm.graph.inserting_after(wait_tensor_node):
                    fused_node = gm.graph.create_node(
                        op="call_function",
                        target=tensorrt_fused_nccl_all_gather_op,
                        args=(node.args[0], node.args[1], node.args[2]),
                    )
            elif (
                node.target == torch.ops._c10d_functional.reduce_scatter_tensor.default
            ):
                with gm.graph.inserting_after(wait_tensor_node):
                    fused_node = gm.graph.create_node(
                        op="call_function",
                        target=tensorrt_fused_nccl_reduce_scatter_op,
                        args=(node.args[0], node.args[1], node.args[2], node.args[3]),
                    )
            elif node.target == torch.ops._c10d_functional.all_to_all_single.default:
                with gm.graph.inserting_after(wait_tensor_node):
                    fused_node = gm.graph.create_node(
                        op="call_function",
                        target=tensorrt_fused_nccl_all_to_all_op,
                        args=(node.args[0], node.args[1], node.args[2], node.args[3]),
                    )
            else:
                with gm.graph.inserting_after(wait_tensor_node):
                    fused_node = gm.graph.create_node(
                        op="call_function",
                        target=tensorrt_fused_nccl_all_reduce_op,
                        args=(node.args[0], node.args[1], node.args[2]),
                    )

            wait_tensor_node.replace_all_uses_with(fused_node)
            fused_node.meta.update(node.meta)
            modified_graph = True
            gm.graph.erase_node(wait_tensor_node)
            gm.graph.erase_node(node)

    # If graph was modified, clean it up
    if modified_graph:
        gm = clean_up_graph_after_modifications(gm)
        logger.debug(
            f"Graph after fusing wait_tensor and distributed op tensor:\n{gm.graph}"
        )

    return gm
