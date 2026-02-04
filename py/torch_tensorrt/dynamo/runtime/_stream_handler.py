import logging
from typing import Any

import torch
import torch.fx
from torch._higher_order_ops.effects import (
    _EffectType,
)

logger = logging.getLogger(__name__)


def handle_cuda_stream(
    partitioned_module: torch.fx.GraphModule,
    sample_arg_inputs: tuple[Any, ...],
    sample_kwarg_inputs: dict[str, Any],
) -> torch.fx.GraphModule:

    for i, node in enumerate(partitioned_module.graph.nodes):
        if i == 0:
            with partitioned_module.graph.inserting_after(node):
                is_default_stream = partitioned_module.graph.call_function(
                    torch.ops.tensorrt.enter_compute_stream_guard.default,
                )
                is_default_stream.meta["val"] = tuple()

    output_node = list(partitioned_module.graph.nodes)[-1]
    with partitioned_module.graph.inserting_before(output_node):
        partitioned_module.graph.call_function(
            torch.ops.tensorrt.exit_compute_stream_guard.default,
            args=(is_default_stream,),
        )
        partitioned_module.meta["val"] = tuple()

    partitioned_module.graph.lint()
    partitioned_module.recompile()
    ep = torch.export.export(partitioned_module, sample_arg_inputs, sample_kwarg_inputs)
    ep = ep.run_decompositions(torch.export.default_decompositions())
    logger.debug(f"After inserting stream guard:\n{ep}")
    return ep


@torch.library.custom_op("tensorrt::enter_compute_stream_guard", mutates_args=())  # type: ignore[misc]
def enter_compute_stream_guard() -> bool:
    default_stream = torch.cuda.default_stream()
    if default_stream != torch.cuda.current_stream():
        return False

    else:
        stream = torch.cuda.Stream()
        stream.wait_stream(default_stream)
        torch.cuda.set_stream(stream)
        return True


@torch.library.custom_op("tensorrt::exit_compute_stream_guard", mutates_args=())  # type: ignore[misc]
def exit_compute_stream_guard(is_default_stream: bool) -> None:
    if is_default_stream:
        stream = torch.cuda.current_stream()
        torch.cuda.default_stream().wait_stream(stream)
        torch.cuda.set_stream(torch.cuda.default_stream())


@torch.library.register_fake("tensorrt::enter_compute_stream_guard")  # type: ignore[misc]
def fake_enter_compute_stream() -> None:
    pass


@torch.library.register_fake("tensorrt::exit_compute_stream_guard")  # type: ignore[misc]
def fake_exit_compute_stream(is_default_stream: bool) -> None:
    pass


torch.library._register_effectful_op(
    torch.ops.tensorrt.enter_compute_stream_guard.default, _EffectType.ORDERED
)
torch.library._register_effectful_op(
    torch.ops.tensorrt.exit_compute_stream_guard.default, _EffectType.ORDERED
)
