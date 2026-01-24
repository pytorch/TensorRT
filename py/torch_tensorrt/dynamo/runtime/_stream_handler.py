import torch
import torch.fx
from torch._higher_order_ops.effects import (
    _EffectType,
)


def handle_cuda_stream(
    partitioned_module: torch.fx.GraphModule,
) -> torch.fx.GraphModule:

    for i, node in enumerate(partitioned_module.graph.nodes):
        if i == 0:
            with partitioned_module.graph.inserting_after(node):
                is_default_stream = partitioned_module.graph.call_function(
                    torch.ops.tensorrt.enter_compute_stream_guard.default,
                )
        elif node.op == "call_module" and "_run_on_acc" in node.name:
            node.args = (*node.args, is_default_stream)

    output_node = list(partitioned_module.graph.nodes)[-1]
    with partitioned_module.graph.inserting_before(output_node):
        partitioned_module.graph.call_function(
            torch.ops.tensorrt.exit_compute_stream_guard.default,
            args=(is_default_stream, output_node.args[0][0]),
        )

    partitioned_module.graph.lint()
    partitioned_module.recompile()
    # new_graph = torch.fx.symbolic_trace(partitioned_module)
    # breakpoint()
    return partitioned_module


@torch.library.custom_op("tensorrt::enter_compute_stream_guard", mutates_args=())
def enter_compute_stream_guard() -> bool:
    default_stream = torch.cuda.default_stream()
    if default_stream != torch.cuda.current_stream():
        return False

    else:
        stream = torch.cuda.Stream()
        stream.wait_stream(default_stream)
        torch.cuda.set_stream(stream)
        return True


@torch.library.custom_op("tensorrt::exit_compute_stream_guard", mutates_args=())
def exit_compute_stream_guard(is_default_stream: bool, output: torch.Tensor) -> None:
    if is_default_stream:
        stream = torch.cuda.current_stream()
        torch.cuda.default_stream().wait_stream(stream)
        torch.cuda.set_stream(torch.cuda.default_stream())


@torch.library.register_fake("tensorrt::enter_compute_stream_guard")
def fake_enter_compute_stream() -> None:
    pass


@torch.library.register_fake("tensorrt::exit_compute_stream_guard")
def fake_exit_compute_stream(is_default_stream: bool, output: torch.Tensor) -> None:
    pass


torch.library._register_effectful_op(
    torch.ops.tensorrt.enter_compute_stream_guard.default, _EffectType.ORDERED
)
torch.library._register_effectful_op(
    torch.ops.tensorrt.exit_compute_stream_guard.default, _EffectType.ORDERED
)
