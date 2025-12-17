import torch
import torch.fx


def handle_cuda_stream(
    partitioned_module: torch.fx.GraphModule,
) -> torch.fx.GraphModule:
    for node in partitioned_module.graph.nodes:
        if node.op == "placeholder":
            with partitioned_module.graph.inserting_before(node):
                partitioned_module.graph.call_function(
                    torch.ops.tensorrt.enter_compute_stream
                )
        elif node.op == "output":
            with partitioned_module.graph.inserting_before(node):
                partitioned_module.graph.call_function(
                    torch.ops.tensorrt.exit_compute_stream
                )

    partitioned_module.graph.lint()
    partitioned_module.recompile()
    return partitioned_module


@torch.library.custom_op("tensorrt::enter_compute_stream", mutates_args=())
def enter_compute_stream() -> None:
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.default_stream())
    torch.cuda.set_stream(stream)


@torch.library.custom_op("tensorrt::exit_compute_stream", mutates_args=())
def exit_compute_stream() -> None:
    stream = torch.cuda.current_stream()
    torch.cuda.default_stream().wait_stream(stream)
    torch.cuda.set_stream(torch.cuda.default_stream())


@torch.library.register_fake("tensorrt::enter_compute_stream")
def fake_enter_compute_stream() -> None:
    pass


@torch.library.register_fake("tensorrt::exit_compute_stream")
def fake_exit_compute_stream() -> None:
    pass
