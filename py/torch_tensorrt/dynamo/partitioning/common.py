import logging
from typing import Any, Dict, Optional, Sequence, Set, Tuple

import torch
from torch_tensorrt._Input import Input
from torch_tensorrt.dynamo._defaults import DEBUG
from torch_tensorrt.dynamo.utils import get_torch_inputs, input_is_dynamic

logger = logging.getLogger(__name__)


def run_shape_analysis(
    parent_module: torch.fx.GraphModule, inputs: Sequence[Input]
) -> Tuple[Dict[Any, Sequence[Any]], Dict[Any, Sequence[Any]]]:
    submod_inputs_shape_map: Dict[Any, Sequence[Any]] = {}
    submod_outputs_shape_map: Dict[Any, Sequence[Any]] = {}
    sub_inputs: Sequence[torch.Tensor] = []
    sub_outputs: Sequence[torch.Tensor] = []

    # Register a hook to capture IO shapes for submodules
    def get_submodule_io(
        self: Any, inputs: Sequence[torch.Tensor], outputs: Sequence[torch.Tensor]
    ) -> None:
        nonlocal sub_inputs, sub_outputs
        sub_inputs = inputs
        sub_outputs = outputs
        return

    # Iterate through submodules (both Torch and TRT) and store IO shapes
    for name, _ in parent_module.named_children():
        submodule = getattr(parent_module, name)
        handle = submodule.register_forward_hook(get_submodule_io)
        parent_module(*inputs)
        handle.remove()
        submod_inputs_shape_map[name] = (
            [input.shape for input in sub_inputs]
            if isinstance(sub_inputs, (tuple, list))
            else [sub_inputs.shape]
        )
        submod_outputs_shape_map[name] = (
            [output.shape for output in sub_outputs]
            if isinstance(sub_outputs, (tuple, list))
            else [sub_outputs.shape]
        )

    return submod_inputs_shape_map, submod_outputs_shape_map


def get_submod_inputs(
    mod: torch.fx.GraphModule,
    submod: torch.fx.GraphModule,
    inputs: Sequence[Input],
    device: torch.device,
) -> Optional[Sequence[torch.Tensor]]:
    """Helper function to get inputs to a Torch submodule

    Args:
        mod: Parent FX GraphModule
        submod: Child FX GraphModule
        inputs: Sample inputs to parent module
    Returns:
        Sequence of Tensors representing inputs to child module
    """
    acc_inputs: Any = None

    def get_input(self: Any, inputs: Sequence[torch.Tensor]) -> None:
        nonlocal acc_inputs
        acc_inputs = inputs
        return

    # Register a hook to capture submodule input
    handle = submod.register_forward_pre_hook(get_input)
    # Iterate over min, opt, max shapes for dynamic inputs
    inputs_map = {}

    if input_is_dynamic(inputs):
        for mode in ["min_shape", "opt_shape", "max_shape"]:
            torch_inputs = get_torch_inputs(inputs, device, mode)
            mod(*torch_inputs)
            inputs_map[mode] = acc_inputs
        handle.remove()
    else:
        torch_inputs = get_torch_inputs(inputs, device)
        mod(*torch_inputs)
        handle.remove()
        assert isinstance(acc_inputs, tuple)
        return [
            Input(shape=acc_input.shape, dtype=acc_input.dtype)
            for acc_input in acc_inputs
        ]

    num_submodule_inputs = (
        len(inputs_map["min_shape"]) if inputs_map["min_shape"] else 0
    )
    submodule_inputs = []
    for idx in range(num_submodule_inputs):
        if not isinstance(inputs_map["min_shape"][idx], torch.Tensor):
            input_val = torch.tensor(inputs_map["opt_shape"][idx], dtype=torch.int32)
            logger.warning(
                "Detected a zero-dimensional input. This might be a shape tensor input which is not currently supported. This might result in undefined behavior"
            )
            submodule_inputs.append(
                Input(
                    shape=[1],
                    torch_tensor=input_val,
                    dtype=input_val.dtype,
                )
            )
        else:
            submodule_inputs.append(
                Input(
                    min_shape=inputs_map["min_shape"][idx].shape,
                    opt_shape=inputs_map["opt_shape"][idx].shape,
                    max_shape=inputs_map["max_shape"][idx].shape,
                    torch_tensor=inputs_map["opt_shape"][idx],
                    dtype=inputs_map["opt_shape"][idx].dtype,
                )
            )

    return submodule_inputs


def get_graph_converter_support(
    graph_module: torch.fx.GraphModule,
    verbose: bool = DEBUG,
    torch_executed_ops: Optional[Set[str]] = None,
) -> Tuple[int, int]:
    """Helper function to get converter support overview pre-partitioning

    Args:
        graph_module: FX GraphModule to determine support for
        verbose: Bool representing whether to print operator support
        torch_executed_ops: Collection of operations to run in Torch, regardless of converter coverage
    Returns:
        The number of supported call_function nodes in the graph
    """
    from ._global_partitioner import TorchTensorRTOperatorSupport

    # Instantiate operator support object and module dictionary
    op_support = TorchTensorRTOperatorSupport(torch_executed_ops=torch_executed_ops)
    module_dict = dict(graph_module.named_modules())

    number_of_supported_nodes = 0
    total_functional_nodes = 0

    # Iterate over all nodes in the graph, enumerating call_function nodes
    for node in graph_module.graph.nodes:
        if node.op == "call_function":
            total_functional_nodes += 1

            if op_support.is_node_supported(module_dict, node):
                number_of_supported_nodes += 1

    # Print node support overview prior to partitioning
    if verbose:
        op_support.print_support_overview(print_node_support=True)

    return number_of_supported_nodes, total_functional_nodes
