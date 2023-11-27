import logging
from typing import Any, Sequence

import torch
from torch_tensorrt._utils import sanitized_torch_version
from torch_tensorrt.dynamo.lowering.passes.pass_utils import (
    clean_up_graph_after_modifications,
)

from packaging import version

# Modify import location of utilities based on Torch version
if version.parse(sanitized_torch_version()) < version.parse("2.1.1"):
    from torch._inductor.freezing import ConstantFolder, replace_node_with_constant
else:
    from torch._inductor.constant_folding import (
        ConstantFolder,
        replace_node_with_constant,
    )

logger = logging.getLogger(__name__)


"""
The below is adapted from
https://github.com/pytorch/pytorch/blob/4c794f2ef183c7bf28d6973cd0fd946dfe73a97c/torch/_inductor/constant_folding.py
"""
import collections
from typing import Any, Callable, Dict, Optional

import torch
import torch.utils._pytree as pytree

aten = torch.ops.aten


def replace_node_with_constant(gm, node, constant):
    g = gm.graph

    if not hasattr(gm, "_frozen_param_count"):
        gm._frozen_param_count = 0

    i = gm._frozen_param_count

    while True:
        qualname = f"_frozen_param{i}"
        if not hasattr(gm, qualname):
            break
        i += 1

    gm._frozen_param_count = i + 1

    with g.inserting_before(node):
        new_input_node = g.create_node("get_attr", qualname, (), {})
        node.replace_all_uses_with(new_input_node)
        new_input_node.meta.update(node.meta)
        g.erase_node(node)

    # needed to suppress `does not reference an nn.Module, nn.Parameter, or buffer` warning
    gm.register_buffer(qualname, constant)
    setattr(gm, qualname, constant)


class ConstantFolder(torch.fx.Interpreter):
    def __init__(
        self,
        gm,
        skip_constructors=False,
    ):
        super().__init__(gm)
        self.node_replacements: Dict[torch.fx.Node, Any] = {}
        self.replaced_uses: Dict[torch.fx.Node, int] = collections.Counter()
        self.unknown_value = object()
        self.skip_constructors: bool = skip_constructors

        # overwrite this to deallocate env values if their only remaining use
        # is the output
        self.user_to_last_uses = self.node_to_last_non_output_use()

    def is_impure(self, node: torch.fx.node.Node):
        if node.target in [
            torch.ops.quantized_decomposed.dequantize_per_channel.default,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default,
            torch.ops.quantized_decomposed.dequantize_per_tensor.tensor,
        ]:
            # For the pattern fp32_weight -> q -> dq
            # We only folding fp32_weight -> q
            # int8_weight and leave dq in graph to be fused
            return True
        return False

    def node_to_last_non_output_use(self):
        last_non_output_use = collections.defaultdict(list)
        seen_uses = set()
        output_node = next(iter(reversed(self.module.graph.nodes)))

        for node in reversed(self.module.graph.nodes):
            if node.target == "output":
                continue

            def add_use(inp):
                if inp in seen_uses:
                    return

                seen_uses.add(inp)
                last_non_output_use[node].append(inp)

            pytree.tree_map_only(torch.fx.Node, add_use, (node.args, node.kwargs))

            # if this node is only used in output, we want to gc it right away
            if len(node.users) == 1 and output_node in node.users:
                last_non_output_use[node].append(node)

        return last_non_output_use

    def run_node(self, node):
        if node.target == "output":
            # because we remove nodes from env on last non output use,
            # re-define them now or we'll get error in interpreter
            def set_env(arg):
                self.env[arg] = self.unknown_value

            pytree.tree_map_only(torch.fx.Node, set_env, node.args)
            return super().run_node(node)

        args, kwargs = self.fetch_args_kwargs_from_env(node)
        flattened_inputs = pytree.arg_tree_leaves(*args, **kwargs)

        if self.unknown_value in flattened_inputs:
            return self.unknown_value

        # TODO - fix errors with this
        if (
            node.op == "call_function"
            and node.target == aten._efficientzerotensor.default
        ):
            return self.unknown_value

        # skip constructors, since inductor generates optimal code for them already
        # and turning into tensor would result in an additional global memory read
        # TODO - more complicated strategy
        if (
            self.skip_constructors
            and node.op != "get_attr"
            and not any(isinstance(e, torch.Tensor) for e in flattened_inputs)
        ):
            return self.unknown_value

        # All mutations should either be removed or on inputs which we did not make constant
        if (
            isinstance(node.target, torch._ops.OpOverload)
            and torch.Tag.nondeterministic_seeded in node.target.tags
        ):
            return self.unknown_value

        out = super().run_node(node)

        if node.op != "get_attr" and isinstance(out, torch.Tensor):
            if not self.insertable_tensor_check(out):
                return out

            if self.is_impure(node):
                return self.unknown_value

            self.add_node_replacement(node, out)

            flattened_node_inps = pytree.arg_tree_leaves(*node.args, **node.kwargs)

            for n in flattened_node_inps:
                if not isinstance(n, torch.fx.Node):
                    continue

                self.replaced_uses[n] += 1

            for to_delete in self.user_to_last_uses.get(node, []):
                if self.replaced_uses[to_delete] == len(to_delete.users):
                    self.node_replacements.pop(to_delete, None)

        return out

    def insertable_tensor_check(self, tensor: torch.Tensor) -> bool:
        return True

    def add_node_replacement(self, node: torch.fx.Node, tensor: torch.Tensor) -> None:
        self.node_replacements[node] = tensor

    def run(self):
        env = {}
        for n in self.module.graph.nodes:
            if n.op == "placeholder":
                env[n] = self.unknown_value
        return super().run(initial_env=env)


@torch.utils._python_dispatch._disable_current_modes()
def constant_fold(gm, constraint_fn: Optional[Callable[[torch.fx.Node], bool]] = None):
    cf = ConstantFolder(gm, skip_constructors=True)
    cf.run()

    for node, constant in cf.node_replacements.items():
        if constraint_fn is not None and not constraint_fn(node):
            continue
        replace_node_with_constant(gm, node, constant)

    erased_params = []
    for node in gm.graph.nodes:
        if node.op == "get_attr" and len(node.users) == 0:
            if hasattr(gm, node.target):
                delattr(gm, node.target)
            erased_params.append(node)

    for node in erased_params:
        gm.graph.erase_node(node)

    gm.graph.eliminate_dead_code()
    gm.graph.lint()
    gm.recompile()


"""
The above is adapted from
https://github.com/pytorch/pytorch/blob/4c794f2ef183c7bf28d6973cd0fd946dfe73a97c/torch/_inductor/constant_folding.py
"""


@torch.utils._python_dispatch._disable_current_modes()  # type: ignore
def constant_fold(
    gm: torch.fx.GraphModule, sample_inputs: Sequence[torch.Tensor]
) -> torch.fx.GraphModule:
    """Adapted from:
    https://github.com/pytorch/pytorch/blob/3a79621c9dce17f77fbddc06aab21f6bc477f313/torch/_inductor/freezing.py#L178-L197

    Folds constants in the graph module, not skipping constructors

    Modifies the graph in-place and replaces node with constants
    """
    cf = _TorchTensorRTConstantFolder(gm, skip_constructors=False)
    cf.run()

    for node, constant in cf.node_replacements.items():
        replace_node_with_constant(gm, node, constant)

    erased_params = []
    for node in gm.graph.nodes:
        # If get_attr node has no users, mark it for deletion
        if node.op == "get_attr" and len(node.users) == 0:
            # If the node's parameter is not a parameter of any other node, remove it
            if not any(
                other.target == node.target for other in gm.graph.nodes if other != node
            ):
                delattr(gm, node.target)
            erased_params.append(node)

    # Remove unused nodes from the graph
    for node in erased_params:
        gm.graph.erase_node(node)

    gm = clean_up_graph_after_modifications(gm)

    logger.debug(f"Graph after constant folding:\n{gm.graph}")

    return gm


# TODO: Delete this class when the following code is fixed in nightly:
# https://github.com/pytorch/pytorch/blob/4b881b0da390c1290bb12850ef9daad6f6eb2cb6/torch/_inductor/constant_folding.py#L53-L63
class _TorchTensorRTConstantFolder(ConstantFolder):  # type: ignore[misc]
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    # TODO: Update this function when quantization is added
    def is_impure(self, node: torch.fx.node.Node) -> bool:
        return False
