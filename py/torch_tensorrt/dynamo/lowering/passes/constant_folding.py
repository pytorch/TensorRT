import inspect
import logging
from operator import attrgetter
from typing import Any, Callable, Optional, Set

import torch
from torch_tensorrt._utils import sanitized_torch_version
from torch_tensorrt.dynamo._settings import CompilationSettings
from torch_tensorrt.dynamo.lowering.passes.pass_utils import (
    clean_up_graph_after_modifications,
)

from packaging import version

# Modify import location of utilities based on Torch version
if version.parse(sanitized_torch_version()) < version.parse("2.1.1"):
    from torch._inductor.freezing import ConstantFolder
else:
    from torch._inductor.constant_folding import ConstantFolder

logger = logging.getLogger(__name__)

# Skip installing large folded tensors that already share storage with an
# existing module attribute (typically weight permutes/views). On Flux NVFP4
# those dominate replace cost via cpu().contiguous() under offload_module_to_cpu.
# Do NOT skip materialized folds (e.g. VLA position embedding outputs) — those
# can be required for TRT legality (int64 indices never reach the converter).
_MAX_CONSTANT_FOLD_BYTES = 1 << 20  # 1 MiB

# True views that never allocate. ``aten.contiguous`` is ``is_view`` but can
# materialize; keep executing those so 4531 can still install the copy.
_ALIASING_VIEW_OP_NAMES = (
    "permute.default",
    "transpose.int",
    "t.default",
    "view.default",
    "reshape.default",
    "squeeze.default",
    "squeeze.dim",
    "squeeze.dims",
    "unsqueeze.default",
    "expand.default",
    "slice.Tensor",
    "select.int",
    "detach.default",
    "alias.default",
    "as_strided.default",
    "flatten.using_ints",
    "unflatten.int",
    "movedim.int",
    "movedim.intlist",
)


def _resolve_aten_op(name: str) -> Optional[Any]:
    op: Any = torch.ops.aten
    for part in name.split("."):
        op = getattr(op, part, None)
        if op is None:
            return None
    return op


_ALIASING_VIEW_OPS: Set[Any] = {
    op for name in _ALIASING_VIEW_OP_NAMES if (op := _resolve_aten_op(name)) is not None
}


def _named_attr(gm: torch.fx.GraphModule, target: Any) -> Any:
    if not isinstance(target, str):
        return None
    try:
        return attrgetter(target)(gm)
    except (AttributeError, ValueError):
        return None


def _source_attr_nbytes(gm: torch.fx.GraphModule, node: torch.fx.Node) -> int:
    """Bytes of the get_attr tensor at the root of an aliasing-view chain, else 0."""
    seen: Set[torch.fx.Node] = set()
    cur: Optional[torch.fx.Node] = node
    while cur is not None and cur not in seen:
        seen.add(cur)
        if cur.op == "get_attr":
            tensor = _named_attr(gm, cur.target)
            if isinstance(tensor, torch.Tensor):
                return int(tensor.numel() * tensor.element_size())
            return 0
        if cur.op != "call_function" or cur.target not in _ALIASING_VIEW_OPS:
            return 0
        tensor_args = [arg for arg in cur.args if isinstance(arg, torch.fx.Node)]
        if len(tensor_args) != 1:
            return 0
        cur = tensor_args[0]
    return 0


def skip_large_aliased_view_fold(gm: torch.fx.GraphModule, node: torch.fx.Node) -> bool:
    """Inductor ``skip_folding_node_fn``: True → do not execute this fold.

    Matches the 4531 install skip: large view/permute of a module tensor that
    would only alias existing storage. Materializing ops (add, contiguous, …)
    still run.
    """
    if node.op != "call_function" or node.target not in _ALIASING_VIEW_OPS:
        return False
    return _source_attr_nbytes(gm, node) > _MAX_CONSTANT_FOLD_BYTES


def _tensor_reuses_module_storage(
    gm: torch.fx.GraphModule, constant: torch.Tensor
) -> bool:
    try:
        ptr = constant.untyped_storage().data_ptr()
    except Exception:
        return False
    for tensor in gm.state_dict().values():
        if isinstance(tensor, torch.Tensor):
            try:
                if tensor.untyped_storage().data_ptr() == ptr:
                    return True
            except Exception:
                continue
    return False


@torch.utils._python_dispatch._disable_current_modes()  # type: ignore
def constant_fold(
    gm: torch.fx.GraphModule, settings: CompilationSettings
) -> torch.fx.GraphModule:
    """Adapted from:
    https://github.com/pytorch/pytorch/blob/3a79621c9dce17f77fbddc06aab21f6bc477f313/torch/_inductor/freezing.py#L178-L197

    Folds constants in the graph module, not skipping constructors

    Modifies the graph in-place and replaces node with constants
    """
    cf = _TorchTensorRTConstantFolder(
        gm,
        skip_constructors=False,
        skip_folding_node_fn=lambda node: skip_large_aliased_view_fold(gm, node),
    )
    cf.run()

    # The constants are created on CPU to save GPU memory for TensorRT compilation.
    # For TRT INetwork construction the constants are moved to CPU in get_attr call.
    skipped_alias = 0
    for node, constant in cf.node_replacements.items():
        if isinstance(constant, torch.Tensor):
            nbytes = int(constant.numel() * constant.element_size())
            if nbytes > _MAX_CONSTANT_FOLD_BYTES and _tensor_reuses_module_storage(
                gm, constant
            ):
                skipped_alias += 1
                logger.debug(
                    "Skipping constant-fold install for aliased %s (%d bytes > %d)",
                    node.name,
                    nbytes,
                    _MAX_CONSTANT_FOLD_BYTES,
                )
                continue
        # Register folded values as plain tensors (buffers), matching Inductor.
        if settings.offload_module_to_cpu:
            replace_node_with_constant(
                gm,
                node,
                constant.cpu().contiguous(),
            )
        else:
            replace_node_with_constant(gm, node, constant)

    if skipped_alias:
        logger.info(
            "Skipped installing %d large aliased folded constant(s) (>%d bytes); "
            "leaving original view/permute ops in the graph",
            skipped_alias,
            _MAX_CONSTANT_FOLD_BYTES,
        )

    erased_params = []
    for node in gm.graph.nodes:
        # If get_attr node has no users, mark it for deletion
        if node.op == "get_attr" and len(node.users) == 0:
            erased_params.append(node)

    # Remove unused nodes from the graph
    for node in erased_params:
        gm.graph.erase_node(node)

    gm = clean_up_graph_after_modifications(gm)
    # Delete the constant folder instance which holds GPU memory
    del cf

    logger.debug(f"Graph after constant folding:\n{gm.graph}")
    return gm


def replace_node_with_constant(
    gm: torch.fx.GraphModule, node: torch.fx.Node, constant: torch.Tensor
) -> None:
    """Adapted from:
    https://github.com/pytorch/pytorch/blob/bcf35c6ae62bb6560befa3550e37a8283944e5f4/torch/_inductor/constant_folding.py#L17-L43

    Registers frozen constants as buffers (same as Inductor), not Parameters.
    """
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

    # Needed to suppress `does not reference an nn.Module, nn.Parameter, or buffer` warning
    gm.register_buffer(qualname, constant)
    setattr(gm, qualname, constant)


# TODO: Delete this class when the following code is fixed in nightly:
# https://github.com/pytorch/pytorch/blob/4b881b0da390c1290bb12850ef9daad6f6eb2cb6/torch/_inductor/constant_folding.py#L53-L63
class _TorchTensorRTConstantFolder(ConstantFolder):  # type: ignore[misc]
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        skip_fn = kwargs.get("skip_folding_node_fn")
        init_params = inspect.signature(ConstantFolder.__init__).parameters
        if "skip_folding_node_fn" not in init_params:
            kwargs.pop("skip_folding_node_fn", None)
            super().__init__(*args, **kwargs)
            self.skip_folding_node_fn = skip_fn
        else:
            super().__init__(*args, **kwargs)
        # Set of known quantization ops to be excluded from constant folding.
        # Currently, we exclude all quantization ops coming from modelopt library.
        self.quantization_ops: Set[torch._ops.OpOverload] = set()
        try:
            # modelopt import ensures torch.ops.tensorrt.quantize_op.default is registered
            import modelopt.torch.quantization as mtq  # noqa: F401

            assert torch.ops.tensorrt.quantize_op.default
            assert torch.ops.tensorrt.dynamic_block_quantize_op.default
            self.quantization_ops.add(torch.ops.tensorrt.quantize_op.default)
            self.quantization_ops.add(
                torch.ops.tensorrt.dynamic_block_quantize_op.default
            )
        except Exception as e:
            pass

    def run_node(self, node: torch.fx.node.Node) -> Any:
        # Inductor only consults skip_folding_node_fn when lifted_constant_names
        # is set. Our cf.run() path has none, so honor the callback here and
        # return unknown without executing the op.
        skip_fn: Optional[Callable[[torch.fx.Node], bool]] = getattr(
            self, "skip_folding_node_fn", None
        )
        if skip_fn is not None and node.op == "call_function" and skip_fn(node):
            logger.debug(
                "Skipping constant-fold execute for aliased view %s", node.name
            )
            return self.unknown_value
        return super().run_node(node)

    # TODO: Update this function when quantization is added
    def is_impure(self, node: torch.fx.node.Node) -> bool:

        if node.target in self.quantization_ops:
            return True
        return False
