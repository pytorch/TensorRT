# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import contextlib
import logging
import threading
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

import torch
from torch_tensorrt._utils import sanitized_torch_version
from torch_tensorrt.dynamo._settings import CompilationSettings
from torch_tensorrt.dynamo.conversion.aten_ops_converters import _PASS_THROUGH_OPS
from torch_tensorrt.dynamo.lowering.constant_fold_exclusions import (
    CONSTANT_FOLD_EXCLUSION_META_KEY,
)
from torch_tensorrt.dynamo.lowering.passes.pass_utils import (
    clean_up_graph_after_modifications,
)
from torch_tensorrt.dynamo.lowering.passes.reset_folded_constructors import (
    FOLDED_CONSTRUCTOR_META,
    FOLDED_WITHOUT_PARAMETERS_META,
    _aliases_input,
    _storage_escapes,
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

# Install only the source of a folded broadcast when that saves more than this.
_MIN_BROADCAST_SAVING_BYTES = 1 << 20  # 1 MiB
_SAME_SIZE_INTEGER = {1: torch.int8, 2: torch.int16, 4: torch.int32, 8: torch.int64}
_SOURCE_VIEW_OPS = {torch.ops.aten.slice.Tensor, torch.ops.aten.permute.default}
_MATMUL_OPS = {
    torch.ops.aten.matmul.default,
    torch.ops.aten.mm.default,
    torch.ops.aten.bmm.default,
    torch.ops.aten.linear.default,
}
# Readers after which adding, subtracting, multiplying or dividing a broadcast and a matmul
# result was measured to run as fast.
_AFTER_MATMUL_ADD_OPS = {
    torch.ops.aten.softmax.int,
    torch.ops.aten._softmax.default,
    torch.ops.aten.layer_norm.default,
    torch.ops.aten.native_layer_norm.default,
    torch.ops.aten._assert_tensor_metadata.default,
}
_SCALE_LAYER_OPS = {
    torch.ops.aten.add.Tensor,
    torch.ops.aten.sub.Tensor,
    torch.ops.aten.mul.Tensor,
    torch.ops.aten.div.Tensor,
}

# Only skip installing aliased folds whose converters are verified to accept
# constant inputs. Unknown operations retain the established install behavior.
_SAFE_ALIASED_FOLD_SKIP_OPS: Set[torch._ops.OpOverload] = {
    torch.ops.aten.permute.default,
}

# Compared by value in fold keys; any other argument is compared by identity.
_VALUE_TYPES = (
    bool,
    int,
    float,
    str,
    type(None),
    slice,
    torch.dtype,
    torch.device,
    torch.layout,
    torch.memory_format,
)


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


def _has_offloaded_constants(gm: torch.fx.GraphModule) -> bool:
    """Whether a constant sits on the CPU while its metadata names a device."""
    for node in gm.graph.nodes:
        meta = node.meta.get("val")
        if node.op != "get_attr" or not isinstance(meta, torch.Tensor):
            continue
        value: Any = gm
        for atom in str(node.target).split("."):
            value = getattr(value, atom, None)
        if (
            isinstance(value, torch.Tensor)
            and value.device.type == "cpu"
            and meta.device.type != "cpu"
        ):
            return True
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
    folder = (
        _OffloadedConstantFolder
        if _has_offloaded_constants(gm)
        else _TorchTensorRTConstantFolder
    )
    cf = folder(gm, skip_constructors=False)
    # With TF32 allowed, TensorRT computes in TF32 too, so folding keeps the caller's settings.
    with _ieee_float32() if settings.disable_tf32 else contextlib.nullcontext():
        cf.run()

    # The constants are created on CPU to save GPU memory for TensorRT compilation.
    # For TRT INetwork construction the constants are moved to CPU in get_attr call.
    skipped_alias = 0
    # Slices and permutes rewritten to read an installed source, not installed on their own.
    rewritten: Set[torch.fx.Node] = set()
    # Repeats of an op share the tensor folded first, so install each form of it once.
    installed: Dict[Tuple[int, bool], torch.fx.Node] = {}
    # Repeats of an op take the source only if all of them can read it.
    repeats: Dict[int, List[torch.fx.Node]] = {}
    for node, constant in cf.node_replacements.items():
        repeats.setdefault(id(constant), []).append(node)
    sources: Dict[int, Optional[torch.Tensor]] = {}
    # Folded nodes and permutes that take a slice, so later checks read them at that shape.
    sliced: Dict[torch.fx.Node, torch.Tensor] = {}
    # Whether each value reads a parameter, in one pass in graph order, which puts a node's inputs
    # before it. The constant installed for a value that reads none inherits the mark from its meta.
    parameters = {name for name, _ in gm.named_parameters(remove_duplicate=False)}
    reads_parameter: Dict[torch.fx.Node, bool] = {}
    for node in gm.graph.nodes:
        if node.op == "get_attr":
            reads_parameter[node] = (
                not node.meta.get(FOLDED_WITHOUT_PARAMETERS_META)
                if node.meta.get(FOLDED_CONSTRUCTOR_META)
                else str(node.target) in parameters
            )
        else:
            reads_parameter[node] = any(
                reads_parameter[i] for i in node.all_input_nodes
            )
        if node in cf.node_replacements and not reads_parameter[node]:
            node.meta[FOLDED_WITHOUT_PARAMETERS_META] = True
    for node, constant in list(cf.node_replacements.items()):
        if node in rewritten:
            continue
        # TensorRT only accepts a quantized constant directly in front of a
        # DQ layer, so a permute feeding a dequantize must still be folded.
        if (
            isinstance(constant, torch.Tensor)
            and node.target in _SAFE_ALIASED_FOLD_SKIP_OPS
            and not any(user.target in cf.quantization_ops for user in node.users)
        ):
            nbytes = int(constant.numel() * constant.element_size())
            if nbytes > _MAX_CONSTANT_FOLD_BYTES and _tensor_reuses_module_storage(
                gm, constant
            ):
                skipped_alias += 1
                logger.debug(
                    "Skipping constant-fold install for supported aliased op %s "
                    "(%d bytes > %d)",
                    node.target,
                    nbytes,
                    _MAX_CONSTANT_FOLD_BYTES,
                )
                continue

        if id(constant) not in sources:
            # A refit folds the new weights again, and their values or strides can differ, so a
            # refittable engine keeps every folded value at its full shape.
            sources[id(constant)] = (
                _shared_source(
                    repeats[id(constant)], constant, cf.node_replacements, sliced
                )
                if settings.immutable_weights
                else None
            )
        source = sources[id(constant)]
        readers = list(node.users)
        key = (id(constant), source is not None)
        attr = installed.get(key)
        if attr is not None:
            node.replace_all_uses_with(attr)
            gm.graph.erase_node(node)
        else:
            value = constant if source is None else source
            if settings.offload_module_to_cpu:
                value = value.cpu().contiguous()
            attr = installed[key] = replace_node_with_constant(
                gm, node, torch.nn.Parameter(value, requires_grad=False)
            )
            if source is not None:
                logger.debug(
                    "Installed %s, a %s slice of the folded %s value of %s",
                    attr.target,
                    tuple(source.shape),
                    tuple(constant.shape),
                    node.name,
                )
                attr.meta.pop("tensor_meta", None)
                if isinstance(full := attr.meta.get("val"), torch.Tensor):
                    attr.meta["val"] = full[tuple(slice(0, n) for n in source.shape)]
        if source is not None:
            for reader in readers:
                if reader.target in _SOURCE_VIEW_OPS:
                    _read_source_view(gm, reader, attr)
                    rewritten.add(reader)

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


# Two folds that overlap in time would otherwise restore each other's TF32 flags.
_TF32_FLAGS_LOCK = threading.Lock()


@contextlib.contextmanager
def _ieee_float32() -> Iterator[None]:
    """A folded value is baked into the engine as it was computed, so with disable_tf32
    compute it in full float32. PyTorch lets cuDNN convolutions use TF32 by default, which
    disable_tf32 does not reach. The flags are process-wide, so other threads also run
    without TF32 while a fold holds them."""
    matmul = torch.backends.cuda.matmul
    with _TF32_FLAGS_LOCK:
        if hasattr(matmul, "fp32_precision"):
            conv = torch.backends.cudnn.conv
            saved = conv.fp32_precision, matmul.fp32_precision
            conv.fp32_precision = matmul.fp32_precision = "ieee"
            try:
                yield
            finally:
                conv.fp32_precision, matmul.fp32_precision = saved
        else:
            # PyTorch before 2.9, as on the JetPack build, has only these flags. The matmul one is
            # kept by name: allow_tf32 reads True for both "high" and "medium".
            cudnn = torch.backends.cudnn
            saved_conv = cudnn.allow_tf32
            saved_matmul = torch.get_float32_matmul_precision()
            cudnn.allow_tf32 = False
            torch.set_float32_matmul_precision("highest")
            try:
                yield
            finally:
                cudnn.allow_tf32 = saved_conv
                torch.set_float32_matmul_precision(saved_matmul)


def _broadcast_source(constant: Any) -> Optional[torch.Tensor]:
    """One slice along each dimension a folded constant repeats along, or None.

    A broadcast folds to a tensor that repeats its source, as a stride-0 view after an
    expand or as a full copy after a clone, repeat or cat. Installed as is, the engine
    stores every copy. The slices keep every value while storing only the source.
    """
    if (
        not isinstance(constant, torch.Tensor)
        or constant.layout != torch.strided
        or constant.is_nested
        or constant.numel() * constant.element_size() <= _MIN_BROADCAST_SAVING_BYTES
    ):
        return None
    source = constant
    for dim, size in enumerate(constant.shape):
        if size > 1 and (source.stride(dim) == 0 or _repeats_along(source, dim)):
            source = source.narrow(dim, 0, 1)
    saving = (constant.numel() - source.numel()) * constant.element_size()
    # A copy, so the installed slice does not keep the full storage alive.
    return source.clone() if saving > _MIN_BROADCAST_SAVING_BYTES else None


def _shared_source(
    nodes: List[torch.fx.Node],
    constant: Any,
    replacements: Dict[torch.fx.Node, Any],
    sliced: Dict[torch.fx.Node, torch.Tensor],
) -> Optional[torch.Tensor]:
    """The broadcast source of a folded constant if every node folded to it can read the
    source, or None. If they can, adds them and their permutes to sliced.

    The attention mask, a slice of one copy and some pointwise ops read the source directly,
    and a permute read only by those reads a permute of it. Measured, other readers run slower
    on a broadcast: a matmul, and a pointwise op that TensorRT then no longer fuses with its
    neighbors.
    """
    if not all(
        not _storage_escapes(node)
        and all(
            user.target in _SOURCE_VIEW_OPS
            or _reads_as_attention_mask(user, node)
            or _pointwise_reads_broadcast(user, node)
            for user in node.users
        )
        for node in nodes
    ):
        return None
    source = _broadcast_source(constant)
    if source is None:
        return None
    # Checked with every repeat, permute and earlier value that takes a slice read as it, as
    # one op can read several.
    substitutes = dict(sliced)
    for node in nodes:
        substitutes[node] = source
        for user in node.users:
            if user.target == torch.ops.aten.permute.default:
                substitutes[user] = source.permute(user.args[1])
    if not all(
        _reads_source(user, node, substitutes, replacements)
        for node in nodes
        for user in node.users
    ):
        return None
    sliced.update(substitutes)
    return source


def _repeats_along(tensor: torch.Tensor, dim: int) -> bool:
    """Whether every slice of the tensor along dim holds the same bits as the first."""
    integer = _SAME_SIZE_INTEGER.get(tensor.element_size())
    if integer is None or tensor.is_quantized:
        return False
    # Comparing bits tells -0.0 from 0.0 and matches equal NaNs, so the check is exact.
    bits = tensor.view(integer)
    # A corner of the first two slices rules out most tensors before the full pass.
    corner = bits[
        tuple(slice(0, 2) if d == dim else slice(0, 8) for d in range(bits.dim()))
    ]
    if not torch.equal(corner, corner.narrow(dim, 0, 1).expand_as(corner)):
        return False
    return bool(torch.equal(bits, bits.narrow(dim, 0, 1).expand_as(bits)))


def _reads_as_attention_mask(user: torch.fx.Node, folded: torch.fx.Node) -> bool:
    """Whether user is scaled_dot_product_attention reading the folded value only as its mask.

    Its mask only has to broadcast to the scores. The fused efficient and cuDNN attention
    ops are left out, as their eager kernels are not known to accept a broadcast mask.
    """
    if user.target != torch.ops.aten.scaled_dot_product_attention.default:
        return False
    mask = user.kwargs.get("attn_mask", user.args[3] if len(user.args) > 3 else None)
    others = [arg for i, arg in enumerate(user.args) if i != 3]
    others += [arg for name, arg in user.kwargs.items() if name != "attn_mask"]
    return mask is folded and folded not in others


def _reads_source(
    user: torch.fx.Node,
    folded: torch.fx.Node,
    substitutes: Dict[torch.fx.Node, torch.Tensor],
    replacements: Dict[torch.fx.Node, Any],
) -> bool:
    """Whether a user of a folded broadcast can read its source, or a permute of it, when
    each node in substitutes reads its value there."""
    if user.target == torch.ops.aten.permute.default:
        # Readers that a folded constant replaces never see the permute.
        return all(
            _reads_as_attention_mask(reader, user)
            or _folded_away(reader, replacements)
            or (
                _pointwise_reads_broadcast(reader, user)
                and _same_output_from_source(reader, substitutes)
                # Two shrunk reads of the folded value could shrink the output.
                and all(
                    node is user or (node is not folded and node not in folded.users)
                    for node in reader.all_input_nodes
                )
            )
            for reader in user.users
        )
    if user.target == torch.ops.aten.slice.Tensor:
        # A slice that takes one copy has the source's shape, so its readers take the source.
        output = user.meta.get("val")
        return (
            isinstance(output, torch.Tensor)
            and all(isinstance(n, int) for n in output.shape)
            and output.shape == substitutes[folded].shape
        )
    return _reads_as_attention_mask(user, folded) or (
        _pointwise_reads_broadcast(user, folded)
        and _same_output_from_source(user, substitutes)
    )


def _read_source_view(
    gm: torch.fx.GraphModule, view: torch.fx.Node, source: torch.fx.Node
) -> None:
    """Point a slice or permute of a folded broadcast at its installed source."""
    if view.target == torch.ops.aten.slice.Tensor:
        view.replace_all_uses_with(source)
        gm.graph.erase_node(view)
        return
    view.meta.pop("tensor_meta", None)
    if isinstance(small := source.meta.get("val"), torch.Tensor):
        view.meta["val"] = small.permute(view.args[1])


def _folded_away(node: torch.fx.Node, replacements: Dict[torch.fx.Node, Any]) -> bool:
    """Whether a folded constant replaces the node, or every node it feeds."""
    # Each node is decided once, as paths through shared readers multiply, and without
    # recursion, as a chain of readers can be longer than the recursion limit.
    decided: Dict[torch.fx.Node, bool] = {}
    pending = [node]
    while pending:
        current = pending[-1]
        if current in decided:
            pending.pop()
        elif current.target in _SAFE_ALIASED_FOLD_SKIP_OPS:
            decided[current] = False
        elif current in replacements:
            decided[current] = True
        elif current.op != "call_function" or not current.users:
            decided[current] = False
        else:
            undecided = [user for user in current.users if user not in decided]
            if undecided:
                pending.extend(undecided)
            else:
                decided[current] = all(decided[user] for user in current.users)
    return decided[node]


def _same_output_from_source(
    user: torch.fx.Node, substitutes: Dict[torch.fx.Node, torch.Tensor]
) -> bool:
    """Whether a pointwise user gives an output of the same shape and strides when each of
    its inputs in substitutes reads the value there. The strides matter to a later view or
    as_strided of the output, and running the op on meta tensors also rejects an op that
    does not broadcast."""
    output = user.meta.get("val")
    if not isinstance(output, torch.Tensor):
        return False
    inputs = {}
    for node in user.all_input_nodes:
        value = substitutes[node] if node in substitutes else node.meta.get("val")
        if not isinstance(value, torch.Tensor) or not all(
            isinstance(n, int) for n in (*value.shape, *value.stride())
        ):
            return False
        inputs[node] = torch.empty_strided(
            value.shape, value.stride(), dtype=value.dtype, device="meta"
        )
    args, kwargs = torch.fx.node.map_arg((user.args, user.kwargs), inputs.__getitem__)
    try:
        result = user.target(*args, **kwargs)
    except Exception:
        return False
    return bool(
        isinstance(result, torch.Tensor)
        and result.shape == output.shape
        and result.stride() == output.stride()
    )


def _pointwise_reads_broadcast(user: torch.fx.Node, folded: torch.fx.Node) -> bool:
    """Whether TensorRT loses no speed when a pointwise user reads a broadcast of folded."""
    if user.target in _SCALE_LAYER_OPS:
        return _keeps_fusion(user, folded)
    return _ends_pointwise_run(user)


def _ends_pointwise_run(user: torch.fx.Node) -> bool:
    """Whether user is a pointwise op whose readers are neither pointwise ops nor views.

    Keeping pointwise readers and views out keeps the full constant wherever TensorRT might
    fuse the op with the ops after it, which a broadcast was measured to prevent.
    """
    return (
        torch.Tag.pointwise in getattr(user.target, "tags", ())
        and bool(user.users)
        and all(
            reader.op == "call_function"
            and torch.Tag.pointwise not in getattr(reader.target, "tags", ())
            and not _aliases_input(reader)
            for reader in user.users
        )
    )


def _keeps_fusion(user: torch.fx.Node, folded: torch.fx.Node) -> bool:
    """Whether TensorRT loses no speed when an add, subtract, multiply or divide reads a
    broadcast of folded.

    TensorRT runs an fp16 or fp32 add, subtract, multiply or divide with a broadcast constant
    as a Scale layer. Its kernel is slower than the fused pointwise kernel and doesn't fuse
    with pointwise ops before or after it. Measured, that costs nothing when the op adds a
    bias to attention scores, and when the result goes only to matmuls or only adds a bias to
    attention scores and the other operand is made by neither a pointwise op nor a matmul, as
    TensorRT fuses the op into those. Ops that only reshape, copy or cast are looked through
    after the op and before a matmul, as TensorRT folds them into the ops around them.
    """
    if _adds_bias_to_scores(user, folded):
        return True
    others = [node for node in user.all_input_nodes if node is not folded]
    if (
        len(others) != 1
        or torch.Tag.pointwise in getattr(others[0].target, "tags", ())
        or _producer_past_views(others[0]).target in _MATMUL_OPS
    ):
        return False
    return all(
        reader.target in _MATMUL_OPS for reader in _readers_past_views(user)
    ) or all(_adds_bias_to_scores(reader, user) for reader in user.users)


def _adds_bias_to_scores(node: torch.fx.Node, bias: torch.fx.Node) -> bool:
    """Whether node combines bias with the output of a matmul, scaled or not, and its result
    goes to a softmax or a layer norm, as TensorRT fuses the scores, the bias and the softmax
    of attention into one kernel."""
    others = [other for other in node.all_input_nodes if other is not bias]
    if node.target not in _SCALE_LAYER_OPS or len(others) != 1:
        return False
    scores = others[0]
    while (
        scores.target in (torch.ops.aten.mul.Tensor, torch.ops.aten.div.Tensor)
        and isinstance(scores.args[0], torch.fx.Node)
        and _is_number(scores.args[1])
    ):
        scores = scores.args[0]
    return scores.target in _MATMUL_OPS and all(
        reader.target in _AFTER_MATMUL_ADD_OPS for reader in _readers_past_views(node)
    )


def _producer_past_views(node: torch.fx.Node) -> torch.fx.Node:
    """The node that makes node's value, past ops that only reshape, copy or cast it."""
    while node.target in _PASS_THROUGH_OPS and isinstance(node.args[0], torch.fx.Node):
        node = node.args[0]
    return node


def _readers_past_views(node: torch.fx.Node) -> Set[torch.fx.Node]:
    """The readers of node, with each op that only reshapes, copies or casts replaced by its
    own readers. TensorRT folds those ops into the ones around them."""
    readers: Set[torch.fx.Node] = set()
    seen: Set[torch.fx.Node] = set()
    pending = list(node.users)
    while pending:
        reader = pending.pop()
        if reader in seen:
            continue
        seen.add(reader)
        if reader.target in _PASS_THROUGH_OPS and reader.users:
            pending.extend(reader.users)
        else:
            readers.add(reader)
    return readers


def _is_number(arg: Any) -> bool:
    """Whether arg is a Python number or a node holding a tensor of one element."""
    if isinstance(arg, (int, float)):
        return True
    if not isinstance(arg, torch.fx.Node):
        return False
    value = arg.meta.get("val")
    if value is None and arg.op == "get_attr":
        value = getattr(arg.graph.owning_module, arg.target, None)
    return isinstance(value, torch.Tensor) and value.numel() == 1


def replace_node_with_constant(
    gm: torch.fx.GraphModule, node: torch.fx.Node, constant: torch.Tensor
) -> torch.fx.Node:
    """Adapted from:
    https://github.com/pytorch/pytorch/blob/bcf35c6ae62bb6560befa3550e37a8283944e5f4/torch/_inductor/constant_folding.py#L17-L43

    Registers frozen constants as non-trainable parameters.
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
        # Distinguishes this from module state that existed before folding: the
        # value came from an op in the graph body, so eager rebuilds it per call.
        new_input_node.meta[FOLDED_CONSTRUCTOR_META] = True
        g.erase_node(node)

    # Needed to suppress `does not reference an nn.Module, nn.Parameter, or buffer` warning
    gm.register_parameter(qualname, constant)
    setattr(gm, qualname, constant)
    return new_input_node


def _describe_args(arg: Any, repeat_of: Dict[torch.fx.Node, torch.fx.Node]) -> Any:
    """A hashable description of node arguments, naming each input by the first node
    that computes its value."""
    if isinstance(arg, torch.fx.Node):
        return repeat_of.get(arg, arg)
    if isinstance(arg, (list, tuple)):
        return type(arg), tuple(_describe_args(a, repeat_of) for a in arg)
    if isinstance(arg, dict):
        return dict, tuple((k, _describe_args(v, repeat_of)) for k, v in arg.items())
    if isinstance(arg, _VALUE_TYPES):
        # The type and repr keep apart values that compare equal, such as 1, 1.0
        # and True, or 0.0 and -0.0.
        return type(arg), repr(arg)
    return type(arg), id(arg)


# TODO: Delete this class when the following code is fixed in nightly:
# https://github.com/pytorch/pytorch/blob/4b881b0da390c1290bb12850ef9daad6f6eb2cb6/torch/_inductor/constant_folding.py#L53-L63
class _TorchTensorRTConstantFolder(ConstantFolder):  # type: ignore[misc]
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._first_with_key: Dict[Tuple[Any, ...], torch.fx.Node] = {}
        self._repeat_of: Dict[torch.fx.Node, torch.fx.Node] = {}
        # Quantization ops excluded from constant folding so TRT sees QDQ
        # (ModelOpt tensorrt.quantize_op, TorchAO dequantize_affine, TorchAO
        # NVFP4 dequantize_nvfp4, and TorchAO MXFP4 dequantize_mxfp4 when
        # those custom ops are registered).
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

        try:
            import torchao  # noqa: F401

            assert torch.ops.torchao.dequantize_affine.default
            self.quantization_ops.add(torch.ops.torchao.dequantize_affine.default)
        except Exception:
            pass

        try:
            from torchao.quantization.quant_primitives import (  # noqa: F401
                _dequantize_affine_float8_non_decomposed,
                _quantize_affine_float8_non_decomposed,
            )

            self.quantization_ops.add(
                torch.ops.torchao.quantize_affine_float8_non_decomposed.default
            )
            self.quantization_ops.add(
                torch.ops.torchao.dequantize_affine_float8_non_decomposed.default
            )
        except Exception:
            pass

        try:
            from torch_tensorrt.dynamo.conversion import nvfp4_custom_op  # noqa: F401

            assert torch.ops.torchao_trt.dequantize_nvfp4.default
            self.quantization_ops.add(torch.ops.torchao_trt.dequantize_nvfp4.default)
        except Exception:
            pass

        try:
            from torch_tensorrt.dynamo.conversion import mxfp4_custom_op  # noqa: F401

            assert torch.ops.torchao_trt.dequantize_mxfp4.default
            self.quantization_ops.add(torch.ops.torchao_trt.dequantize_mxfp4.default)
        except Exception:
            pass

    def _deduce_value(self, node: torch.fx.Node) -> Any:
        # Readers of a constant often repeat the same op on it, such as one cast per
        # step of an unrolled loop under autocast. Keeping a folded result per repeat
        # would hold one copy per reader, so a repeat reuses the first result while
        # it is still held.
        key = self._fold_key(node)
        if key is not None:
            first = self._first_with_key.setdefault(key, node)
            if first is not node:
                self._repeat_of[node] = first
                if first in self.node_replacements:
                    return self.node_replacements[first]
        # The parent computes a node's value before it asks is_impure, so a node that
        # stays in the graph would still run here. A quantize kernel that cannot run
        # in this environment would then fail the compile for a value never used.
        # This comes after the repeat bookkeeping, so repeats share as before.
        if self.is_impure(node):
            return self.unknown_value
        return super()._deduce_value(node)

    def _fold_key(self, node: torch.fx.Node) -> Optional[Tuple[Any, ...]]:
        # A value that escapes or is mutated must stay a tensor of its own.
        if node.op != "call_function" or node.is_impure() or _storage_escapes(node):
            return None
        return node.target, _describe_args((node.args, node.kwargs), self._repeat_of)

    # TODO: Update this function when quantization is added
    def is_impure(self, node: torch.fx.node.Node) -> bool:

        # Folding an observable view loses its relation to the original storage.
        # Keep the view so the reset pass can clone a fresh allocation once and
        # rebuild its views, while views of module state keep their aliases.
        if _aliases_input(node) and _storage_escapes(node):
            return True

        if node.target in self.quantization_ops:
            return True
        # The meta value holds the IDs of the rules that marked this node; it is
        # empty once every rule that claimed it has been disabled.
        return bool(node.meta.get(CONSTANT_FOLD_EXCLUSION_META_KEY))


class _OffloadedConstantFolder(_TorchTensorRTConstantFolder):
    """Folds a graph holding constants that offload_module_to_cpu left on the CPU."""

    def run_node(self, node: torch.fx.Node) -> Any:
        try:
            return super().run_node(node)
        except RuntimeError as error:
            # offload_module_to_cpu leaves folded constants on the CPU while their
            # metadata keeps the device that computed them. An op that also reads a
            # tensor on that device, such as a scalar the complex lowering's retrace
            # created there, can fail on the mix. Only then does it run again with
            # those constants on their device, so every op that ran before still
            # computes where it did, and an op that fails either way keeps its error.
            inputs = {inp: self.env.get(inp) for inp in node.all_input_nodes}
            devices = {v.device for v in inputs.values() if isinstance(v, torch.Tensor)}
            offloaded = {
                inp: meta.device
                for inp, value in inputs.items()
                if isinstance(value, torch.Tensor)
                and value.device.type == "cpu"
                and isinstance(meta := inp.meta.get("val"), torch.Tensor)
                and meta.device.type != "cpu"
                and meta.device in devices
            }
            if not offloaded:
                raise
            try:
                self.env.update(
                    {inp: inputs[inp].to(device) for inp, device in offloaded.items()}
                )
                return super().run_node(node)
            except RuntimeError:
                raise error from None
            finally:
                self.env.update({inp: inputs[inp] for inp in offloaded})
