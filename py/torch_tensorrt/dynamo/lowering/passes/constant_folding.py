# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import contextlib
import logging
import threading
from typing import Any, Dict, Iterator, Optional, Set, Tuple

import torch
from torch_tensorrt._utils import sanitized_torch_version
from torch_tensorrt.dynamo._settings import CompilationSettings
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
    # Repeats of an op share the tensor folded first, so install it only once.
    installed: Dict[int, torch.fx.Node] = {}
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
    for node, constant in cf.node_replacements.items():
        if (
            isinstance(constant, torch.Tensor)
            and node.target in _SAFE_ALIASED_FOLD_SKIP_OPS
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

        shared = installed.get(id(constant))
        if shared is not None:
            node.replace_all_uses_with(shared)
            gm.graph.erase_node(node)
        elif settings.offload_module_to_cpu:
            installed[id(constant)] = replace_node_with_constant(
                gm,
                node,
                torch.nn.Parameter(constant.cpu().contiguous(), requires_grad=False),
            )
        else:
            installed[id(constant)] = replace_node_with_constant(
                gm, node, torch.nn.Parameter(constant, requires_grad=False)
            )

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
