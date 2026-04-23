from __future__ import annotations

import functools
import logging
import unittest
from typing import Any, Callable, Sequence

import torch
import torch._dynamo as td
import torch_tensorrt.logging as torchtrt_logging
from torch._dynamo.backends.common import aot_autograd
from torch._dynamo.utils import detect_fake_mode
from torch._functorch.aot_autograd import aot_export_joint_simple
from torch_tensorrt._utils import is_tegra_platform
from torch_tensorrt.dynamo import CompilationSettings
from torch_tensorrt.dynamo._compiler import compile_module
from torch_tensorrt.dynamo.lowering import (
    get_decompositions,
    post_lowering,
    remove_detach,
    remove_sym_nodes,
    repair_input_aliasing,
)
from torch_tensorrt.dynamo.lowering.passes.fold_get_attr_item_calls import (
    fold_get_attr_item_calls,
)
from torch_tensorrt.dynamo.utils import (
    parse_dynamo_kwargs,
    prepare_inputs,
)

logger = logging.getLogger(__name__)


@td.register_backend(name="tensorrt")  # type: ignore[misc]
@td.register_backend(name="torch_tensorrt")  # type: ignore[misc]
def torch_tensorrt_backend(
    gm: torch.fx.GraphModule, sample_inputs: Sequence[Any], **kwargs: Any
) -> torch.nn.Module:
    # Set log level at the top of compilation (torch_tensorrt.dynamo)
    if (
        "options" in kwargs
        and "debug" in kwargs["options"]
        and kwargs["options"]["debug"]
    ) or ("debug" in kwargs and kwargs["debug"]):
        torchtrt_logging.set_level(logging.DEBUG)

    DEFAULT_BACKEND = aot_torch_tensorrt_aten_backend

    return DEFAULT_BACKEND(gm, sample_inputs, **kwargs)


@td.register_backend(name="aot_torch_tensorrt_aten")  # type: ignore[misc]
def aot_torch_tensorrt_aten_backend(
    gm: torch.fx.GraphModule, sample_inputs: Sequence[Any], **kwargs: Any
) -> torch.nn.Module:
    import torch.distributed as dist

    settings, engine_cache = parse_dynamo_kwargs(kwargs)

    # Auto-enable distributed tracing when running inside an active distributed
    # context — mirrors how DistributedDataParallel works: no explicit flag needed.
    if (
        not settings.use_distributed_mode_trace
        and dist.is_available()
        and dist.is_initialized()
        and dist.get_world_size() > 1
    ):
        logger.debug(
            "Detected active distributed context (world_size=%d); "
            "enabling use_distributed_mode_trace automatically.",
            dist.get_world_size(),
        )
        settings.use_distributed_mode_trace = True

    if settings.use_distributed_mode_trace:
        logger.debug(
            "Wrapping the backend with aot_autograd for Distributed examples\n"
        )
        _pretraced_backend_autograd = functools.partial(
            _pretraced_backend, settings=settings, engine_cache=engine_cache
        )
        settings_aot_autograd = {}
        settings_aot_autograd["decompositions"] = get_decompositions(
            settings.enable_experimental_decompositions,
            settings.decompose_attention,
            settings.use_distributed_mode_trace,
        )
        # This is added since detach lowering leads to alias nodes
        # Error - View operation returned a tensor that is the same as the input base tensor
        # torch nop_decompositions in torch/_decomp/decompositions.py
        # transpose key deleted since not desirable to lower it to permute
        to_delete = {
            key
            for key in settings_aot_autograd["decompositions"]
            if "detach" in key._name
        }
        for key in to_delete:
            del settings_aot_autograd["decompositions"][key]

        return aot_autograd(
            fw_compiler=_pretraced_backend_autograd,
            decompositions=settings_aot_autograd["decompositions"],
        )(gm, sample_inputs)

    if not is_tegra_platform():
        from torch.distributed.tensor import DTensor

        if any(isinstance(tensor, DTensor) for tensor in sample_inputs):
            logger.warning(
                "It is recommended to run the model with use_distributed_mode_trace = True since there are distributed tensors in the input which is not supported in aot_export_joint_simple"
            )

    if settings.offload_module_to_cpu:
        logger.warning(
            "The offload_module_to_cpu option is set, but it is being ignored since the torch_compile backend does not support this feature"
        )

    # If the dynamo-traced graph contains higher-order ops (vmap, etc.) that are
    # incompatible with aot_export_joint_simple, fall back to the aot_autograd
    # path which handles them correctly.
    if _graph_has_higher_order_ops(gm):
        logger.debug(
            "Graph contains higher-order ops (e.g. vmap); "
            "using aot_autograd path instead of aot_export_joint_simple"
        )
        import copy

        aot_settings = copy.copy(settings)
        aot_settings.use_distributed_mode_trace = True
        _pretraced_backend_autograd = functools.partial(
            _pretraced_backend, settings=aot_settings, engine_cache=engine_cache
        )
        aot_decomps = get_decompositions(
            settings.enable_experimental_decompositions, settings.decompose_attention
        )
        # Remove detach decompositions to avoid alias node errors.
        to_delete = {k for k in aot_decomps if "detach" in k._name}
        for k in to_delete:
            del aot_decomps[k]
        return aot_autograd(
            fw_compiler=_pretraced_backend_autograd,
            decompositions=aot_decomps,
        )(gm, sample_inputs)

    return _pretraced_backend(gm, sample_inputs, settings, engine_cache)


def _strip_trt_sdpa_kwargs(gm: torch.fx.GraphModule) -> None:
    """Remove TRT-specific kwargs (use_fp32_acc, sliding_window_size) from SDPA nodes.

    Called both on the failure fallback path (original_gm) and on the compiled
    result (trt_compiled) to clean up any SDPA nodes that escaped TRT compilation
    and will be executed by PyTorch.  Recurses into all submodules so that nodes
    inside nested GraphModules (e.g. from aot_autograd subgraph splitting) are
    also cleaned up.
    """
    # Recurse into submodules first.
    for _, submod in gm.named_children():
        if isinstance(submod, torch.fx.GraphModule):
            _strip_trt_sdpa_kwargs(submod)

    _TRT_SDPA_KWARGS = frozenset({"use_fp32_acc", "sliding_window_size"})
    modified = False
    for node in gm.graph.nodes:
        if (
            node.op == "call_function"
            and node.target is torch.nn.functional.scaled_dot_product_attention
        ):
            bad_kwargs = {k: v for k, v in node.kwargs.items() if k in _TRT_SDPA_KWARGS}
            if bad_kwargs:
                node.kwargs = {
                    k: v for k, v in node.kwargs.items() if k not in _TRT_SDPA_KWARGS
                }
                modified = True
    if modified:
        gm.graph.lint()
        gm.recompile()


def _sdpa_attn_mask_has_bool_ancestry(
    node: torch.fx.Node, max_depth: int = 30, visited: Any = None
) -> bool:
    """Return True if the node's computation chain uses boolean-valued operations.

    Used to distinguish a bool gate mask (built with and_ / new_ones(dtype=bool)
    / ge / etc.) from a float additive logit-bias mask.  We only inject a
    .to(dtype=bool) cast when the mask is clearly boolean-valued, so that float
    additive masks used by causal models (Llama, GPT-2, …) are left untouched.
    """
    import operator

    if visited is None:
        visited = set()
    if id(node) in visited or max_depth <= 0:
        return False
    visited.add(id(node))

    if node.op == "call_function" and node.target in (
        operator.and_,
        operator.or_,
        operator.xor,
    ):
        return True

    if node.op == "call_method" and node.target == "new_ones":
        if node.kwargs.get("dtype") == torch.bool:
            return True

    for inp in node.all_input_nodes:
        if _sdpa_attn_mask_has_bool_ancestry(inp, max_depth - 1, visited):
            return True

    return False


def _cast_bool_sdpa_attn_masks(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
    """Insert .to(dtype=torch.bool) before SDPA attn_mask when mask is boolean-valued.

    FakeTensor mode inside aot_export_joint_simple occasionally loses dtype
    tracking through Python-level boolean ops (operator.and_, new_ones, etc.),
    causing SDPA to receive an int-dtype mask and raise:
        RuntimeError: Expected attn_mask dtype to be bool or float …

    Inserting an explicit cast in the pre-AOT dynamo graph forces correct dtype
    through FakeTensor dispatch without affecting float additive masks used by
    other models.
    """
    modified = False
    for node in list(gm.graph.nodes):
        if (
            node.op == "call_function"
            and node.target is torch.nn.functional.scaled_dot_product_attention
            and "attn_mask" in node.kwargs
        ):
            mask_node = node.kwargs["attn_mask"]
            if isinstance(
                mask_node, torch.fx.Node
            ) and _sdpa_attn_mask_has_bool_ancestry(mask_node):
                with gm.graph.inserting_before(node):
                    cast_node = gm.graph.call_method(
                        "to", args=(mask_node,), kwargs={"dtype": torch.bool}
                    )
                node.kwargs = {**node.kwargs, "attn_mask": cast_node}
                modified = True
                logger.debug(
                    "Inserted bool cast on SDPA attn_mask for node %s", node.name
                )

    if modified:
        gm.graph.lint()
        gm.recompile()

    return gm


def _graph_has_higher_order_ops(gm: torch.fx.GraphModule) -> bool:
    """Return True if the graph contains vmap or other higher-order ops."""
    for node in gm.graph.nodes:
        if node.op != "call_function":
            continue
        name = str(node.target)
        if (
            "_vmap_increment_nesting" in name
            or "_add_batch_dim" in name
            or "_remove_batch_dim" in name
        ):
            return True
    return False


def _pretraced_backend(
    gm: torch.fx.GraphModule,
    sample_inputs: Sequence[Any],
    settings: CompilationSettings = CompilationSettings(),
    engine_cache: Any = None,
) -> torch.fx.GraphModule | Callable[..., Any]:
    """Helper function to manage translation of traced FX module to TRT engines

    Args:
        module: FX GraphModule to convert
        inputs: Inputs to the module
        settings: Compilation settings
        engine_cache: Engine cache instance
    Returns:
        Compiled FX GraphModule
    """
    # Save the original graph for use in the failure fallback path.  Lowering
    # passes (pre_aot_lowering, post_lowering) modify the graph in-place; if TRT
    # compilation later fails we must return an unmodified graph so that the
    # PyTorch fallback can execute it without encountering custom TRT-only kwargs
    # (e.g. use_fp32_acc) that standard torch ops don't accept.
    original_gm = gm

    try:
        logger.debug("Pre-AOT Autograd graph:\n" + str(gm.graph))

        fake_mode = detect_fake_mode(sample_inputs)

        # Fold get_attr(...).item() / placeholder(...).item() patterns into Python
        # scalars BEFORE entering FakeTensorMode.  Inside FakeTensorMode, even
        # real-tensor .item() calls raise DataDependentOutputException.  We access
        # the actual values via grapharg.example (weakref held by dynamo) which is
        # safe to dereference outside of fake mode.
        gm = fold_get_attr_item_calls(gm, sample_inputs)

        # Place backend tracing within FakeTensor context allowing nonfake Tensors
        with (
            unittest.mock.patch.object(fake_mode, "allow_non_fake_inputs", True),
            fake_mode,
        ):
            repair_input_aliasing(gm, settings)

            # Remove detach nodes
            remove_detach(gm, settings)

            # Ensure SDPA attn_mask tensors computed via boolean ops are
            # explicitly cast to bool before AOT tracing.  FakeTensor mode
            # inside aot_export_joint_simple can lose dtype tracking through
            # Python-level boolean ops (and_, new_ones, ge), causing SDPA to
            # receive an int-typed mask and fail at dispatch time.
            _cast_bool_sdpa_attn_masks(gm)

            # Invoke AOTAutograd to translate operators to aten.
            # SymInt placeholders are kept so that aot_export_joint_simple
            # can handle dynamic shapes natively (mirroring how torch
            # inductor lets aot_autograd resolve them).
            if not settings.use_distributed_mode_trace:
                gm = aot_export_joint_simple(
                    gm,
                    sample_inputs,
                    trace_joint=False,
                    decompositions=get_decompositions(
                        settings.enable_experimental_decompositions,
                        settings.decompose_attention,
                        settings.use_distributed_mode_trace,
                    ),
                )

            # Remove sym_int placeholders and inputs *after* AOT tracing
            # so we don't inject sym_size nodes that confuse the tracer.
            remove_sym_nodes(gm, sample_inputs, settings)

            torch_inputs = [
                input for input in sample_inputs if isinstance(input, torch.Tensor)
            ]

            logger.debug("Post-AOT Autograd graph:\n" + str(gm.graph))

            gm = post_lowering(gm, settings)

            logger.debug("Lowered Input graph:\n " + str(gm.graph))

            torchtrt_inputs = prepare_inputs(
                torch_inputs, disable_memory_format_check=True
            )
            if settings.require_full_compilation:
                logger.warning(
                    "require_full_compilation arg is not applicable for torch.compile with backend='torch_tensorrt"
                )
            if settings.strip_engine_weights:
                logger.warning(
                    "strip_engine_weights=True is not supported for torch.compile(). It will be set to False automatically."
                )
                settings.strip_engine_weights = False

            trt_compiled = compile_module(
                gm,
                torchtrt_inputs,
                settings=settings,
                engine_cache=engine_cache,
            )
            # Strip TRT-only SDPA kwargs from any nodes that were not captured
            # into a TRT engine (e.g. because a subgraph fell back to PyTorch
            # due to an unsupported op).  These nodes will be executed by
            # PyTorch which does not accept use_fp32_acc / sliding_window_size.
            _strip_trt_sdpa_kwargs(trt_compiled)
            return trt_compiled
    except (AssertionError, RuntimeError, TypeError):
        if not settings.pass_through_build_failures:
            logger.warning(
                "TRT conversion failed on the subgraph. See trace above. "
                + "Returning GraphModule forward instead.",
                exc_info=True,
            )
            _strip_trt_sdpa_kwargs(original_gm)
            return original_gm
        else:
            logger.critical(
                "Halting compilation on build failure since "
                + "pass_through_build_failures was specified as True. "
                + "To return the default Torch implementation and avoid "
                + "halting compilation on engine build failures, "
                + "specify pass_through_build_failures=False."
            )
            raise
