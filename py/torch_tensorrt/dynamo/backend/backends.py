from __future__ import annotations

import logging
import unittest
from typing import Any, Callable, Dict, Optional, Sequence

import torch
import torch._dynamo as td
import torch.utils._pytree as pytree
from torch._dynamo.utils import detect_fake_mode
from torch._functorch.aot_autograd import _aot_export_function
from torch._ops import OpOverload
from torch_tensorrt.dynamo import CompilationSettings
from torch_tensorrt.dynamo.compile import compile_module
from torch_tensorrt.dynamo.lowering import apply_lowering_passes, get_decompositions
from torch_tensorrt.dynamo.lowering._pre_aot_lowering import pre_aot_substitutions
from torch_tensorrt.dynamo.utils import (
    parse_dynamo_kwargs,
    prepare_inputs,
    set_log_level,
)

logger = logging.getLogger(__name__)


@td.register_backend(name="tensorrt")  # type: ignore[misc]
@td.register_backend(name="torch_tensorrt")  # type: ignore[misc]
def torch_tensorrt_backend(
    gm: torch.fx.GraphModule, sample_inputs: Sequence[torch.Tensor], **kwargs: Any
) -> torch.nn.Module:
    # Set log level at the top of compilation (torch_tensorrt.dynamo)
    if (
        "options" in kwargs
        and "debug" in kwargs["options"]
        and kwargs["options"]["debug"]
    ) or ("debug" in kwargs and kwargs["debug"]):
        set_log_level(logger.parent, logging.DEBUG)

    DEFAULT_BACKEND = aot_torch_tensorrt_aten_backend

    return DEFAULT_BACKEND(gm, sample_inputs, **kwargs)


@td.register_backend(name="aot_torch_tensorrt_aten")  # type: ignore[misc]
def aot_torch_tensorrt_aten_backend(
    gm: torch.fx.GraphModule, sample_inputs: Sequence[torch.Tensor], **kwargs: Any
) -> torch.nn.Module:
    settings = parse_dynamo_kwargs(kwargs)
    return _pretraced_backend(gm, sample_inputs, settings)


def _pretraced_backend(
    gm: torch.fx.GraphModule,
    sample_inputs: Sequence[torch.Tensor],
    settings: CompilationSettings = CompilationSettings(),
) -> torch.fx.GraphModule | Callable[..., Any]:
    """Helper function to manage translation of traced FX module to TRT engines

    Args:
        module: FX GraphModule to convert
        inputs: Inputs to the module
        settings: Compilation settings
    Returns:
        Compiled FX GraphModule
    """
    try:
        logger.debug("Pre-AOT Autograd graph:\n" + str(gm.graph))

        # Perform Pre-AOT Lowering for Module-Level Replacement
        gm = pre_aot_substitutions(gm)

        fake_mode = detect_fake_mode(sample_inputs)

        # Place backend tracing within FakeTensor context allowing nonfake Tensors
        with unittest.mock.patch.object(
            fake_mode, "allow_non_fake_inputs", True
        ), fake_mode:
            # Invoke AOTAutograd to translate operators to aten
            gm = aot_export_for_compile(
                gm,
                sample_inputs,
                decompositions=get_decompositions(
                    settings.enable_experimental_decompositions
                ),
            )

            logger.debug("Post-AOT Autograd graph:\n" + str(gm.graph))

            gm = apply_lowering_passes(gm)

            torchtrt_inputs = prepare_inputs(sample_inputs)
            trt_compiled = compile_module(
                gm,
                torchtrt_inputs,
                settings=settings,
            )
            return trt_compiled
    except (AssertionError, RuntimeError):
        if not settings.pass_through_build_failures:
            logger.warning(
                "TRT conversion failed on the subgraph. See trace above. "
                + "Returning GraphModule forward instead.",
                exc_info=True,
            )
            return gm
        else:
            logger.critical(
                "Halting compilation on build failure since "
                + "pass_through_build_failures was specified as True. "
                + "To return the default Torch implementation and avoid "
                + "halting compilation on engine build failures, "
                + "specify pass_through_build_failures=False."
            )
            raise


def aot_export_for_compile(
    func: torch.fx.GraphModule,
    args: Sequence[torch.Tensor],
    *,
    decompositions: Optional[Dict[OpOverload, Callable[[Any], Any]]] = None,
) -> torch.fx.GraphModule:
    """Adapted from:
    https://github.com/pytorch/pytorch/blob/1a5fdc2458b98697c75c32eb6f4b8b34d76429cf/torch/_functorch/aot_autograd.py#L4084-L4158

    Removed check for input aliasing in resultant subgraph - TRT is functional-only

    Exports the function to ATen for torch compile
    """
    # Trace function with input arguments and decompositions
    with torch.no_grad():
        fx_g, metadata, in_spec, out_spec = _aot_export_function(
            func,
            args,
            decompositions=decompositions,
        )

    # No input mutations
    if (
        len([x for x in metadata.input_info if x.mutates_data or x.mutates_metadata])
        != 0
    ):
        raise RuntimeError(
            f"aot_export_joint_simple does not support input mutations. {str(metadata)}"
        )
    # No pytrees
    if type(in_spec) == pytree.LeafSpec:
        raise RuntimeError(
            f"aot_export_for_compile requires inputs to be a single list/tuple. in_spec={str(in_spec)}"
        )
    if len([x for x in in_spec.children_specs if type(x) != pytree.LeafSpec]) != 0:
        raise RuntimeError(
            f"aot_export_for_compile requires individual inputs not to be pytrees. in_spec={str(in_spec)}"
        )
    if type(out_spec) == pytree.LeafSpec:
        raise RuntimeError(
            f"aot_export_for_compile requires outputs to be a single list/tuple. out_spec={str(out_spec)}"
        )
    if len([x for x in out_spec.children_specs if type(x) != pytree.LeafSpec]) != 0:
        raise RuntimeError(
            f"aot_export_for_compile requires individual outputs not to be pytrees. out_spec={str(out_spec)}"
        )

    return fx_g
