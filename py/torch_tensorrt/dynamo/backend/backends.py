from __future__ import annotations

import logging
from functools import partial
from typing import Any, Callable, Sequence

import torch
import torch._dynamo as td
from torch._functorch.aot_autograd import aot_module_simplified, make_boxed_compiler
from torch_tensorrt.dynamo import CompilationSettings
from torch_tensorrt.dynamo.compile import compile_module
from torch_tensorrt.dynamo.lowering._decompositions import get_decompositions
from torch_tensorrt.dynamo.lowering._pre_aot_lowering import pre_aot_substitutions
from torch_tensorrt.dynamo.utils import parse_dynamo_kwargs

logger = logging.getLogger(__name__)


@td.register_backend(name="torch_tensorrt")  # type: ignore[misc]
def torch_tensorrt_backend(
    gm: torch.fx.GraphModule, sample_inputs: Sequence[torch.Tensor], **kwargs: Any
) -> torch.nn.Module:
    # Set log level at the top of compilation (torch_tensorrt.dynamo)
    if (
        (
            "options" in kwargs
            and "debug" in kwargs["options"]
            and kwargs["options"]["debug"]
        )
        or ("debug" in kwargs and kwargs["debug"])
    ) and logger.parent:
        logger.parent.setLevel(logging.DEBUG)

    DEFAULT_BACKEND = aot_torch_tensorrt_aten_backend

    compiled_mod: torch.nn.Module = DEFAULT_BACKEND(gm, sample_inputs, **kwargs)
    return compiled_mod


@td.register_backend(name="aot_torch_tensorrt_aten")  # type: ignore[misc]
def aot_torch_tensorrt_aten_backend(
    gm: torch.fx.GraphModule, sample_inputs: Sequence[torch.Tensor], **kwargs: Any
) -> torch.nn.Module:
    settings = parse_dynamo_kwargs(kwargs)

    custom_backend = partial(
        _pretraced_backend,
        settings=settings,
    )

    # Perform Pre-AOT Lowering for Module-Level Replacement
    gm = pre_aot_substitutions(gm)

    # Invoke AOTAutograd to translate operators to aten
    return aot_module_simplified(
        gm,
        sample_inputs,
        fw_compiler=make_boxed_compiler(custom_backend),
        decompositions=get_decompositions(settings.enable_experimental_decompositions),
    )


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
        logger.debug("Post-AOT Autograd graph:\n" + str(gm.graph))

        trt_compiled = compile_module(
            gm,
            sample_inputs,
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
            if settings.fallback_to_inductor:
                pass
            else:
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

    # If Inductor fallback is desired, attempt model compilation with inductor
    try:
        inductor_compiled = torch._inductor.compile(
            gm,
            sample_inputs,
        )
        return inductor_compiled
    except (AssertionError, RuntimeError):
        logger.warning(
            "Inductor compilation failed on the subgraph. See trace above. "
            + "Returning GraphModule forward instead.",
            exc_info=True,
        )
        return gm
