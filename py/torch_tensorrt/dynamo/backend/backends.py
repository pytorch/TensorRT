from __future__ import annotations

import logging
from functools import partial
from typing import Any, Callable, Sequence

import torch
import torch._dynamo as td
from torch._functorch.aot_autograd import make_boxed_compiler
from torch._guards import TracingContext
from torch_tensorrt.dynamo import CompilationSettings
from torch_tensorrt.dynamo.compile import compile_module
from torch_tensorrt.dynamo.lowering._decompositions import get_decompositions
from torch_tensorrt.dynamo.lowering._freeze_aot_graph import freeze_autograd_gm
from torch_tensorrt.dynamo.lowering._pre_aot_lowering import pre_aot_substitutions
from torch_tensorrt.dynamo.utils import parse_dynamo_kwargs

from .aot_module import aot_module

logger = logging.getLogger(__name__)


@td.register_backend(name="torch_tensorrt")  # type: ignore[misc]
def torch_tensorrt_backend(
    gm: torch.fx.GraphModule, sample_inputs: Sequence[torch.Tensor], **kwargs: Any
) -> torch.nn.Module:
    DEFAULT_BACKEND = aot_torch_tensorrt_aten_backend

    TracingContext.get().fake_mode.allow_non_fake_inputs = True

    return DEFAULT_BACKEND(gm, sample_inputs, **kwargs)


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
    return aot_module(
        gm,
        sample_inputs,
        fw_compiler=make_boxed_compiler(custom_backend),
        decompositions=get_decompositions(),
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

        frozen_gm, unfrozen_indices = freeze_autograd_gm(gm, sample_inputs)
        nonfrozen_inputs = [sample_inputs[idx] for idx in unfrozen_indices]

        frozen_gm.graph.eliminate_dead_code()
        frozen_gm.graph.lint()
        frozen_gm.recompile()

        trt_compiled = compile_module(
            frozen_gm,
            nonfrozen_inputs,
            settings=settings,
        )
        return trt_compiled
    except AssertionError:
        if not settings.pass_through_build_failures:
            logger.warning(
                "TRT conversion failed on the subgraph. See trace above. "
                + "Returning GraphModule forward instead.",
                exc_info=True,
            )
            return gm.forward
        else:
            logger.critical(
                "Halting compilation on build failure since "
                + "pass_through_build_failures was specified as True. "
                + "To return the default Torch implementation and avoid "
                + "halting compilation on engine build failures, "
                + "specify pass_through_build_failures=False."
            )
            raise
