from __future__ import annotations

import functools
import logging
import unittest
from typing import Any, Callable, Sequence

import torch
import torch._dynamo as td
from torch._dynamo.backends.common import aot_autograd
from torch._dynamo.utils import detect_fake_mode
from torch._functorch.aot_autograd import aot_export_joint_simple
from torch_tensorrt.dynamo import CompilationSettings
from torch_tensorrt.dynamo._compiler import compile_module
from torch_tensorrt.dynamo.lowering import (
    get_decompositions,
    post_lowering,
    remove_detach,
    remove_sym_nodes,
    repair_input_aliasing,
)
from torch_tensorrt.dynamo.utils import (
    parse_dynamo_kwargs,
    prepare_inputs,
    set_log_level,
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
        set_log_level(logger.parent, logging.DEBUG)

    DEFAULT_BACKEND = aot_torch_tensorrt_aten_backend

    return DEFAULT_BACKEND(gm, sample_inputs, **kwargs)


@td.register_backend(name="aot_torch_tensorrt_aten")  # type: ignore[misc]
def aot_torch_tensorrt_aten_backend(
    gm: torch.fx.GraphModule, sample_inputs: Sequence[Any], **kwargs: Any
) -> torch.nn.Module:
    settings, engine_cache = parse_dynamo_kwargs(kwargs)
    if settings.use_aot_joint_export:
        return _pretraced_backend(gm, sample_inputs, settings, engine_cache)
    logger.debug("Wrapping the backend with aot_autograd\n")
    _pretraced_backend_autograd = functools.partial(
        _pretraced_backend, settings=settings, engine_cache=engine_cache
    )
    settings_aot_autograd = {}
    settings_aot_autograd["decompostions"] = get_decompositions(
        settings.enable_experimental_decompositions
    )
    return aot_autograd(fw_compiler=_pretraced_backend_autograd)(
        gm, sample_inputs, **settings_aot_autograd
    )


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
    try:
        logger.debug("Pre-AOT Autograd graph:\n" + str(gm.graph))

        fake_mode = detect_fake_mode(sample_inputs)

        # Place backend tracing within FakeTensor context allowing nonfake Tensors
        with unittest.mock.patch.object(
            fake_mode, "allow_non_fake_inputs", True
        ), fake_mode:
            repair_input_aliasing(gm, settings)

            # Remove sym_int placeholders and inputs
            remove_sym_nodes(gm, sample_inputs, settings)

            torch_inputs = [
                input for input in sample_inputs if isinstance(input, torch.Tensor)
            ]

            # Remove detach nodes
            remove_detach(gm, settings)

            # Invoke AOTAutograd to translate operators to aten
            if settings.use_aot_joint_export:
                gm = aot_export_joint_simple(
                    gm,
                    sample_inputs,
                    trace_joint=False,
                    decompositions=get_decompositions(
                        settings.enable_experimental_decompositions
                    ),
                )

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
                logger.error(
                    "strip_engine_weights arg is not supported for torch.compile()"
                )
            trt_compiled = compile_module(
                gm,
                torchtrt_inputs,
                settings=settings,
                engine_cache=engine_cache,
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
