import torch
import logging
import collections.abc
import torch_tensorrt
from functools import partial

from typing import Any, Sequence
from torch_tensorrt import EngineCapability, Device
from torch_tensorrt.fx.utils import LowerPrecision

from torch_tensorrt.dynamo.backend._settings import CompilationSettings
from torch_tensorrt.dynamo.backend.utils import prepare_inputs, prepare_device
from torch_tensorrt.dynamo.backend.backends import torch_tensorrt_backend
from torch_tensorrt.dynamo.backend._defaults import (
    PRECISION,
    DEBUG,
    MAX_WORKSPACE_SIZE,
    MIN_BLOCK_SIZE,
)


logger = logging.getLogger(__name__)


def compile(
    gm: torch.nn.Module,
    inputs: Any,
    *,
    device=Device._current_device(),
    disable_tf32=False,
    sparse_weights=False,
    enabled_precisions=set(),
    refit=False,
    debug=DEBUG,
    capability=EngineCapability.default,
    num_avg_timing_iters=1,
    workspace_size=MAX_WORKSPACE_SIZE,
    dla_sram_size=1048576,
    dla_local_dram_size=1073741824,
    dla_global_dram_size=536870912,
    calibrator=None,
    truncate_long_and_double=False,
    require_full_compilation=False,
    min_block_size=MIN_BLOCK_SIZE,
    torch_executed_ops=[],
    torch_executed_modules=[],
    **kwargs,
):
    if debug:
        logger.setLevel(logging.DEBUG)

    logger.warn(
        "The Dynamo backend is an experimental feature, for which only the "
        + "following arguments are supported: "
        + "{enabled_precisions, debug, workspace_size, min_block_size, torch_executed_ops}"
    )

    if not isinstance(inputs, collections.abc.Sequence):
        inputs = [inputs]

    inputs = prepare_inputs(inputs, prepare_device(device))

    if (
        torch.float16 in enabled_precisions
        or torch_tensorrt.dtype.half in enabled_precisions
    ):
        lower_precision = LowerPrecision.FP16
    elif (
        torch.float32 in enabled_precisions
        or torch_tensorrt.dtype.float in enabled_precisions
    ):
        lower_precision = LowerPrecision.FP32
    elif len(enabled_precisions) == 0:
        logger.info(f"No precision specified, defaulting to {PRECISION}")
        lower_precision = PRECISION
    else:
        raise ValueError(
            f"Precision {enabled_precisions} not supported in the Dynamo Path"
        )

    custom_backend = create_backend(
        precision=lower_precision,
        debug=debug,
        workspace_size=workspace_size,
        min_block_size=min_block_size,
        torch_executed_ops=torch_executed_ops,
        **kwargs,
    )

    model = torch.compile(gm, backend=custom_backend)

    # Ensure compilation occurs by calling the function with provided inputs
    model(*inputs)

    return model


from torch_tensorrt.fx.utils import LowerPrecision

logger = logging.getLogger(__name__)


def create_backend(
    precision: LowerPrecision = PRECISION,
    debug: bool = DEBUG,
    workspace_size: int = MAX_WORKSPACE_SIZE,
    min_block_size: int = MIN_BLOCK_SIZE,
    torch_executed_ops: Sequence[str] = set(),
    **kwargs,
):
    """Create torch.compile backend given specified arguments

    Args:
        precision:
        debug: Whether to print out verbose debugging information
        workspace_size: Maximum workspace TRT is allowed to use for the module
        precision: Model Layer precision
    Returns:
        Backend for torch.compile
    """
    settings = CompilationSettings(
        debug=debug,
        precision=precision,
        workspace_size=workspace_size,
        min_block_size=min_block_size,
        torch_executed_ops=torch_executed_ops,
    )

    return partial(
        torch_tensorrt_backend,
        settings=settings,
    )
