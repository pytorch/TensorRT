import torch
import logging
import collections.abc
import torch_tensorrt
from functools import partial

from typing import Any, Optional, Sequence
from torch_tensorrt import EngineCapability, Device
from torch_tensorrt.fx.utils import LowerPrecision

from torch_tensorrt.dynamo.backend.utils import prepare_inputs, prepare_device
from torch_tensorrt.dynamo.backend.backends import torch_tensorrt_backend
from torch_tensorrt.dynamo.backend._defaults import (
    PRECISION,
    DEBUG,
    WORKSPACE_SIZE,
    MIN_BLOCK_SIZE,
    PASS_THROUGH_BUILD_FAILURES,
    MAX_AUX_STREAMS,
    VERSION_COMPATIBLE,
    OPTIMIZATION_LEVEL,
    USE_PYTHON_RUNTIME,
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
    workspace_size=WORKSPACE_SIZE,
    dla_sram_size=1048576,
    dla_local_dram_size=1073741824,
    dla_global_dram_size=536870912,
    calibrator=None,
    truncate_long_and_double=False,
    require_full_compilation=False,
    min_block_size=MIN_BLOCK_SIZE,
    torch_executed_ops=[],
    torch_executed_modules=[],
    pass_through_build_failures=PASS_THROUGH_BUILD_FAILURES,
    max_aux_streams=MAX_AUX_STREAMS,
    version_compatible=VERSION_COMPATIBLE,
    optimization_level=OPTIMIZATION_LEVEL,
    use_python_runtime=USE_PYTHON_RUNTIME,
    **kwargs,
):
    if debug:
        logger.setLevel(logging.DEBUG)

    logger.warn(
        "The Dynamo backend is an experimental feature, for which only the "
        + "following arguments are supported: "
        + "{enabled_precisions, debug, workspace_size, min_block_size, "
        + "torch_executed_ops, pass_through_build_failures}"
    )

    if not isinstance(inputs, collections.abc.Sequence):
        inputs = [inputs]

    inputs = prepare_inputs(inputs, prepare_device(device))

    if not isinstance(enabled_precisions, collections.abc.Collection):
        enabled_precisions = [enabled_precisions]

    # Parse user-specified enabled precisions
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
        pass_through_build_failures=pass_through_build_failures,
        max_aux_streams=max_aux_streams,
        version_compatible=version_compatible,
        optimization_level=optimization_level,
        use_python_runtime=use_python_runtime,
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
    workspace_size: int = WORKSPACE_SIZE,
    min_block_size: int = MIN_BLOCK_SIZE,
    torch_executed_ops: Sequence[str] = set(),
    pass_through_build_failures: bool = PASS_THROUGH_BUILD_FAILURES,
    max_aux_streams: Optional[int] = MAX_AUX_STREAMS,
    version_compatible: bool = VERSION_COMPATIBLE,
    optimization_level: Optional[int] = OPTIMIZATION_LEVEL,
    use_python_runtime: Optional[bool] = USE_PYTHON_RUNTIME,
    **kwargs,
):
    """Create torch.compile backend given specified arguments

    Args:
        precision: Model Layer precision
        debug: Whether to print out verbose debugging information
        workspace_size: Workspace TRT is allowed to use for the module (0 is default)
        min_block_size: Minimum number of operators per TRT-Engine Block
        torch_executed_ops: Sequence of operations to run in Torch, regardless of converter coverage
        pass_through_build_failures: Whether to fail on TRT engine build errors (True) or not (False)
        max_aux_streams: Maximum number of allowed auxiliary TRT streams for each engine
        version_compatible: Provide version forward-compatibility for engine plan files
        optimization_level: Builder optimization 0-5, higher levels imply longer build time,
            searching for more optimization options. TRT defaults to 3
        use_python_runtime: Whether to strictly use Python runtime or C++ runtime. To auto-select a runtime
            based on C++ dependency presence (preferentially choosing C++ runtime if available), leave the
            argument as None
    Returns:
        Backend for torch.compile
    """
    return partial(
        torch_tensorrt_backend,
        debug=debug,
        precision=precision,
        workspace_size=workspace_size,
        min_block_size=min_block_size,
        torch_executed_ops=torch_executed_ops,
        pass_through_build_failures=pass_through_build_failures,
        max_aux_streams=max_aux_streams,
        version_compatible=version_compatible,
        optimization_level=optimization_level,
        use_python_runtime=use_python_runtime,
        **kwargs,
    )
