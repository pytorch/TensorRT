import torch
import logging
import collections.abc
import torch_tensorrt
from functools import partial

from typing import Any, Optional, Sequence
from torch_tensorrt import EngineCapability, Device
from torch_tensorrt.fx.utils import LowerPrecision
from torch.fx.passes.pass_manager import inplace_wrapper, PassManager
from torch.fx.passes.shape_prop import ShapeProp
import torch_tensorrt.fx.tracer.dispatch_tracer.aten_tracer as aten_tracer
from torch_tensorrt.fx.tools.trt_splitter import TRTSplitter, TRTSplitterSetting
from torch_tensorrt.fx.passes.remove_duplicate_output_args import (
    remove_duplicate_output_args,
)
from torch_tensorrt.dynamo.backend.lowering import (
    fuse_permute_linear,
    fuse_permute_matmul,
)
from torch_tensorrt.dynamo import CompilationSettings
from torch_tensorrt.dynamo.utils import prepare_inputs, prepare_device
from torch_tensorrt.dynamo.backend.backends import torch_tensorrt_backend
from torch_tensorrt.dynamo.backend.conversion import convert_module

from torch_tensorrt.dynamo._defaults import (
    PRECISION,
    DEBUG,
    WORKSPACE_SIZE,
    MIN_BLOCK_SIZE,
    PASS_THROUGH_BUILD_FAILURES,
    MAX_AUX_STREAMS,
    VERSION_COMPATIBLE,
    OPTIMIZATION_LEVEL,
    USE_EXPERIMENTAL_RT,
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
    max_aux_streams=MAX_AUX_STREAMS,
    version_compatible=VERSION_COMPATIBLE,
    optimization_level=OPTIMIZATION_LEVEL,
    use_experimental_rt=USE_EXPERIMENTAL_RT,
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

    if "use_experimental_fx_rt" in kwargs:
        use_experimental_rt = kwargs["use_experimental_fx_rt"]

    logger.info(f"Using {'C++' if use_experimental_rt else 'Python'} TRT Runtime")

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

    settings = CompilationSettings(
        debug=debug,
        precision=lower_precision,
        workspace_size=workspace_size,
        min_block_size=min_block_size,
        torch_executed_ops=torch_executed_ops,
        pass_through_build_failures=False,
        max_aux_streams=max_aux_streams,
        version_compatible=version_compatible,
        optimization_level=optimization_level,
        use_experimental_rt=use_experimental_rt,
    )

    model = trace(gm, inputs, **kwargs)

    if kwargs.get("use_capability_partitioner", None):
        traced_model = trace(model)
        model = lower_model(traced_model, inputs)
        return _compile_module(model, inputs, settings)
    else:
        split_result = lower_model_using_trt_splitter(model, inputs)
        trt_module = _compile_graph(split_result, inputs, settings)

        return trt_module


def _compile_graph(
    split_result: TRTSplitter,
    inputs: Any,
    settings: CompilationSettings = CompilationSettings(),
    **kwargs,
):

    for submod_name, submod_inputs in split_result.submodule_inputs.items():
        submod = getattr(split_result.split_module, submod_name)
        # Only acc submodules will be lowered.
        if not submod_name.startswith(split_result.non_acc_submodule_prefix):
            # Create TRT Module from submodule
            trt_mod = convert_module(
                submod,
                submod_inputs,
                settings=settings,
                name=submod_name,
            )
            setattr(split_result.split_module, submod_name, trt_mod)

    return split_result.split_module


def trace(
    model: torch.nn.Module,
    inputs: Any,
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
        use_experimental_rt: Whether to use the new experimental TRTModuleNext for TRT engines
    Returns:
        Backend for torch.compile
    """
    model = aten_tracer.opt_trace(model, inputs)

    return model


def lower_model_using_trt_splitter(model: torch.nn.Module, inputs: Any, **kwargs):
    # Perform basic lowering
    model = lower_model(model, inputs)
    splitter_setting = TRTSplitterSetting()
    splitter_setting.use_implicit_batch_dim = False
    splitter_setting.min_acc_module_size = 1
    splitter_setting.use_experimental_rt = False
    splitter = TRTSplitter(model, inputs, settings=splitter_setting)
    splitter.node_support_preview()
    split_result = splitter.generate_split_results()

    return split_result


def lower_model(model: torch.nn.Module, inputs: Any, **kwargs):

    graph_optimization_pm = PassManager.build_from_passlist(
        [fuse_permute_matmul, fuse_permute_linear]
    )
    lowered_model = graph_optimization_pm(model)
    if isinstance(lowered_model, torch.fx.GraphModule):
        ShapeProp(lowered_model).propagate(*inputs)

    return lowered_model
