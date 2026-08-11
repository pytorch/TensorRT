"""TensorRT Global Performance Tuner support for Torch-TensorRT Dynamo."""

from torch_tensorrt.dynamo.tuning._capability import (
    get_all_build_routes,
    get_all_build_routes_raw,
    gpt_settings_requested,
    is_global_perf_tuner_available,
    require_global_perf_tuner,
)
from torch_tensorrt.dynamo.tuning.accuracy import (
    compute_output_losses,
    compute_tensor_loss,
    loss_cos,
    loss_l0,
    loss_l1,
    loss_l2,
    loss_linf,
)
from torch_tensorrt.dynamo.tuning.cache import (
    resolve_partition_tuning_cache_path,
    subgraph_partition_key,
)
from torch_tensorrt.dynamo.tuning.routes import (
    BuildRouteExprParser,
    BuildRouteKnobDatabase,
    expand_build_routes,
    expand_routes_fast,
    expand_routes_full,
    expand_routes_mixed,
)
from torch_tensorrt.dynamo.tuning.sweeper import (
    should_run_tuning,
    tune_subgraph,
    validate_tuning_options,
)

__all__ = [
    "BuildRouteExprParser",
    "BuildRouteKnobDatabase",
    "compute_output_losses",
    "compute_tensor_loss",
    "expand_build_routes",
    "expand_routes_fast",
    "expand_routes_full",
    "expand_routes_mixed",
    "get_all_build_routes",
    "get_all_build_routes_raw",
    "gpt_settings_requested",
    "is_global_perf_tuner_available",
    "loss_cos",
    "loss_l0",
    "loss_l1",
    "loss_l2",
    "loss_linf",
    "require_global_perf_tuner",
    "resolve_partition_tuning_cache_path",
    "should_run_tuning",
    "subgraph_partition_key",
    "tune_subgraph",
    "validate_tuning_options",
]
