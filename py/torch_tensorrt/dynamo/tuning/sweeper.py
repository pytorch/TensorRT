"""In-process Global Performance Tuner sweep for Dynamo TRT subgraphs."""

from __future__ import annotations

import logging
import statistics
import time
from dataclasses import replace
from typing import Any, List, Optional, Sequence, Tuple

import torch
from torch_tensorrt._Input import Input
from torch_tensorrt.dynamo._engine_cache import BaseEngineCache
from torch_tensorrt.dynamo._settings import CompilationSettings
from torch_tensorrt.dynamo.conversion._conversion import SerializedInterpreterResult
from torch_tensorrt.dynamo.tuning import cache as tuning_cache
from torch_tensorrt.dynamo.tuning._capability import (
    get_all_build_routes_raw,
    require_global_perf_tuner,
)
from torch_tensorrt.dynamo.tuning.accuracy import (
    accuracy_failed,
    compute_output_losses,
)
from torch_tensorrt.dynamo.tuning.routes import (
    BuildRouteKnobDatabase,
    expand_build_routes,
    expand_routes_mixed,
    identify_positive_knobs,
    resolve_tuning_expression,
)

_LOGGER = logging.getLogger(__name__)

_WARMUP_ITERS = 3
_BENCH_ITERS = 10


def _inputs_to_tensors(
    inputs: Sequence[Input], device: torch.device
) -> List[torch.Tensor]:
    tensors: List[torch.Tensor] = []
    for inp in inputs:
        if getattr(inp, "torch_tensor", None) is not None:
            t = inp.torch_tensor
        elif inp.shape_mode == Input._ShapeMode.STATIC:
            t = inp.example_tensor()
        else:
            t = inp.example_tensor("opt_shape")
        tensors.append(t.to(device))
    return tensors


def _benchmark_callable(
    fn: Any,
    args: Sequence[torch.Tensor],
    *,
    warmup: int = _WARMUP_ITERS,
    iters: int = _BENCH_ITERS,
) -> float:
    """Return median GPU latency in milliseconds."""
    for _ in range(max(0, warmup)):
        fn(*args)
    torch.cuda.synchronize()
    times: List[float] = []
    for _ in range(max(1, iters)):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn(*args)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return float(statistics.median(times))


def validate_tuning_options(settings: CompilationSettings) -> None:
    """Validate GPT-related CompilationSettings (trtexec-like rules)."""
    if settings.tune_build_routes and settings.tune_build_route_file:
        raise ValueError(
            "Cannot specify both tune_build_routes and tune_build_route_file."
        )
    if settings.tuning_continue:
        if not settings.tuning_cache_file:
            raise ValueError("tuning_continue requires tuning_cache_file.")
        if (
            settings.tune_build_routes
            or settings.tune_build_route_file
            or settings.tuning_dry_run
        ):
            raise ValueError(
                "tuning_continue cannot be combined with tune_build_routes, "
                "tune_build_route_file, or tuning_dry_run; recover the sweep "
                "from tuning_cache_file."
            )
    if settings.tuning_dry_run and settings.tuning_search == "mixed":
        raise ValueError("tuning_dry_run is incompatible with tuning_search='mixed'.")
    if settings.accuracy_atol != 1e-5 or settings.accuracy_rtol != 1e-5:
        if settings.accuracy_algorithm.lower() != "l0":
            raise ValueError(
                "accuracy_atol/accuracy_rtol are only valid when accuracy_algorithm='l0'."
            )


def should_run_tuning(settings: CompilationSettings) -> bool:
    if settings.tuning_continue:
        return True
    if settings.tune_build_routes or settings.tune_build_route_file:
        return True
    return False


def tune_subgraph(
    module: torch.fx.GraphModule,
    inputs: Sequence[Input],
    settings: CompilationSettings,
    engine_cache: Optional[BaseEngineCache] = None,
    *,
    input_binding_names: Optional[Sequence[str]] = None,
    output_binding_names: Optional[Sequence[str]] = None,
) -> SerializedInterpreterResult:
    """Sweep build routes and return the best SerializedInterpreterResult."""
    from torch_tensorrt.dynamo.conversion._conversion import (
        _interpret_module_to_result_impl,
    )
    from torch_tensorrt.dynamo.runtime import TorchTensorRTModule

    require_global_perf_tuner("Global Performance Tuner sweep")
    validate_tuning_options(settings)

    # Per-partition cache path so multi-subgraph compiles do not overwrite each other.
    user_cache_file = settings.tuning_cache_file
    cache_file = tuning_cache.resolve_partition_tuning_cache_path(
        user_cache_file, module
    )
    if cache_file and cache_file != user_cache_file:
        _LOGGER.info(
            f"Using per-partition tuning cache {cache_file} (from {user_cache_file})",
        )

    db = BuildRouteKnobDatabase()
    raw = get_all_build_routes_raw()
    if not db.load_from_json(raw):
        raise RuntimeError("Failed to load Global Performance Tuner knob database.")

    start_iter = 0
    if settings.tuning_continue:
        assert cache_file is not None
        header = tuning_cache.read_cache(cache_file)
        expression = header.tuning_expr
        start_iter = header.completed_iterations
        _LOGGER.info(
            f"Resuming tuning from {cache_file} at iteration {start_iter}",
        )
    else:
        expression = resolve_tuning_expression(
            settings.tune_build_routes, settings.tune_build_route_file
        )
        if not expression:
            raise ValueError("tune_build_routes expression is empty.")

    exprs, routes = expand_build_routes(
        expression,
        settings.tuning_search,
        db,
        dry_run=settings.tuning_dry_run,
    )

    if settings.tuning_dry_run:
        for i, route in enumerate(routes):
            _LOGGER.info(f"[Tuning Dry Run] iter={i} BuildRoute = '{route}'")
        raise RuntimeError(
            f"tuning_dry_run enumerated {len(routes)} build routes; "
            "no engines were built."
        )

    if cache_file and not settings.tuning_continue:
        tuning_cache.write_header(
            cache_file,
            {
                "tuner_version": db.tuner_version,
                "accuracy_algorithm": settings.accuracy_algorithm,
                "accuracy_parameter": {
                    "atol": settings.accuracy_atol,
                    "rtol": settings.accuracy_rtol,
                    "epsilon": settings.accuracy_threshold,
                },
                "searching_algorithm": settings.tuning_search,
                "tuning_expr": expression,
                "default_build_route": db.build_default_path(),
                "partition_key": tuning_cache.subgraph_partition_key(module),
                "user_tuning_cache_file": user_cache_file,
            },
        )

    device = torch.device(
        f"cuda:{settings.device.gpu_id}"
        if settings.device.gpu_id is not None
        else "cuda"
    )
    sample_args = _inputs_to_tensors(inputs, device)
    module = module.to(device).eval()
    with torch.no_grad():
        ref_outputs = module(*sample_args)

    best_result = None
    best_time: Optional[float] = None
    best_route = ""
    gpu_times: List[Optional[float]] = []
    sweep_start = time.monotonic()

    def _trial(
        route: str, iter_idx: int, *, record_cache: bool = True
    ) -> Tuple[Optional[Any], Optional[float]]:
        nonlocal best_result, best_time, best_route
        _LOGGER.info(f"&&&& TASK_BEGIN [iter={iter_idx}] BuildRoute = '{route}'")
        trial_settings = replace(
            settings,
            build_route=route,
            tune_build_routes="",
            tune_build_route_file=None,
            tuning_continue=False,
            tuning_dry_run=False,
            reuse_cached_engines=False,
            cache_built_engines=False,
        )
        crashed = False
        error_message = ""
        accuracy_loss = None
        gpu_time: Optional[float] = None
        result = None
        try:
            result = _interpret_module_to_result_impl(
                module,
                inputs,
                trial_settings,
                engine_cache=None,
                input_binding_names=input_binding_names,
                output_binding_names=output_binding_names,
            )
            trt_mod = TorchTensorRTModule(
                serialized_engine=result.serialized_engine,
                input_binding_names=list(result.input_names),
                output_binding_names=list(result.output_names),
                name=f"tune_iter_{iter_idx}",
                settings=trial_settings,
                requires_output_allocator=result.requires_output_allocator,
                requires_native_multidevice=result.requires_native_multidevice,
                symbolic_shape_expressions=result.symbolic_shape_expressions,
                aliased_io=result.aliased_io,
            )
            trt_mod.eval()
            with torch.no_grad():
                actual = trt_mod(*sample_args)
                if settings.accuracy_threshold is not None:
                    accuracy_loss = compute_output_losses(
                        actual,
                        ref_outputs,
                        algorithm=settings.accuracy_algorithm,
                        atol=settings.accuracy_atol,
                        rtol=settings.accuracy_rtol,
                    )
                    if accuracy_failed(accuracy_loss, settings.accuracy_threshold):
                        error_message = f"accuracy threshold exceeded: {accuracy_loss}"
                        _LOGGER.warning(
                            "iter=%d route failed accuracy: %s",
                            iter_idx,
                            accuracy_loss,
                        )
                    else:
                        gpu_time = _benchmark_callable(trt_mod, sample_args)
                else:
                    gpu_time = _benchmark_callable(trt_mod, sample_args)

            if (
                gpu_time is not None
                and not error_message
                and (best_time is None or gpu_time < best_time)
            ):
                best_time = gpu_time
                best_result = result
                best_route = route
            del trt_mod
            torch.cuda.empty_cache()
            _LOGGER.info(f"&&&& TASK_END [iter={iter_idx}] BuildRoute = '{route}'")
        except Exception as exc:
            crashed = True
            _LOGGER.warning(
                f"&&&& TASK_ABORT [iter={iter_idx}] BuildRoute = '{route}': {str(exc)}"
            )

        if record_cache and cache_file:
            tuning_cache.append_iteration(
                cache_file,
                iter_idx=iter_idx,
                build_route=route,
                crashed=crashed,
                error_message=error_message,
                accuracy_loss=accuracy_loss,
                gpu_time_ms=gpu_time,
            )
        return result, gpu_time

    if settings.tuning_continue and start_iter > 0 and cache_file:
        cached_times = tuning_cache.read_iteration_gpu_times(cache_file, start_iter)
        gpu_times.extend(cached_times)
        best_cached_idx = None
        best_cached_time: Optional[float] = None
        for idx, t in enumerate(cached_times):
            if t is not None and (best_cached_time is None or t < best_cached_time):
                best_cached_time = t
                best_cached_idx = idx
        if best_cached_idx is not None and best_cached_idx < len(routes):
            _LOGGER.info(
                f"Rebuilding best cached route from iter {best_cached_idx} for resume",
            )
            # TODO (@Evan): Consider rebuilding the cached best engine later if it's still the best after the remaining trials.
            # Need to think about how to deal with timeout.
            _trial(routes[best_cached_idx], best_cached_idx, record_cache=False)

    for i, route in enumerate(routes):
        if i < start_iter:
            continue
        if settings.tuning_timeout_s >= 0:
            elapsed = time.monotonic() - sweep_start
            if elapsed >= settings.tuning_timeout_s:
                _LOGGER.info(
                    f"Tuning timeout reached after {elapsed:.1f}s; stopping before iter {i}",
                )
                break
        _, gpu_time = _trial(route, i)
        gpu_times.append(gpu_time)

    if settings.tuning_search == "mixed" and start_iter < len(routes):
        # Only run phase-2 if we completed (or nearly) phase-1
        if len(gpu_times) >= len(routes):
            positive = identify_positive_knobs(exprs, gpu_times[: len(routes)], db)
            phase2 = expand_routes_mixed(exprs, db, positive)
            # Skip routes already evaluated in phase 1
            phase1_set = set(routes)
            phase2_new = [r for r in phase2 if r not in phase1_set]
            base_idx = len(routes)
            for j, route in enumerate(phase2_new):
                iter_idx = base_idx + j
                if settings.tuning_timeout_s >= 0:
                    elapsed = time.monotonic() - sweep_start
                    if elapsed >= settings.tuning_timeout_s:
                        _LOGGER.info(
                            f"Tuning timeout reached during mixed phase-2 at iter {iter_idx}",
                        )
                        break
                _trial(route, iter_idx)

    if best_result is None:
        raise RuntimeError(
            "Global Performance Tuner sweep completed without a valid engine "
            "(all routes crashed or failed accuracy checks)."
        )

    _LOGGER.info(
        f"Selected best build route '{best_route}' with gpu_time={best_time if best_time is not None else float('nan'):.3f} ms"
    )
    # TODO (@Evan): Consider persisting winner on settings for engine-cache hashing / introspection
    settings.build_route = best_route
    return best_result
