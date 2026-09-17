"""Global Performance Tuning sweep for Dynamo TRT subgraphs.

Each candidate ``build_route`` is built and measured in a **spawned child
process**, matching trtexec's per-trial isolation. TensorRT / driver aborts
(SIGABRT, CUDA context death, etc.) kill only the child; the parent records
``crash=True`` and continues the sweep.
"""

from __future__ import annotations

import logging
import math
import multiprocessing
import os
import pickle
import statistics
import tempfile
import time
from dataclasses import replace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch_tensorrt._Input import Input
from torch_tensorrt.dynamo._engine_cache import BaseEngineCache
from torch_tensorrt.dynamo._settings import CompilationSettings
from torch_tensorrt.dynamo.conversion._conversion import SerializedInterpreterResult
from torch_tensorrt.dynamo.tuning import cache as tuning_cache
from torch_tensorrt.dynamo.tuning._capability import (
    get_all_build_routes_raw,
    require_global_perf_tuning,
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
_SAMPLE_SEED = 0


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


def _materialize_sample_tensors(inputs: Sequence[Input]) -> List[torch.Tensor]:
    """Create one CPU sample set which is reused by every spawned trial."""
    # A fixed local seed makes generated Input samples reproducible across
    # continuation invocations without mutating the caller's global RNG state.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(_SAMPLE_SEED)
        return [
            tensor.detach().to("cpu").clone()
            for tensor in _inputs_to_tensors(inputs, torch.device("cpu"))
        ]


def _tensor_tree_to_device(tree: Any, device: torch.device) -> Any:
    """Move a nested tensor output to a device while preserving its structure."""
    if isinstance(tree, torch.Tensor):
        return tree.detach().to(device)
    if isinstance(tree, dict):
        return {
            key: _tensor_tree_to_device(value, device) for key, value in tree.items()
        }
    if isinstance(tree, list):
        return [_tensor_tree_to_device(value, device) for value in tree]
    if isinstance(tree, tuple):
        return tuple(_tensor_tree_to_device(value, device) for value in tree)
    raise TypeError(f"Unsupported output type for accuracy: {type(tree)}")


def _compute_reference_outputs(
    module: torch.fx.GraphModule,
    sample_tensors: Sequence[torch.Tensor],
    device: torch.device,
) -> Any:
    """Compute reference outputs once and return a CPU tensor tree."""
    module = module.to(device).eval()
    sample_args = [tensor.to(device) for tensor in sample_tensors]
    with torch.no_grad():
        outputs = module(*sample_args)
    return _tensor_tree_to_device(outputs, torch.device("cpu"))


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


def _child_exit_message(exitcode: Optional[int]) -> str:
    if exitcode is None:
        return "child process did not exit"
    if exitcode < 0:
        return f"child process terminated by signal {-exitcode}"
    return f"child process exited with code {exitcode}"


def _tensor_as_cpu_empty(t: torch.Tensor) -> torch.Tensor:
    """Replace FakeTensors (unpicklable FakeTensorMode) with a real CPU tensor."""
    sizes = []
    for dim in t.shape:
        if isinstance(dim, torch.SymInt):
            hint = getattr(dim.node, "hint", None)
            sizes.append(int(hint) if hint is not None else 1)
        else:
            sizes.append(int(dim))
    return torch.empty(tuple(sizes), dtype=t.dtype, device="cpu")


def _sanitize_meta_value(value: Any) -> Any:
    """Drop or rewrite FX meta values that cannot be pickled into the child."""
    if isinstance(value, torch.SymInt):
        hint = getattr(value.node, "hint", None)
        return int(hint) if hint is not None else 0
    if isinstance(value, torch.SymFloat):
        hint = getattr(value.node, "hint", None)
        return float(hint) if hint is not None else 0.0
    if isinstance(value, torch.Tensor):
        if type(value) is torch.Tensor:
            return value.detach().to("cpu")
        return _tensor_as_cpu_empty(value)
    if isinstance(value, dict):
        return {
            k: _sanitize_meta_value(v)
            for k, v in value.items()
            if _meta_value_is_picklable_after_sanitize(v)
        }
    if isinstance(value, (list, tuple)):
        converted = [_sanitize_meta_value(v) for v in value]
        return type(value)(converted)
    return value


def _meta_value_is_picklable_after_sanitize(value: Any) -> bool:
    try:
        pickle.dumps(_sanitize_meta_value(value))
        return True
    except Exception:
        return False


def _sanitize_meta_dict(meta: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in meta.items():
        try:
            sanitized = _sanitize_meta_value(value)
            pickle.dumps(sanitized)
        except Exception:
            _LOGGER.debug(
                "Skipping unpicklable FX meta key %s (%s)", key, type(value).__name__
            )
            continue
        out[key] = sanitized
    return out


def _snapshot_fx_meta(module: torch.fx.GraphModule) -> Dict[str, Any]:
    """FX pickle/torch.save drops ``node.meta``; snapshot picklable fields for the child.

    Node names can change across ``torch.save``/``torch.load`` (e.g. ``relu`` ->
    ``relu_default``), so metas are stored in graph order rather than by name.
    """
    return {
        "module_meta": _sanitize_meta_dict(dict(getattr(module, "meta", {}) or {})),
        "node_meta_list": [
            _sanitize_meta_dict(dict(node.meta)) for node in module.graph.nodes
        ],
    }


def _restore_fx_meta(module: torch.fx.GraphModule, snapshot: Dict[str, Any]) -> None:
    module_meta = snapshot["module_meta"]
    if module_meta:
        if not hasattr(module, "meta") or module.meta is None:
            module.meta = {}
        module.meta.update(module_meta)
    node_meta_list = snapshot["node_meta_list"]
    nodes = list(module.graph.nodes)
    if len(node_meta_list) != len(nodes):
        raise ValueError(
            "FX meta snapshot node count does not match the deserialized graph: "
            f"{len(node_meta_list)} != {len(nodes)}"
        )
    for node, saved in zip(nodes, node_meta_list):
        if saved:
            node.meta.update(saved)


def _empty_trial_result(
    *, crashed: bool, error_message: str = "", gpu_time: Optional[float] = None
) -> Dict[str, Any]:
    return {
        "crashed": crashed,
        "error_message": error_message,
        "accuracy_loss": None,
        "gpu_time": gpu_time,
        "serialized_engine": None,
        "input_names": [],
        "output_names": [],
        "requires_output_allocator": False,
        "symbolic_shape_expressions": {},
        "requires_native_multidevice": False,
        "aliased_io": {},
    }


def _serialized_result_from_trial(
    trial: Dict[str, Any],
) -> Optional[SerializedInterpreterResult]:
    engine = trial.get("serialized_engine")
    if trial.get("crashed") or engine is None:
        return None
    return SerializedInterpreterResult(
        serialized_engine=engine,
        input_names=list(trial.get("input_names") or []),
        output_names=list(trial.get("output_names") or []),
        requires_output_allocator=bool(trial.get("requires_output_allocator", False)),
        symbolic_shape_expressions=dict(trial.get("symbolic_shape_expressions") or {}),
        requires_native_multidevice=bool(
            trial.get("requires_native_multidevice", False)
        ),
        aliased_io=dict(trial.get("aliased_io") or {}),
    )


def _trial_result_validation_error(result: Dict[str, Any]) -> Optional[str]:
    """This is a sanity check to ensure the result will not crash the parent process."""
    if not isinstance(result.get("crashed"), bool):
        return "'crashed' is not a bool"
    if not isinstance(result.get("error_message", ""), str):
        return "'error_message' is not a string"
    gpu_time = result.get("gpu_time")
    if gpu_time is not None and (
        isinstance(gpu_time, bool) or not isinstance(gpu_time, (int, float))
    ):
        return "'gpu_time' is not numeric"
    losses = result.get("accuracy_loss")
    if losses is not None and (
        not isinstance(losses, dict)
        or any(
            not isinstance(name, str)
            or isinstance(value, bool)
            or not isinstance(value, (int, float))
            for name, value in losses.items()
        )
    ):
        return "'accuracy_loss' is malformed"
    engine = result.get("serialized_engine")
    if engine is not None and not isinstance(engine, bytes):
        return "'serialized_engine' is not bytes"
    successful = not result["crashed"] and not result.get("error_message")
    if successful and engine is None:
        return "successful result has no serialized engine"
    for field in ("input_names", "output_names"):
        names = result.get(field)
        if not isinstance(names, list) or any(
            not isinstance(name, str) for name in names
        ):
            return f"'{field}' is not a list of strings"
    for field in ("requires_output_allocator", "requires_native_multidevice"):
        if not isinstance(result.get(field), bool):
            return f"'{field}' is not a bool"
    symbolic_shapes = result.get("symbolic_shape_expressions")
    if not isinstance(symbolic_shapes, dict) or any(
        not isinstance(name, str)
        or not isinstance(expressions, list)
        or any(not isinstance(expression, dict) for expression in expressions)
        for name, expressions in symbolic_shapes.items()
    ):
        return "'symbolic_shape_expressions' is malformed"
    aliased_io = result.get("aliased_io")
    if not isinstance(aliased_io, dict) or any(
        not isinstance(output_name, str)
        or not isinstance(alias, (tuple, list))
        or len(alias) != 2
        or any(not isinstance(value, str) for value in alias)
        for output_name, alias in aliased_io.items()
    ):
        return "'aliased_io' is malformed"
    return None


def _execute_route_trial(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Build, check accuracy, and bench one route. Runs in the child process."""
    from torch_tensorrt.dynamo.conversion._conversion import (
        _interpret_module_to_result_impl,
    )
    from torch_tensorrt.dynamo.runtime import TorchTensorRTModule

    settings: CompilationSettings = payload["settings"]
    inputs: Sequence[Input] = payload["inputs"]
    module: torch.fx.GraphModule = payload["module"]
    _restore_fx_meta(module, payload["fx_meta"])
    route: str = payload["route"]
    iter_idx: int = payload["iter_idx"]
    gpu_id = payload.get("gpu_id")
    device = torch.device(f"cuda:{gpu_id}" if gpu_id is not None else "cuda")
    if gpu_id is not None:
        torch.cuda.set_device(gpu_id)

    module = module.to(device).eval()
    sample_args = [tensor.to(device) for tensor in payload["sample_tensors"]]
    reference_outputs = None
    if settings.accuracy_threshold is not None:
        reference_outputs = _tensor_tree_to_device(payload["reference_outputs"], device)

    result = _interpret_module_to_result_impl(
        module,
        inputs,
        settings,
        engine_cache=None,
        input_binding_names=payload.get("input_binding_names"),
        output_binding_names=payload.get("output_binding_names"),
    )
    trt_mod = TorchTensorRTModule(
        serialized_engine=result.serialized_engine,
        input_binding_names=list(result.input_names),
        output_binding_names=list(result.output_names),
        name=f"tune_iter_{iter_idx}",
        settings=settings,
        requires_output_allocator=result.requires_output_allocator,
        requires_native_multidevice=result.requires_native_multidevice,
        symbolic_shape_expressions=result.symbolic_shape_expressions,
        aliased_io=result.aliased_io,
    )
    trt_mod.eval()
    error_message = ""
    accuracy_loss = None
    gpu_time: Optional[float] = None
    with torch.no_grad():
        actual = trt_mod(*sample_args)
        if settings.accuracy_threshold is not None:
            assert reference_outputs is not None
            accuracy_loss = compute_output_losses(
                actual,
                reference_outputs,
                algorithm=settings.accuracy_algorithm,
                atol=settings.accuracy_atol,
                rtol=settings.accuracy_rtol,
            )
            if accuracy_failed(accuracy_loss, settings.accuracy_threshold):
                error_message = f"accuracy threshold exceeded: {accuracy_loss}"
            else:
                gpu_time = _benchmark_callable(trt_mod, sample_args)
        else:
            gpu_time = _benchmark_callable(trt_mod, sample_args)
    del trt_mod
    torch.cuda.empty_cache()
    return {
        "crashed": False,
        "error_message": error_message,
        "accuracy_loss": accuracy_loss,
        "gpu_time": gpu_time,
        "serialized_engine": result.serialized_engine,
        "input_names": list(result.input_names),
        "output_names": list(result.output_names),
        "requires_output_allocator": result.requires_output_allocator,
        "symbolic_shape_expressions": result.symbolic_shape_expressions,
        "requires_native_multidevice": result.requires_native_multidevice,
        "aliased_io": result.aliased_io,
        "route": route,
        "iter_idx": iter_idx,
    }


def _tuning_trial_entry(payload_path: str, result_path: str) -> None:
    """Spawn target: must be a top-level function so ``spawn`` can pickle it."""
    out = _empty_trial_result(
        crashed=True, error_message="child exited before writing a result"
    )
    try:
        payload = torch.load(payload_path, map_location="cpu", weights_only=False)
        out = _execute_route_trial(payload)
    except Exception as exc:
        out = _empty_trial_result(crashed=True, error_message=str(exc))
    tmp_partial_path = f"{result_path}.partial.{os.getpid()}"
    try:
        torch.save(out, tmp_partial_path)
        os.replace(tmp_partial_path, result_path)
    finally:
        if os.path.exists(tmp_partial_path):
            os.remove(tmp_partial_path)


def _spawn_route_trial(
    *,
    module: torch.fx.GraphModule,
    inputs: Sequence[Input],
    sample_tensors: Sequence[torch.Tensor],
    trial_settings: CompilationSettings,
    route: str,
    iter_idx: int,
    input_binding_names: Optional[Sequence[str]],
    output_binding_names: Optional[Sequence[str]],
    join_timeout_s: Optional[float],
    reference_outputs: Optional[Any] = None,
) -> Dict[str, Any]:
    """Run one trial in a spawned child and return its result dict."""
    payload = {
        "module": module,
        "fx_meta": _snapshot_fx_meta(module),
        "inputs": list(inputs),
        "sample_tensors": list(sample_tensors),
        "reference_outputs": reference_outputs,
        "settings": trial_settings,
        "route": route,
        "iter_idx": iter_idx,
        "input_binding_names": (
            list(input_binding_names) if input_binding_names is not None else None
        ),
        "output_binding_names": (
            list(output_binding_names) if output_binding_names is not None else None
        ),
        "gpu_id": trial_settings.device.gpu_id,
    }
    # spawn (not fork): CUDA is already initialized in the parent.
    ctx = multiprocessing.get_context("spawn")
    with tempfile.TemporaryDirectory(prefix="torch_trt_tune_") as tmp:
        payload_path = os.path.join(tmp, "payload.pt")
        result_path = os.path.join(tmp, "result.pt")
        try:
            torch.save(payload, payload_path)
            proc = ctx.Process(
                target=_tuning_trial_entry,
                args=(payload_path, result_path),
                daemon=False,
            )
            proc.start()
        except Exception as exc:
            return _empty_trial_result(
                crashed=True,
                error_message=f"failed to start child process: {exc}",
            )
        proc.join(join_timeout_s)
        if proc.is_alive():
            proc.kill()
            proc.join()
            return _empty_trial_result(
                crashed=True,
                error_message="child process exceeded remaining tuning_timeout_s and was killed",
            )
        if proc.exitcode != 0:
            return _empty_trial_result(
                crashed=True, error_message=_child_exit_message(proc.exitcode)
            )
        if not os.path.isfile(result_path):
            return _empty_trial_result(
                crashed=True, error_message="child exited without a result file"
            )
        try:
            result = torch.load(result_path, map_location="cpu", weights_only=False)
        except Exception as exc:
            return _empty_trial_result(
                crashed=True,
                error_message=f"failed to load child result: {exc}",
            )
        if not isinstance(result, dict):
            return _empty_trial_result(
                crashed=True, error_message="child result is not a dictionary"
            )
        validation_error = _trial_result_validation_error(result)
        if validation_error:
            return _empty_trial_result(
                crashed=True,
                error_message=f"malformed child result: {validation_error}",
            )
        return result


def validate_tuning_options(settings: CompilationSettings) -> None:
    """Validate Global Performance Tuning-related CompilationSettings."""
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
    if settings.tuning_dry_run and not (
        settings.tune_build_routes or settings.tune_build_route_file
    ):
        raise ValueError(
            "tuning_dry_run requires tune_build_routes or tune_build_route_file."
        )
    if settings.accuracy_threshold is not None and (
        not math.isfinite(settings.accuracy_threshold)
        or settings.accuracy_threshold < 0
    ):
        raise ValueError("accuracy_threshold must be finite and non-negative.")
    if (
        not math.isfinite(settings.accuracy_atol)
        or settings.accuracy_atol < 0
        or not math.isfinite(settings.accuracy_rtol)
        or settings.accuracy_rtol < 0
    ):
        raise ValueError(
            "accuracy_atol and accuracy_rtol must be finite and non-negative."
        )
    if settings.accuracy_atol != 1e-5 or settings.accuracy_rtol != 1e-5:
        if settings.accuracy_algorithm.lower() != "l0":
            raise ValueError(
                "accuracy_atol/accuracy_rtol are only valid when accuracy_algorithm='l0'."
            )


def should_run_tuning(settings: CompilationSettings) -> bool:
    if settings.tuning_continue or settings.tuning_dry_run:
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
    require_global_perf_tuning("Global Performance Tuning sweep")
    validate_tuning_options(settings)

    db = BuildRouteKnobDatabase()
    raw = get_all_build_routes_raw()
    if not db.load_from_json(raw):
        raise RuntimeError("Failed to load Global Performance Tuning knob database.")

    sample_tensors = _materialize_sample_tensors(inputs)
    user_cache_file = settings.tuning_cache_file
    cache_file = tuning_cache.resolve_partition_tuning_cache_path(
        user_cache_file, module
    )
    if cache_file and cache_file != user_cache_file:
        _LOGGER.info(
            f"Using per-partition tuning cache {cache_file} (from {user_cache_file})",
        )

    effective_settings = settings
    cached_rows: List[Dict[str, Any]] = []
    if settings.tuning_continue:
        assert cache_file is not None
        header = tuning_cache.read_cache(cache_file)
        if header.tuner_version != db.tuner_version:
            raise ValueError(
                "Tuning cache tuner version mismatch: "
                f"{header.tuner_version!r} != {db.tuner_version!r}"
            )
        expression = header.tuning_expr
        effective_settings = replace(
            settings,
            tuning_search=header.searching_algorithm,
            accuracy_algorithm=header.accuracy_algorithm,
            accuracy_threshold=header.accuracy_threshold,
            accuracy_atol=header.accuracy_atol,
            accuracy_rtol=header.accuracy_rtol,
        )
        validate_tuning_options(effective_settings)
        cached_rows = tuning_cache.read_iterations(cache_file)
        _LOGGER.info(
            f"Resuming tuning from {cache_file} with {len(cached_rows)} completed trials",
        )
    else:
        expression = resolve_tuning_expression(
            settings.tune_build_routes, settings.tune_build_route_file
        )
        if not expression:
            raise ValueError("tune_build_routes expression is empty.")

    exprs, routes = expand_build_routes(
        expression,
        effective_settings.tuning_search,
        db,
        dry_run=effective_settings.tuning_dry_run,
    )

    if effective_settings.tuning_dry_run:
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
                "accuracy_algorithm": effective_settings.accuracy_algorithm,
                "accuracy_parameter": {
                    "atol": effective_settings.accuracy_atol,
                    "rtol": effective_settings.accuracy_rtol,
                    "epsilon": effective_settings.accuracy_threshold,
                },
                "searching_algorithm": effective_settings.tuning_search,
                "tuning_expr": expression,
                "default_build_route": db.build_default_path(),
                "partition_key": tuning_cache.subgraph_partition_key(module),
                "user_tuning_cache_file": user_cache_file,
            },
        )

    reference_outputs = None
    if effective_settings.accuracy_threshold is not None:
        gpu_id = effective_settings.device.gpu_id
        device = torch.device(f"cuda:{gpu_id}" if gpu_id is not None else "cuda")
        if gpu_id is not None:
            torch.cuda.set_device(gpu_id)
        reference_outputs = _compute_reference_outputs(module, sample_tensors, device)

    best_result: Optional[SerializedInterpreterResult] = None
    best_time: Optional[float] = None
    best_route = ""
    best_result_route = ""
    candidates: Dict[Tuple[str, int], Tuple[float, str, int]] = {}  # successful trials
    completed_trials: set[Tuple[str, int]] = (
        set()
    )  # trials that have tried (might be crashed)
    sweep_start = time.monotonic()

    def _remaining_join_timeout(*, ignore_timeout: bool = False) -> Optional[float]:
        if ignore_timeout or effective_settings.tuning_timeout_s < 0:
            return None
        remaining = effective_settings.tuning_timeout_s - (
            time.monotonic() - sweep_start
        )
        return max(0.0, remaining)

    def _trial(
        route: str,
        iter_idx: int,
        *,
        phase: str,
        phase_iter: int,
        record_cache: bool = True,
        select_winner: bool = True,
        ignore_timeout: bool = False,
    ) -> Tuple[Optional[float], Optional[SerializedInterpreterResult]]:
        """Run a single trial and return its result."""
        nonlocal best_result, best_time, best_route, best_result_route
        _LOGGER.info(f"&&&& TASK_BEGIN [iter={iter_idx}] BuildRoute = '{route}'")
        trial_settings = replace(
            effective_settings,
            build_route=route,
            tune_build_routes="",
            tune_build_route_file=None,
            tuning_continue=False,
            tuning_dry_run=False,
            reuse_cached_engines=False,
            cache_built_engines=False,
        )
        trial = _spawn_route_trial(
            module=module,
            inputs=inputs,
            sample_tensors=sample_tensors,
            trial_settings=trial_settings,
            route=route,
            iter_idx=iter_idx,
            input_binding_names=input_binding_names,
            output_binding_names=output_binding_names,
            join_timeout_s=_remaining_join_timeout(ignore_timeout=ignore_timeout),
            reference_outputs=reference_outputs,
        )
        crashed = bool(trial.get("crashed"))
        error_message = trial.get("error_message") or ""
        accuracy_loss = trial.get("accuracy_loss")
        gpu_time = trial.get("gpu_time")
        if gpu_time is not None and not math.isfinite(gpu_time):
            error_message = f"non-finite gpu_time: {gpu_time}"
            gpu_time = None
        result = _serialized_result_from_trial(trial)
        if error_message:
            result = None
        if crashed:
            _LOGGER.warning(
                f"&&&& TASK_ABORT [iter={iter_idx}] BuildRoute = '{route}': {error_message}"
            )
        elif error_message:
            _LOGGER.warning(
                f"iter={iter_idx} route failed accuracy: {error_message}",
            )
        else:
            _LOGGER.info(f"&&&& TASK_END [iter={iter_idx}] BuildRoute = '{route}'")

        if select_winner and (
            gpu_time is not None
            and not crashed
            and not error_message
            and (best_time is None or gpu_time < best_time)
        ):
            best_time = gpu_time
            best_result = result
            best_route = route
            best_result_route = route

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
        if record_cache:
            completed_trials.add((phase, phase_iter))
        if gpu_time is not None and not crashed and not error_message:
            candidates[(phase, phase_iter)] = (gpu_time, route, iter_idx)
        del trial
        return gpu_time, result

    def _timeout_reached() -> bool:
        return (
            effective_settings.tuning_timeout_s >= 0
            and time.monotonic() - sweep_start >= effective_settings.tuning_timeout_s
        )

    phase_rows: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for row in cached_rows:
        iter_idx = row.get("iter")
        if (
            isinstance(iter_idx, bool)
            or not isinstance(iter_idx, int)
            or iter_idx < 0
            or not isinstance(row.get("build_route"), str)
        ):
            raise ValueError("Invalid tuning cache iteration")
        key = (
            ("phase1", iter_idx)
            if iter_idx < len(routes)
            else ("phase2", iter_idx - len(routes))
        )
        if key in phase_rows:
            raise ValueError(f"Duplicate tuning cache iteration {key}")
        phase_rows[key] = row
        completed_trials.add(key)
        gpu_time = row.get("gpu_time")
        if (
            not row.get("crash")
            and not row.get("error_message")
            and isinstance(gpu_time, (int, float))
            and math.isfinite(gpu_time)
        ):
            candidates[key] = (float(gpu_time), row["build_route"], row["iter"])
            if best_time is None or gpu_time < best_time:
                best_time = float(gpu_time)
                best_route = row["build_route"]

    for (phase, phase_iter), row in phase_rows.items():
        if phase == "phase1":
            expected_routes = routes
        elif effective_settings.tuning_search == "mixed":
            continue
        else:
            raise ValueError("Non-mixed tuning cache contains phase2 records")
        if phase_iter < 0 or phase_iter >= len(expected_routes):
            raise ValueError(f"Invalid {phase} iteration index {phase_iter}")
        if phase == "phase1" and row["iter"] != phase_iter:
            raise ValueError(
                f"Invalid global iteration for phase1 iteration {phase_iter}"
            )
        if row["build_route"] != expected_routes[phase_iter]:
            raise ValueError(
                f"Tuning cache route mismatch for {phase} iteration {phase_iter}"
            )

    phase1_times: List[Optional[float]] = [None] * len(routes)
    for i, route in enumerate(routes):
        existing = phase_rows.get(("phase1", i))
        if existing is not None:
            value = existing.get("gpu_time")
            if (
                not existing.get("crash")
                and not existing.get("error_message")
                and isinstance(value, (int, float))
                and math.isfinite(value)
            ):
                phase1_times[i] = float(value)
            continue
        if _timeout_reached():
            _LOGGER.info(
                f"Tuning timeout reached; stopping before phase1 iter {i}",
            )
            break
        trial_time, trial_result = _trial(route, i, phase="phase1", phase_iter=i)
        phase1_times[i] = trial_time
        del trial_result

    phase1_complete = all(("phase1", i) in completed_trials for i in range(len(routes)))
    if not phase1_complete and any(phase == "phase2" for phase, _ in phase_rows):
        raise ValueError("Tuning cache contains phase2 records before phase1 completed")
    if effective_settings.tuning_search == "mixed" and phase1_complete:
        positive = identify_positive_knobs(exprs, phase1_times, db)
        phase2 = expand_routes_mixed(exprs, db, positive)
        phase1_set = set(routes)
        phase2_new = [route for route in phase2 if route not in phase1_set]
        for (phase, phase_iter), row in phase_rows.items():
            if phase != "phase2":
                continue
            if phase_iter < 0 or phase_iter >= len(phase2_new):
                raise ValueError(f"Invalid phase2 iteration index {phase_iter}")
            if row["iter"] != len(routes) + phase_iter:
                raise ValueError(
                    f"Invalid global iteration for phase2 iteration {phase_iter}"
                )
            if row["build_route"] != phase2_new[phase_iter]:
                raise ValueError(
                    f"Tuning cache route mismatch for phase2 iteration {phase_iter}"
                )
        for j, route in enumerate(phase2_new):
            if ("phase2", j) in phase_rows:
                continue
            if _timeout_reached():
                _LOGGER.info(
                    f"Tuning timeout reached during mixed phase2 iter {j}",
                )
                break
            _trial(
                route,
                len(routes) + j,
                phase="phase2",
                phase_iter=j,
            )

    # The only engine bytes retained from this run are for the current winner.
    # Historical winners carry only statistics and are rebuilt once, after all
    # remaining trials have run.
    while candidates:
        # Find the best candidate (lowest gpu_time)
        _, (candidate_time, candidate_route, candidate_iter) = min(
            candidates.items(),
            key=lambda item: item[1][0],  # find the candidate with the minimum gpu_time
        )
        best_time = candidate_time
        best_route = candidate_route
        if best_result is not None and best_result_route == candidate_route:
            # don't need to rebuild
            break
        _LOGGER.info(
            f"Rebuilding historical best route from iter {candidate_iter}",
        )
        _, rebuilt = _trial(
            candidate_route,
            candidate_iter,
            phase="phase1",
            phase_iter=0,
            record_cache=False,
            select_winner=False,
            ignore_timeout=True,
        )
        if rebuilt is not None:
            best_result = rebuilt
            best_result_route = candidate_route
            break
        candidates = {
            key: value
            for key, value in candidates.items()
            if value[1] != candidate_route
        }

    if best_result is None:
        raise RuntimeError(
            "Global Performance Tuning sweep completed without a valid engine "
            "(all routes crashed or failed accuracy checks)."
        )

    _LOGGER.info(
        f"Selected best build route '{best_route}' with gpu_time={best_time if best_time is not None else float('nan'):.3f} ms"
    )
    settings.build_route = best_route
    if settings.cache_built_engines and engine_cache is not None:
        import tensorrt as trt
        from torch_tensorrt._features import ENABLED_FEATURES
        from torch_tensorrt.dynamo.conversion._conversion import (
            insert_engine_to_cache,
        )
        from torch_tensorrt.dynamo.conversion._TRTInterpreter import (
            TRTInterpreterResult,
        )
        from torch_tensorrt.logging import TRT_LOGGER

        winner_settings = replace(settings, build_route=best_route)
        if not ENABLED_FEATURES.refit:
            _LOGGER.warning(
                "Refit feature is not available, so the tuned engine will not be cached"
            )
        elif winner_settings.immutable_weights:
            _LOGGER.warning(
                "The engine weights are immutable, so the tuned engine will not be cached"
            )
        else:
            runtime = trt.Runtime(TRT_LOGGER)
            engine = runtime.deserialize_cuda_engine(best_result.serialized_engine)
            if engine is None:
                _LOGGER.warning(
                    "Failed to deserialize the tuned winner for engine caching"
                )
            else:
                hash_val = engine_cache.get_hash(module, inputs, winner_settings)
                insert_engine_to_cache(
                    hash_val,
                    TRTInterpreterResult(
                        engine=engine,
                        input_names=best_result.input_names,
                        output_names=best_result.output_names,
                        requires_output_allocator=best_result.requires_output_allocator,
                        requires_native_multidevice=best_result.requires_native_multidevice,
                        aliased_io=best_result.aliased_io,
                    ),
                    engine_cache,
                    winner_settings,
                    inputs,
                )
    return best_result
