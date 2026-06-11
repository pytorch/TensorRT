"""Runtime settings + the TRTRuntimeConfig shim + the ``runtime_config`` CM.

This module groups three closely related concepts together:

* :class:`RuntimeSettings` -- the user-facing, frozen dataclass of runtime-only
  knobs sampled at IExecutionContext creation (cuda_graph_strategy,
  dynamic_shapes_kernel_specialization_strategy, runtime_cache).
* :class:`TRTRuntimeConfig` -- the engine-internal shim that owns a
  :class:`RuntimeSettings` and the live ``trt.IRuntimeConfig`` derived from
  it. All ``ENABLED_FEATURES.tensorrt_rtx`` branching lives inside; callers in
  ``_TRTEngine`` and ``_TorchTensorRTModule`` stay uniform. Mirrors the C++
  ``torch_tensorrt::core::runtime::TRTRuntimeConfig`` struct.
* :func:`runtime_config` -- the runtime-mode context manager that toggles
  settings on every TRT submodule under a target for the duration of a
  ``with`` block.

Three ways to use ``RuntimeSettings``:

1. **Runtime context manager** -- toggle settings inside a ``with`` block.
2. **Programmatic** -- assign ``module.runtime_settings = rs`` directly.

``RuntimeSettings`` is intentionally NOT part of ``CompilationSettings`` and is
NOT serialized into the engine tuple. It's purely an in-memory initialization
parameter / runtime override state.
"""

from __future__ import annotations

import dataclasses
import logging
import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Optional, Sequence, Tuple, Union

import torch
from torch_tensorrt._features import ENABLED_FEATURES
from torch_tensorrt.dynamo._defaults import (
    CUDA_GRAPH_STRATEGY,
    DYNAMIC_SHAPES_KERNEL_SPECIALIZATION_STRATEGY,
    RUNTIME_CACHE_PATH,
)

if TYPE_CHECKING:
    from torch_tensorrt.runtime._runtime_cache import RuntimeCacheHandle

logger = logging.getLogger(__name__)

# Validation maps for the dataclass post-init. The TRT enum mappings live
# inside ``TRTRuntimeConfig._apply_settings`` so non-RTX imports don't trip
# over RTX-only enum symbols at module load time.
_DYNAMIC_SHAPES_KERNEL_STRATEGY_MAP: Dict[str, int] = {
    "lazy": 0,
    "eager": 1,
    "none": 2,
}
_CUDA_GRAPH_STRATEGY_MAP: Dict[str, int] = {
    "disabled": 0,
    "whole_graph_capture": 1,
}


@dataclass(frozen=True)
class RuntimeSettings:
    """Per-engine runtime-only knobs sampled at IExecutionContext creation.

    Fields:
        dynamic_shapes_kernel_specialization_strategy: ``"lazy" | "eager" | "none"``.
            TRT-RTX-only; no-op on standard TensorRT.
        cuda_graph_strategy: ``"disabled" | "whole_graph_capture"``. TRT-RTX-only.
        runtime_cache: ``None``, a disk path string, or a
            :class:`RuntimeCacheHandle`. ``None`` ⇒ no cache attached. A
            string is honored at engine construction time and primes a
            per-engine disk-backed cache (engine owns the implicit handle and
            it saves on ``__del__``). A handle is the shared-cache form,
            typically obtained from :func:`torch_tensorrt.runtime.runtime_cache`
            -- multiple engines attaching the same handle share one
            ``IRuntimeCache``.

    Equality compares all fields; for ``runtime_cache``, handle equality is
    by identity (same handle ⇒ same cache).

    Note on the default disk path: the default ``runtime_cache`` points at a
    per-user temp file (see ``torch_tensorrt.dynamo._defaults.RUNTIME_CACHE_PATH``).
    Concurrent processes for the same user share that file via filelock --
    that prevents corruption, but **not** lost-update races (process A's
    save can clobber kernels process B just generated). For
    multi-process / CI / hyperparameter-sweep workloads where lost kernels
    materially slow you down, either give each worker its own
    ``runtime_cache="..."`` path or pass ``runtime_cache=None`` to opt out.
    """

    dynamic_shapes_kernel_specialization_strategy: str = (
        DYNAMIC_SHAPES_KERNEL_SPECIALIZATION_STRATEGY
    )
    cuda_graph_strategy: str = CUDA_GRAPH_STRATEGY
    runtime_cache: Optional[Union[str, "RuntimeCacheHandle"]] = field(  # noqa: F821
        default_factory=lambda: RUNTIME_CACHE_PATH
    )

    def __post_init__(self) -> None:
        if (
            self.dynamic_shapes_kernel_specialization_strategy
            not in _DYNAMIC_SHAPES_KERNEL_STRATEGY_MAP
        ):
            raise ValueError(
                "Invalid dynamic_shapes_kernel_specialization_strategy: "
                f"{self.dynamic_shapes_kernel_specialization_strategy!r}. "
                f"Expected one of {list(_DYNAMIC_SHAPES_KERNEL_STRATEGY_MAP)}."
            )
        if self.cuda_graph_strategy not in _CUDA_GRAPH_STRATEGY_MAP:
            raise ValueError(
                f"Invalid cuda_graph_strategy: {self.cuda_graph_strategy!r}. "
                f"Expected one of {list(_CUDA_GRAPH_STRATEGY_MAP)}."
            )
        # RuntimeSettings only takes effect on TRT-RTX builds. Warn on every
        # construction on regular TRT so users don't silently expect cache /
        # strategy plumbing to do anything. Callers that want once-per-process
        # behavior can filter via ``warnings.simplefilter("once", UserWarning)``.
        if not ENABLED_FEATURES.tensorrt_rtx:
            warnings.warn(
                "RuntimeSettings is only honored on TRT-RTX builds; "
                "constructing it on regular TensorRT has no effect.",
                UserWarning,
                stacklevel=2,
            )

    def merge(self, **overrides: Any) -> "RuntimeSettings":
        """Return a new ``RuntimeSettings`` with ``overrides`` applied on top of self."""
        unknown = set(overrides) - {f.name for f in dataclasses.fields(self)}
        if unknown:
            raise TypeError(
                f"Unknown RuntimeSettings field(s): {sorted(unknown)}. "
                f"Valid fields: {[f.name for f in dataclasses.fields(self)]}."
            )
        return dataclasses.replace(self, **overrides)


class TRTRuntimeConfig:
    """Owns ``RuntimeSettings`` + the live ``trt.IRuntimeConfig`` for one engine.

    All TRT-RTX feature-flag branching lives in this class -- callers in
    ``_TRTEngine`` and ``_TorchTensorRTModule`` stay uniform. Mirrors the C++
    ``TRTRuntimeConfig`` struct.

    On non-RTX builds, ``self._live`` stays ``None`` and the strategy /
    runtime-cache plumbing is short-circuited; ``create_execution_context``
    picks the legacy ``cuda_engine.create_execution_context(strategy)``
    overload.
    """

    def __init__(self, settings: Optional[RuntimeSettings] = None) -> None:
        self._settings: RuntimeSettings = settings or RuntimeSettings()
        # Live trt.IRuntimeConfig (RTX) or None (non-RTX / pre-init).
        self._live: Any = None

    @property
    def settings(self) -> RuntimeSettings:
        """The current ``RuntimeSettings``. Mutate only via :meth:`set_settings`."""
        return self._settings

    def set_settings(self, new: RuntimeSettings) -> bool:
        """Apply ``new`` settings. Returns True iff the value actually changed.

        On change, invalidates the live ``IRuntimeConfig`` and signals callers
        to recreate the ``IExecutionContext``. Disk persistence of any prior
        implicit cache handle is the module's responsibility (see
        ``TorchTensorRTModule._materialize_implicit_handle``); this method is
        a pure-execution swap.
        """
        if new == self._settings:
            return False
        self._settings = new
        self.reset()
        return True

    def ensure_initialized(self, cuda_engine: Any) -> None:
        """Lazy-create the live ``trt.IRuntimeConfig`` and apply settings.

        No-op on non-TRT-RTX builds, where there is no ``IRuntimeConfig`` to
        configure.
        """
        if not ENABLED_FEATURES.tensorrt_rtx:
            return
        if self._live is not None:
            return
        self._live = cuda_engine.create_runtime_config()
        self._apply_settings()

    def reset(self) -> None:
        """Drop the live ``IRuntimeConfig``; the next ``ensure_initialized`` rebuilds."""
        self._live = None

    def create_execution_context(
        self,
        cuda_engine: Any,
        allocation_strategy: Any,
    ) -> Any:
        """Lazy-init + create a fresh ``IExecutionContext``.

        Picks the right ``cuda_engine.create_execution_context`` overload
        (``IRuntimeConfig`` vs ``ExecutionContextAllocationStrategy``) so
        callers stay free of any ``ENABLED_FEATURES.tensorrt_rtx`` branching.
        """
        if ENABLED_FEATURES.tensorrt_rtx:
            self.ensure_initialized(cuda_engine)
            assert self._live is not None
            self._live.set_execution_context_allocation_strategy(allocation_strategy)
            return cuda_engine.create_execution_context(self._live)
        return cuda_engine.create_execution_context(allocation_strategy)

    def uses_internal_capture(self, cudagraphs_enabled: bool) -> bool:
        """Returns True if TRT-RTX owns capture/replay for the current settings.

        Caller should then bypass its own ``torch.cuda.CUDAGraph`` capture
        around ``execute_async_v3``. Always False on non-RTX builds.
        """
        if not ENABLED_FEATURES.tensorrt_rtx:
            return False
        return self._settings.cuda_graph_strategy != "disabled" or cudagraphs_enabled

    def is_monolithic_capturable(
        self,
        has_dynamic_inputs: bool,
        context: Any,
        stream: Any,
    ) -> bool:
        """Returns True iff this engine can be safely wrapped by an outer monolithic capture.

        Mirrors C++ ``TRTRuntimeConfig::is_monolithic_capturable``. Non-RTX
        builds always return True.
        """
        if not ENABLED_FEATURES.tensorrt_rtx:
            return True
        if not context.is_stream_capturable(stream.cuda_stream):
            return False
        # "lazy" kernel specialization swaps specialized kernels mid-run when an
        # input has a dynamic dimension; for static-shape engines the kernels
        # are fixed at setup and the captured graph stays valid.
        return not (
            self._settings.dynamic_shapes_kernel_specialization_strategy == "lazy"
            and has_dynamic_inputs
        )

    # ------------------------------------------------------------------
    # Internal: apply self._settings to self._live
    # ------------------------------------------------------------------

    def _apply_settings(self) -> None:
        """Apply ``self._settings`` to the live ``trt.IRuntimeConfig``.

        Resolves ``runtime_cache``:
        - ``None`` ⇒ no cache attached.
        - ``RuntimeCacheHandle`` ⇒ caller owns lifecycle (handle holds the
          IRuntimeCache reference; ``ensure_cache`` materializes it on first
          use). String paths are pre-wrapped into handles by the upstream
          :py:meth:`TorchTensorRTModule._materialize_implicit_handle`; raw
          strings are not accepted here.
        """
        # Deferred imports: trt is import-aliased to tensorrt_rtx on RTX builds,
        # and _runtime_cache imports this module's RuntimeSettings.
        import tensorrt as trt
        from torch_tensorrt.runtime._runtime_cache import RuntimeCacheHandle

        self._live.dynamic_shapes_kernel_specialization_strategy = (
            self._to_trt_ds_strategy(trt)
        )
        logger.info(
            "Dynamic shapes kernel specialization strategy: "
            f"{self._settings.dynamic_shapes_kernel_specialization_strategy}"
        )
        self._live.cuda_graph_strategy = self._to_trt_cg_strategy(trt)
        logger.info(f"CUDA graph strategy: {self._settings.cuda_graph_strategy}")

        rc = self._settings.runtime_cache
        if rc is None:
            logger.debug(
                "Runtime cache disabled (no RuntimeCacheHandle / path provided)."
            )
        elif isinstance(rc, RuntimeCacheHandle):
            cache = rc.ensure_cache(self._live)
            self._live.set_runtime_cache(cache)
            # Engine-implicit handles need disk-warm on first attach (the
            # module's post-dispatch load fires while the pybind cache is
            # still un-materialized due to lazy IExecutionContext creation;
            # by the time we're here, ``ensure_cache`` has materialized it).
            # External + CM-yielded handles (autosave_on_del=False) are loaded
            # by their owner, not us.
            if rc.autosave_on_del and rc.path:
                try:
                    rc.load()
                except Exception as e:
                    logger.warning(f"Failed to load implicit runtime cache: {e}")
        else:
            raise TypeError(
                f"runtime_cache must be None or RuntimeCacheHandle by the time "
                f"it reaches TRTRuntimeConfig; got {type(rc).__name__}. "
                f"Path strings should be pre-wrapped by the module."
            )
        logger.info("TensorRT-RTX runtime config configured")

    def _to_trt_ds_strategy(self, trt: Any) -> Any:
        return {
            "lazy": trt.DynamicShapesKernelSpecializationStrategy.LAZY,
            "eager": trt.DynamicShapesKernelSpecializationStrategy.EAGER,
            "none": trt.DynamicShapesKernelSpecializationStrategy.NONE,
        }[self._settings.dynamic_shapes_kernel_specialization_strategy]

    def _to_trt_cg_strategy(self, trt: Any) -> Any:
        return {
            "disabled": trt.CudaGraphStrategy.DISABLED,
            "whole_graph_capture": trt.CudaGraphStrategy.WHOLE_GRAPH_CAPTURE,
        }[self._settings.cuda_graph_strategy]


# ---------------------------------------------------------------------------
# runtime_config(...) CM and factory
# ---------------------------------------------------------------------------


class _RuntimeConfigContextManager:
    """Pool CM that applies ``RuntimeSettings`` overrides to every TRT submodule.

    Walks ``named_modules()`` once on enter, snapshots prior settings per
    engine, sets ``mod.runtime_settings = merged`` per engine. Restores on
    exit using the same snapshot dict.

    Yields the target (or tuple of targets) so users can write
    ``with runtime_config(model, ...) as m: m(*inputs)``.
    """

    def __init__(
        self,
        target_or_targets: Union["torch.nn.Module", Sequence["torch.nn.Module"]],
        **overrides: Any,
    ) -> None:
        # Validate keys against RuntimeSettings field names (typo => raise here,
        # not silently no-op later).
        valid_fields = {f.name for f in dataclasses.fields(RuntimeSettings)}
        unknown = set(overrides) - valid_fields
        if unknown:
            raise TypeError(
                f"Unknown RuntimeSettings field(s): {sorted(unknown)}. "
                f"Valid fields: {sorted(valid_fields)}."
            )

        if isinstance(target_or_targets, torch.nn.Module):
            self._targets: Tuple[torch.nn.Module, ...] = (target_or_targets,)
            self._yield_tuple = False
        else:
            self._targets = tuple(target_or_targets)
            self._yield_tuple = True
        self._overrides = overrides
        # Engine ↔ prior RuntimeSettings snapshot; populated on enter.
        self._saved: Dict[Any, RuntimeSettings] = {}

    def __enter__(self) -> Union["torch.nn.Module", Tuple["torch.nn.Module", ...]]:
        # Deferred import to avoid a circular dependency at module-load time.
        from torch_tensorrt.dynamo.runtime._TorchTensorRTModule import (
            TorchTensorRTModule,
        )

        for target in self._targets:
            for _, mod in target.named_modules():
                if isinstance(mod, TorchTensorRTModule) and mod.engine is not None:
                    current = mod.runtime_settings
                    if mod in self._saved:
                        # The same TRTModule appears under multiple targets in the
                        # list (or the tree contains a cycle). Don't snapshot twice.
                        continue
                    self._saved[mod] = current
                    merged = current.merge(**self._overrides)
                    mod.runtime_settings = merged
        return self._targets if self._yield_tuple else self._targets[0]

    def __exit__(self, *args: Any) -> None:
        for mod, prior in self._saved.items():
            mod.runtime_settings = prior


def runtime_config(
    target_or_targets: Union["torch.nn.Module", Sequence["torch.nn.Module"]],
    **overrides: Any,
) -> _RuntimeConfigContextManager:
    """Context manager that applies ``RuntimeSettings`` overrides to all TRT
    engines under ``target_or_targets`` for the duration of the ``with`` block.

    Accepts the same kwargs as :class:`RuntimeSettings` fields. The pool
    semantics collapse N knob changes into one ``update_runtime_settings`` call
    per engine, which means exactly two ``IExecutionContext`` recreates per
    engine (one on enter, one on exit) regardless of how many overrides are
    passed.

    Yields the target module (single form) or a tuple of targets (list form),
    by-reference -- same object the caller passed in.
    """
    return _RuntimeConfigContextManager(target_or_targets, **overrides)


# ---------------------------------------------------------------------------
# Sugar wrappers for the two strategy knobs
# ---------------------------------------------------------------------------


def set_dynamic_shapes_kernel_strategy(
    target_or_targets: Union["torch.nn.Module", Sequence["torch.nn.Module"]],
    strategy: str,
) -> _RuntimeConfigContextManager:
    """Context manager that sets the dynamic-shapes kernel specialization
    strategy on all TRT engines under ``target_or_targets``.

    Accepts ``"lazy"``, ``"eager"``, or ``"none"``. Delegates to
    :func:`runtime_config`.
    """
    return runtime_config(
        target_or_targets, dynamic_shapes_kernel_specialization_strategy=strategy
    )


def set_cuda_graph_strategy(
    target_or_targets: Union["torch.nn.Module", Sequence["torch.nn.Module"]],
    strategy: str,
) -> _RuntimeConfigContextManager:
    """Context manager that sets the cuda-graph strategy on all TRT engines
    under ``target_or_targets``.

    Accepts ``"disabled"`` or ``"whole_graph_capture"``. Delegates to
    :func:`runtime_config`.
    """
    return runtime_config(target_or_targets, cuda_graph_strategy=strategy)
