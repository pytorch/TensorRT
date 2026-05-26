"""Python-side TensorRT engine: deserialize and execute TRT engines without the C++ runtime.

Serialization layout lives in :mod:`torch_tensorrt.dynamo.runtime._serialized_engine_layout`.
When the C++ Torch-TensorRT runtime is unavailable, :class:`TRTEngine` is registered as an
opaque type and ``tensorrt::execute_engine`` is registered as a Python custom op so that the
same compiled graph can run on either the C++ or Python runtime transparently.
"""

from __future__ import annotations

import base64
import copy
import logging
import os
import pickle
import tempfile
from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any, ContextManager, Dict, List, Optional, Sequence, Tuple, cast

import torch
import torch.distributed as dist
import torch_tensorrt
from torch._library.opaque_object import register_opaque_type
from torch._opaque_base import OpaqueBase
from torch_tensorrt._enums import dtype
from torch_tensorrt._features import ENABLED_FEATURES
from torch_tensorrt.dynamo._defaults import DEBUG_LOGGING_DIR
from torch_tensorrt.dynamo._settings import CompilationSettings
from torch_tensorrt.dynamo.runtime._serialized_engine_layout import (
    ABI_TARGET_IDX,
    DEVICE_IDX,
    ENGINE_IDX,
    HW_COMPATIBLE_IDX,
    INPUT_BINDING_NAMES_IDX,
    NAME_IDX,
    OUTPUT_BINDING_NAMES_IDX,
    REQUIRES_NATIVE_MULTIDEVICE_IDX,
    REQUIRES_OUTPUT_ALLOCATOR_IDX,
    RESOURCE_ALLOCATION_STRATEGY_IDX,
    SERIALIZATION_LEN,
    SERIALIZED_METADATA_IDX,
    TARGET_PLATFORM_IDX,
    SerializedTensorRTEngineFmt,
    deserialize_binding_names,
    parse_device_info,
)
from torch_tensorrt.logging import TRT_LOGGER
from torch_tensorrt.runtime._utils import (
    _is_switch_required,
    _select_rt_device,
    multi_gpu_device_check,
)

# must import after torch_tensorrt to resolve tensorrt_rtx alias
import tensorrt as trt  # isort: skip

logger = logging.getLogger(__name__)


def _get_dynamic_shapes_kernel_strategy(strategy_str: str) -> Any:
    """Map strategy string to TRT enum. Only meaningful on TensorRT-RTX builds."""
    return {
        "lazy": trt.DynamicShapesKernelSpecializationStrategy.LAZY,
        "eager": trt.DynamicShapesKernelSpecializationStrategy.EAGER,
        "none": trt.DynamicShapesKernelSpecializationStrategy.NONE,
    }.get(strategy_str, trt.DynamicShapesKernelSpecializationStrategy.LAZY)


def _get_cuda_graph_strategy(strategy_str: str) -> Any:
    """Map strategy string to TRT CudaGraphStrategy enum. Only meaningful on RTX."""
    return {
        "disabled": trt.CudaGraphStrategy.DISABLED,
        "whole_graph_capture": trt.CudaGraphStrategy.WHOLE_GRAPH_CAPTURE,
    }.get(strategy_str, trt.CudaGraphStrategy.DISABLED)


# ---------------------------------------------------------------------------
# TRT I/O helpers
# ---------------------------------------------------------------------------


class DynamicOutputAllocator(trt.IOutputAllocator):  # type: ignore[misc]
    def __init__(self, output_dtypes: Dict[str, torch.dtype]) -> None:
        trt.IOutputAllocator.__init__(self)
        self.buffers: Dict[str, torch.Tensor] = {}
        self.shapes: Dict[str, Tuple[int, ...]] = {}
        self.dtypes: Dict[str, torch.dtype] = output_dtypes

    def reallocate_output_async(
        self,
        tensor_name: str,
        memory: int,
        size: int,
        alignment: int,
        stream: torch.cuda.Stream,
    ) -> Any:
        shape = (size,)
        if tensor_name not in self.buffers or self.buffers[tensor_name].shape != shape:
            self.buffers[tensor_name] = torch.empty(
                shape,
                dtype=self.dtypes[tensor_name],
                device=torch.cuda.current_device(),
            )
        return self.buffers[tensor_name].data_ptr()

    def notify_shape(self, tensor_name: str, shape: Tuple[int, ...]) -> None:
        self.shapes[tensor_name] = tuple(shape)


class TorchTRTRuntimeStates:
    """Tracks CUDA graph / pre-allocated-output state across invocations."""

    def __init__(self, new_cudagraphs: bool):
        self.old_cudagraphs = new_cudagraphs
        self.old_pre_allocated_outputs = False
        self.context_changed = False

    def set_runtime_states(
        self,
        new_cudagraphs: bool,
        new_pre_allocated_output: bool,
        shape_changed: bool,
    ) -> Tuple[bool, bool, bool]:
        need_cudagraphs_record = False
        can_use_pre_allocated_outputs = False
        need_cudagraphs_reset = False

        if new_cudagraphs and (
            not self.old_cudagraphs or shape_changed or self.context_changed
        ):
            need_cudagraphs_record = True

        if (
            self.old_pre_allocated_outputs
            and new_pre_allocated_output
            and (not shape_changed)
        ):
            can_use_pre_allocated_outputs = True

        if not new_cudagraphs or shape_changed or self.context_changed:
            need_cudagraphs_reset = True

        self.old_cudagraphs = new_cudagraphs
        self.old_pre_allocated_outputs = new_pre_allocated_output
        self.context_changed = False

        return (
            need_cudagraphs_record,
            can_use_pre_allocated_outputs,
            need_cudagraphs_reset,
        )


# ---------------------------------------------------------------------------
# Pickle reconstruction — returns the right engine type for the current runtime
# ---------------------------------------------------------------------------

# TODO: Uncomment this when cross serialization is enabled

# def _reconstruct_trt_engine(serialized_info: List[Any]) -> Any:
#     """Reconstruct a TRT engine from its serialized info list.

#     Called by pickle when deserializing a ``TRTEngine``.  Checks which runtime
#     is available and returns either a C++ ``torch.classes.tensorrt.Engine`` or
#     a Python ``TRTEngine``, so a single ``.pt2`` artifact is portable across
#     runtimes.
#     """
#     serialized_info = list(serialized_info)
#     engine_field = serialized_info[ENGINE_IDX]
#     if isinstance(engine_field, str):
#         serialized_info[ENGINE_IDX] = base64.b64decode(engine_field.encode("utf-8"))
#     elif isinstance(engine_field, bytes) and not engine_field.startswith(b"ftrt"):
#         serialized_info[ENGINE_IDX] = base64.b64decode(engine_field)

#     if torch_tensorrt.ENABLED_FEATURES.torch_tensorrt_runtime:
#         return torch.classes.tensorrt.Engine(tuple(serialized_info))

#     return TRTEngine(serialized_info)


# class EngineSerializer(OpaqueBase):  # type: ignore[misc]
#     def __init__(self, serialized_info: SerializedTensorRTEngineFmt) -> None:
#         self.serialized_info = serialized_info

#     def __reduce__(self) -> Tuple[Any, Tuple[List[Any]]]:
#         """Pickle protocol: delegates to :func:`_reconstruct_trt_engine`.

#         The reconstruction function checks which runtime is available at
#         load time and returns either a C++ ``torch.classes.tensorrt.Engine``
#         or a Python ``TRTEngine``, so a single saved artifact works on both.
#         """
#         state = list(self.serialized_info)
#         state[ENGINE_IDX] = base64.b64encode(state[ENGINE_IDX]).decode("utf-8")
#         return (_reconstruct_trt_engine, (state,))


# ---------------------------------------------------------------------------
# TRTEngine (Python implementation)
# ---------------------------------------------------------------------------


class TRTEngine(OpaqueBase):  # type: ignore[misc]
    """TensorRT engine + execution context, driven from Python TRT APIs.

    Exposes the same surface as the C++ ``torch.classes.tensorrt.Engine`` TorchBind
    class so that :class:`~torch_tensorrt.dynamo.runtime.TorchTensorRTModule` can use
    either implementation without branching.  When the C++ runtime is unavailable this
    class is registered as an opaque type and ``tensorrt::execute_engine`` is registered
    as a Python custom op pointing to :func:`execute_engine`.
    """

    # --- construction / teardown ---

    def __init__(
        self,
        serialized_info: SerializedTensorRTEngineFmt,
        *,
        profile_execution: bool = False,
    ) -> None:
        self._profile_execution = profile_execution
        self.profile_path_prefix = tempfile.gettempdir()
        self.use_pre_allocated_outputs = False
        self.use_output_allocator_outputs = False
        self.output_tensors_are_unowned = False
        self.output_allocator: Optional[DynamicOutputAllocator] = None
        self.pre_allocated_outputs: List[torch.Tensor] = []
        self._input_buffers: List[torch.Tensor] = []
        self._output_buffers: List[torch.Tensor] = []
        self._caller_stream: Optional[torch.cuda.Stream] = None
        self._engine_stream: Optional[torch.cuda.Stream] = None
        self._owned_pool_stream: Optional[torch.cuda.Stream] = None
        self.cudagraph: Optional[torch.cuda.CUDAGraph] = None
        self.shape_key: Optional[str] = None
        self._empty_tensor_placeholder: Optional[torch.Tensor] = None
        self._dynamic_workspace: Optional[torch.Tensor] = None
        self.runtime_states = TorchTRTRuntimeStates(
            torch_tensorrt.runtime.get_cudagraphs_mode()
        )
        self.resource_allocation_strategy = 0
        # TensorRT-RTX runtime cache state. ``runtime_cache_path`` is filled in
        # by ``_load_serialized_info`` once compilation settings are available;
        # ``runtime_config`` and ``runtime_cache`` are populated by
        # ``_setup_runtime_config`` on RTX builds. Initialized to ``None`` here
        # so the destructor can safely save the cache even if ``_setup_engine``
        # never runs.
        self.runtime_config: Any = None
        self.runtime_cache: Any = None
        # True once an IRuntimeConfig.cuda_graph_strategy other than
        # ``"disabled"`` is in effect (either set at compile time or installed
        # at runtime by ``_enable_rtx_native_cudagraphs``). When true,
        # ``_execute_standard`` must skip manual torch.cuda.CUDAGraph capture
        # because TRT-RTX handles it internally.
        self._rtx_native_cudagraphs: bool = False
        # NCCL communicator is bound lazily on the first forward pass for
        # engines compiled with native multi-device collective layers.
        self._nccl_comm: Optional[Any] = None

        self._load_serialized_info(serialized_info)
        self._setup_engine()

    def __del__(self) -> None:
        # Persist the TensorRT-RTX runtime cache before tearing the engine
        # down; no-op when ``runtime_cache`` was never populated.
        try:
            self._save_runtime_cache()
        except Exception:
            pass
        self.reset_captured_graph()

    def __deepcopy__(self, memo: dict[int, Any]) -> "TRTEngine":
        """Rebuild from serialized layout so ``copy.deepcopy`` skips unpickleable TRT handles."""
        if id(self) in memo:
            return memo[id(self)]  # type: ignore
        serialized_copy = copy.deepcopy(self.serialized_info, memo)
        dup = type(self)(serialized_copy, profile_execution=self._profile_execution)
        memo[id(self)] = dup
        return dup

    def __str__(self) -> str:
        return f"TRTEngine(name={self.name}, device={self.serialized_device_info})"

    def __repr__(self) -> str:
        return self.__str__()

    def __getstate__(self) -> Tuple[List[Any], str]:
        """Return pickle state in the same shape as C++ ``ScriptObject.__getstate__``.

        Outer tuple with a single element: the ``serialize()``-style list of string
        slots, with ``ENGINE_IDX`` base64-encoded (matches ``def_pickle`` getter).
        """
        serialized_info = list(self.serialized_info)
        serialized_info[ENGINE_IDX] = base64.b64encode(
            serialized_info[ENGINE_IDX]
        ).decode("utf-8")
        return (serialized_info, "TRTEngine")

    def __setstate__(self, state: Any) -> None:
        """Restore from C++-matching pickle state ``(serialized_info,)``."""
        self._profile_execution = False
        self.profile_path_prefix = tempfile.gettempdir()
        self.use_pre_allocated_outputs = False
        self.use_output_allocator_outputs = False
        self.output_tensors_are_unowned = False
        self.output_allocator = None
        self.pre_allocated_outputs = []
        self._input_buffers = []
        self._output_buffers = []
        self._caller_stream = None
        self._engine_stream = None
        self._owned_pool_stream = None
        self.cudagraph = None
        self.shape_key = None
        self._empty_tensor_placeholder = None
        self._dynamic_workspace = None
        self.runtime_states = TorchTRTRuntimeStates(
            torch_tensorrt.runtime.get_cudagraphs_mode()
        )
        self.resource_allocation_strategy = 0
        # See ``__init__`` for the rationale: pre-init these so a destructor
        # firing on a partially-loaded engine never trips an ``AttributeError``.
        self.runtime_config = None
        self.runtime_cache = None
        self._rtx_native_cudagraphs = False
        # NCCL communicators cannot be pickled; rebind lazily on the next
        # forward pass via setup_nccl_comm().
        self._nccl_comm = None

        serialized_info = list(state[0])
        engine_field = serialized_info[ENGINE_IDX]
        if isinstance(engine_field, str):
            serialized_info[ENGINE_IDX] = base64.b64decode(engine_field.encode("utf-8"))
        elif isinstance(engine_field, bytes) and not engine_field.startswith(b"ftrt"):
            serialized_info[ENGINE_IDX] = base64.b64decode(engine_field)
        self._load_serialized_info(serialized_info)
        self._setup_engine()

    def tracing_mode(self) -> str:
        """Return ``"real"`` so FakeTensor/export pass the real engine into meta kernels.

        Mirrors TorchBind ``tracing_with_real`` behavior (see
        :func:`torch._library.fake_class_registry.maybe_to_fake_obj`).
        """

        return "real"

    def _load_serialized_info(
        self, serialized_info: SerializedTensorRTEngineFmt
    ) -> None:
        if len(serialized_info) != SERIALIZATION_LEN:
            raise RuntimeError(
                f"Expected serialized info length {SERIALIZATION_LEN}, got {len(serialized_info)}"
            )

        self.serialized_info: SerializedTensorRTEngineFmt = list(serialized_info)
        self.version = str(self.serialized_info[ABI_TARGET_IDX])
        self.name = str(self.serialized_info[NAME_IDX]).replace(".", "_")
        self.serialized_device_info = str(self.serialized_info[DEVICE_IDX])
        self.serialized_engine = self.serialized_info[ENGINE_IDX]
        if not isinstance(self.serialized_engine, (bytes, bytearray)):
            raise TypeError("Expected serialized engine as bytes")

        self.in_binding_names = deserialize_binding_names(
            str(self.serialized_info[INPUT_BINDING_NAMES_IDX])
        )
        self.out_binding_names = deserialize_binding_names(
            str(self.serialized_info[OUTPUT_BINDING_NAMES_IDX])
        )
        self.hardware_compatible = bool(int(self.serialized_info[HW_COMPATIBLE_IDX]))
        self.serialized_metadata = str(self.serialized_info[SERIALIZED_METADATA_IDX])
        self.serialized_target_platform = str(self.serialized_info[TARGET_PLATFORM_IDX])
        self.requires_output_allocator = bool(
            int(self.serialized_info[REQUIRES_OUTPUT_ALLOCATOR_IDX])
        )
        self.resource_allocation_strategy = int(
            self.serialized_info[RESOURCE_ALLOCATION_STRATEGY_IDX]
        )
        # Mirrors the C++ TRTEngine::requires_native_multidevice field; consumed
        # by TorchTensorRTModule.setup_engine() and the distributed helpers.
        self.requires_native_multidevice = bool(
            int(self.serialized_info[REQUIRES_NATIVE_MULTIDEVICE_IDX])
        )
        # Internal alias used by the NCCL setup paths (matches the original
        # _PythonTorchTensorRTModule attribute name).
        self._has_nccl_ops: bool = self.requires_native_multidevice

        metadata = self.decode_metadata(self.serialized_metadata)
        self.settings = metadata.get("settings", CompilationSettings())
        # Path used by ``_load_runtime_cache`` / ``_save_runtime_cache`` on
        # TensorRT-RTX. Always set so non-RTX engines also expose it.
        self.runtime_cache_path = self.settings.runtime_cache_path
        self.weight_name_map = metadata.get("weight_name_map")
        self.symbolic_shape_expressions = metadata.get("inout_symexprs")
        self.output_tensors_are_unowned = metadata.get(
            "output_tensors_are_unowned", False
        )

        device_info = parse_device_info(self.serialized_device_info)
        self.target_device_id = device_info["id"]
        self._target_device = torch.device("cuda", self.target_device_id)
        self._default_stream = torch.cuda.default_stream(self._target_device)
        # Serialized major/minor/name only — not ``_CudaDeviceProperties`` — so deepcopy/refit
        # can copy the owning ``GraphModule`` without pickle errors.
        self.target_device_properties = SimpleNamespace(
            major=device_info["major"],
            minor=device_info["minor"],
            name=device_info["name"],
        )

    @staticmethod
    def decode_metadata(encoded_metadata: str) -> Any:
        dumped_metadata = base64.b64decode(encoded_metadata.encode("utf-8"))
        return pickle.loads(dumped_metadata)

    def get_serialized_metadata(self) -> str:
        return self.serialized_metadata

    def close(self) -> None:
        """Release CUDA graph resources (called explicitly or via __del__)."""
        self.reset_captured_graph()

    def _create_execution_context(self) -> trt.IExecutionContext:
        # On TensorRT-RTX builds the allocation strategy lives on the
        # ``IRuntimeConfig`` (set by ``_setup_runtime_config``), so once the
        # runtime config is built we route context creation through it. The
        # first call from ``_setup_engine`` precedes ``_setup_runtime_config``
        # and falls through to the strategy-based path below.
        if (
            ENABLED_FEATURES.tensorrt_rtx
            and getattr(self, "runtime_config", None) is not None
        ):
            context = self.cuda_engine.create_execution_context(self.runtime_config)
            assert context is not None, "Failed to create execution context"
            return context
        strategy = trt.ExecutionContextAllocationStrategy.STATIC
        if self.resource_allocation_strategy:
            strategy = trt.ExecutionContextAllocationStrategy.USER_MANAGED
        context = self.cuda_engine.create_execution_context(strategy)
        assert context is not None, "Failed to create execution context"
        return context

    def _setup_engine(self) -> None:
        multi_gpu_device_check()
        self.runtime = trt.Runtime(TRT_LOGGER)
        self.cuda_engine = self.runtime.deserialize_cuda_engine(self.serialized_engine)
        assert self.cuda_engine is not None, "Failed to deserialize TensorRT engine"

        if self.cuda_engine.streamable_weights_size > 0:
            budget_bytes = self.cuda_engine.get_weight_streaming_automatic_budget()
            logger.debug(f"Weight streaming budget set to {budget_bytes}B")
            self.cuda_engine.weight_streaming_budget_v2 = budget_bytes

        self.context = self._create_execution_context()

        if self._has_nccl_ops:
            from torch_tensorrt.distributed._nccl_utils import (
                check_nccl_engine_requirements,
            )

            check_nccl_engine_requirements()

            # For engines with native NCCL collective layers, all ranks must
            # have a live IExecutionContext before any rank executes a
            # collective. Barrier here so a fast-compiling rank does not race
            # ahead and issue an NCCL op while another rank is still inside
            # deserialize_cuda_engine / create_execution_context.
            if (
                dist.is_available()
                and dist.is_initialized()
                and dist.get_world_size() > 1
            ):
                logger.debug(
                    "Barrier after execution context creation (distributed NCCL engine)"
                )
                dist.barrier()

        # On TensorRT-RTX, build the IRuntimeConfig (with runtime cache,
        # dynamic-shape kernel specialization strategy, and CUDA graph
        # strategy) and rebuild the execution context so it picks them up.
        # The NCCL barrier above runs against the initial strategy-based
        # context.
        if ENABLED_FEATURES.tensorrt_rtx:
            self._setup_runtime_config()
            self._rtx_native_cudagraphs = (
                self.settings.cuda_graph_strategy != "disabled"
            )
            self.context = self._create_execution_context()

        if not self.in_binding_names and not self.out_binding_names:
            input_names: List[str] = []
            output_names: List[str] = []
            for idx in range(self.cuda_engine.num_io_tensors):
                bind_name = self.cuda_engine.get_tensor_name(idx)
                if (
                    self.cuda_engine.get_tensor_mode(bind_name)
                    == trt.TensorIOMode.INPUT
                ):
                    input_names.append(bind_name)
                else:
                    output_names.append(bind_name)
            self.in_binding_names = input_names
            self.out_binding_names = output_names

        self._input_buffers = [None] * len(self.in_binding_names)
        self._output_buffers = [None] * len(self.out_binding_names)
        self.input_dtypes = [
            dtype._from(self.cuda_engine.get_tensor_dtype(input_name)).to(torch.dtype)
            for input_name in self.in_binding_names
        ]
        self.output_dtypes = [
            dtype._from(self.cuda_engine.get_tensor_dtype(output_name)).to(torch.dtype)
            for output_name in self.out_binding_names
        ]
        self.output_shapes = [
            self.cuda_engine.get_tensor_shape(output_name)
            for output_name in self.out_binding_names
        ]
        self.is_shape_inference_io = {
            input_name: self.cuda_engine.is_shape_inference_io(input_name)
            for input_name in self.in_binding_names
        }
        if self.requires_output_allocator:
            self.create_output_allocator()

    # --- TensorRT-RTX runtime cache / dynamic shapes strategy ---

    def _setup_runtime_config(self) -> None:
        """Build an ``IRuntimeConfig`` with runtime cache and dynamic-shape strategy.

        The runtime cache stores JIT compilation results so kernel/graph
        compilation is not repeated across inference runs. The dynamic-shape
        kernel specialization strategy controls how the JIT compiles
        shape-specialized kernels (``lazy``, ``eager``, or ``none``).
        """
        self.runtime_config = self.cuda_engine.create_runtime_config()
        alloc_strategy = trt.ExecutionContextAllocationStrategy.STATIC
        if self.resource_allocation_strategy:
            alloc_strategy = trt.ExecutionContextAllocationStrategy.USER_MANAGED
        self.runtime_config.set_execution_context_allocation_strategy(alloc_strategy)
        self.runtime_config.dynamic_shapes_kernel_specialization_strategy = (
            _get_dynamic_shapes_kernel_strategy(
                self.settings.dynamic_shapes_kernel_specialization_strategy
            )
        )
        logger.info(
            "Dynamic shapes kernel specialization strategy: "
            f"{self.settings.dynamic_shapes_kernel_specialization_strategy}"
        )
        self.runtime_config.cuda_graph_strategy = _get_cuda_graph_strategy(
            self.settings.cuda_graph_strategy
        )
        logger.info(f"CUDA graph strategy: {self.settings.cuda_graph_strategy}")
        self.runtime_cache = self.runtime_config.create_runtime_cache()
        self._load_runtime_cache()
        self.runtime_config.set_runtime_cache(self.runtime_cache)
        logger.info("TensorRT-RTX runtime cache configured")

    def _load_runtime_cache(self) -> None:
        """Load runtime cache from disk if it exists (with a shared file lock)."""
        if self.runtime_cache is None:
            return
        if not os.path.isfile(self.runtime_cache_path):
            logger.debug(f"No existing runtime cache at {self.runtime_cache_path}")
            return
        try:
            from filelock import FileLock

            lock = FileLock(self.runtime_cache_path + ".lock")
            with lock.acquire(timeout=10):
                with open(self.runtime_cache_path, "rb") as f:
                    data = f.read()
            if data:
                self.runtime_cache.deserialize(data)
                logger.info(f"Loaded runtime cache from {self.runtime_cache_path}")
        except Exception as e:
            logger.warning(f"Failed to load runtime cache: {e}")

    def _save_runtime_cache(self) -> None:
        """Save runtime cache to disk (with an exclusive file lock)."""
        if getattr(self, "runtime_cache", None) is None:
            return
        try:
            host_mem = self.runtime_cache.serialize()
            if host_mem is None:
                return
            os.makedirs(os.path.dirname(self.runtime_cache_path), exist_ok=True)

            from filelock import FileLock

            lock = FileLock(self.runtime_cache_path + ".lock")
            with lock.acquire(timeout=10):
                with open(self.runtime_cache_path, "wb") as f:
                    f.write(memoryview(host_mem))
            logger.info(f"Saved runtime cache to {self.runtime_cache_path}")
        except Exception as e:
            logger.warning(f"Failed to save runtime cache: {e}")

    def _is_monolithic_capturable(self, stream: torch.cuda.Stream) -> bool:
        """Return True iff manual ``torch.cuda.CUDAGraph`` capture is safe.

        Non-RTX builds always return True (existing behavior). On RTX,
        capture is unsafe when the TRT-RTX context cannot be stream-captured
        (e.g. due to runtime allocation or data-dependent shapes) or when
        the dynamic-shape strategy is ``"lazy"`` -- a later lazy-compiled
        specialized kernel would invalidate the captured graph.
        """
        if not ENABLED_FEATURES.tensorrt_rtx:
            return True
        if not self.context.is_stream_capturable(stream.cuda_stream):
            return False
        if self.settings.dynamic_shapes_kernel_specialization_strategy == "lazy":
            return False
        return True

    def _enable_rtx_native_cudagraphs(self) -> None:
        """Switch this engine to TRT-RTX native CUDA graphs.

        Sets the runtime config's ``cuda_graph_strategy`` to
        ``WHOLE_GRAPH_CAPTURE`` and rebuilds the execution context so it
        picks up the new strategy. No-op on non-RTX or when the runtime
        config is not present.
        """
        if self.runtime_config is None:
            return
        self.runtime_config.cuda_graph_strategy = _get_cuda_graph_strategy(
            "whole_graph_capture"
        )
        self.context = self._create_execution_context()
        self._rtx_native_cudagraphs = True
        logger.info("Switched to TRT-RTX native CUDA graphs")

    # --- distributed / NCCL ---

    @property
    def is_distributed(self) -> bool:
        """Check if this engine is running inside an active distributed context."""
        return bool(
            dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1
        )

    def setup_nccl_comm(self) -> None:
        """Set up NCCL communicator from the active ProcessGroup.

        Uses the process group set by torch_tensorrt.distributed.distributed_context() if
        active, otherwise falls back to the default world group.
        Called lazily on first forward pass for distributed engines.
        """
        from torch_tensorrt.distributed._distributed import get_active_group

        if not self.is_distributed:
            return

        pg = get_active_group()
        if pg is None or dist.get_backend(pg) != "nccl":
            raise RuntimeError(
                "Active ProcessGroup must use NCCL backend. "
                "Use torch_tensorrt.distributed.distributed_context(group) to select a non-default group."
            )

        backend = pg._get_backend(torch.device("cuda"))

        # Force NCCL communicator initialization with a dummy collective.
        # Must use group=pg so the correct group's comm is initialized;
        # dist.all_reduce without group= only initializes the default world group.
        dummy = torch.zeros(1, device="cuda")
        dist.all_reduce(dummy, group=pg)

        comm_ptr = backend._comm_ptr()
        if comm_ptr is None or comm_ptr == 0:
            raise RuntimeError("Failed to get NCCL communicator from ProcessGroup")

        self._nccl_comm = comm_ptr

        # Bind communicator to TRT execution context (PyCapsule required by TRT Python API)
        if self.context is not None:
            import ctypes

            ctypes.pythonapi.PyCapsule_New.restype = ctypes.py_object
            ctypes.pythonapi.PyCapsule_New.argtypes = [
                ctypes.c_void_p,
                ctypes.c_char_p,
                ctypes.c_void_p,
            ]
            comm_capsule = ctypes.pythonapi.PyCapsule_New(comm_ptr, None, None)
            ok = self.context.set_communicator(comm_capsule)
            if not ok:
                raise RuntimeError(
                    f"TRT context.set_communicator() returned False for rank={dist.get_rank()}. "
                    f"comm_ptr={comm_ptr:#x}. Failed to bind NCCL communicator to TRT execution context."
                )

        logger.info(
            f"NCCL comm set up (rank={dist.get_rank()}, world_size={dist.get_world_size()})"
        )

    # --- weight streaming (mirrors C++ engine surface) ---

    @property
    def streamable_device_memory_budget(self) -> Any:
        return self.cuda_engine.streamable_weights_size

    @property
    def automatic_device_memory_budget(self) -> Any:
        return self.cuda_engine.get_weight_streaming_automatic_budget()

    @property
    def device_memory_budget(self) -> Any:
        return self.cuda_engine.weight_streaming_budget_v2

    @device_memory_budget.setter
    def device_memory_budget(self, budget_bytes: int) -> None:
        if budget_bytes < 0:
            budget_bytes = self.streamable_device_memory_budget
        self.cuda_engine.weight_streaming_budget_v2 = budget_bytes
        if self.cuda_engine.weight_streaming_budget_v2 != budget_bytes:
            logger.error(f"Failed to set weight streaming budget to {budget_bytes}")
        self.context = self._create_execution_context()
        self.runtime_states.context_changed = True

    def reset_captured_graph(self) -> None:
        if self.cudagraph:
            self.cudagraph.reset()
            self.cudagraph = None

    def use_dynamically_allocated_resources(self, dynamic: bool = False) -> None:
        new_strategy = 1 if dynamic else 0
        if self.resource_allocation_strategy == new_strategy:
            return
        self.resource_allocation_strategy = new_strategy
        self.context = self._create_execution_context()
        self.runtime_states.context_changed = True

    def set_output_tensors_as_unowned(self, enabled: bool) -> None:
        self.output_tensors_are_unowned = enabled

    def are_output_tensors_unowned(self) -> bool:
        return self.output_tensors_are_unowned

    # --- profiling / inspection ---

    def enable_profiling(self) -> None:
        if not self.context.profiler:
            self.context.profiler = trt.Profiler()
        self._profile_execution = True

    def set_profile_format(self, profile_format: str) -> None:
        if profile_format not in ["cudagraph", "trex", "perfetto"]:
            raise ValueError(f"Invalid profile format: {profile_format}")

    def disable_profiling(self) -> None:
        torch.cuda.synchronize()
        self.context = self._create_execution_context()
        self._profile_execution = False
        self.runtime_states.context_changed = True

    def get_engine_layer_info(self) -> str:
        inspector = self.cuda_engine.create_engine_inspector()
        return str(inspector.get_engine_information(trt.LayerInformationFormat.JSON))

    def dump_engine_layer_info(self) -> None:
        print(self.get_engine_layer_info())

    def dump_engine_layer_info_to_file(self, path: str) -> None:
        with open(path, "w") as f:
            f.write(self.get_engine_layer_info())

    def infer_outputs(self, input_shapes: List[Any]) -> List[Any]:
        """Return output shapes inferred for the given input shapes."""
        results = []
        for i, input_name in enumerate(self.in_binding_names):
            if i < len(input_shapes):
                self.context.set_input_shape(input_name, tuple(input_shapes[i]))
        for output_name in self.out_binding_names:
            results.append(tuple(self.context.get_tensor_shape(output_name)))
        return results

    # --- tensor binding helpers ---

    def validate_input_shapes(self, inputs: Sequence[torch.Tensor]) -> bool:
        new_shape_key = "".join(str(tuple(t.shape)).replace(" ", "") for t in inputs)
        if new_shape_key != self.shape_key:
            logger.debug(f"Input shape changed {self.shape_key} -> {new_shape_key}")
            self.shape_key = new_shape_key
            return True
        return False

    def create_output_allocator(self) -> None:
        if self.output_allocator is None:
            self.output_allocator = DynamicOutputAllocator(
                {
                    name: self.output_dtypes[idx]
                    for idx, name in enumerate(self.out_binding_names)
                }
            )

    def create_output_tensors(self) -> List[torch.Tensor]:
        return [
            torch.empty(
                size=self.output_shapes[idx],
                dtype=self.output_dtypes[idx],
                device=torch.cuda.current_device(),
            )
            for idx, _ in enumerate(self.out_binding_names)
        ]

    def setup_input_tensors(
        self,
        contiguous_inputs: List[torch.Tensor],
        cudagraphs_enabled: bool,
        need_cudagraphs_record: bool,
    ) -> None:
        for i, input_name in enumerate(self.in_binding_names):

            assert (
                contiguous_inputs[i].dtype == self.input_dtypes[i]
            ), f"Dtype mismatch for input {input_name}. Expect {self.input_dtypes[i]}, got {contiguous_inputs[i].dtype}."

            if need_cudagraphs_record:
                self._input_buffers[i] = contiguous_inputs[i].clone()

            if self.is_shape_inference_io[input_name]:
                inputs_cpu = contiguous_inputs[i].cpu().to(torch.int64).numpy().copy()
                self.context.set_tensor_address(input_name, inputs_cpu.ctypes.data)
            else:
                self.context.set_input_shape(
                    input_name, tuple(contiguous_inputs[i].shape)
                )
                tensor_to_bind = contiguous_inputs[i]
                if tensor_to_bind.numel() == 0:
                    if self._empty_tensor_placeholder is None:
                        self._empty_tensor_placeholder = torch.empty(
                            1,
                            dtype=tensor_to_bind.dtype,
                            device=torch.cuda.current_device(),
                        )
                    tensor_to_bind = self._empty_tensor_placeholder

                if cudagraphs_enabled:
                    self._input_buffers[i].copy_(contiguous_inputs[i])
                    self.context.set_tensor_address(
                        input_name, self._input_buffers[i].data_ptr()
                    )
                else:
                    self.context.set_tensor_address(
                        input_name, tensor_to_bind.data_ptr()
                    )

    def _profile_section(self, label: str) -> ContextManager[None]:
        if self._profile_execution:
            return cast(
                ContextManager[None],
                torch.autograd.profiler.record_function(label),
            )
        return nullcontext()

    # --- execution ---

    def _prepare_streams(self, contiguous_inputs: List[torch.Tensor]) -> bool:
        """Pick the engine stream relative to the caller's current stream.

        If the caller is on the default stream we keep the legacy behavior of
        running the engine on a dedicated pool stream (and synchronising via
        wait_stream).  If the caller is on a non-default stream (e.g. a stream
        attached to a CUDA Green Context) we honor it by reusing the caller's
        stream for the engine, so the caller's scheduling choice (e.g. SM
        partitioning) is preserved end to end and no wait_stream sync is
        needed.

        Returns ``caller_on_default`` so call sites can gate the wait_stream
        pair on it.  Also flips ``runtime_states.context_changed`` whenever
        the engine stream changes while CUDA graphs are enabled so the
        current invocation re-records the graph against the new stream.
        Call sites MUST invoke this before ``runtime_states.set_runtime_states``
        because that call consumes and resets ``context_changed``.
        """
        current_device = (
            contiguous_inputs[0].device
            if contiguous_inputs and contiguous_inputs[0].is_cuda
            else self._target_device
        )
        default_stream = self._default_stream
        previous_engine_stream = self._engine_stream
        self._caller_stream = torch.cuda.current_stream(current_device)
        caller_on_default = self._caller_stream == default_stream
        if caller_on_default:
            if self._owned_pool_stream is None:
                self._owned_pool_stream = torch.cuda.Stream(self._target_device)
            self._engine_stream = self._owned_pool_stream
        else:
            # Honor caller's non-default stream so its scheduling choice (e.g.
            # SM partitioning via a CUDA Green Context) is preserved end to
            # end.
            self._engine_stream = self._caller_stream
        if (
            torch_tensorrt.runtime.get_cudagraphs_mode()
            and self._engine_stream != previous_engine_stream
        ):
            # Captured CUDA graph was recorded against the old stream.
            self.runtime_states.context_changed = True
        return caller_on_default

    def _execute_standard(
        self, contiguous_inputs: List[torch.Tensor]
    ) -> torch.Tensor | Tuple[torch.Tensor, ...]:
        # On RTX, manual ``torch.cuda.CUDAGraph`` capture is not safe
        # (lazy kernel specialization can invalidate captured graphs and
        # runtime allocation can prevent stream capture). If the user
        # requested SUBGRAPH cudagraphs without explicitly setting
        # ``cuda_graph_strategy="whole_graph_capture"``, transparently
        # switch to RTX-native CUDA graphs and warn.
        cudagraphs_enabled = torch_tensorrt.runtime.get_cudagraphs_mode()
        if (
            ENABLED_FEATURES.tensorrt_rtx
            and cudagraphs_enabled
            and not self._rtx_native_cudagraphs
        ):
            logger.warning(
                "Manual CUDA graph capture is not guaranteed to work on "
                "TRT-RTX (lazy kernel specialization or non-capturable "
                "stream). Switching to TRT-RTX native CUDA graphs. Set "
                'cuda_graph_strategy="whole_graph_capture" at compile '
                "time to avoid this warning."
            )
            self._enable_rtx_native_cudagraphs()

        # ``effective_cudagraphs`` is the value the downstream record/replay
        # paths should react to. When RTX native is active, TRT-RTX is
        # already handling capture/replay internally, so the manual
        # ``torch.cuda.CUDAGraph`` machinery must stay quiet.
        effective_cudagraphs = cudagraphs_enabled and not self._rtx_native_cudagraphs

        # Pick the engine stream BEFORE set_runtime_states so that any
        # stream-identity change observed this call flips
        # runtime_states.context_changed in time to trigger same-call
        # cudagraph recapture (set_runtime_states consumes and resets the
        # flag).
        caller_on_default = self._prepare_streams(contiguous_inputs)
        shape_changed = self.validate_input_shapes(contiguous_inputs)
        (
            need_cudagraphs_record,
            can_use_pre_allocated_outputs,
            need_cudagraphs_reset,
        ) = self.runtime_states.set_runtime_states(
            effective_cudagraphs,
            self.use_pre_allocated_outputs,
            shape_changed,
        )

        if need_cudagraphs_reset:
            self.reset_captured_graph()

        if need_cudagraphs_record:
            self._input_buffers = [None] * len(self.in_binding_names)
            self._output_buffers = [None] * len(self.out_binding_names)

        with self._profile_section("TRTEngine:ProcessInputs"):
            self.setup_input_tensors(
                contiguous_inputs,
                effective_cudagraphs,
                need_cudagraphs_record,
            )
            if shape_changed:
                uninferred_input_names = self.context.infer_shapes()
                if uninferred_input_names:
                    logger.warning(
                        f"The shapes of the inputs: {uninferred_input_names} cannot be inferred and could lead to undefined behavior."
                    )

        with self._profile_section("TRTEngine:ProcessOutputs"):
            if can_use_pre_allocated_outputs:
                outputs = self.pre_allocated_outputs
            else:
                self.output_shapes = [
                    tuple(self.context.get_tensor_shape(output_name))
                    for output_name in self.out_binding_names
                ]
                if any(-1 in shape for shape in self.output_shapes):
                    raise ValueError(
                        "Encountered dynamic output shapes during runtime. This could mean the network has data-dependent output shapes which is not currently supported."
                    )
                outputs = self.create_output_tensors()

            for o, output_name in enumerate(self.out_binding_names):
                if need_cudagraphs_record:
                    self._output_buffers[o] = outputs[o].clone()
                if effective_cudagraphs:
                    self.context.set_tensor_address(
                        output_name, self._output_buffers[o].data_ptr()
                    )
                else:
                    self.context.set_tensor_address(output_name, outputs[o].data_ptr())

        with self._profile_section("TRTEngine:TensorRTRuntime"):
            if caller_on_default:
                self._engine_stream.wait_stream(self._caller_stream)
            with torch.cuda.stream(self._engine_stream):
                if self.resource_allocation_strategy:
                    self._dynamic_workspace = torch.empty(
                        self.cuda_engine.device_memory_size_v2,
                        dtype=torch.uint8,
                        device=torch.cuda.current_device(),
                    )
                    self.context.set_device_memory(self._dynamic_workspace.data_ptr())

                if effective_cudagraphs:
                    if need_cudagraphs_record:
                        self.cudagraph = torch.cuda.CUDAGraph()
                        if self._profile_execution:
                            self.cudagraph.enable_debug_mode()
                        with torch.cuda.graph(
                            self.cudagraph, stream=self._engine_stream
                        ):
                            self.context.execute_async_v3(
                                self._engine_stream.cuda_stream
                            )
                        if self._profile_execution:
                            self.cudagraph.debug_dump(
                                f"{DEBUG_LOGGING_DIR}/{self.name}_cudagraph.dot"
                            )
                    self.cudagraph.replay()  # type: ignore[union-attr]
                else:
                    self.context.execute_async_v3(self._engine_stream.cuda_stream)

            if caller_on_default:
                self._caller_stream.wait_stream(self._engine_stream)

        if self.use_pre_allocated_outputs and (
            self.output_tensors_are_unowned
            or not self.pre_allocated_outputs
            or shape_changed
        ):
            self.pre_allocated_outputs = self.create_output_tensors()

        if effective_cudagraphs:
            for idx, output in enumerate(outputs):
                output.copy_(self._output_buffers[idx])

        if len(outputs) == 1:
            return outputs[0]
        return tuple(outputs)

    def _execute_output_allocator(
        self, contiguous_inputs: List[torch.Tensor]
    ) -> torch.Tensor | Tuple[torch.Tensor, ...]:
        if torch_tensorrt.runtime.get_cudagraphs_mode():
            raise RuntimeError(
                "Both CUDA Graphs and dynamic output allocation are enabled, which are "
                "incompatible runtime modes. Please disable one of the two."
            )

        with self._profile_section("TRTEngine:ProcessInputs"):
            self.setup_input_tensors(contiguous_inputs, False, False)

        with self._profile_section("TRTEngine:SetupOutputAllocator"):
            self.create_output_allocator()
            for output_name in self.out_binding_names:
                if not self.context.set_output_allocator(
                    output_name, self.output_allocator
                ):
                    raise RuntimeError(
                        f"Failed to set output allocator for {output_name}"
                    )

        caller_on_default = self._prepare_streams(contiguous_inputs)

        with self._profile_section("TRTEngine:TensorRTRuntime"):
            if caller_on_default:
                self._engine_stream.wait_stream(self._caller_stream)
            with torch.cuda.stream(self._engine_stream):
                self.context.execute_async_v3(self._engine_stream.cuda_stream)
            if caller_on_default:
                self._caller_stream.wait_stream(self._engine_stream)

        outputs = []
        assert self.output_allocator is not None
        for idx, output_name in enumerate(self.out_binding_names):
            shape = self.output_allocator.shapes.get(output_name, None)
            dtype_ = self.output_dtypes[idx]
            buffer_tensor = self.output_allocator.buffers.get(output_name)
            assert buffer_tensor is not None
            output = buffer_tensor.clone().detach()
            prod = int(torch.prod(torch.tensor(shape)))
            output = output.reshape(-1).view(dtype_)[:prod].reshape(shape)
            outputs.append(output)

        if len(outputs) == 1:
            return outputs[0]
        return tuple(outputs)

    def execute(
        self, inputs: Sequence[torch.Tensor]
    ) -> torch.Tensor | Tuple[torch.Tensor, ...]:
        contiguous_inputs = [tensor.contiguous() for tensor in inputs]

        if self._has_nccl_ops and self._nccl_comm is None:
            nccl_type = (
                "native TRT collectives"
                if ENABLED_FEATURES.native_trt_collectives
                else (
                    "TRT-LLM NCCL plugins"
                    if ENABLED_FEATURES.trtllm_for_nccl
                    else "unknown backend"
                )
            )
            logger.info(
                f"Setting up NCCL for distributed execution using {nccl_type} "
                f"(rank={dist.get_rank()}, world_size={dist.get_world_size()})"
            )
            self.setup_nccl_comm()
            logger.info(f"NCCL setup complete, comm={self._nccl_comm}")

        if torch_tensorrt.runtime._multi_device_safe_mode._PY_RT_MULTI_DEVICE_SAFE_MODE:
            curr_device_id = torch.cuda.current_device()
            curr_device_properties = torch.cuda.get_device_properties(curr_device_id)
            if _is_switch_required(
                curr_device_id,
                self.target_device_id,
                curr_device_properties,
                self.target_device_properties,
            ):
                device_id, _ = _select_rt_device(
                    curr_device_id,
                    self.target_device_id,
                    self.target_device_properties,
                )
                device = torch.device(device_id)
                torch.cuda.set_device(device_id)
                contiguous_inputs = [tensor.to(device) for tensor in contiguous_inputs]
                logger.warning(f"Moved all input Tensors to cuda:{device_id}")

        if self.requires_output_allocator or self.use_output_allocator_outputs:
            logger.debug("Using the dynamic allocator runtime mode.")
            return self._execute_output_allocator(contiguous_inputs)

        logger.debug(
            f"Using the standard execution runtime mode with cudagraphs={torch_tensorrt.runtime.get_cudagraphs_mode()}"
            + (" (RTX native)" if self._rtx_native_cudagraphs else "")
        )
        return self._execute_standard(contiguous_inputs)


# register_opaque_type(EngineSerializer, typ="reference")


register_opaque_type(TRTEngine, typ="reference")


@torch.library.custom_op(  # type: ignore[misc]
    "tensorrt::execute_engine_python", mutates_args=()
)
def execute_engine_python(
    input_tensors: List[torch.Tensor], engine: TRTEngine
) -> List[torch.Tensor]:
    outputs = engine.execute(input_tensors)
    return [outputs] if isinstance(outputs, torch.Tensor) else list(outputs)


@execute_engine_python.register_fake  # type: ignore[misc]
def execute_engine_python_fake(
    input_tensors: List[torch.Tensor], engine: TRTEngine
) -> List[torch.Tensor]:
    """Abstract/fake kernel for ``tensorrt::execute_engine``.

    Called by FakeTensor propagation and ``torch.export`` to infer output
    shapes and dtypes without executing the real TRT engine.  Output shapes
    are obtained by asking the engine's execution context to propagate the
    concrete input shapes symbolically; dtypes come from the engine's
    pre-parsed output dtype list.
    """
    input_shapes = [list(t.shape) for t in input_tensors]
    try:
        output_shapes = engine.infer_outputs(input_shapes)
    except Exception:
        # Fall back to the statically-stored shapes when shape inference is
        # unavailable (e.g. engine context not yet initialised in meta mode).
        output_shapes = [list(s) for s in engine.output_shapes]

    return [
        torch.empty(
            shape, dtype=engine.output_dtypes[i], device=input_tensors[0].device
        )
        for i, shape in enumerate(output_shapes)
    ]
