from __future__ import annotations

import base64
import copy
import logging
import pickle
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import torch
from torch_tensorrt._Device import Device
from torch_tensorrt._enums import Platform
from torch_tensorrt._features import ENABLED_FEATURES
from torch_tensorrt.dynamo._settings import CompilationSettings
from torch_tensorrt.dynamo.runtime._PythonTRTEngine import PythonTRTEngine
from torch_tensorrt.dynamo.runtime._RuntimeBackendSelection import (
    RuntimeBackend,
    _normalize_runtime_backend,
    get_runtime_backend,
)
from torch_tensorrt.dynamo.runtime._serialized_engine_layout import (
    ABI_TARGET_IDX,
    ABI_VERSION,
    DEVICE_IDX,
    ENGINE_IDX,
    HW_COMPATIBLE_IDX,
    INPUT_BINDING_NAMES_IDX,
    NAME_IDX,
    OUTPUT_BINDING_NAMES_IDX,
    REQUIRES_OUTPUT_ALLOCATOR_IDX,
    RESOURCE_ALLOCATION_STRATEGY_IDX,
    SERIALIZATION_LEN,
    SERIALIZED_METADATA_IDX,
    TARGET_PLATFORM_IDX,
    SerializedTensorRTEngineFmt,
    serialize_binding_names,
    serialize_device_info,
)

logger = logging.getLogger(__name__)

SerializedTorchTensorRTModuleFmt = Tuple[
    str,
    Optional[SerializedTensorRTEngineFmt],
    List[str],
    List[str],
    Optional[str],
]
# Checkpoints written before the trailing ``runtime_backend`` slot used four elements.
_LegacyTorchTensorRTModuleExtraState = Tuple[
    str, Optional[SerializedTensorRTEngineFmt], List[str], List[str]
]
TorchTensorRTModuleExtraState = Union[
    SerializedTorchTensorRTModuleFmt,
    _LegacyTorchTensorRTModuleExtraState,
]


class TorchTensorRTModule(torch.nn.Module):  # type: ignore[misc]
    """``nn.Module`` that runs a TensorRT engine inside PyTorch.

    Execution uses either the C++ Torch-TensorRT runtime (``torch.classes.tensorrt.Engine``)
    or the Python TRT stack (``tensorrt`` + ``execute_engine_python``), depending on
    :func:`~torch_tensorrt.runtime.get_runtime_backend` (set via
    :func:`~torch_tensorrt.runtime.set_runtime_backend` as a context manager for scoped
    changes). The backend is read from :func:`~torch_tensorrt.runtime.get_runtime_backend`
    when the module is constructed (and from checkpoint metadata on load).

    Supports ``torch.save`` / ``torch.load`` via ``get_extra_state`` / ``set_extra_state``.
    Extra state is a 5-tuple; the last element is ``runtime_backend`` (enum value as
    ``str``) when an engine is saved, or ``None`` when there is no engine. If the fifth
    element is missing (legacy 4-tuple with an engine), the C++ backend is used.
    """

    def __init__(
        self,
        serialized_engine: Optional[bytes] = None,
        input_binding_names: Optional[List[str]] = None,
        output_binding_names: Optional[List[str]] = None,
        *,
        name: str = "",
        settings: CompilationSettings = CompilationSettings(),
        weight_name_map: Optional[dict[Any, Any]] = None,
        requires_output_allocator: bool = False,
        symbolic_shape_expressions: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    ) -> None:
        """Build the module from serialized engine bytes and binding metadata.

        Args:
            serialized_engine: Raw TRT engine bytes (``None`` if restoring state only).
            input_binding_names: Input tensor names in ``forward`` order.
            output_binding_names: Output tensor names in return order.
            name: Logical name for logging and serialization.
            settings: Compilation/runtime settings (device, lazy init, cross-compile, etc.).
            weight_name_map: Engine weight name to ``state_dict`` key mapping (refit).
            requires_output_allocator: Engine needs TRT dynamic output allocation.
            symbolic_shape_expressions: Optional symbolic shape metadata from compile.
        """
        super().__init__()

        self.input_binding_names = (
            input_binding_names if input_binding_names is not None else []
        )
        self.output_binding_names = (
            output_binding_names if output_binding_names is not None else []
        )
        self.name = name
        self.hardware_compatible = settings.hardware_compatible
        self.settings = copy.deepcopy(settings)
        self.weight_name_map = weight_name_map
        self.serialized_engine = serialized_engine
        self.engine: Optional[Any] = None
        self.requires_output_allocator = requires_output_allocator
        self.dynamically_allocate_resources = settings.dynamically_allocate_resources
        self.symbolic_shape_expressions = symbolic_shape_expressions
        self.target_platform = (
            Platform.current_platform()
            if not self.settings.enable_cross_compile_for_windows
            else Platform.WIN_X86_64
        )
        self._runtime_backend = get_runtime_backend()
        self.profiling_enabled = False

        if (
            serialized_engine
            and not self.settings.lazy_engine_init
            and not self.settings.enable_cross_compile_for_windows
        ):
            self.setup_engine()

    def _require_engine(self) -> Any:
        if self.engine is None:
            raise RuntimeError("Engine has not been setup yet.")
        return self.engine

    @property
    def _is_python_runtime(self) -> bool:
        return self._runtime_backend is RuntimeBackend.PYTHON

    def _cleanup_engine(self) -> None:
        engine = getattr(self, "engine", None)
        if engine is not None and hasattr(engine, "close"):
            engine.close()
        self.engine = None

    def _pack_engine_info(self) -> List[str | bytes]:
        target_device = (
            self.settings.device
            if self.settings.device is not None
            else Device._current_device()
        )
        metadata = {
            "settings": self.settings,
            "weight_name_map": self.weight_name_map,
            "inout_symexprs": self.symbolic_shape_expressions,
            "output_tensors_are_unowned": (
                False
                if self.engine is None
                else self.engine.are_output_tensors_unowned()
            ),
        }

        engine_info: List[str | bytes] = [""] * SERIALIZATION_LEN
        engine_info[ABI_TARGET_IDX] = (
            torch.ops.tensorrt.ABI_VERSION()
            if ENABLED_FEATURES.torch_tensorrt_runtime
            else ABI_VERSION
        )
        engine_info[NAME_IDX] = (
            self.name + "_engine" if self.name != "" else "tensorrt_engine"
        )
        engine_info[DEVICE_IDX] = (
            target_device._to_serialized_rt_device()
            if ENABLED_FEATURES.torch_tensorrt_runtime
            else serialize_device_info(target_device)
        )
        assert self.serialized_engine is not None
        engine_info[ENGINE_IDX] = self.serialized_engine
        engine_info[INPUT_BINDING_NAMES_IDX] = serialize_binding_names(
            self.input_binding_names
        )
        engine_info[OUTPUT_BINDING_NAMES_IDX] = serialize_binding_names(
            self.output_binding_names
        )
        engine_info[HW_COMPATIBLE_IDX] = str(int(self.hardware_compatible))
        engine_info[SERIALIZED_METADATA_IDX] = self.encode_metadata(metadata)
        engine_info[TARGET_PLATFORM_IDX] = (
            self.target_platform._to_serialized_rt_platform()
            if ENABLED_FEATURES.torch_tensorrt_runtime
            else str(self.target_platform)
        )
        engine_info[REQUIRES_OUTPUT_ALLOCATOR_IDX] = str(
            int(self.requires_output_allocator)
        )
        logger.info(
            f"PROVIDED RESOURCE ALLOCATION STRATEGY: {self.dynamically_allocate_resources}"
        )
        engine_info[RESOURCE_ALLOCATION_STRATEGY_IDX] = str(
            int(self.dynamically_allocate_resources)
        )
        return engine_info

    def get_streamable_device_memory_budget(self) -> Any:
        return self._require_engine().streamable_device_memory_budget

    def get_automatic_device_memory_budget(self) -> Any:
        return self._require_engine().automatic_device_memory_budget

    def get_device_memory_budget(self) -> Any:
        return self._require_engine().device_memory_budget

    def set_device_memory_budget(self, budget_bytes: int) -> int:
        engine = self._require_engine()
        if budget_bytes < 0:
            budget_bytes = self.get_streamable_device_memory_budget()
        engine.device_memory_budget = budget_bytes
        if engine.device_memory_budget != budget_bytes:
            logger.error(f"Failed to set weight streaming budget to {budget_bytes}")
            budget_bytes = engine.device_memory_budget
        if self.get_streamable_device_memory_budget() == budget_bytes:
            logger.warning("Weight streaming is disabled")
        return budget_bytes

    def _reset_captured_graph(self) -> None:
        self._require_engine().reset_captured_graph()

    def use_dynamically_allocated_resources(
        self, dynamically_allocate_resources: bool = False
    ) -> None:
        self.dynamically_allocate_resources = dynamically_allocate_resources
        self._require_engine().use_dynamically_allocated_resources(
            self.dynamically_allocate_resources
        )

    def setup_engine(self) -> None:
        """
        Setup engine for a module which has deferred engine setup.

        Will setup the TensorRT engine for this module in the case that setup has been
        deferred. In the case that the engine has already been setup, will return without
        changing anything. Assumes that serialized engine and settings have already been passed
        to the module.
        """
        if self.engine is not None:
            return

        if self._is_python_runtime:
            self.engine = PythonTRTEngine(
                self._pack_engine_info(),
                profile_execution=self.profiling_enabled,
            )
            return

        if not ENABLED_FEATURES.torch_tensorrt_runtime:
            raise NotImplementedError("Torch-TensorRT Runtime is not available")
        self.engine = torch.classes.tensorrt.Engine(self._pack_engine_info())

    def encode_metadata(self, metadata: Any) -> str:
        metadata = copy.deepcopy(metadata)
        dumped_metadata = pickle.dumps(metadata)
        encoded_metadata = base64.b64encode(dumped_metadata).decode("utf-8")
        return encoded_metadata

    @staticmethod
    def decode_metadata(encoded_metadata: bytes | str) -> Any:
        if isinstance(encoded_metadata, str):
            encoded_metadata = encoded_metadata.encode("utf-8")
        return pickle.loads(base64.b64decode(encoded_metadata))

    def get_extra_state(self) -> SerializedTorchTensorRTModuleFmt:
        """Return payload for ``torch.save`` (engine blob base64-encoded in the packed list)."""
        if self.engine or self.serialized_engine:
            engine_info = self._pack_engine_info()
            raw_engine_blob = engine_info[ENGINE_IDX]
            assert isinstance(raw_engine_blob, (bytes, bytearray))
            engine_info[ENGINE_IDX] = base64.b64encode(raw_engine_blob)
            return (
                self.name,
                engine_info,
                self.input_binding_names,
                self.output_binding_names,
                self._runtime_backend.value,
            )
        return (
            self.name,
            None,
            self.input_binding_names,
            self.output_binding_names,
            None,
        )

    def set_extra_state(self, state: TorchTensorRTModuleExtraState) -> None:
        """Restore module fields and engine from ``torch.load`` extra state."""
        self._cleanup_engine()
        self.name = state[0]
        self.input_binding_names = state[2]
        self.output_binding_names = state[3]
        if len(state) not in (4, 5):
            raise ValueError(
                "Invalid TorchTensorRTModule extra_state: expected 4 (legacy) or 5 "
                f"elements when engine_info is None, got {len(state)}"
            )

        if state[1] is None:
            self.serialized_engine = None
            self.settings = CompilationSettings()
            self.weight_name_map = None
            self.hardware_compatible = False
            self.requires_output_allocator = False
            self.dynamically_allocate_resources = False
            self.symbolic_shape_expressions = None
            self.target_platform = Platform.current_platform()
            self.profiling_enabled = False
            return

        serialized_engine_info: SerializedTensorRTEngineFmt = list(state[1])
        metadata = TorchTensorRTModule.decode_metadata(
            serialized_engine_info[SERIALIZED_METADATA_IDX]
        )
        raw_backend = state[4] if len(state) == 5 else None
        if raw_backend is None:
            raw_backend = RuntimeBackend.CPP
        runtime_backend = _normalize_runtime_backend(raw_backend)
        self._runtime_backend = runtime_backend

        encoded_engine = serialized_engine_info[ENGINE_IDX]
        decoded_engine = base64.b64decode(encoded_engine)
        serialized_engine_info[ENGINE_IDX] = decoded_engine
        self.serialized_engine = decoded_engine
        self.hardware_compatible = bool(int(serialized_engine_info[HW_COMPATIBLE_IDX]))
        self.requires_output_allocator = bool(
            int(serialized_engine_info[REQUIRES_OUTPUT_ALLOCATOR_IDX])
        )
        self.dynamically_allocate_resources = bool(
            int(serialized_engine_info[RESOURCE_ALLOCATION_STRATEGY_IDX])
        )
        self.settings = metadata["settings"]
        self.weight_name_map = metadata["weight_name_map"]
        self.symbolic_shape_expressions = metadata["inout_symexprs"]
        self.target_platform = (
            Platform.WIN_X86_64
            if self.settings.enable_cross_compile_for_windows
            else Platform.current_platform()
        )
        self.profiling_enabled = False

        if runtime_backend is RuntimeBackend.PYTHON:
            self.engine = PythonTRTEngine(serialized_engine_info)
        else:
            if not ENABLED_FEATURES.torch_tensorrt_runtime:
                raise NotImplementedError("Torch-TensorRT Runtime is not available")
            self.engine = torch.classes.tensorrt.Engine(serialized_engine_info)

        self.engine.set_output_tensors_as_unowned(
            metadata["output_tensors_are_unowned"]
        )

    def __del__(self) -> None:
        self._cleanup_engine()

    def set_pre_allocated_outputs(self, enable: bool) -> None:
        self._require_engine().use_pre_allocated_outputs = enable

    def set_use_output_allocator(self, enable: bool) -> None:
        self._require_engine().use_output_allocator_outputs = enable

    def _execute_engine(self, input_tensors: List[torch.Tensor]) -> List[torch.Tensor]:
        """Dispatch to ``execute_engine`` or ``execute_engine_python``."""
        engine = self._require_engine()
        if self._is_python_runtime:
            return cast(
                List[torch.Tensor],
                torch.ops.tensorrt.execute_engine_python(list(input_tensors), engine),
            )
        return cast(
            List[torch.Tensor],
            torch.ops.tensorrt.execute_engine(list(input_tensors), engine),
        )

    def forward(self, *inputs: Any) -> torch.Tensor | Tuple[torch.Tensor, ...]:
        """Run the TensorRT engine on GPU tensors (non-tensor args are cast to CUDA tensors)."""
        if self.engine is None:
            raise RuntimeError("Engine has not been setup yet.")

        assert len(inputs) == len(
            self.input_binding_names
        ), f"Wrong number of inputs, expected {len(self.input_binding_names)} got {len(inputs)}."

        input_tensors: List[torch.Tensor] = [
            (value if isinstance(value, torch.Tensor) else torch.tensor(value).cuda())
            for value in inputs
        ]
        outputs = self._execute_engine(input_tensors)
        if len(outputs) == 1:
            return outputs[0]
        return tuple(outputs)

    def enable_profiling(
        self,
        profiling_results_dir: Optional[str] = None,
        profile_format: str = "perfetto",
    ) -> None:
        """Enable engine profiling (C++: optional Perfetto/TREx path prefix on disk)."""
        if self.engine is None:
            raise RuntimeError("Engine has not been initialized yet.")

        if not self._is_python_runtime and profiling_results_dir is not None:
            self.engine.profile_path_prefix = profiling_results_dir

        self.engine.enable_profiling()
        if hasattr(self.engine, "set_profile_format"):
            self.engine.set_profile_format(profile_format)
        self.profiling_enabled = True

    def set_output_tensors_as_unowned(self, enabled: bool) -> None:
        self._require_engine().set_output_tensors_as_unowned(enabled)

    def are_output_tensors_unowned(self) -> bool:
        return cast(bool, self._require_engine().are_output_tensors_unowned())

    def disable_profiling(self) -> None:
        """Disable engine profiling and clear the profiling flag on this module."""
        if self.engine is None:
            raise RuntimeError("Engine has not been initialized yet.")
        self.engine.disable_profiling()
        self.profiling_enabled = False

    def get_layer_info(self) -> str:
        """Return TRT layer information as a JSON string (TRT version dependent)."""
        return cast(str, self._require_engine().get_engine_layer_info())

    def dump_layer_info(self) -> None:
        """Print layer information for this engine to stdout."""
        if self.engine is None:
            raise RuntimeError("Engine has not been initialized yet.")
        self.engine.dump_engine_layer_info()
