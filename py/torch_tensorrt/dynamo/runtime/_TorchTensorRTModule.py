from __future__ import annotations

import base64
import copy
import logging
import pickle
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch_tensorrt._Device import Device
from torch_tensorrt._enums import Platform
from torch_tensorrt._features import ENABLED_FEATURES
from torch_tensorrt.dynamo._settings import CompilationSettings
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
]


class TorchTensorRTModule(torch.nn.Module):  # type: ignore[misc]
    """``nn.Module`` that runs a TensorRT engine inside PyTorch.

    When the C++ Torch-TensorRT runtime is available, execution uses
    ``torch.classes.tensorrt.Engine`` and ``torch.ops.tensorrt.execute_engine``.
    When only the Python runtime is available, a Python :class:`TRTEngine` is
    registered under the same ``tensorrt::execute_engine`` op so that the same
    compiled graph works with either runtime transparently.

    Supports ``torch.save`` / ``torch.load`` via ``get_extra_state`` / ``set_extra_state``.
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
        self.engine = None
        self.requires_output_allocator = requires_output_allocator
        self.dynamically_allocate_resources = settings.dynamically_allocate_resources
        self.symbolic_shape_expressions = symbolic_shape_expressions
        self.target_platform = (
            Platform.current_platform()
            if not self.settings.enable_cross_compile_for_windows
            else Platform.WIN_X86_64
        )
        self.profiling_enabled = False

        if (
            serialized_engine
            and not self.settings.lazy_engine_init
            and not self.settings.enable_cross_compile_for_windows
        ):
            self.setup_engine()

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
        return self.engine.streamable_device_memory_budget

    def get_automatic_device_memory_budget(self) -> Any:
        return self.engine.automatic_device_memory_budget

    def get_device_memory_budget(self) -> Any:
        return self.engine.device_memory_budget

    def set_device_memory_budget(self, budget_bytes: int) -> int:
        if budget_bytes < 0:
            budget_bytes = self.get_streamable_device_memory_budget()
        self.engine.device_memory_budget = budget_bytes
        if self.engine.device_memory_budget != budget_bytes:
            logger.error(f"Failed to set weight streaming budget to {budget_bytes}")
            budget_bytes = self.engine.device_memory_budget
        if self.get_streamable_device_memory_budget() == budget_bytes:
            logger.warning("Weight streaming is disabled")
        return budget_bytes

    def _reset_captured_graph(self) -> None:
        self.engine.reset_captured_graph()

    def use_dynamically_allocated_resources(
        self, dynamically_allocate_resources: bool = False
    ) -> None:
        self.dynamically_allocate_resources = dynamically_allocate_resources
        self.engine.use_dynamically_allocated_resources(
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

        if ENABLED_FEATURES.torch_tensorrt_runtime:
            self.engine = torch.classes.tensorrt.Engine(self._pack_engine_info())
        else:
            from torch_tensorrt.dynamo.runtime._PythonTRTEngine import TRTEngine

            self.engine = TRTEngine(
                self._pack_engine_info(),
                profile_execution=self.profiling_enabled,
            )  # type: ignore[assignment]

    def encode_metadata(self, metadata: Any) -> str:
        metadata = copy.deepcopy(metadata)
        dumped_metadata = pickle.dumps(metadata)
        encoded_metadata = base64.b64encode(dumped_metadata).decode("utf-8")
        return encoded_metadata

    @staticmethod
    def decode_metadata(encoded_metadata: bytes) -> Any:
        dumped_metadata = base64.b64decode(encoded_metadata.encode("utf-8"))
        metadata = pickle.loads(dumped_metadata)
        return metadata

    def get_extra_state(self) -> SerializedTorchTensorRTModuleFmt:
        if self.engine:
            engine_info = self._pack_engine_info()
            assert isinstance(engine_info[ENGINE_IDX], (bytes, bytearray))
            engine_info[ENGINE_IDX] = base64.b64encode(engine_info[ENGINE_IDX])
            return (
                self.name,
                engine_info,
                self.input_binding_names,
                self.output_binding_names,
            )
        elif self.serialized_engine:
            engine_info = self._pack_engine_info()
            assert isinstance(engine_info[ENGINE_IDX], bytes)
            engine_info[ENGINE_IDX] = base64.b64encode(engine_info[ENGINE_IDX])  # type: ignore[arg-type]
            return (
                self.name,
                engine_info,
                self.input_binding_names,
                self.output_binding_names,
            )
        else:
            return (
                self.name,
                None,
                self.input_binding_names,
                self.output_binding_names,
            )

    def set_extra_state(self, state: SerializedTorchTensorRTModuleFmt) -> None:
        self.name = state[0]

        if state[1] is not None:
            serialized_engine_info: SerializedTensorRTEngineFmt = list(state[1])
            serialized_engine_info[ENGINE_IDX] = base64.b64decode(
                serialized_engine_info[ENGINE_IDX]
            )
            self.hardware_compatible = bool(
                int(serialized_engine_info[HW_COMPATIBLE_IDX])
            )
            self.requires_output_allocator = bool(
                int(serialized_engine_info[REQUIRES_OUTPUT_ALLOCATOR_IDX])
            )

            serialized_metadata = serialized_engine_info[SERIALIZED_METADATA_IDX]
            assert isinstance(serialized_metadata, bytes)
            metadata = TorchTensorRTModule.decode_metadata(serialized_metadata)
            self.settings = metadata["settings"]
            self.weight_name_map = metadata["weight_name_map"]
            self.symbolic_shape_expressions = metadata["inout_symexprs"]

            if ENABLED_FEATURES.torch_tensorrt_runtime:
                self.engine = torch.classes.tensorrt.Engine(serialized_engine_info)
            else:
                from torch_tensorrt.dynamo.runtime._PythonTRTEngine import TRTEngine

                self.engine = TRTEngine(serialized_engine_info)  # type: ignore[assignment]

            self.engine.set_output_tensors_as_unowned(
                metadata["output_tensors_are_unowned"]
            )
        else:
            self.engine = None
            self.settings = CompilationSettings()
            self.hardware_compatible = False

        self.input_binding_names = state[2]
        self.output_binding_names = state[3]

    def set_pre_allocated_outputs(self, enable: bool) -> None:
        self.engine.use_pre_allocated_outputs = enable

    def set_use_output_allocator(self, enable: bool) -> None:
        self.engine.use_output_allocator_outputs = enable

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
        outputs = list(torch.ops.tensorrt.execute_engine(input_tensors, self.engine))
        if len(outputs) == 1:
            return outputs[0]
        return tuple(outputs)

    def enable_profiling(
        self,
        profiling_results_dir: Optional[str] = None,
        profile_format: str = "perfetto",
    ) -> None:
        """Enable engine profiling (optional path prefix and format for tracing output)."""
        if self.engine is None:
            raise RuntimeError("Engine has not been initialized yet.")

        if profiling_results_dir is not None:
            self.engine.profile_path_prefix = profiling_results_dir

        self.engine.enable_profiling()
        if hasattr(self.engine, "set_profile_format"):
            self.engine.set_profile_format(profile_format)
        self.profiling_enabled = True

    def set_output_tensors_as_unowned(self, enabled: bool) -> None:
        self.engine.set_output_tensors_as_unowned(enabled)

    def are_output_tensors_unowned(self) -> bool:
        return bool(self.engine.are_output_tensors_unowned())

    def disable_profiling(self) -> None:
        """Disable engine profiling and clear the profiling flag on this module."""
        if self.engine is None:
            raise RuntimeError("Engine has not been initialized yet.")
        self.engine.disable_profiling()
        self.profiling_enabled = False

    def get_layer_info(self) -> str:
        """Get a JSON string containing the layer information encoded by the TensorRT engine in this module

        Returns:

            str: A JSON string which contains the layer information of the engine incapsulated in this module
        """
        if self.engine is None:
            raise RuntimeError("Engine has not been initialized yet.")

        layer_info: str = self.engine.get_engine_layer_info()
        return layer_info

    def dump_layer_info(self) -> None:
        """Dump layer information encoded by the TensorRT engine in this module to STDOUT"""
        if self.engine is None:
            raise RuntimeError("Engine has not been initialized yet.")

        self.engine.dump_engine_layer_info()
