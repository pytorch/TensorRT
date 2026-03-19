"""Python-side TensorRT engine: deserialize, execute, and drive ``execute_engine_python``.

Serialization layout lives in :mod:`torch_tensorrt.dynamo.runtime._serialized_engine_layout`.
The engine is passed into ``tensorrt::execute_engine_python`` as an opaque reference (see
``register_opaque_type``), analogous to ``tensorrt::Engine`` for the C++ ``execute_engine`` op.
"""

from __future__ import annotations

import base64
import copy
import logging
import pickle
import tempfile
from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any, ContextManager, Dict, List, Optional, Sequence, Tuple, cast

import tensorrt as trt
import torch
import torch_tensorrt
from torch._library.opaque_object import register_opaque_type
from torch_tensorrt._enums import dtype
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

logger = logging.getLogger(__name__)

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
# PythonTRTEngine
# ---------------------------------------------------------------------------


class PythonTRTEngine:
    """TensorRT engine + execution context, driven from Python TRT APIs."""

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
        self.cudagraph: Optional[torch.cuda.CUDAGraph] = None
        self.shape_key: Optional[str] = None
        self._empty_tensor_placeholder: Optional[torch.Tensor] = None
        self._dynamic_workspace: Optional[torch.Tensor] = None
        self.runtime_states = TorchTRTRuntimeStates(
            torch_tensorrt.runtime.get_cudagraphs_mode()
        )
        self.resource_allocation_strategy = 0
        self._runtime_config = None

        self._load_serialized_info(serialized_info)
        self._setup_engine()

    def __deepcopy__(self, memo: dict[int, Any]) -> PythonTRTEngine:
        """Rebuild from serialized layout so ``copy.deepcopy`` skips unpickleable TRT handles."""
        if id(self) in memo:
            return memo[id(self)]  # type: ignore
        serialized_copy = copy.deepcopy(self.serialized_info, memo)
        dup = type(self)(serialized_copy, profile_execution=self._profile_execution)
        memo[id(self)] = dup
        return dup

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

        metadata = self.decode_metadata(self.serialized_metadata)
        self.settings = metadata.get("settings", CompilationSettings())
        self.weight_name_map = metadata.get("weight_name_map")
        self.symbolic_shape_expressions = metadata.get("inout_symexprs")
        self.output_tensors_are_unowned = metadata.get(
            "output_tensors_are_unowned", False
        )

        device_info = parse_device_info(self.serialized_device_info)
        self.target_device_id = device_info["id"]
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
        self.reset_captured_graph()

    def _create_execution_context(self) -> trt.IExecutionContext:
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
            if not contiguous_inputs[i].is_cuda:
                logger.warning(
                    f"Detected input {input_name} of engine {self.name} is not on a cuda device. "
                    "This tensor is being moved by the runtime but for performance considerations, "
                    "ensure your inputs are all on GPU and open an issue here "
                    "(https://github.com/pytorch/TensorRT/issues) if this warning persists."
                )
                contiguous_inputs[i] = contiguous_inputs[i].cuda()

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

    def _execute_standard(
        self, contiguous_inputs: List[torch.Tensor]
    ) -> torch.Tensor | Tuple[torch.Tensor, ...]:
        shape_changed = self.validate_input_shapes(contiguous_inputs)
        (
            need_cudagraphs_record,
            can_use_pre_allocated_outputs,
            need_cudagraphs_reset,
        ) = self.runtime_states.set_runtime_states(
            torch_tensorrt.runtime.get_cudagraphs_mode(),
            self.use_pre_allocated_outputs,
            shape_changed,
        )

        if need_cudagraphs_reset:
            self.reset_captured_graph()

        if need_cudagraphs_record:
            self._input_buffers = [None] * len(self.in_binding_names)
            self._output_buffers = [None] * len(self.out_binding_names)

        with self._profile_section("PythonTRTEngine:ProcessInputs"):
            self.setup_input_tensors(
                contiguous_inputs,
                torch_tensorrt.runtime.get_cudagraphs_mode(),
                need_cudagraphs_record,
            )
            if shape_changed:
                uninferred_input_names = self.context.infer_shapes()
                if uninferred_input_names:
                    logger.warning(
                        f"The shapes of the inputs: {uninferred_input_names} cannot be inferred and could lead to undefined behavior."
                    )

        with self._profile_section("PythonTRTEngine:ProcessOutputs"):
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
                if torch_tensorrt.runtime.get_cudagraphs_mode():
                    self.context.set_tensor_address(
                        output_name, self._output_buffers[o].data_ptr()
                    )
                else:
                    self.context.set_tensor_address(output_name, outputs[o].data_ptr())

        with self._profile_section("PythonTRTEngine:TensorRTRuntime"):
            self._caller_stream = torch.cuda.current_stream()
            if (
                self._engine_stream == torch.cuda.default_stream()
                or self._engine_stream is None
            ):
                self._engine_stream = torch.cuda.Stream()

            self._engine_stream.wait_stream(self._caller_stream)
            with torch.cuda.stream(self._engine_stream):
                if self.resource_allocation_strategy:
                    self._dynamic_workspace = torch.empty(
                        self.cuda_engine.device_memory_size_v2,
                        dtype=torch.uint8,
                        device=torch.cuda.current_device(),
                    )
                    self.context.set_device_memory(self._dynamic_workspace.data_ptr())

                if torch_tensorrt.runtime.get_cudagraphs_mode():
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

            self._caller_stream.wait_stream(self._engine_stream)

        if self.use_pre_allocated_outputs and (
            self.output_tensors_are_unowned
            or not self.pre_allocated_outputs
            or shape_changed
        ):
            self.pre_allocated_outputs = self.create_output_tensors()

        if torch_tensorrt.runtime.get_cudagraphs_mode():
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

        with self._profile_section("PythonTRTEngine:ProcessInputs"):
            self.setup_input_tensors(contiguous_inputs, False, False)

        with self._profile_section("PythonTRTEngine:SetupOutputAllocator"):
            self.create_output_allocator()
            for output_name in self.out_binding_names:
                if not self.context.set_output_allocator(
                    output_name, self.output_allocator
                ):
                    raise RuntimeError(
                        f"Failed to set output allocator for {output_name}"
                    )

        with self._profile_section("PythonTRTEngine:TensorRTRuntime"):
            self._caller_stream = torch.cuda.current_stream()
            if (
                self._engine_stream == torch.cuda.default_stream()
                or self._engine_stream is None
            ):
                self._engine_stream = torch.cuda.Stream()

            self._engine_stream.wait_stream(self._caller_stream)
            with torch.cuda.stream(self._engine_stream):
                self.context.execute_async_v3(self._engine_stream.cuda_stream)
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
            f"Using the standard execution runtime mode with cudagraphs={torch_tensorrt.runtime.get_cudagraphs_mode()}."
        )
        return self._execute_standard(contiguous_inputs)


register_opaque_type(PythonTRTEngine, typ="reference")


@torch.library.custom_op(  # type: ignore[misc]
    "tensorrt::execute_engine_python", mutates_args=()
)
def execute_engine_python(
    input_tensors: List[torch.Tensor], engine: PythonTRTEngine
) -> List[torch.Tensor]:
    outputs = engine.execute(input_tensors)
    return [outputs] if isinstance(outputs, torch.Tensor) else list(outputs)
