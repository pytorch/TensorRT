from __future__ import annotations

import logging
from contextlib import nullcontext
from tempfile import tempdir
from typing import Any, Dict, List, Optional, Sequence, Tuple

import mlir_tensorrt.runtime.api as runtime
import numpy as np
import tensorrt as trt
import torch
import torch_tensorrt
from torch.nn import Module
from torch_tensorrt._Device import Device
from torch_tensorrt._enums import Platform, dtype
from torch_tensorrt.dynamo._settings import CompilationSettings
from torch_tensorrt.dynamo.utils import DYNAMIC_DIM
from torch_tensorrt.logging import TRT_LOGGER
from torch_tensorrt.runtime._utils import (
    _is_switch_required,
    _select_rt_device,
    multi_gpu_device_check,
)

logger = logging.getLogger(__name__)


class MLIRTensorRTModule(Module):  # type: ignore[misc]
    """MLIRTensorRTModule is a PyTorch module which encompasses an MLIR-TRT executable.

    This module is backed by the Torch-TensorRT runtime and is only compatible with
    FX / Dynamo / Python deployments. This module cannot be serialized for C++ deployment.
    """

    def __init__(
        self,
        mlir_exec: Optional[bytes] = None,
        input_binding_names: Optional[List[str]] = None,
        output_binding_names: Optional[List[str]] = None,
        *,
        name: str = "",
        settings: CompilationSettings = CompilationSettings(),
        weight_name_map: Optional[dict[Any, Any]] = None,
    ):
        """Takes a name, target device, serialized TensorRT engine, and binding names / order and constructs
        a PyTorch ``torch.nn.Module`` around it. Uses TensorRT Python APIs to run the engine

        Arguments:
            serialized_engine (bytes): Serialized TensorRT engine in the form of a bytearray
            input_binding_names (List[str]): List of input TensorRT engine binding names in the order they would be passed to the TRT modules
            output_binding_names (List[str]): List of output TensorRT engine binding names in the order they should be returned

        Keyword Arguments:
            name (str): Name for module
            settings (torch_tensorrt.dynamo.CompilationSettings): Settings used to compile engine, assumes engine was built with default compilation settings if object not passed
            weight_name_map (dict): Mapping of engine weight name to state_dict weight name

        Example:

            .. code-block:: py

                trt_module = PythonTorchTensorRTModule(
                    engine_str,
                    input_binding_names=["x"],
                    output_binding_names=["output"],
                    name="my_module",
                    settings=CompilationSettings(device=torch.cuda.current_device)
                )

        """
        self.context: Any
        super(MLIRTensorRTModule, self).__init__()
        self._register_state_dict_hook(MLIRTensorRTModule._on_state_dict)

        self.name = name
        self.engine = mlir_exec

    def get_streamable_device_memory_budget(self) -> Any:
        return self.engine.streamable_weights_size

    def get_automatic_device_memory_budget(self) -> Any:
        return self.engine.get_weight_streaming_automatic_budget()

    def get_device_memory_budget(self) -> Any:
        return self.engine.weight_streaming_budget_v2

    def set_device_memory_budget(self, budget_bytes: int) -> int:
        # Recreating the context because weight streaming budget cannot be modified while there are active context.
        if self.context is not None:
            del self.context
        budget_bytes = self._set_device_memory_budget(budget_bytes)
        self.context = self.engine.create_execution_context()
        self.runtime_states.context_changed = True
        return budget_bytes

    def _set_device_memory_budget(self, budget_bytes: int) -> int:
        # Disable weight streaming for invalid budget size
        if budget_bytes < 0:
            budget_bytes = self.get_streamable_device_memory_budget()
        self.engine.weight_streaming_budget_v2 = budget_bytes
        if self.engine.weight_streaming_budget_v2 != budget_bytes:
            logger.error(f"Failed to set weight streaming budget to {budget_bytes}")
            budget_bytes = self.engine.weight_streaming_budget_v2
        if self.get_streamable_device_memory_budget() == budget_bytes:
            logger.warning("Weight streaming is disabled")

        return budget_bytes

    def set_default_device_memory_budget(self) -> int:
        budget_bytes = self.get_automatic_device_memory_budget()
        # Set automatic weight streaming budget as default when context is created
        logger.debug(f"Weight streaming budget set to {budget_bytes}B")
        return self._set_device_memory_budget(budget_bytes)

    def setup_engine(self) -> None:
        assert (
            self.target_platform == Platform.current_platform()
        ), f"TensorRT engine was not built to target current platform (target: {self.target_platform}, current: {Platform.current_platform()})"

        self.initialized = True
        runtime = trt.Runtime(TRT_LOGGER)
        self.engine = runtime.deserialize_cuda_engine(self.serialized_engine)
        if self.settings.enable_weight_streaming:
            self.set_default_device_memory_budget()
        self.context = self.engine.create_execution_context()
        assert self.engine.num_io_tensors == (
            len(self.input_names) + len(self.output_names)
        )

        self.input_dtypes = [
            dtype._from(self.engine.get_tensor_dtype(input_name))
            for input_name in self.input_names
        ]
        self.input_shapes = [
            self.engine.get_tensor_shape(input_name) for input_name in self.input_names
        ]
        self.output_dtypes = [
            dtype._from(self.engine.get_tensor_dtype(output_name)).to(torch.dtype)
            for output_name in self.output_names
        ]
        self.output_shapes = [
            self.engine.get_tensor_shape(output_name)
            for output_name in self.output_names
        ]

        if torch_tensorrt.runtime.get_cudagraphs_mode():
            self.cudagraph = torch.cuda.CUDAGraph()

    def _check_initialized(self) -> None:
        if not self.initialized:
            raise RuntimeError("PythonTorchTensorRTModule is not initialized.")

    def _on_state_dict(self, state_dict: Dict[str, Any], prefix: str, _: Any) -> None:
        state_dict[prefix + "engine"] = self.serialized_engine
        state_dict[prefix + "input_names"] = self.input_names
        state_dict[prefix + "output_names"] = self.output_names
        state_dict[prefix + "platform"] = self.target_platform

    def _load_from_state_dict(
        self,
        state_dict: Dict[str, Any],
        prefix: str,
        local_metadata: Any,
        strict: Any,
        missing_keys: Any,
        unexpected_keys: Any,
        error_msgs: Any,
    ) -> None:
        self.serialized_engine = state_dict[prefix + "engine"]
        self.input_names = state_dict[prefix + "input_names"]
        self.output_names = state_dict[prefix + "output_names"]
        self.target_platform = state_dict[prefix + "platform"]

        # Run multi-gpu device check to validate engine instantiation
        multi_gpu_device_check()
        self.setup_engine()

    def __getstate__(self) -> Dict[str, Any]:
        state = self.__dict__.copy()
        state.pop("engine", None)
        state.pop("context", None)
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)
        self.setup_engine()

    def __deepcopy__(self, memo: Any) -> MLIRTensorRTModule:
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        result.__setstate__(self.__getstate__())
        return result

    def __del__(self) -> None:
        # if self.cudagraph:
        #     self.cudagraph.reset()
        pass

    def setup_input_tensors(
        self,
        contiguous_inputs: List[torch.Tensor],
        cudagraphs_enabled: bool,
        need_cudagraphs_record: bool,
    ) -> None:
        for i, input_name in enumerate(self.input_names):
            if not contiguous_inputs[i].is_cuda:
                logger.warning(
                    f"Detected input {input_name} of engine {self.engine.name} is not on a cuda device. "
                    "This tensor is being moved by the runtime but for performance considerations, "
                    "ensure your inputs are all on GPU and open an issue here "
                    "(https://github.com/pytorch/TensorRT/issues) if this warning persists."
                )
                contiguous_inputs = (
                    contiguous_inputs[:i]
                    + [contiguous_inputs[i].cuda()]
                    + contiguous_inputs[i + 1 :]
                )

            assert (
                contiguous_inputs[i].dtype == self.input_dtypes[i]
            ), f"Dtype mismatch for {i}th input({input_name}). Expect {self.input_dtypes[i]}, got {contiguous_inputs[i].dtype}."

            if need_cudagraphs_record:
                # If cudagraphs is enabled, this memory is reserved for future cudagraph runs
                # Clone is required to avoid re-using user-provided GPU memory
                self._input_buffers[i] = contiguous_inputs[i].clone()

            # For shape tensors, we use CPU pointers and for data tensors, we use GPU pointers
            # as per TensorRT requirements
            if self.engine.is_shape_inference_io(input_name):
                # Shape tensor inputs are casted to int64 explicitly
                # Currently Torch CPU pointers are not working; numpy pointers are used instead
                # to refer to underlying memory
                inputs_cpu = contiguous_inputs[i].cpu().to(torch.int64).numpy().copy()
                self.context.set_tensor_address(input_name, inputs_cpu.ctypes.data)
            else:
                self.context.set_input_shape(
                    input_name, tuple(contiguous_inputs[i].shape)
                )
                if cudagraphs_enabled:
                    self._input_buffers[i].copy_(contiguous_inputs[i])
                    self.context.set_tensor_address(
                        input_name, self._input_buffers[i].data_ptr()
                    )
                else:
                    self.context.set_tensor_address(
                        input_name, contiguous_inputs[i].data_ptr()
                    )

    def create_output_tensors(self) -> List[torch.Tensor]:
        # create output tensors
        outputs: List[torch.Tensor] = []

        for o, _ in enumerate(self.output_names):
            output = torch.empty(
                size=self.output_shapes[o],
                dtype=self.output_dtypes[o],
                device=torch.cuda.current_device(),
            )
            outputs.append(output)
        return outputs

    def set_pre_allocated_outputs(self, enable: bool) -> None:
        self.use_pre_allocated_outputs = enable

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor | Tuple[torch.Tensor, ...]:
        # Ensure inputs are available in all scopes and cast symbolic integers to Tensors
        contiguous_inputs: List[torch.Tensor] = [
            (i.contiguous() if isinstance(i, torch.Tensor) else torch.tensor(i).cuda())
            for i in inputs
        ]

        client = runtime.RuntimeClient()
        stream = client.create_stream()
        devices = client.get_devices()
        if len(devices) == 0:
            return

        session_options = runtime.RuntimeSessionOptions(num_devices=1, device_id=0)
        session = runtime.RuntimeSession(session_options, self.engine)
        runtime_inputs = []
        for input in inputs:
            arg0 = client.create_memref_view_from_dlpack(input.__dlpack__())
            runtime_inputs.append(arg0)

        torch_arg1 = torch.zeros(2, 2, dtype=torch.float32)
        arg1 = client.create_memref_view_from_dlpack(torch_arg1.__dlpack__())
        arg1 = client.copy_to_device(arg1, device=devices[0])

        session.execute_function(
            "main", in_args=runtime_inputs, out_args=[arg1], stream=stream
        )
        breakpoint()
        outputs = torch.from_blob(client.copy_to_host(arg1, stream=stream))
        stream.sync()
        return outputs

    def enable_profiling(self, profiler: "trt.IProfiler" = None) -> None:
        """
        Enable TensorRT profiling. After calling this function, TensorRT will report
        time spent on each layer in stdout for each forward run.
        """
        self._check_initialized()

        if not self.context.profiler:
            self.context.profiler = trt.Profiler() if profiler is None else profiler

        self.profiling_enabled = True

    def disable_profiling(self) -> None:
        """
        Disable TensorRT profiling.
        """
        self._check_initialized()
        torch.cuda.synchronize()
        del self.context
        self.context = self.engine.create_execution_context()
        self.profiling_enabled = False

    def get_layer_info(self) -> str:
        """
        Get layer info of the engine. Only support for TRT > 8.2.
        """
        inspector = self.engine.create_engine_inspector()
        engine_json: str = inspector.get_engine_information(
            trt.LayerInformationFormat.JSON
        )
        return engine_json

    def validate_input_shapes(self, inputs: Sequence[torch.Tensor]) -> bool:
        """
        Validates the input shapes of the forward function has changed
        """
        # Representation of input shapes to a given model
        # Shapes are concatenated as so:
        # x: (3, 4), y: (4, 5) --> Key: (3,4)(4,5)
        new_shape_key = "".join(str(tuple(t.shape)).replace(" ", "") for t in inputs)

        # If the new shape key differs from the existing one,
        # invalidate the old shape key and remove the CUDAGraph
        if new_shape_key != self.shape_key:
            logger.debug(f"Input shape changed {self.shape_key} -> {new_shape_key}")
            self.shape_key = new_shape_key
            return True

        return False
