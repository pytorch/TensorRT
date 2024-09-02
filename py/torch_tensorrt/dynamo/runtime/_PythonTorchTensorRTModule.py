from __future__ import annotations

import logging
from contextlib import nullcontext
from functools import wraps
from tempfile import tempdir
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

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


def recreate_context_decorator(method: Callable[..., Any]) -> Callable[..., Any]:
    """
    A decorator that destroys a context before a method execution and
    creates it after the method execution within the same class instance.
    """

    @wraps(method)
    def wrapper(self: object, *args: Any, **kwargs: Any) -> Any:
        self.reset_context()
        result = method(self, *args, **kwargs)
        self.init_context()
        return result

    return wrapper


class PythonTorchTensorRTModule(Module):  # type: ignore[misc]
    """PythonTorchTensorRTModule is a PyTorch module which encompasses an arbitrary TensorRT Engine.

    This module is backed by the Torch-TensorRT runtime and is only compatible with
    FX / Dynamo / Python deployments. This module cannot be serialized to torchscript via torch.jit.trace for C++ deployment.
    """

    def __init__(
        self,
        serialized_engine: Optional[bytes] = None,
        input_binding_names: Optional[List[str]] = None,
        output_binding_names: Optional[List[str]] = None,
        *,
        name: str = "",
        settings: CompilationSettings = CompilationSettings(),
        weight_name_map: Any = None,
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
        super(PythonTorchTensorRTModule, self).__init__()
        self._register_state_dict_hook(PythonTorchTensorRTModule._on_state_dict)

        # Run multi-gpu device check to validate engine instantiation
        multi_gpu_device_check()

        self.name = name
        self._input_buffers: List[torch.Tensor] = []
        self._output_buffers: List[torch.Tensor] = []
        self.cudagraph: Optional[torch.cuda.CUDAGraph] = None
        self._caller_stream: Optional[torch.cuda.Stream] = None
        self._engine_stream: Optional[torch.cuda.Stream] = None

        # TODO: Make the below a Dictionary {shape: cudagraph}
        self.shape_key: Optional[str] = None

        # See https://github.com/pytorch/pytorch/blob/acfe237a71af609e837a34bb38048aa8acb8eb4d/torch/cuda/graphs.py#L92-L98
        # Unused currently - to be used by Dynamic Shape support implementation
        self.memory_pool = None

        self.serialized_engine = serialized_engine
        self.input_names = (
            input_binding_names if input_binding_names is not None else []
        )
        self.output_names = (
            output_binding_names if output_binding_names is not None else []
        )
        self.initialized = False
        self.target_device_id = (
            settings.device.gpu_id
            if settings.device is not None
            else Device._current_device().gpu_id
        )
        self.target_device_properties = torch.cuda.get_device_properties(
            self.target_device_id
        )
        self.profiling_enabled = settings.debug if settings.debug is not None else False
        self.settings = settings
        self.engine = None
        self.weight_name_map = weight_name_map
        self.target_platform = Platform.current_platform()
        self.min_required_device_budget = 0

        if self.serialized_engine is not None and not self.settings.lazy_engine_init:
            self.setup_engine()

    def init_context(self) -> None:
        assert self.engine, "Context is used before setting up the engine"
        if self.context is None:
            self.context = self.engine.create_execution_context()

    def reset_context(self) -> None:
        if self.context is not None:
            del self.context
            self.context = None

    def get_streamable_weights_size(self) -> Any:
        return self.engine.streamable_weights_size

    def get_weight_streaming_budget(self) -> Any:
        return self.engine.weight_streaming_budget_v2

    @recreate_context_decorator
    def set_device_memory_budget(self, budget_bytes: int) -> int:
        return self._set_device_memory_budget(budget_bytes)

    def _set_device_memory_budget(self, budget_bytes: int) -> int:
        # Disable weight streaming for invalid budget size
        if budget_bytes < 0:
            budget_bytes = self.engine.streamable_weights_size
        self.engine.weight_streaming_budget_v2 = budget_bytes
        if self.get_weight_streaming_budget() != budget_bytes:
            logger.error(f"Failed to set weight streaming budget to {budget_bytes}")
            budget_bytes = self.get_weight_streaming_budget()
        if self.engine.streamable_weights_size == budget_bytes:
            logger.warning("Weight streaming is disabled")

        return budget_bytes

    def set_default_streaming_budget(self) -> int:
        # Scratch memory size may change based on the current weight streaming budget
        # Required memory for full streaming is used to minimum weight budget
        self.engine.weight_streaming_budget_v2 = 0
        self.min_required_device_budget = (
            self.engine.weight_streaming_scratch_memory_size
        )
        budget_bytes = self.engine.get_weight_streaming_automatic_budget()
        # Set automatic weight streaming budget as default when context is created
        return self._set_device_memory_budget(budget_bytes)

    def setup_engine(self) -> None:
        assert (
            self.target_platform == Platform.current_platform()
        ), f"TensorRT engine was not built to target current platform (target: {self.target_platform}, current: {Platform.current_platform()})"

        self.initialized = True
        runtime = trt.Runtime(TRT_LOGGER)
        self.engine = runtime.deserialize_cuda_engine(self.serialized_engine)
        if self.settings.enable_weight_streaming:
            self.set_default_streaming_budget()
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
            dtype._from(self.engine.get_tensor_dtype(output_name))
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

    def __deepcopy__(self, memo: Any) -> PythonTorchTensorRTModule:
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        result.__setstate__(self.__getstate__())
        return result

    def __del__(self) -> None:
        if self.cudagraph:
            self.cudagraph.reset()

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor | Tuple[torch.Tensor, ...]:
        # Ensure inputs are available in all scopes and cast symbolic integers to Tensors
        contiguous_inputs: List[torch.Tensor] = [
            (i.contiguous() if isinstance(i, torch.Tensor) else torch.tensor(i).cuda())
            for i in inputs
        ]

        with (
            torch.autograd.profiler.record_function("PythonTorchTensorRTModule:Forward")
            if self.profiling_enabled
            else nullcontext()
        ):
            self._check_initialized()

            cudagraphs_enabled = torch_tensorrt.runtime.get_cudagraphs_mode()
            need_cudagraphs_record = (
                cudagraphs_enabled and not self.cudagraphs_validate_shapes(inputs)
            )

            if need_cudagraphs_record:
                self._input_buffers = [None] * len(self.input_names)
                self._output_buffers = [None] * len(self.output_names)

            if not cudagraphs_enabled and self.cudagraph:
                self.cudagraph.reset()
                self.cudagraph = None

            # If in safe mode, check at each iteration for for whether a switch is required
            if (
                torch_tensorrt.runtime._multi_device_safe_mode._PY_RT_MULTI_DEVICE_SAFE_MODE
            ):
                curr_device_id = torch.cuda.current_device()
                curr_device_properties = torch.cuda.get_device_properties(
                    curr_device_id
                )
                logger.debug(f"Current Device: cuda:{curr_device_id}")

                # If a switch is required, move all inputs to new device and set as active device
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

                    # Update current device
                    device = torch.device(device_id)
                    torch.cuda.set_device(device_id)

                    contiguous_inputs = [
                        tensor.to(device) for tensor in contiguous_inputs
                    ]
                    logger.warning(f"Moved all input Tensors to cuda:{device_id}")

            with (
                torch.autograd.profiler.record_function(
                    "PythonTorchTensorRTModule:ProcessInputs"
                )
                if self.profiling_enabled
                else nullcontext()
            ):
                assert len(contiguous_inputs) == len(
                    self.input_names
                ), f"Wrong number of inputs, expect {len(self.input_names)} get {len(contiguous_inputs)}."

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
                        inputs_cpu = (
                            contiguous_inputs[i].cpu().to(torch.int64).numpy().copy()
                        )
                        self.context.set_tensor_address(
                            input_name, inputs_cpu.ctypes.data
                        )
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

                # Check if input shapes can be inferred.
                uninferred_input_names = self.context.infer_shapes()
                if uninferred_input_names:
                    logger.warning(
                        f"The shapes of the inputs: {uninferred_input_names} cannot be inferred and could lead to undefined behavior. \
                                This could happen if the input tensor addresses/shapes haven't been configured correctly"
                    )

            with (
                torch.autograd.profiler.record_function(
                    "PythonTorchTensorRTModule:ProcessOutputs"
                )
                if self.profiling_enabled
                else nullcontext()
            ):
                # create output tensors
                outputs: List[torch.Tensor] = []

                for o, output_name in enumerate(self.output_names):
                    shape = tuple(self.context.get_tensor_shape(output_name))

                    if DYNAMIC_DIM in shape:
                        raise ValueError(
                            "Encountered dynamic output shapes during runtime. This could mean the network has data-dependent output shapes which is not currently supported."
                        )

                    output = torch.empty(
                        size=shape,
                        dtype=self.output_dtypes[o].to(torch.dtype),
                        device=torch.cuda.current_device(),
                    )

                    outputs.append(output)

                    if need_cudagraphs_record:
                        self._output_buffers[o] = outputs[o].clone()

                    if cudagraphs_enabled:
                        self.context.set_tensor_address(
                            output_name, self._output_buffers[o].data_ptr()
                        )
                    else:
                        self.context.set_tensor_address(
                            output_name, outputs[o].data_ptr()
                        )

            with (
                torch.autograd.profiler.record_function(
                    "PythonTorchTensorRTModule:TensorRTRuntime"
                )
                if self.profiling_enabled
                else nullcontext()
            ):
                self._caller_stream = torch.cuda.current_stream()
                if (
                    self._engine_stream == torch.cuda.default_stream()
                    or self._engine_stream is None
                ):
                    self._engine_stream = torch.cuda.Stream()

                self._engine_stream.wait_stream(self._caller_stream)

                with torch.cuda.stream(self._engine_stream):

                    if cudagraphs_enabled:
                        if need_cudagraphs_record:
                            self.cudagraph = torch.cuda.CUDAGraph()

                            if self.profiling_enabled:
                                self.cudagraph.enable_debug_mode()

                            with torch.cuda.graph(
                                self.cudagraph, stream=self._engine_stream
                            ):
                                self.context.execute_async_v3(
                                    self._engine_stream.cuda_stream
                                )

                            if self.profiling_enabled:
                                import tempfile

                                with tempfile.TemporaryDirectory() as tmpdir:
                                    self.cudagraph.debug_dump(
                                        f"{tempdir}/{self.name}_cudagraph.dot"
                                    )

                        self.cudagraph.replay()  # type: ignore

                    else:
                        self.context.execute_async_v3(self._engine_stream.cuda_stream)

                self._caller_stream.wait_stream(self._engine_stream)

            if cudagraphs_enabled:
                for idx, o in enumerate(outputs):
                    o.copy_(self._output_buffers[idx])

            if len(outputs) == 1:
                return outputs[0]

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

    def cudagraphs_validate_shapes(self, inputs: Sequence[torch.Tensor]) -> bool:
        """
        Validates the input shapes of the forward function
        versus the version currently active for the
        """
        # Representation of input shapes to a given model
        # Shapes are concatenated as so:
        # x: (3, 4), y: (4, 5) --> Key: (3,4)(4,5)
        new_shape_key = "".join(str(tuple(t.shape)).replace(" ", "") for t in inputs)

        # If the new shape key differs from the existing one,
        # invalidate the old shape key and remove the CUDAGraph
        if new_shape_key != self.shape_key:
            logger.debug(f"Resetting Cudagraph on new shape key {new_shape_key}")
            self.shape_key = new_shape_key
            if self.cudagraph:
                self.cudagraph.reset()
            return False

        return True
