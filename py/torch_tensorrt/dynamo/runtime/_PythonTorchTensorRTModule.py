from __future__ import annotations

import logging
from contextlib import nullcontext
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch_tensorrt
from torch.nn import Module
from torch_tensorrt._Device import Device
from torch_tensorrt._enums import dtype
from torch_tensorrt.dynamo._settings import CompilationSettings
from torch_tensorrt.dynamo.runtime.tools import (
    _is_switch_required,
    _select_rt_device,
    multi_gpu_device_check,
)
from torch_tensorrt.dynamo.utils import DYNAMIC_DIM
from torch_tensorrt.logging import TRT_LOGGER

import tensorrt as trt

logger = logging.getLogger(__name__)


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
        super(PythonTorchTensorRTModule, self).__init__()
        self._register_state_dict_hook(PythonTorchTensorRTModule._on_state_dict)

        # Run multi-gpu device check to validate engine instantiation
        multi_gpu_device_check()

        self.name = name
        self.input_buffers: List[torch.Tensor] = []
        self.output_buffers: List[torch.Tensor] = []
        self.cudagraph: Optional[torch.cuda.CUDAGraph] = None
        self.active_stream: Optional[torch.cuda.Stream] = None

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
        self._initialize()

        if self.serialized_engine is not None and not self.settings.lazy_engine_init:
            self.setup_engine()

    def setup_engine(self) -> None:
        self.initialized = True
        runtime = trt.Runtime(TRT_LOGGER)
        self.engine = runtime.deserialize_cuda_engine(self.serialized_engine)
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
            self.graph_capturer = torch.cuda.graphs.graph(self.cudagraph)

        # Set the active stream using the current device
        current_stream = torch.cuda.current_stream()
        if current_stream == torch.cuda.default_stream():
            self.active_stream = torch.cuda.Stream()
            torch.cuda.set_stream(self.active_stream)
        else:
            self.active_stream = current_stream

    def _check_initialized(self) -> None:
        if not self.initialized:
            raise RuntimeError("PythonTorchTensorRTModule is not initialized.")

    def _on_state_dict(self, state_dict: Dict[str, Any], prefix: str, _: Any) -> None:
        state_dict[prefix + "engine"] = self.serialized_engine
        state_dict[prefix + "input_names"] = self.input_names
        state_dict[prefix + "output_names"] = self.output_names

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

        # Run multi-gpu device check to validate engine instantiation
        multi_gpu_device_check()
        self.setup_engine()

    def __getstate__(self) -> Dict[str, Any]:
        state = self.__dict__.copy()
        state["engine"] = bytearray(self.engine.serialize())
        state.pop("context", None)
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        logger = trt.Logger()
        runtime = trt.Runtime(logger)
        state["engine"] = runtime.deserialize_cuda_engine(state["engine"])
        self.__dict__.update(state)
        if self.engine:
            self.context = self.engine.create_execution_context()

    def __deepcopy__(self, memo: Any) -> PythonTorchTensorRTModule:
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        result.__setstate__(self.__getstate__())
        return result

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

            # If cudagraphs is not enabled or the recorded graph shapes are either uninitialized or invalid
            if not cudagraphs_enabled or need_cudagraphs_record:
                # If in safe mode, check at each iteration for for whether a switch is required
                if (
                    torch_tensorrt.runtime.multi_device_safe_mode._PY_RT_MULTI_DEVICE_SAFE_MODE
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

                        # Update current stream
                        current_stream = torch.cuda.current_stream(device)
                        if current_stream == torch.cuda.default_stream(device):
                            self.active_stream = torch.cuda.Stream(device)
                            torch.cuda.set_stream(self.active_stream)
                        else:
                            self.active_stream = current_stream

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

                    if cudagraphs_enabled:
                        # If cudagraphs is enabled, this memory is reserved for future cudagraph runs
                        # Clone is required to avoid re-using user-provided GPU memory
                        contiguous_inputs = [i.clone() for i in contiguous_inputs]

                    bindings = []
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

                        # For shape tensors, we use CPU pointers and for data tensors, we use GPU pointers
                        # as per TensorRT requirements
                        if self.engine.is_shape_inference_io(input_name):
                            # Shape tensor inputs are casted to int64 explicitly
                            # Currently Torch CPU pointers are not working; numpy pointers are used instead
                            # to refer to underlying memory
                            inputs_cpu = (
                                contiguous_inputs[i]
                                .cpu()
                                .to(torch.int64)
                                .numpy()
                                .copy()
                            )
                            self.context.set_tensor_address(
                                input_name, inputs_cpu.ctypes.data
                            )
                            bindings.append(inputs_cpu.ctypes.data)
                        else:
                            self.context.set_input_shape(
                                input_name, tuple(contiguous_inputs[i].shape)
                            )
                            self.context.set_tensor_address(
                                input_name, contiguous_inputs[i].data_ptr()
                            )
                            bindings.append(contiguous_inputs[i].data_ptr())

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

                    for i, output_name in enumerate(self.output_names):
                        shape = tuple(self.context.get_tensor_shape(output_name))

                        if DYNAMIC_DIM in shape:
                            raise ValueError(
                                "Encountered dynamic output shapes during runtime. This could mean the network has data-dependent output shapes which is not currently supported."
                            )

                        output = torch.empty(
                            size=shape,
                            dtype=self.output_dtypes[i].to(torch.dtype),
                            device=torch.cuda.current_device(),
                        )
                        bindings.append(output.data_ptr())
                        outputs.append(output)

                # Assign tensor address appropriately
                for idx in range(self.engine.num_io_tensors):
                    self.context.set_tensor_address(
                        self.engine.get_tensor_name(idx), bindings[idx]
                    )

            with (
                torch.autograd.profiler.record_function(
                    "PythonTorchTensorRTModule:TensorRTRuntime"
                )
                if self.profiling_enabled
                else nullcontext()
            ):

                if not cudagraphs_enabled:
                    self.context.execute_async_v3(self.active_stream.cuda_stream)  # type: ignore

                elif need_cudagraphs_record:
                    self.input_buffers = list(contiguous_inputs)
                    self.output_buffers = list(outputs)

                    graph_capturer_stream = self.graph_capturer.capture_stream

                    self.context.execute_async_v3(graph_capturer_stream.cuda_stream)
                    graph_capturer_stream.synchronize()

                    with self.graph_capturer:
                        self.context.execute_async_v3(graph_capturer_stream.cuda_stream)

                else:
                    for idx, input_tensor in enumerate(inputs):
                        self.input_buffers[idx].copy_(input_tensor, non_blocking=True)

                    self.cudagraph.replay()  # type: ignore

            if cudagraphs_enabled:
                model_outputs = tuple(output.clone() for output in self.output_buffers)
            else:
                model_outputs = tuple(outputs)

            if len(model_outputs) == 1:
                return model_outputs[0]

            return model_outputs

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
            self.cudagraph.reset()  # type: ignore
            return False

        return True
