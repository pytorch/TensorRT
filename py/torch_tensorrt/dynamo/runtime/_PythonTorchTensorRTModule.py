from __future__ import annotations

import logging
from contextlib import nullcontext
from typing import Any, Dict, List, Optional, Sequence, Tuple

import tensorrt as trt
import torch
import torch_tensorrt
from torch.nn import Module
from torch_tensorrt._Device import Device
from torch_tensorrt.dynamo.runtime.tools import (
    _is_switch_required,
    _select_rt_device,
    multi_gpu_device_check,
)
from torch_tensorrt.fx.utils import Frameworks, unified_dtype_converter

logger = logging.getLogger(__name__)


class PythonTorchTensorRTModule(Module):  # type: ignore[misc]
    """PythonTorchTensorRTModule is a PyTorch module which encompasses an arbitrary TensorRT Engine.

    This module is backed by the Torch-TensorRT runtime and is only compatibile with
    FX / Dynamo / Python deployments. This module cannot be serialized to torchscript via torch.jit.trace for C++ deployment.
    """

    def __init__(
        self,
        engine: trt.ICudaEngine,
        input_names: Optional[List[str]] = None,
        output_names: Optional[List[str]] = None,
        target_device: Device = Device._current_device(),
        profiling_enabled: Optional[bool] = None,
    ):
        super(PythonTorchTensorRTModule, self).__init__()
        self._register_state_dict_hook(PythonTorchTensorRTModule._on_state_dict)

        # Run multi-gpu device check to validate engine instantiation
        multi_gpu_device_check()

        self.engine = engine
        self.input_names = input_names if input_names is not None else []
        self.output_names = output_names if output_names is not None else []
        self.initialized = False
        self.target_device_id = target_device.gpu_id
        self.target_device_properties = torch.cuda.get_device_properties(
            self.target_device_id
        )
        self.profiling_enabled = (
            profiling_enabled if profiling_enabled is not None else False
        )
        self._initialize()

    def _initialize(self) -> None:
        self.initialized = True
        self.context = self.engine.create_execution_context()

        # Indices of inputs/outputs in the trt engine bindings, in the order
        # as they are in the original PyTorch model.
        self.input_binding_indices_in_order: Sequence[int] = [
            self.engine.get_binding_index(name) for name in self.input_names
        ]
        self.output_binding_indices_in_order: Sequence[int] = [
            self.engine.get_binding_index(name) for name in self.output_names
        ]
        primary_input_outputs = set()
        primary_input_outputs.update(self.input_binding_indices_in_order)
        primary_input_outputs.update(self.output_binding_indices_in_order)
        self.hidden_output_binding_indices_in_order: Sequence[int] = []
        self.hidden_output_names: Sequence[str] = []
        for i in range(
            self.engine.num_bindings // self.engine.num_optimization_profiles
        ):
            if i not in primary_input_outputs:
                self.hidden_output_binding_indices_in_order.append(i)
                self.hidden_output_names.append(self.engine.get_binding_name(i))

        assert (self.engine.num_bindings // self.engine.num_optimization_profiles) == (
            len(self.input_names)
            + len(self.output_names)
            + len(self.hidden_output_names)
        )

        self.input_dtypes = [
            unified_dtype_converter(
                self.engine.get_binding_dtype(idx), Frameworks.TORCH
            )
            for idx in self.input_binding_indices_in_order
        ]
        self.input_shapes: Sequence[Sequence[int]] = [
            tuple(self.engine.get_binding_shape(idx))
            for idx in self.input_binding_indices_in_order
        ]
        self.output_dtypes = [
            unified_dtype_converter(
                self.engine.get_binding_dtype(idx), Frameworks.TORCH
            )
            for idx in self.output_binding_indices_in_order
        ]
        self.output_shapes = [
            (
                tuple(self.engine.get_binding_shape(idx))
                if self.engine.has_implicit_batch_dimension
                else tuple()
            )
            for idx in self.output_binding_indices_in_order
        ]
        self.hidden_output_dtypes = [
            unified_dtype_converter(
                self.engine.get_binding_dtype(idx), Frameworks.TORCH
            )
            for idx in self.hidden_output_binding_indices_in_order
        ]
        self.hidden_output_shapes = [
            (
                tuple(self.engine.get_binding_shape(idx))
                if self.engine.has_implicit_batch_dimension
                else tuple()
            )
            for idx in self.hidden_output_binding_indices_in_order
        ]

    def _check_initialized(self) -> None:
        if not self.initialized:
            raise RuntimeError("PythonTorchTensorRTModule is not initialized.")

    def _on_state_dict(self, state_dict: Dict[str, Any], prefix: str, _: Any) -> None:
        self._check_initialized()
        state_dict[prefix + "engine"] = bytearray(self.engine.serialize())
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
        engine_bytes = state_dict[prefix + "engine"]

        # Run multi-gpu device check to validate engine instantiation
        multi_gpu_device_check()

        logger = trt.Logger()
        runtime = trt.Runtime(logger)
        self.engine = runtime.deserialize_cuda_engine(engine_bytes)

        self.input_names = state_dict[prefix + "input_names"]
        self.output_names = state_dict[prefix + "output_names"]
        self._initialize()

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

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor | Tuple[torch.Tensor, ...]:
        with (
            torch.autograd.profiler.record_function("PythonTorchTensorRTModule:Forward")
            if self.profiling_enabled
            else nullcontext()
        ):
            self._check_initialized()

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
                    device = torch.device(device_id)
                    torch.cuda.set_device(device_id)

                    inputs = tuple([tensor.to(device) for tensor in inputs])
                    logger.warning(f"Moved all input Tensors to cuda:{device_id}")

            with (
                torch.autograd.profiler.record_function(
                    "PythonTorchTensorRTModule:ProcessInputs"
                )
                if self.profiling_enabled
                else nullcontext()
            ):
                assert len(inputs) == len(
                    self.input_names
                ), f"Wrong number of inputs, expect {len(self.input_names)} get {len(inputs)}."

                contiguous_inputs: List[torch.Tensor] = [i.contiguous() for i in inputs]
                bindings: List[Any] = [None] * (
                    len(self.input_names)
                    + len(self.output_names)
                    + len(self.hidden_output_names)
                )

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

                    idx = self.input_binding_indices_in_order[i]
                    bindings[idx] = contiguous_inputs[i].data_ptr()

                    self.context.set_binding_shape(
                        idx, tuple(contiguous_inputs[i].shape)
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

                for i, idx in enumerate(self.output_binding_indices_in_order):
                    shape = tuple(self.context.get_binding_shape(idx))

                    output = torch.empty(
                        size=shape,
                        dtype=self.output_dtypes[i],
                        device=torch.cuda.current_device(),
                    )
                    outputs.append(output)
                    bindings[idx] = output.data_ptr()

                for i, idx in enumerate(self.hidden_output_binding_indices_in_order):
                    shape = tuple(self.context.get_binding_shape(idx))

                    output = torch.empty(
                        size=shape,
                        dtype=self.hidden_output_dtypes[i],
                        device=torch.cuda.current_device(),
                    )
                    bindings[idx] = output.data_ptr()

            with (
                torch.autograd.profiler.record_function(
                    "PythonTorchTensorRTModule:TensorRTRuntime"
                )
                if self.profiling_enabled
                else nullcontext()
            ):
                self.context.execute_async_v2(
                    bindings, torch.cuda.current_stream().cuda_stream
                )

            if len(outputs) == 1:
                return outputs[0]

            return tuple(outputs)

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
