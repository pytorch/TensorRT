from __future__ import annotations

import logging
from contextlib import nullcontext
from tempfile import tempdir
from typing import List, Optional, Sequence, Tuple

import torch
import torch_tensorrt
from torch.fx.experimental.proxy_tensor import unset_fake_temporarily
from torch_tensorrt.dynamo import partitioning
from torch_tensorrt.runtime._utils import _is_switch_required, _select_rt_device

logger = logging.getLogger(__name__)


class WrapperTorchTensorRTModule(torch.nn.Module):  # type: ignore[misc]
    """This Wrapper runtime module is to record/replay whole cuda graph in sub modules

    Args:
        compiled_module: Complied fx graphModule that will be wrapped
    Returns:
        Output tensor or tensor list
    """

    def __init__(
        self,
        compiled_module: torch.nn.Module,
    ):
        super(WrapperTorchTensorRTModule, self).__init__()
        self.compiled_module = compiled_module
        self.inputs = partitioning.construct_submodule_inputs(compiled_module)

        self._input_buffers: List[torch.Tensor] = []
        self._output_buffers: List[torch.Tensor] = []
        self.cudagraph: Optional[torch.cuda.CUDAGraph] = None
        self.shape_key: Optional[str] = None
        self.profiling_enabled = False
        self.prev_cudagraphs_enabled = False
        self._caller_stream: Optional[torch.cuda.Stream] = None
        self._engine_stream: Optional[torch.cuda.Stream] = None

        # Disable cudagrphs in submodules as it will be enabled in wrapper
        for name, rt_mod in self.compiled_module.named_children():
            if "_run_on_acc" in name:
                rt_mod.set_whole_cudagraphs(True)
        self.warm_up()

    def warm_up(self) -> None:
        """
        Warm up is necessary to ensure that memory allocations and initializations
        are not recorded in cuda graphs
        """
        with torch_tensorrt.logging.errors():
            with unset_fake_temporarily():
                inputs_tensor = [spec.torch_tensor.cuda() for spec in self.inputs]
                s = torch.cuda.Stream()
                s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):
                    for _ in range(3):
                        self.compiled_module(*inputs_tensor)
                torch.cuda.current_stream().wait_stream(s)

    def validate_input_shapes(self, inputs: Sequence[torch.Tensor]) -> bool:
        """
        Validates the input shapes of the forward function has changed
        And infer output shapes if dynamic input shape has changed.
        """
        # Representation of input shapes to a given model
        # Shapes are concatenated as so:
        # x: (3, 4), y: (4, 5) --> Key: (3,4)(4,5)
        new_shape_key = "".join(str(tuple(t.shape)).replace(" ", "") for t in inputs)

        if new_shape_key != self.shape_key:
            logger.debug(f"Input shape changed {self.shape_key} -> {new_shape_key}")
            self.shape_key = new_shape_key
            return True

        return False

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
            torch.autograd.profiler.record_function(
                "WrapperTorchTensorRTModule:Forward"
            )
            if self.profiling_enabled
            else nullcontext()
        ):
            shape_changed = self.validate_input_shapes(inputs)
            cudagraphs_enabled = torch_tensorrt.runtime.get_cudagraphs_mode()
            # Cudagraphs record is required if cudagraphs_enabled is toggled to True regardless of shape change
            need_cudagraphs_record = cudagraphs_enabled and (
                (not self.prev_cudagraphs_enabled) or shape_changed
            )
            self.prev_cudagraphs_enabled = cudagraphs_enabled

            if need_cudagraphs_record:
                if self.cudagraph:
                    self.cudagraph.reset()

                self._input_buffers = [None] * len(self.inputs)

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
                    "WrapperTorchTensorRTModule:ProcessInputs"
                )
                if self.profiling_enabled
                else nullcontext()
            ):
                assert len(contiguous_inputs) == len(
                    self.inputs
                ), f"Wrong number of inputs, expect {len(self.inputs)} get {len(contiguous_inputs)}."

                for i, _ in enumerate(self.inputs):
                    if not contiguous_inputs[i].is_cuda:
                        logger.warning(
                            f"Detected input[{i}] of engine {self.engine.name} is not on a cuda device. "
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
                        contiguous_inputs[i].dtype == self.inputs[i].dtype
                    ), f"Dtype mismatch for {i}th input. Expect {self.inputs[i].dtype}, got {contiguous_inputs[i].dtype}."

                    if need_cudagraphs_record:
                        # If cudagraphs is enabled, this memory is reserved for future cudagraph runs
                        # Clone is required to avoid re-using user-provided GPU memory
                        self._input_buffers[i] = contiguous_inputs[i].clone()
                    elif cudagraphs_enabled:
                        self._input_buffers[i].copy_(contiguous_inputs[i])

            with (
                torch.autograd.profiler.record_function(
                    "WrapperTorchTensorRTModule:TensorRTRuntime"
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
                                self._output_buffers = self.compiled_module(
                                    *self._input_buffers
                                )

                            if self.profiling_enabled:
                                import tempfile

                                with tempfile.TemporaryDirectory() as tmpdir:
                                    self.cudagraph.debug_dump(
                                        f"{tempdir}/{self.name}_cudagraph.dot"
                                    )
                        self.cudagraph.replay()  # type: ignore

                    else:
                        outputs = self.compiled_module(*inputs)

                self._caller_stream.wait_stream(self._engine_stream)

            if cudagraphs_enabled:
                if isinstance(self._output_buffers, (list, tuple)):
                    output_buffers = self._output_buffers
                else:
                    output_buffers = [self._output_buffers]
                outputs = [output.clone() for output in output_buffers]
                if len(outputs) == 1:
                    return outputs[0]

                return outputs
            else:
                return outputs
