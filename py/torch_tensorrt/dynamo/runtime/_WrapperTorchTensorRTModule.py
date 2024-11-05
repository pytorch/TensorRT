from __future__ import annotations

import logging
from tempfile import tempdir
from typing import List, Optional, Sequence, Tuple

import nvtx
import torch
import torch_tensorrt
from torch.fx.experimental.proxy_tensor import unset_fake_temporarily
from torch_tensorrt.dynamo import partitioning
from torch_tensorrt.dynamo.conversion import DYNAMIC_DIM
from torch_tensorrt.runtime._utils import _is_switch_required, _select_rt_device

logger = logging.getLogger(__name__)


class WrapperTorchTensorRTModule(torch.nn.Module):  # type: ignore[misc]
    """This Wrapper runtime module to record/replay cuda graph in sub modules"""

    def __init__(
        self,
        original_module: torch.nn.Module,
        output_dtypes: List[torch.dtype],
    ):
        super(WrapperTorchTensorRTModule, self).__init__()
        self.original_module = original_module
        self.inputs = partitioning.construct_submodule_inputs(original_module)
        self.output_shapes: List[torch.Tensor] = []
        self.output_dtypes = output_dtypes

        self._input_buffers: List[torch.Tensor] = []
        self._output_buffers: List[torch.Tensor] = []
        self.cudagraph: Optional[torch.cuda.CUDAGraph] = None
        self.shape_key: Optional[str] = None
        self.profiling_enabled = False
        self.cudagraphs_enabled = False
        self._caller_stream: Optional[torch.cuda.Stream] = None
        self._engine_stream: Optional[torch.cuda.Stream] = None

        # Disable cudagrphs in submodules as it will be enabled in wrapper
        for name, rt_mod in self.original_module.named_children():
            if "_run_on_acc" in name:
                rt_mod.cudagraphs_enabled_parent_module = True

        # TODO: check if only torch needs warm up.
        with unset_fake_temporarily():
            inputs_tensor = [spec.torch_tensor.cuda() for spec in self.inputs]
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    self.original_module(*inputs_tensor)
            torch.cuda.current_stream().wait_stream(s)

    def validate_input_shapes(self, inputs: Sequence[torch.Tensor]) -> bool:
        """
        Validates the input shapes of the forward function has changed
        """
        # Representation of input shapes to a given model
        # Shapes are concatenated as so:
        # x: (3, 4), y: (4, 5) --> Key: (3,4)(4,5)
        new_shape_key = "".join(str(tuple(t.shape)).replace(" ", "") for t in inputs)

        # If the new shape key differs from the existing one, infer new output shape
        if new_shape_key != self.shape_key:
            logger.debug(f"Input shape changed {self.shape_key} -> {new_shape_key}")
            self.shape_key = new_shape_key

            # TODO: avoid it for static input shape
            outputs = self.original_module(*inputs)
            if not isinstance(outputs, (list, tuple)):
                outputs = [outputs]
            self.output_shapes = [tuple(output.shape) for output in outputs]
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
        with nvtx.annotate("Wrapper:Forward", color="orange"):

            shape_changed = self.validate_input_shapes(inputs)
            cudagraphs_enabled = torch_tensorrt.runtime.get_cudagraphs_mode()
            # Cudagraphs record is required if cudagraphs_enabled is toggled to True regardless of shape change
            if not self.cudagraphs_enabled and cudagraphs_enabled:
                need_cudagraphs_record = True
            else:
                need_cudagraphs_record = cudagraphs_enabled and shape_changed
            self.cudagraphs_enabled = cudagraphs_enabled

            if need_cudagraphs_record:
                if self.cudagraph:
                    self.cudagraph.reset()
                self._input_buffers = [None] * len(self.inputs)
                self._output_buffers = [None] * len(self.output_shapes)

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

            with nvtx.annotate("Wrapper:ProcessInputs", color="orange"):
                assert len(contiguous_inputs) == len(
                    self.inputs
                ), f"Wrong number of inputs, expect {len(self.inputs)} get {len(contiguous_inputs)}."

                for i, input_name in enumerate(self.inputs):
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
                        contiguous_inputs[i].dtype == self.inputs[i].dtype
                    ), f"Dtype mismatch for {i}th input. Expect {self.inputs[i].dtype}, got {contiguous_inputs[i].dtype}."

                    if need_cudagraphs_record:
                        # If cudagraphs is enabled, this memory is reserved for future cudagraph runs
                        # Clone is required to avoid re-using user-provided GPU memory
                        self._input_buffers[i] = contiguous_inputs[i].clone()
                    elif cudagraphs_enabled:
                        self._input_buffers[i].copy_(contiguous_inputs[i])

            with nvtx.annotate("ProcessOutputs", color="red"):
                # create output tensors
                outputs: List[torch.Tensor] = []

                for o, shape in enumerate(self.output_shapes):
                    if DYNAMIC_DIM in shape:
                        raise ValueError(
                            "Encountered dynamic output shapes during runtime. This could mean the network has data-dependent output shapes which is not currently supported."
                        )

                    output = torch.empty(
                        size=shape,
                        dtype=self.output_dtypes[o],
                        device=torch.cuda.current_device(),
                    )

                    outputs.append(output)

                    if need_cudagraphs_record:
                        self._output_buffers[o] = outputs[o].clone()

            with nvtx.annotate("Wrapper:TensorRTRuntime", color="orange"):
                self._caller_stream = torch.cuda.current_stream()
                if (
                    self._engine_stream == torch.cuda.default_stream()
                    or self._engine_stream is None
                ):
                    self._engine_stream = torch.cuda.Stream()

                with nvtx.annotate("wait_stream", color="green"):
                    self._engine_stream.wait_stream(self._caller_stream)

                with torch.cuda.stream(self._engine_stream):
                    if cudagraphs_enabled:
                        if need_cudagraphs_record:
                            with nvtx.annotate("CUDAGraph", color="green"):
                                self.cudagraph = torch.cuda.CUDAGraph()

                            if self.profiling_enabled:
                                self.cudagraph.enable_debug_mode()
                            with nvtx.annotate("torch.cuda.graph", color="green"):
                                with torch.cuda.graph(
                                    self.cudagraph, stream=self._engine_stream
                                ):
                                    with nvtx.annotate("record", color="green"):
                                        self._output_buffers = self.original_module(
                                            *self._input_buffers
                                        )

                            if self.profiling_enabled:
                                import tempfile

                                with tempfile.TemporaryDirectory() as tmpdir:
                                    self.cudagraph.debug_dump(
                                        f"{tempdir}/{self.name}_cudagraph.dot"
                                    )
                        with nvtx.annotate("replay", color="green"):
                            self.cudagraph.replay()  # type: ignore

                    else:
                        outputs = self.original_module(*inputs)

                self._caller_stream.wait_stream(self._engine_stream)

            if cudagraphs_enabled:
                # TODO: submodule to return list only
                if isinstance(self._output_buffers, (list, tuple)):
                    output_buffers = self._output_buffers
                else:
                    output_buffers = [self._output_buffers]
                for idx, o in enumerate(outputs):
                    o.copy_(output_buffers[idx])

                if len(outputs) == 1:
                    return outputs[0]

                return outputs
            else:
                return outputs
