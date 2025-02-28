from __future__ import annotations

import logging
from typing import List, Optional, Sequence, Tuple

import torch
import torch_tensorrt
from torch.fx.experimental.proxy_tensor import unset_fake_temporarily
from torch_tensorrt.dynamo import partitioning

logger = logging.getLogger(__name__)


class CudaGraphsTorchTensorRTModule(torch.nn.Module):  # type: ignore[misc]
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
        super(CudaGraphsTorchTensorRTModule, self).__init__()
        self.compiled_module = compiled_module
        self.inputs = partitioning.construct_submodule_inputs(compiled_module)
        self.is_weight_streaming_set = False

        self._input_buffers: List[torch.Tensor] = []
        self._output_buffers: List[torch.Tensor] = []
        self.cudagraph: Optional[torch.cuda.CUDAGraph] = None
        self.use_output_allocator_outputs = False
        self.shape_key: Optional[str] = None
        self._caller_stream: Optional[torch.cuda.Stream] = None
        self._engine_stream: Optional[torch.cuda.Stream] = None
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

    def set_output_allocator_outputs(self, enable: bool) -> None:
        self.use_output_allocator_outputs = enable

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor | Tuple[torch.Tensor, ...]:
        cudagraphs_enabled = torch_tensorrt.runtime.get_whole_cudagraphs_mode()
        if cudagraphs_enabled:
            shape_changed = self.validate_input_shapes(inputs)
            need_cudagraphs_record = shape_changed or self.is_weight_streaming_set
            if need_cudagraphs_record:
                if self.cudagraph:
                    self.cudagraph.reset()
                self._input_buffers = [None] * len(self.inputs)

            self.is_weight_streaming_set = False
            # Ensure inputs are available in all scopes and cast symbolic integers to Tensors
            contiguous_inputs: List[torch.Tensor] = [
                (
                    i.contiguous()
                    if isinstance(i, torch.Tensor)
                    else torch.tensor(i).cuda()
                )
                for i in inputs
            ]
            assert len(contiguous_inputs) == len(
                self.inputs
            ), f"Wrong number of inputs, expect {len(self.inputs)} get {len(contiguous_inputs)}."

            for i, _ in enumerate(self.inputs):
                if not contiguous_inputs[i].is_cuda:
                    logger.warning(
                        f"Detected input[{i}] is not on a cuda device. "
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
                else:
                    self._input_buffers[i].copy_(contiguous_inputs[i])

            self._caller_stream = torch.cuda.current_stream()
            if (
                self._engine_stream == torch.cuda.default_stream()
                or self._engine_stream is None
            ):
                self._engine_stream = torch.cuda.Stream()

            self._engine_stream.wait_stream(self._caller_stream)

            with torch.cuda.stream(self._engine_stream):
                if need_cudagraphs_record:
                    self.cudagraph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(self.cudagraph, stream=self._engine_stream):
                        self._output_buffers = self.compiled_module(
                            *self._input_buffers
                        )

                self.cudagraph.replay()  # type: ignore
            self._caller_stream.wait_stream(self._engine_stream)

            if isinstance(self._output_buffers, (list, tuple)):
                output_buffers = self._output_buffers
            else:
                output_buffers = [self._output_buffers]
            outputs = [output.clone() for output in output_buffers]
            if len(outputs) == 1:
                return outputs[0]
            return outputs
        else:
            if self.cudagraph:
                self.cudagraph.reset()
                self.cudagraph = None
            return self.compiled_module(*inputs)
