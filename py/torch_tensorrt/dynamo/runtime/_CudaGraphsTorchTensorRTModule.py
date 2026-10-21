# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import logging
from typing import Any, List, Optional, Sequence, Set, Tuple

import torch
import torch_tensorrt
from torch.fx._pytree import tree_flatten_spec
from torch.fx.experimental.proxy_tensor import unset_fake_temporarily
from torch.utils._pytree import tree_map, tree_unflatten
from torch_tensorrt.dynamo import partitioning

logger = logging.getLogger(__name__)


def _unflatten_inputs(
    flattened_inputs: Sequence[torch_tensorrt.Input],
    compiled_module: torch.fx.GraphModule,
) -> Tuple[Any, Any]:
    """
    Create the input tensors and use tree_unflatten to reconstruct inputs

    Args:
        flattened_inputs: Flattened input tensors to process
        compiled_module: The compiled GraphModule containing input specifications

    Returns:
        Tuple of (args, kwargs) containing reconstructed input tensors
    """

    def convert_input_to_cuda_tensor(input: Any) -> torch.Tensor:
        if isinstance(input, torch_tensorrt.Input):
            return input.torch_tensor.cuda()
        else:
            raise RuntimeError("Input is not a torch_tensorrt.Input")

    # Create the tensors before unflattening: a custom pytree node (e.g. a Hugging
    # Face cache) may only keep tensors as leaves, as in the exported program
    tensors = [convert_input_to_cuda_tensor(i) for i in flattened_inputs]
    # Reconstruct the (args, kwargs) structure that was flattened during export
    args, kwargs = tree_unflatten(tensors, compiled_module._in_spec)
    return args, kwargs


def _inputs_updated_in_place(compiled_module: torch.fx.GraphModule) -> Set[int]:
    """
    Find the inputs the module updates in place (e.g. a KV cache): those written by
    a PyTorch op or aliased by a TensorRT engine's output

    Args:
        compiled_module: The compiled GraphModule

    Returns:
        Positions of those inputs among the module's flattened inputs
    """

    def written(gm: torch.fx.GraphModule) -> Set[torch.fx.Node]:
        nodes = set()
        for node in gm.graph.nodes:
            if not isinstance(node.target, torch._ops.OpOverload):
                continue
            for i, arg in enumerate(node.target._schema.arguments):
                if arg.alias_info is not None and arg.alias_info.is_write:
                    value = (
                        node.args[i]
                        if i < len(node.args)
                        else node.kwargs.get(arg.name)
                    )
                    if isinstance(value, torch.fx.Node):
                        nodes.add(value)
        return nodes

    targets = written(compiled_module)
    for node in compiled_module.graph.find_nodes(op="call_module"):
        submodule = compiled_module.get_submodule(node.target)
        if isinstance(submodule, torch.fx.GraphModule):
            sub_targets = written(submodule)
            sub_inputs = submodule.graph.find_nodes(op="placeholder")
            targets.update(a for a, p in zip(node.args, sub_inputs) if p in sub_targets)
        elif getattr(submodule, "aliased_io", None):
            aliased = {name for name, _ in submodule.aliased_io.values()}
            targets.update(
                a
                for a, name in zip(node.args, submodule.input_binding_names)
                if name in aliased
            )
    placeholders = compiled_module.graph.find_nodes(op="placeholder")
    return {i for i, p in enumerate(placeholders) if p in targets}


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
        self._inputs_updated_in_place = _inputs_updated_in_place(compiled_module)
        self.is_weight_streaming_set = False

        self._input_buffers: List[torch.Tensor] = []
        self._output_buffers: List[torch.Tensor] = []
        self.cudagraph: Optional[torch.cuda.CUDAGraph] = None
        self.use_output_allocator_outputs = False
        self.shape_key: Optional[str] = None
        self._caller_stream: Optional[torch.cuda.Stream] = None
        self._engine_stream: Optional[torch.cuda.Stream] = None
        self._owned_pool_stream: Optional[torch.cuda.Stream] = None
        # Bind to the device of the wrapped module's first CUDA parameter so
        # multi-GPU deployments cache the right default stream.
        target = next(
            (p.device for p in self.compiled_module.parameters() if p.is_cuda),
            None,
        )
        self._target_device = (
            target
            if target is not None
            else torch.device("cuda", torch.cuda.current_device())
        )
        self._default_stream = torch.cuda.default_stream(self._target_device)
        self.warm_up()

    def warm_up(self) -> None:
        """
        Warm up is necessary to ensure that memory allocations and initializations
        are not recorded in cuda graphs
        """

        with torch_tensorrt.logging.errors():
            with unset_fake_temporarily():
                args, kwargs = _unflatten_inputs(self.inputs, self.compiled_module)
                s = torch.cuda.Stream()
                s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):
                    for _ in range(3):
                        self.compiled_module(*args, **kwargs)
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
        # An input updated in place is recorded as the caller's tensor when it is
        # contiguous, so the graph writes to that address: a new one needs a new graph
        new_shape_key += str(
            [
                (inputs[i].data_ptr(), inputs[i].is_contiguous())
                for i in sorted(self._inputs_updated_in_place)
            ]
        )

        if new_shape_key != self.shape_key:
            logger.debug(f"Input shape changed {self.shape_key} -> {new_shape_key}")
            self.shape_key = new_shape_key
            return True

        return False

    def _reset_captured_graph(self) -> None:
        if self.cudagraph:
            self.cudagraph.reset()
            self.cudagraph = None

    def __del__(self) -> None:
        self._reset_captured_graph()

    def set_use_output_allocator(self, enable: bool) -> None:
        self.use_output_allocator_outputs = enable

    def _check_monolithic_capturability(self, stream: torch.cuda.Stream) -> None:
        """Verify every TRT submodule is safe for monolithic stream capture.

        Whole-graph CUDA graph mode wraps mixed TRT + PyTorch ops in a
        single outer ``torch.cuda.CUDAGraph`` capture. On TRT-RTX, each
        engine must opt out of RTX-native CUDA graphs (which would
        interfere with the outer capture) and must pass the
        ``IExecutionContext.is_stream_capturable`` check. Raises
        ``RuntimeError`` if any TRT engine is not monolithically
        capturable. No-op on non-RTX builds.
        """
        from torch_tensorrt._features import ENABLED_FEATURES

        if not ENABLED_FEATURES.tensorrt_rtx:
            return
        from torch_tensorrt.dynamo.runtime._TorchTensorRTModule import (
            TorchTensorRTModule,
        )
        from torch_tensorrt.dynamo.runtime._TRTEngine import TRTEngine

        for name, mod in self.compiled_module.named_modules():
            if not (
                isinstance(mod, TorchTensorRTModule)
                and isinstance(mod.engine, TRTEngine)
            ):
                continue
            engine = mod.engine
            if not engine._is_monolithic_capturable(stream):
                raise RuntimeError(
                    f"CUDA graph capture failed: TRT submodule '{name}' is "
                    "not monolithically capturable (lazy kernel "
                    "specialization or non-capturable stream). Whole-graph "
                    "CUDA graph mode with mixed TRT + PyTorch ops requires "
                    "all TRT engines to be capturable. Consider using "
                    "cuda_graph_strategy='whole_graph_capture' with "
                    "set_cudagraphs_mode(True) instead of enable_cudagraphs()."
                )
            # Disable RTX-native CUDA graphs on this engine so they don't
            # interfere with the outer monolithic capture.
            if engine._rtx_native_cudagraphs:
                disabled = engine.runtime_settings.merge(cuda_graph_strategy="disabled")
                engine.update_runtime_settings(disabled)
                engine._rtx_native_cudagraphs = False
                logger.info(
                    f"Disabled RTX-native CUDA graphs for '{name}' "
                    "(using outer monolithic capture instead)"
                )

    def forward(
        self, *args: Any, **kwargs: Any
    ) -> torch.Tensor | Tuple[torch.Tensor, ...]:
        # Flatten by the input spec, as the compiled module does: keyword arguments
        # given in another order than at export must still fill the same inputs
        inputs = tree_flatten_spec((args, kwargs), self.compiled_module._in_spec)
        cudagraphs_enabled = torch_tensorrt.runtime.get_whole_cudagraphs_mode()
        if cudagraphs_enabled:
            shape_changed = self.validate_input_shapes(inputs)

            current_device = (
                inputs[0].device
                if inputs and inputs[0].is_cuda
                else self._target_device
            )
            default_stream = self._default_stream
            previous_engine_stream = self._engine_stream
            self._caller_stream = torch.cuda.current_stream(current_device)
            caller_on_default = self._caller_stream == default_stream
            if caller_on_default:
                if self._owned_pool_stream is None:
                    self._owned_pool_stream = torch.cuda.Stream(self._target_device)
                self._engine_stream = self._owned_pool_stream
            else:
                # Honor caller's non-default stream so its scheduling choice (e.g. SM
                # partitioning via a CUDA Green Context) is preserved end to end.
                self._engine_stream = self._caller_stream
            stream_changed = self._engine_stream != previous_engine_stream

            need_cudagraphs_record = (
                shape_changed or self.is_weight_streaming_set or stream_changed
            )
            if need_cudagraphs_record:
                self._reset_captured_graph()
                self._input_buffers = [None] * len(inputs)

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
                inputs
            ), f"Wrong number of inputs, expect {len(inputs)} get {len(contiguous_inputs)}."

            for i, _ in enumerate(inputs):
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
                    contiguous_inputs[i].dtype == inputs[i].dtype
                ), f"Dtype mismatch for {i}th input. Expect {inputs[i].dtype}, got {contiguous_inputs[i].dtype}."

                if (
                    i in self._inputs_updated_in_place
                    and contiguous_inputs[i] is inputs[i]
                ):
                    # The graph updates this input in place (e.g. a KV cache): record it
                    # as the caller's tensor, so the updates land there as they would
                    # without CUDA graphs, with no copies
                    self._input_buffers[i] = inputs[i]
                elif need_cudagraphs_record:
                    # If cudagraphs is enabled, this memory is reserved for future cudagraph runs
                    # Clone is required to avoid re-using user-provided GPU memory
                    self._input_buffers[i] = contiguous_inputs[i].clone()
                else:
                    self._input_buffers[i].copy_(contiguous_inputs[i])

            if need_cudagraphs_record:
                # Reconstruct the original args and kwargs structure from static input buffers
                # using the input specification stored during module compilation
                args, kwargs = tree_unflatten(
                    self._input_buffers, self.compiled_module._in_spec
                )

            if caller_on_default:
                self._engine_stream.wait_stream(self._caller_stream)

            with torch.cuda.stream(self._engine_stream):
                if need_cudagraphs_record:
                    self._check_monolithic_capturability(self._engine_stream)
                    self.cudagraph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(self.cudagraph, stream=self._engine_stream):
                        self._output_buffers = self.compiled_module(*args, **kwargs)

                self.cudagraph.replay()  # type: ignore
            if caller_on_default:
                self._caller_stream.wait_stream(self._engine_stream)

            # An input updated in place that is not contiguous or not on the GPU was
            # recorded as a copy: write the update back
            for i in self._inputs_updated_in_place:
                if self._input_buffers[i] is not inputs[i]:
                    inputs[i].copy_(self._input_buffers[i])
            in_place = {
                inputs[i].untyped_storage().data_ptr()
                for i in self._inputs_updated_in_place
                if self._input_buffers[i] is inputs[i]
            }

            # Copy the outputs out of the static buffers, keeping the module's output
            # structure (e.g. a Hugging Face ModelOutput or nested tuples). An output
            # in an input updated in place is the caller's tensor already
            return tree_map(
                lambda t: (
                    t.clone()
                    if isinstance(t, torch.Tensor)
                    and t.untyped_storage().data_ptr() not in in_place
                    else t
                ),
                self._output_buffers,
            )
        else:
            self._reset_captured_graph()
            return self.compiled_module(*args, **kwargs)
