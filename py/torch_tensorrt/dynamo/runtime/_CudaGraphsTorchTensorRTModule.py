# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import logging
from typing import Any, List, Optional, Sequence, Tuple

import torch
from torch.fx.experimental.proxy_tensor import unset_fake_temporarily
from torch.utils._pytree import tree_flatten, tree_map, tree_unflatten

import torch_tensorrt
from torch_tensorrt.dynamo import partitioning

logger = logging.getLogger(__name__)


def _unflatten_inputs(
    flattened_inputs: Sequence[torch_tensorrt.Input],
    compiled_module: torch.fx.GraphModule,
) -> Tuple[Any, Any]:
    """
    Process inputs using tree_unflatten and tree_map to reconstructe inputs

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

    # Reconstruct the (args, kwargs) structure that was flattened during export
    pytree_inputs = tree_unflatten(flattened_inputs, compiled_module._in_spec)
    # Apply the tensor creation to the reconstructed structure
    processed_inputs = tree_map(convert_input_to_cuda_tensor, pytree_inputs)

    # Since inputs were originally flattened from (args, kwargs),
    # processed_inputs is now that same tuple structure
    return processed_inputs[0], processed_inputs[1]


class CudaGraphsTorchTensorRTModule(torch.nn.Module):  # type: ignore[misc]
    """This Wrapper runtime module is to record/replay whole cuda graph in sub modules

    Args:
        compiled_module: Complied fx graphModule that will be wrapped
    Returns:
        Model outputs with their original container structure
    """

    def __init__(
        self,
        compiled_module: torch.nn.Module,
    ):
        super(CudaGraphsTorchTensorRTModule, self).__init__()
        self.compiled_module = compiled_module
        self.is_weight_streaming_set = False

        self._input_buffers: List[torch.Tensor] = []
        self._input_storage_buffers: List[torch.Tensor] = []
        self._output_buffers: Any = None
        self.cudagraph: Optional[torch.cuda.CUDAGraph] = None
        self.use_output_allocator_outputs = False
        self.shape_key: Optional[Tuple[Any, ...]] = None
        self._caller_stream: Optional[torch.cuda.Stream] = None
        self._engine_stream: Optional[torch.cuda.Stream] = None
        self._owned_pool_stream: Optional[torch.cuda.Stream] = None
        from torch_tensorrt.dynamo.region_compilation import get_region_records

        self._has_torch_regions = any(
            record.route == "torch" for record in get_region_records(compiled_module)
        )
        self._capture_context: Optional[Tuple[Any, ...]] = None
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
        # Region backends need real inputs and the caller's context for warmup.
        if not self._has_torch_regions:
            self.warm_up()

    def warm_up(self) -> None:
        """
        Warm up is necessary to ensure that memory allocations and initializations
        are not recorded in cuda graphs
        """

        with torch_tensorrt.logging.errors():
            with unset_fake_temporarily():
                inputs = partitioning.construct_submodule_inputs(self.compiled_module)
                args, kwargs = _unflatten_inputs(inputs, self.compiled_module)
                s = torch.cuda.Stream()
                s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):
                    for _ in range(3):
                        self.compiled_module(*args, **kwargs)
                torch.cuda.current_stream().wait_stream(s)

    def validate_input_shapes(self, inputs: Sequence[torch.Tensor]) -> bool:
        """Invalidate capture when inputs need different buffers or kernels."""
        if self._has_torch_regions:
            storage_groups: dict[int, int] = {}
            metadata = []
            for tensor in inputs:
                if (
                    not isinstance(tensor, torch.Tensor)
                    or not tensor.is_cuda
                    or tensor.layout != torch.strided
                    or tensor.is_quantized
                    or tensor.is_conj()
                    or tensor.is_neg()
                ):
                    raise RuntimeError(
                        "CUDA graph capture of PyTorch regions requires strided CUDA "
                        "tensor inputs without quantization or conjugate/negative view bits"
                    )
                storage = tensor.untyped_storage()
                group = storage_groups.setdefault(storage._cdata, len(storage_groups))
                metadata.append(
                    (
                        tuple(tensor.shape),
                        tensor.dtype,
                        tensor.device,
                        tensor.stride(),
                        tensor.storage_offset(),
                        storage.nbytes(),
                        group,
                        tensor.requires_grad,
                    )
                )
            if len({tensor.device for tensor in inputs}) > 1:
                raise RuntimeError(
                    "CUDA graph capture of PyTorch regions requires inputs on one CUDA device"
                )
            new_shape_key: Tuple[Any, ...] = tuple(metadata)
        else:
            new_shape_key = tuple((tuple(t.shape), t.dtype, t.device) for t in inputs)

        if new_shape_key != self.shape_key:
            logger.debug(
                "Input metadata changed %s -> %s", self.shape_key, new_shape_key
            )
            self.shape_key = new_shape_key
            return True

        return False

    def _copy_region_inputs(
        self, inputs: Sequence[torch.Tensor], *, recapture: bool
    ) -> None:
        """Copy backing storage once, preserving input views and shared storage."""
        if recapture:
            self._input_buffers = []
            self._input_storage_buffers = []
        storage_groups: dict[int, int] = {}
        for tensor in inputs:
            storage = tensor.untyped_storage()
            group = storage_groups.get(storage._cdata)
            if group is None:
                group = len(storage_groups)
                storage_groups[storage._cdata] = group
                # as_strided may read storage outside the input's visible elements.
                source = torch.empty(0, dtype=torch.uint8, device=tensor.device).set_(
                    storage, 0, (storage.nbytes(),), (1,)
                )
                if recapture:
                    self._input_storage_buffers.append(source.clone())
                else:
                    self._input_storage_buffers[group].copy_(source)
            if recapture:
                view = torch.empty(0, dtype=tensor.dtype, device=tensor.device).set_(
                    self._input_storage_buffers[group].untyped_storage(),
                    tensor.storage_offset(),
                    tensor.shape,
                    tensor.stride(),
                )
                self._input_buffers.append(view.requires_grad_(tensor.requires_grad))

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

    def _warm_up_regions(self, args: Any, kwargs: Any) -> None:
        """Initialize Torch kernels and compiled backends before capture."""
        if not self._has_torch_regions:
            return
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "PyTorch regions must warm up before CUDA graph capture; "
                "enable_cudagraphs cannot be entered inside another capture."
            )
        assert self._engine_stream is not None
        values, _ = tree_flatten((args, kwargs))
        values.extend(self.compiled_module.parameters())
        values.extend(self.compiled_module.buffers())
        devices = {
            value.device
            for value in values
            if isinstance(value, torch.Tensor) and value.is_cuda
        }
        devices.add(self._engine_stream.device)
        # Warmup must not consume the caller's CPU or CUDA random sequence.
        with torch.random.fork_rng(devices=devices, device_type="cuda"):
            for _ in range(3):
                self.compiled_module(*args, **kwargs)
            self._engine_stream.synchronize()

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        inputs, _ = tree_flatten((args, kwargs))
        cudagraphs_enabled = torch_tensorrt.runtime.get_whole_cudagraphs_mode()
        if cudagraphs_enabled:
            shape_changed = self.validate_input_shapes(inputs)

            current_device = (
                inputs[0].device
                if inputs and inputs[0].is_cuda
                else self._target_device
            )
            if self._has_torch_regions and current_device != self._target_device:
                self._target_device = current_device
                self._default_stream = torch.cuda.default_stream(current_device)
                self._owned_pool_stream = None
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

            capture_context = (
                torch.is_grad_enabled(),
                torch.is_inference_mode_enabled(),
                torch.is_autocast_enabled("cuda"),
                torch.get_autocast_dtype("cuda"),
            )
            context_changed = (
                self._has_torch_regions and capture_context != self._capture_context
            )

            need_cudagraphs_record = (
                self.cudagraph is None
                or shape_changed
                or self.is_weight_streaming_set
                or stream_changed
                or context_changed
            )
            if need_cudagraphs_record:
                self._reset_captured_graph()
                self._input_buffers = [None] * len(inputs)

            self.is_weight_streaming_set = False
            if self._has_torch_regions:
                self._copy_region_inputs(inputs, recapture=need_cudagraphs_record)
            else:
                contiguous_inputs: List[torch.Tensor] = [
                    (
                        i.contiguous()
                        if isinstance(i, torch.Tensor)
                        else torch.tensor(i).cuda()
                    )
                    for i in inputs
                ]
                for i, tensor in enumerate(contiguous_inputs):
                    if not tensor.is_cuda:
                        logger.warning(
                            "Input[%s] is not on CUDA and is being moved by the runtime",
                            i,
                        )
                        tensor = tensor.cuda()
                    assert (
                        tensor.dtype == inputs[i].dtype
                    ), f"Dtype mismatch for input {i}: expected {inputs[i].dtype}, got {tensor.dtype}"
                    if need_cudagraphs_record:
                        self._input_buffers[i] = tensor.clone()
                    else:
                        self._input_buffers[i].copy_(tensor)

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
                    self._warm_up_regions(args, kwargs)
                    self._check_monolithic_capturability(self._engine_stream)
                    captured_graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(captured_graph, stream=self._engine_stream):
                        self._output_buffers = self.compiled_module(*args, **kwargs)
                    self.cudagraph = captured_graph
                    self._capture_context = capture_context

                self.cudagraph.replay()  # type: ignore
            if caller_on_default:
                self._caller_stream.wait_stream(self._engine_stream)

            # Copy captured tensors while preserving the model's output structure.
            return tree_map(
                lambda value: (
                    value.clone() if isinstance(value, torch.Tensor) else value
                ),
                self._output_buffers,
            )
        else:
            self._reset_captured_graph()
            return self.compiled_module(*args, **kwargs)
