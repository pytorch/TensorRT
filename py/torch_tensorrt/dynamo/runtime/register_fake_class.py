import base64
from collections import defaultdict
from typing import Any, Dict, List

import tensorrt as trt
import torch
from torch_tensorrt.dynamo.utils import input_is_dynamic, unwrap_tensor_shape
from torch_tensorrt.logging import TRT_LOGGER


@torch.library.register_fake("tensorrt::execute_engine")  # type: ignore
def fake_tensorrt_execute_engine(
    inputs: List[torch.Tensor], fake_trt_engine: Any
) -> Any:
    """
    We infer outputs using the TRT engine and inputs and return fake tensors in this meta kernel.
    """

    # Get the TRT engine from the fake TRTEngine object
    serialized_state = fake_trt_engine.wrapped_obj.state_dict

    # TODO: generalize index
    serialized_engine = base64.b64decode(serialized_state["serialized_engine"])

    # Store input/output names for shape inference
    # TODO: generalize index
    input_names = serialized_state["in_binding_names"]
    output_names = serialized_state["out_binding_names"]
    assert len(input_names) == len(
        inputs
    ), f"Number of inputs serialized in TRTEngine {len(input_names)} doesn't match with the number of inputs found during meta kernel execution {len(inputs)} for execute_engine op"

    # Deserialize the TRT engine
    runtime = trt.Runtime(TRT_LOGGER)
    engine = runtime.deserialize_cuda_engine(serialized_engine)
    context = engine.create_execution_context()

    # Here's what we are doing
    # 1) Check if inputs are dynamic (they have sym ints in their shapes)
    # 2) For dynamic inputs, we gather min_input_shape and max_input shape for all inputs
    # 3) For the above min and max input shape, capture the corresponding min and max output shape using TensorRT's set/get shapes mechanism
    # 4) Create a new symbolic fake tensor using min and max output shape for each output and return them
    # 5) For static inputs, the output shape will be static and we won't need to create sym ints
    is_dynamic_execution = input_is_dynamic(inputs)
    if is_dynamic_execution:
        modes = ["min", "max"]
    else:
        modes = ["opt"]

    outputs_mode_dict = defaultdict(list)
    for mode in modes:
        for input_idx, input in enumerate(inputs):
            # Using TensorRT's infer shape mechanism to infer output shapes
            input_shape = unwrap_tensor_shape(input, mode=mode)
            context.set_input_shape(input_names[input_idx], input_shape)

        for output_name in output_names:
            output_shape = context.get_tensor_shape(output_name)
            outputs_mode_dict[mode].append(output_shape)

    if {"min", "max"}.issubset(outputs_mode_dict):
        assert len(outputs_mode_dict["min"]) == len(outputs_mode_dict["max"])
        num_outputs = len(outputs_mode_dict["min"])
    elif "opt" in outputs_mode_dict:
        num_outputs = len(outputs_mode_dict["opt"])

    assert (
        len(output_names) == num_outputs
    ), f"Number of outputs serialized in TRTEngine {len(output_names)} doesn't match with the number of outputs found during meta kernel execution {num_outputs} for execute_engine op"

    fake_outputs = []
    for out_idx in range(num_outputs):
        output_shape = []
        if is_dynamic_execution:
            output_min_shape = outputs_mode_dict["min"][out_idx]
            output_max_shape = outputs_mode_dict["max"][out_idx]
            # create output symbolic shape
            ctx = torch._custom_ops.get_ctx()
            output_shape = []
            for min_val, max_val in zip(output_min_shape, output_max_shape):
                if min_val != max_val:
                    output_sym_int = ctx.new_dynamic_size(min=min_val, max=max_val)
                    output_shape.append(output_sym_int)
                else:
                    output_shape.append(min_val)
        else:
            output_shape.extend(outputs_mode_dict["opt"][out_idx])

        fake_outputs.append(input.new_empty(output_shape))

    return fake_outputs


@torch._library.register_fake_class("tensorrt::Engine")
class FakeTRTEngine:
    def __init__(self, state_dict: Dict[str, Any]) -> None:
        self.state_dict = state_dict

    @classmethod
    def __obj_unflatten__(cls, flattened_tq: Any) -> Any:
        state_dict = {}
        for key, val in flattened_tq:
            state_dict[key] = val

        return cls(state_dict)
