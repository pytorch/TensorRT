import base64
from typing import Any, List

import tensorrt as trt
import torch
from torch_tensorrt.dynamo.utils import contains_sym_int, unwrap_tensor_shape
from torch_tensorrt.logging import TRT_LOGGER


@torch.library.register_fake("tensorrt::execute_engine")  # type: ignore
def fake_tensorrt_execute_engine(inputs: List[torch.Tensor], trt_engine: Any) -> Any:
    # This will call the FakeTRTEngine.__call__ method which runs inference on real TRT engine and inputs and returns the outputs
    # The output should be fake tensors as per general understanding but we are returning real tensors as outputs here which works.
    return trt_engine.wrapped_obj(inputs)


# namespace::class_name
@torch._library.register_fake_class("tensorrt::Engine")
class FakeTRTEngine:
    def __init__(self, engine_info: List[Any]):
        self.engine = torch.classes.tensorrt.Engine(engine_info)

    @classmethod
    def __obj_unflatten__(cls, flattened_tq: Any) -> Any:
        engine_info = [info[1] for info in flattened_tq]
        engine_info[3] = base64.b64decode(engine_info[3])  # decode engine
        engine_info[4] = str(engine_info[4][0])  # input names
        engine_info[5] = str(engine_info[5][0])  # output names
        engine_info[6] = str(int(engine_info[6]))  # hw compatible
        return cls(engine_info)

    def __call__(self, inputs: List[torch.Tensor]) -> Any:

        serialized_state = self.engine.__getstate__()
        serialized_engine = base64.b64decode(serialized_state[0][3])
        input_name = serialized_state[0][4]
        output_name = serialized_state[0][5]

        runtime = trt.Runtime(TRT_LOGGER)
        engine = runtime.deserialize_cuda_engine(serialized_engine)
        context = engine.create_execution_context()

        fake_outputs = []
        for input in inputs:
            if contains_sym_int(input.shape):
                # Using TensorRT's infer shape mechanism to infer output shapes
                input_min_shape = unwrap_tensor_shape(input, mode="min")
                context.set_input_shape(input_name, input_min_shape)
                output_min_shape = context.get_tensor_shape(output_name)

                input_max_shape = unwrap_tensor_shape(input, mode="max")
                context.set_input_shape(input_name, input_max_shape)
                output_max_shape = context.get_tensor_shape(output_name)

                ctx = torch._custom_ops.get_ctx()

                # create output symbolic shape
                output_shape = []
                for min_val, max_val in zip(output_min_shape, output_max_shape):
                    if min_val != max_val:
                        output_sym_int = ctx.new_dynamic_size(min=min_val, max=max_val)
                        output_shape.append(output_sym_int)
                    else:
                        output_shape.append(min_val)

            else:
                input_shape = unwrap_tensor_shape(input)
                context.set_input_shape(input_name, input_shape)
                output_shape = context.get_tensor_shape(output_name)

            fake_outputs.append(input.new_empty(output_shape))

        return fake_outputs
