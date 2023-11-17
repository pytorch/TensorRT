import unittest

import numpy as np
import pycuda.driver as cuda
import tensorrt as trt
import torch
import torch_tensorrt
from torch_tensorrt.dynamo.utils import COSINE_THRESHOLD, cosine_similarity

try:
    import pycuda.autoprimaryctx
except ModuleNotFoundError:
    import pycuda.autoinit


class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream


def do_inference_v2(context, bindings, inputs, outputs, stream):
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]

    gt_tensor = gt_tensor.flatten().to(torch.float32)
    pred_tensor = pred_tensor.flatten().to(torch.float32)
    if torch.sum(gt_tensor) == 0.0 or torch.sum(pred_tensor) == 0.0:
        if torch.allclose(gt_tensor, pred_tensor, atol=1e-4, rtol=1e-4, equal_nan=True):
            return 1.0
    res = torch.nn.functional.cosine_similarity(gt_tensor, pred_tensor, dim=0, eps=1e-6)
    res = res.cpu().detach().item()

    return res


class TestConvertMethodToTrtEngine(unittest.TestCase):
    def test_convert_module(self):
        class Test(torch.nn.Module):
            def forward(self, a, b):
                return torch.add(a, b)

        # Prepare the input data
        input_data_0, input_data_1 = torch.randn((2, 4)), torch.randn((2, 4))

        # Create a model
        model = Test()
        symbolic_traced_gm = torch.fx.symbolic_trace(model)

        # Convert to TensorRT engine
        trt_engine_str = torch_tensorrt.dynamo.convert_method_to_trt_engine(
            symbolic_traced_gm, "forward", inputs=[input_data_0, input_data_1]
        )

        # Deserialize the TensorRT engine
        with trt.Logger() as logger, trt.Runtime(logger) as runtime:
            engine = runtime.deserialize_cuda_engine(trt_engine_str)

        # Allocate memory for inputs and outputs
        inputs, outputs, bindings, stream = allocate_buffers(engine)
        context = engine.create_execution_context()

        # Copy input data to buffer (need .ravel() here, as the inputs[0] buffer is (4,) not (2, 2))
        np.copyto(inputs[0].host, input_data_0.ravel())
        np.copyto(inputs[1].host, input_data_1.ravel())

        # Inference on TRT Engine
        trt_outputs = do_inference_v2(
            context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream
        )
        trt_output = torch.from_numpy(trt_outputs[0])

        # Inference on PyTorch model
        model_output = model(input_data_0, input_data_1)

        cos_sim = cosine_similarity(model_output, trt_output)
        self.assertTrue(
            cos_sim > COSINE_THRESHOLD,
            msg=f"TRT outputs don't match with the original model. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}",
        )


if __name__ == "__main__":
    unittest.main()
