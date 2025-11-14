"""
Minimal reproducible example demonstrating TensorRT fp16 custom_op() issue.

This module shows the bug where torch_tensorrt.dynamo.conversion.plugins.custom_op()
fails to compile operations that use fp16 (half-precision) tensors.

The issue occurs because the JIT plugin generator doesn't properly declare format
support for fp16 data types in the generated TensorRT plugin.
"""

from typing import List, Tuple, Union

import torch

import torch_tensorrt

# CUDA kernel source (NVRTC) used by the torch custom op
cu_code = """
#include <cuda_fp16.h>

// Simple pointwise Sigmoid kernel: f(x) = 1 / (1 + exp(-x))
__global__ void pointwise_sigmoid_kernel_nvrtc(const __half* __restrict__ input,
                                                __half* __restrict__ output,
                                                const int size) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        const float x = __half2float(input[idx]);
        const float result = 1.0f / (1.0f + expf(-x));
        output[idx] = __float2half(result);
    }
}
"""

# Prepare NVRTC program, kernel, and stream once (simple eager path)
from cuda.core.experimental import (
    Device as _CudaDevice,
    LaunchConfig as _LaunchConfig,
    Program as _CudaProgram,
    ProgramOptions as _CudaProgramOptions,
    launch as _cuda_launch,
)

_cuda_device = _CudaDevice()
_cuda_device.set_current()
_cuda_stream = _cuda_device.create_stream()
_program_options = _CudaProgramOptions(
    std="c++17", arch=f"sm_{_cuda_device.arch}", include_path=["/usr/local/cuda/include"]
)
_program = _CudaProgram(cu_code, code_type="c++", options=_program_options)
_module = _program.compile("ptx", name_expressions=("pointwise_sigmoid_kernel_nvrtc",))
_kernel = _module.get_kernel("pointwise_sigmoid_kernel_nvrtc")

# Eager torch custom_op implemented using the CUDA kernel above (no Triton)


# ============================================================================
# Custom Op Registration
# ============================================================================


@torch.library.custom_op("pointwise_sigmoid_ops::pointwise_sigmoid", mutates_args=())  # type: ignore[misc]
def pointwise_sigmoid(X: torch.Tensor) -> torch.Tensor:
    assert X.is_cuda, "Tensor must be on CUDA device."

    Y = torch.empty_like(X)
    N = int(X.numel())

    block = 256

    grid_x = max(1, (N + block - 1) // block)
    config = _LaunchConfig(grid=(grid_x), block=(block))

    # Use PyTorch's current stream by wrapping it for cuda.core
    class _PyTorchStreamWrapper:
        def __init__(self, pt_stream):
            self.pt_stream = pt_stream

        def __cuda_stream__(self):
            stream_id = self.pt_stream.cuda_stream
            return (0, stream_id)

    pt_stream = torch.cuda.current_stream()
    s = _cuda_device.create_stream(_PyTorchStreamWrapper(pt_stream))

    # Launch kernel with raw pointers as in cuda.core example
    _cuda_launch(
        s,
        config,
        _kernel,
        X.data_ptr(),
        Y.data_ptr(),
        N,
    )

    return Y


@torch.library.register_fake("pointwise_sigmoid_ops::pointwise_sigmoid")
def _(input: torch.Tensor) -> torch.Tensor:
    """Fake implementation for TorchDynamo tracing of base operation."""
    return torch.empty_like(input)


# ============================================================================
# TensorRT Wrapper with custom_op() - THIS FAILS WITH FP16
# ============================================================================

import tensorrt.plugin as trtp
from cuda.core.experimental import Device, LaunchConfig, Program, ProgramOptions


@trtp.register("pointwise_sigmoid_ops::pointwise_sigmoid")
def sigmoid_plugin_desc(input: trtp.TensorDesc) -> Tuple[trtp.TensorDesc]:
    return (input.like(),)


@trtp.autotune("pointwise_sigmoid_ops::pointwise_sigmoid")
def sigmoid_autotune(
    input: trtp.TensorDesc,
    outputs: Tuple[trtp.TensorDesc],
) -> List[trtp.AutoTuneCombination]:
    return [trtp.AutoTuneCombination("FP16, FP16", "LINEAR")]



@trtp.aot_impl("pointwise_sigmoid_ops::pointwise_sigmoid")
def sigmoid_aot_nvrtc_impl(
    input: trtp.TensorDesc,
    outputs: Tuple[trtp.TensorDesc],
    tactic: int,
) -> Tuple[
    Union[str, bytes], Union[str, bytes], trtp.KernelLaunchParams, trtp.SymExprs
]:

    compiled_kernel= _module.code.decode("utf-8")
    print(type(compiled_kernel))
    print(compiled_kernel)

    # import pdb; pdb.set_trace()


    N = input.shape_expr.numel()
    launch_params = trtp.KernelLaunchParams()
    block = 256
    launch_params.grid_x = trtp.cdiv(N, block)
    launch_params.block_x = block
    launch_params.shared_mem = 0

    extra_args = trtp.SymIntExprs(1)
    extra_args[0] = trtp.SymInt32(N)

    return (
        "pointwise_sigmoid_kernel_nvrtc",
        compiled_kernel,
        launch_params,
        extra_args,
    )


torch_tensorrt.dynamo.conversion.plugins.generate_plugin_converter(
    "pointwise_sigmoid_ops::pointwise_sigmoid",
    supports_dynamic_shapes=True,
    requires_output_allocator=False,
)


# ============================================================================
# Test Model
# ============================================================================


class PointwiseSigmoidModel_WithTRTWrapper(torch.nn.Module):
    """
    Test model that uses the TRT wrapper with custom_op() registration.

    When compiled with torch_tensorrt.compile() using fp16 inputs, this will
    fail with: "could not find any supported formats consistent with input/output
    data types"
    """

    def forward(self, input: torch.Tensor) -> torch.Tensor:

        z = torch.ops.pointwise_sigmoid_ops.pointwise_sigmoid(input)
        return z


if __name__ == "__main__":
    model = PointwiseSigmoidModel_WithTRTWrapper().to("cuda").eval()
    input = torch.randn(1, 1024, device="cuda", dtype=torch.float16)

    print(torch.sigmoid(input))

    print(model(input))

    with torch_tensorrt.logging.debug():
        trt_inputs = [input]
        model_trt = torch_tensorrt.compile(
            model,
            inputs=trt_inputs,
            enabled_precisions={torch.float16},
            min_block_size=1,
        )
        print("Model compiled successfully!")
        print("Running inference with compiled model...")
        print("Compiled model output:")
        print(model_trt(input))
        print("Original model output:")
        print(model(input))
        with torch.no_grad():
            for i in range(10):
                res = model_trt(input)
                assert torch.allclose(
                    res, model(input), rtol=1e-2, atol=1e-2
                ), "Results do not match!"

    # print("Inference successful!")
