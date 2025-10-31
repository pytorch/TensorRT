"""
Minimal reproducible example demonstrating TensorRT fp16 custom_op() issue.

This module shows the bug where torch_tensorrt.dynamo.conversion.plugins.custom_op()
fails to compile operations that use fp16 (half-precision) tensors.

The issue occurs because the JIT plugin generator doesn't properly declare format
support for fp16 data types in the generated TensorRT plugin.
"""

from typing import List, Tuple, Union

import torch

# Import triton for kernel implementation
import triton
import triton.language as tl

import torch_tensorrt

# ============================================================================
# Triton Kernel for Eager Execution
# ============================================================================


@triton.jit
def pointwise_sigmoid_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Program ID determines the block of data each thread will process
    pid = tl.program_id(0)
    # Compute the range of elements that this thread block will work on
    block_start = pid * BLOCK_SIZE
    # Range of indices this thread will handle
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Mask for boundary checking
    mask = offsets < n_elements
    # Load elements from the X tensor
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # Convert to float32 for computation
    x_f32 = x.to(tl.float32)
    # Compute sigmoid: 1 / (1 + exp(-x))
    output = tl.sigmoid(x_f32)
    # Convert back to original dtype
    output_casted = output.to(x.dtype)
    # Store the result in Y
    tl.store(y_ptr + offsets, output_casted, mask=mask)


# ============================================================================
# Custom Op Registration
# ============================================================================


@torch.library.custom_op("pointwise_sigmoid_ops::pointwise_sigmoid", mutates_args=())  # type: ignore[misc]
def pointwise_sigmoid(X: torch.Tensor) -> torch.Tensor:
    # Ensure the tensor is on the GPU
    assert X.is_cuda, "Tensor must be on CUDA device."

    # Create output tensor
    Y = torch.empty_like(X)

    # Define block size
    BLOCK_SIZE = 256

    # Grid of programs
    grid = lambda meta: (triton.cdiv(X.numel(), meta["BLOCK_SIZE"]),)

    # Launch the kernel
    pointwise_sigmoid_kernel[grid](X, Y, X.numel(), BLOCK_SIZE=BLOCK_SIZE)

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


# @trtp.aot_impl("pointwise_sigmoid_ops::pointwise_sigmoid")
# def sigmoid_aot_triton_impl(
#     input: trtp.TensorDesc,
#     outputs: Tuple[trtp.TensorDesc],
#     tactic: int,
# ) -> Tuple[
#     Union[str, bytes], Union[str, bytes], trtp.KernelLaunchParams, trtp.SymExprs
# ]:
#     print("WE ARE NOW GENERATING THE PTX FOR THE PLUGIN (Triton)!!!")

#     # Reuse the same Triton kernel we use for eager execution
#     src = triton.compiler.ASTSource(
#         fn=pointwise_sigmoid_kernel,
#         signature={
#             "x_ptr": "*fp16",
#             "y_ptr": "*fp16",
#             "n_elements": "i32",
#             "BLOCK_SIZE": "constexpr",
#         },
#         constexprs={"BLOCK_SIZE": 256},
#     )

#     compiled_kernel = triton.compile(src)

#     N = input.shape_expr.numel()
#     launch_params = trtp.KernelLaunchParams()
#     launch_params.grid_x = trtp.cdiv(N, 256)
#     launch_params.block_x = compiled_kernel.metadata.num_warps * 32
#     launch_params.shared_mem = compiled_kernel.metadata.shared

#     extra_args = trtp.SymIntExprs(1)
#     extra_args[0] = trtp.SymInt32(N)

#     print(compiled_kernel.asm["ptx"])

#     return (
#         compiled_kernel.metadata.name,
#         compiled_kernel.asm["ptx"],
#         launch_params,
#         extra_args,
#     )


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


@trtp.aot_impl("pointwise_sigmoid_ops::pointwise_sigmoid")
def sigmoid_aot_nvrtc_impl(
    input: trtp.TensorDesc,
    outputs: Tuple[trtp.TensorDesc],
    tactic: int,
) -> Tuple[
    Union[str, bytes], Union[str, bytes], trtp.KernelLaunchParams, trtp.SymExprs
]:
    print("WE ARE NOW GENERATING THE PTX FOR THE PLUGIN (NVRTC)!!!")

    dev = Device()
    dev.set_current()
    program_options = ProgramOptions(
        std="c++17", arch=f"sm_{dev.arch}", include_path=["/usr/local/cuda/include"]
    )
    program = Program(cu_code, code_type="c++", options=program_options)
    mod = program.compile("ptx", name_expressions=("pointwise_sigmoid_kernel_nvrtc",))
    compiled_kernel = mod.code.decode("utf-8")
    print(compiled_kernel)

    N = input.shape_expr.numel()
    launch_params = trtp.KernelLaunchParams()
    launch_params.grid_x = trtp.cdiv((N + 256 - 1), 256)
    launch_params.block_x = 256
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
        x = torch.mul(input, 2)
        y = torch.div(x, 2)
        z = torch.ops.pointwise_sigmoid_ops.pointwise_sigmoid(y)
        a = torch.add(z, 1)
        return a


if __name__ == "__main__":
    model = PointwiseSigmoidModel_WithTRTWrapper().to("cuda").eval()
    input = torch.randn(1, 1024, device="cuda", dtype=torch.float16)

    with torch_tensorrt.logging.debug():
        trt_inputs = [input]
        model_trt = torch_tensorrt.compile(
            model,
            inputs=trt_inputs,
            min_block_size=1,
        )
        print("Model compiled successfully!")
        print("Running inference with compiled model...")
        with torch.no_grad():
            for i in range(10):
                res = model_trt(input)
                assert torch.allclose(
                    res, model(input), rtol=1e-2, atol=1e-2
                ), "Results do not match!"

    print("Inference successful!")
