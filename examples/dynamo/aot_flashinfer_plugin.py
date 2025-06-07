"""
.. _auto_generate_converters:

Automatically Generate a Plugin for a Custom Kernel
===================================================================

We are going to demonstrate how to automatically generate a plugin for a custom kernel using Torch-TensorRT using
the new Python based plugin system in TensorRT 10.7.

Torch-TensorRT supports falling back to PyTorch implementations of operations in the case that Torch-TensorRT
does not know how to compile them in TensorRT. However, this comes at the cost of a graph break and will reduce the performance of the model.
The easiest way to fix lack of support for ops is by adding a decomposition (see:
`Writing lowering passes for the Dynamo frontend <https://pytorch.org/TensorRT/contributors/writing_dynamo_aten_lowering_passes.html>`_) - which defines the operator
in terms of PyTorch ops that are supported in Torch-TensorRT or a converter (see:
`Writing converters for the Dynamo frontend <https://pytorch.org/TensorRT/contributors/dynamo_converters.html>`_) - which defines the operator in terms of TensorRT operators.

In some cases there isn't a great way to do either of these, perhaps because the operator is a custom kernel that is not part of standard PyTorch or
TensorRT cannot support it natively.

For these cases, it is possible to use a TensorRT plugin to replace the operator **inside** the TensorRT engine, thereby avoiding
the performance and resource overhead from a graph break.

Previously this involved a complex process in not only building a performant kernel but setting it up to run in TensorRT (see: `Using Custom Kernels within TensorRT Engines with Torch-TensorRT <https://pytorch.org/TensorRT/tutorials/_rendered_examples/dynamo/custom_kernel_plugins.html>`_).
With TensorRT 10.7, there is a new Python native plugin system which greatly streamlines this process. This
plugin system also allows Torch-TensorRT to automatically generate the necessary conversion code to convert the
operation in PyTorch to TensorRT.
"""

# %%
# Writing Custom Operators in PyTorch
# -----------------------------------------
#
#  Pervious tutorials already cover creating custom operators in PyTorch which later get used with Torch-TensorRT.
# Here we define a simple elementwise multiplication operator in Triton. This operator is then registered as a custom op in PyTorch.
# with its host launch code as well as a "meta-kernel", A meta-kernel is a function that describes the shape and data type
# transformations that the operator will perform. This meta-kernel is used by Dynamo and Torch-TensorRT, so it
# is necessary to define.
#

from typing import Tuple, Union

import tensorrt as trt
import tensorrt.plugin as trtp
import torch
import torch_tensorrt
import triton
import triton.language as tl


@triton.jit
def rms_norm_kernel(
    x_ptr,
    w_ptr,
    n,
    x_stride,
    o_stride,
    o_ptr,
    EPS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
) -> None:
    i = tl.program_id(axis=0).to(tl.int64)

    x_row = x_ptr + i * x_stride
    o_row = o_ptr + i * o_stride

    # Find the root mean square for the given row.
    square_sum = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, n, BLOCK_SIZE):
        offsets = off + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n

        x = tl.load(x_row + offsets, mask=mask, other=0.0).to(tl.float32)

        square_sum += x * x

    # Compute the norm.
    rms = tl.rsqrt(tl.sum(square_sum) / n + EPS)

    # x[i] = r[i] + x[i] / rms * weight[i]
    for off in range(0, n, BLOCK_SIZE):
        offsets = off + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n

        x = tl.load(x_row + offsets, mask=mask).to(tl.float32)
        w = tl.load(w_ptr + offsets, mask=mask).to(tl.float32)

        # Multiply x with RMS on float32, but cast to the narrower type before
        # multiplying with the weights to replicate the HF behaviour precisely.
        result = w * (x * rms)

        tl.store(o_row + offsets, result, mask=mask)


@torch.library.custom_op("flashinfer::rmsnorm", mutates_args=())  # type: ignore[misc]
def flashinfer_rmsnorm(
    input: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    # Ensure the tensors are on the GPU
    assert input.is_cuda

    # Create output tensor
    output = torch.empty_like(input)

    # Define block size
    BLOCK_SIZE = 64

    b, n = input.shape

    grid = lambda meta: (triton.cdiv(input.numel(), meta["BLOCK_SIZE"]),)

    rms_norm_kernel[grid](
        input, weight, n, n, n, output, EPS=eps, BLOCK_SIZE=BLOCK_SIZE
    )

    return output


@trtp.register("flashinfer::rmsnorm")
def add_plugin_desc(
    input: trtp.TensorDesc, weight: trtp.TensorDesc, eps: float
) -> Tuple[trtp.TensorDesc]:
    return input.like()


@trtp.aot_impl("flashinfer::rmsnorm")
def flashinfer_rmsnorm(
    input: trtp.TensorDesc,
    weight: trtp.TensorDesc,
    eps: float,
    outputs: Tuple[trtp.TensorDesc],
    tactic: int,
) -> Tuple[
    Union[str, bytes], Union[str, bytes], trtp.KernelLaunchParams, trtp.SymExprs
]:
    assert tactic == 0
    block_size = 64
    # breakpoint()

    type_str = "fp32" if input.dtype == trt.float32 else "fp16"

    src = triton.compiler.ASTSource(
        fn=rms_norm_kernel,
        signature={
            "x_ptr": f"*{type_str}",
            "w_ptr": f"*{type_str}",
            "n": "i32",
            "x_stride": "i32",
            "o_stride": "i32",
            "o_ptr": f"*{type_str}",
            "EPS": "constexpr",
            "BLOCK_SIZE": "constexpr",
        },
        constants={
            "EPS": eps,
            "BLOCK_SIZE": block_size,
        },
    )

    compiled_kernel = triton.compile(src)
    launch_params = trtp.KernelLaunchParams()

    inp_dims = input.shape_expr
    out_dims = outputs[0].shape_expr

    b = inp_dims[0]
    n = inp_dims[1]
    # breakpoint()

    # grid dims
    launch_params.grid_x = trtp.cdiv(out_dims.numel(), block_size)
    # block dims
    launch_params.block_x = compiled_kernel.metadata.num_warps * 32
    # shared memory
    launch_params.shared_mem = compiled_kernel.metadata.shared

    extra_args = trtp.SymIntExprs(3)
    extra_args[0] = trtp.SymInt32(n)
    extra_args[1] = trtp.SymInt32(n)
    extra_args[2] = trtp.SymInt32(n)

    return (
        compiled_kernel.metadata.name,
        compiled_kernel.asm["ptx"],
        launch_params,
        extra_args,
    )


# %%
# The meta kernel for an elementwise operation is just the shape and dtype of one of the inputs since we will not change the shape
# in the course of the operation.


@torch.library.register_fake("flashinfer::rmsnorm")
def _(input: torch.Tensor, weight: torch.Tensor, b: float = 1e-6) -> torch.Tensor:
    return input


# %%
# Here we use automatic plugin creation feature in Torch-TensorRT which enables plugin registration using
# TensorRT QDP APIs
# torch_tensorrt.dynamo.conversion.plugins.generate_plugin(
#     "flashinfer::rmsnorm"
# )


# # %%
# # Generating the Converter
# # -------------------------------------------------------------------
# # Given that we have defined the custom operator in PyTorch and TensorRT, we can now generate the converter for the operation.
# # As long as the namespace and names match, the following function will automatically generate the converter for the operation.


torch_tensorrt.dynamo.conversion.plugins.generate_plugin_converter(
    "flashinfer::rmsnorm",
    supports_dynamic_shapes=False,
    requires_output_allocator=False,
    aot=True,
)


# # %%
# # Above two commands can be replaced with the following single one line:
# torch_tensorrt.dynamo.conversion.plugins.custom_op("torchtrt_ex::elementwise_scale_mul", supports_dynamic_shapes=True)


# %%
# Using our converter with a model
# -------------------------------------------------------------------
#
# Now we can use our custom operator in a model and compile it with Torch-TensorRT.
# We can see that the custom operator is used as one of the operations in the forward pass of the model.
# The process of compiling the model at this point is identical to standard Torch-TensorRT usage.
class MyModel(torch.nn.Module):  # type: ignore[misc]
    def __init__(self):
        super().__init__()

    def forward(self, input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        # z = torch.add(x, y)
        res = torch.ops.flashinfer.rmsnorm.default(input, weight)

        return res


# input_tensor = torch.randn(10, 20, device="cuda", dtype=torch.float16)  # 10 samples, 20 features

# # Weight tensor (usually learnable parameters)
# weight_tensor = torch.ones(20, device = "cuda", dtype=torch.float16)  # Scaling factor for the features

# # Small epsilon for numerical stability
# eps = 1e-5

# Apply RMS Normalization using flashinfer
# output_tensor = flashinfer.norm.rmsnorm(input_tensor, weight_tensor, eps)

# print(output_tensor)


my_model = MyModel().to("cuda")
m = torch.randn((64, 64), device="cuda", dtype=torch.float16)
n = torch.randn((64,), device="cuda", dtype=torch.float16)


with torch_tensorrt.logging.info():
    model_trt = torch_tensorrt.compile(
        my_model,
        inputs=[m, n],
        debug=True,
        min_block_size=1,
        enabled_precisions={torch.float16},
    )
    res = model_trt(m, n)

    print(res)
    print(my_model(m, n))
    # for i in range(300):
    #     res = model_trt(m, n)
    #     assert torch.allclose(res, my_model(m, n))


print("Ran with custom plugin!")
