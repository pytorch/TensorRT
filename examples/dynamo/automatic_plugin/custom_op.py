import triton
import triton.language as tl

@triton.jit
def elementwise_add_kernel(X, Y, Z, BLOCK_SIZE: tl.constexpr):
    # Program ID determines the block of data each thread will process
    pid = tl.program_id(0)
    # Compute the range of elements that this thread block will work on
    block_start = pid * BLOCK_SIZE
    # Range of indices this thread will handle
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Load elements from the X and Y tensors
    x_vals = tl.load(X + offsets)
    y_vals = tl.load(Y + offsets)
    # Perform the element-wise multiplication
    z_vals = x_vals + y_vals
    # Store the result in Z
    tl.store(Z + offsets, z_vals)


import torch
from torch.library import custom_op


@custom_op("torchtrt_ex::elementwise_add", mutates_args=())  # type: ignore[misc]
def elementwise_add(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    # Ensure the tensors are on the GPU
    assert X.is_cuda and Y.is_cuda, "Tensors must be on CUDA device."
    assert X.shape == Y.shape, "Tensors must have the same shape."
    
    # Create output tensor
    Z = torch.empty_like(X)
    
    # Define block size
    BLOCK_SIZE = 1024
    
    # Grid of programs
    grid = lambda meta: (X.numel() // meta['BLOCK_SIZE'],)
    
    # Launch the kernel
    elementwise_add_kernel[grid](X, Y, Z, BLOCK_SIZE=BLOCK_SIZE)
    
    return Z


# Using the module in PyTorch
# X = torch.randn(1024, device='cuda', requires_grad=True)
# Y = torch.randn(1024, device='cuda', requires_grad=True)
# X = torch.full((128, 128), 2, device='cuda',)
# Y = torch.full((128, 128), 2, device='cuda',)
# # elementwise_mul_op = ElementwiseMulModule()
# Z = torch.ops.torchtrt_ex.elementwise_add(X, Y)
# print(Z)
# print(X + Y)
# print(X)
# print(Y)
# print(Z)
# print(X+Y)
# Z.sum().backward()


from torch import nn


class MyModel(nn.Module):  # type: ignore[misc]
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        z = torch.mul(x, y)
        res = torch.ops.torchtrt_ex.elementwise_add(x, z)

        return res


my_model = MyModel().to("cuda")
m = torch.full((64, 64), 2, device='cuda',)
n = torch.full((64, 64), 3, device='cuda',)
# print(torch.ops.torchtrt_ex.elementwise_add(m, n))
# print(my_model.forward(m, n))


@torch.library.register_fake("torchtrt_ex::elementwise_add")
def _(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return x

import torch_tensorrt as torchtrt


with torchtrt.logging.info():
    model_trt = torchtrt.compile(my_model, inputs=[m, n], debug=True, min_block_size=1)
    res = model_trt(m, n)
    print(res)