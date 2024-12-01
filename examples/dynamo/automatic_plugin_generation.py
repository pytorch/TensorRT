import triton
import triton.language as tl

from typing import Tuple
import torch_tensorrt

@triton.jit
def elementwise_mul_kernel(X, Y, Z, BLOCK_SIZE: tl.constexpr):
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
    z_vals = x_vals * y_vals
    # Store the result in Z
    tl.store(Z + offsets, z_vals)


import torch
from torch.library import custom_op

#@torch_tensorrt.dynamo.conversion.plugin.custom_op("torchtrt_ex::elementwise_mul", mutates_args=())
@torch.library.custom_op("torchtrt_ex::elementwise_mul", mutates_args=())  # type: ignore[misc]
def elementwise_mul(X: torch.Tensor, Y: torch.Tensor, b: float=.2, a: int=2) -> torch.Tensor:
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
    elementwise_mul_kernel[grid](X, Y, Z, BLOCK_SIZE=BLOCK_SIZE)

    return Z

from torch import nn


class MyModel(nn.Module):  # type: ignore[misc]
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        z = torch.add(x, y)
        res = torch.ops.torchtrt_ex.elementwise_mul.default(x, z, a=1)

        return res


my_model = MyModel().to("cuda")
m = torch.full((64, 64), 2, device='cuda', dtype=torch.float)
n = torch.full((64, 64), 3, device='cuda', dtype=torch.float)

def mksym(shape_env, value, source, dynamic_dim):
    return shape_env.create_symintnode(
        shape_env.create_symbol(
            value,
            source=source,
            dynamic_dim=dynamic_dim,
        ),
        hint=value,
        source=source,
    )

@torch.library.register_fake("torchtrt_ex::elementwise_mul")
def _(x: torch.Tensor, y: torch.Tensor, b: float=.2, a: int=2) -> torch.Tensor:
    return x

import tensorrt_bindings.plugin as trtp
from torch._dynamo.source import LocalSource
from torch.fx.experimental.symbolic_shapes import DimDynamic, ShapeEnv
from sympy import lambdify

@trtp.register("torchtrt_ex::elementwise_mul")
def _(x: trtp.TensorDesc, y: trtp.TensorDesc, b: float, a: int) -> Tuple[trtp.TensorDesc]:
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch.fx.experimental.symbolic_shapes import ShapeEnv
    from sympy import lambdify
    shape_env = ShapeEnv()
    fake_mode = FakeTensorMode(shape_env=shape_env)
    sample_x = {f"x{i}": 5 for i in range(x.ndim)}
    sample_y = {f"y{i}": 5 for i in range(y.ndim)}
    syms_x = [mksym(shape_env, v, LocalSource(k), DimDynamic.DYNAMIC) for k,v in sample_x.items()]
    syms_y = [mksym(shape_env, v, LocalSource(k), DimDynamic.DYNAMIC) for k,v in sample_y.items()]
    with FakeTensorMode() as fake_mode:
        fake_x = torch.randn(syms_x)
        fake_y = torch.randn(syms_y)
        z = torch.ops.torchtrt_ex.elementwise_mul(fake_x, fake_y, b, a)

    shape_calc_fns = [None] * x.ndim
    for i in range(x.ndim):
        shape_calc_fns[i] = lambdify((syms_x[i].node.expr, syms_y[i].node.expr), z.shape[i].node.expr, "math")

    out_desc = x.like()
    for i in range(out_desc.ndim):
        out_desc.shape_expr[i] = shape_calc_fns[i](x.shape_expr[i], y.shape_expr[i])
    return out_desc


@trtp.impl("torchtrt_ex::elementwise_mul")
def _(x: trtp.Tensor, y: trtp.Tensor, b: float, a: int, outputs: Tuple[trtp.Tensor], stream: int):
    # This should be based on Torch schema
    in_tensors = [
        torch.as_tensor(i, device="cuda") for i in (x, y)
    ]  # What is the right device??
    dest_tensors = [torch.as_tensor(o, device="cuda") for o in outputs]

    stream = torch.cuda.ExternalStream(stream)
    with torch.cuda.stream(stream):
        out_tensors = torch.ops.torchtrt_ex.elementwise_mul(*in_tensors, b, a)
        [d.copy_(o) for (d, o) in zip(dest_tensors, out_tensors)]

# @trtp.impl("torchtrt_ex::elementwise_mul")
# def _(x: trtp.Tensor, y: trtp.Tensor, b: float, a: int, outputs: Tuple[trtp.Tensor], stream: int):
#     # Define block size
#     BLOCK_SIZE = 1024

#     # Grid of programs
#     grid = lambda meta: (x.numel() // meta['BLOCK_SIZE'],)

#     x_t = torch.as_tensor(x, device="cuda")
#     y_t = torch.as_tensor(y, device="cuda")
#     z_t = torch.as_tensor(outputs[0], device="cuda")
#     # Launch the kernel
#     elementwise_mul_kernel[grid](x_t, y_t, z_t, BLOCK_SIZE=BLOCK_SIZE)

_ = torch_tensorrt.dynamo.conversion.plugins.generate_plugin_converter("torchtrt_ex::elementwise_mul", supports_dynamic_shapes=True)

from torch_tensorrt.dynamo.conversion._ConverterRegistry import DYNAMO_CONVERTERS

import torch_tensorrt as torchtrt
import tensorrt as trt
with torchtrt.logging.errors():
    model_trt = torchtrt.compile(my_model, inputs=[m, n], debug=True, min_block_size=1)
    for i in range(300):
        res = model_trt(m, n)
        print(res)
        assert torch.allclose(res, my_model(m,n))
