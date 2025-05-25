import argparse
from typing import Tuple, Union

import tensorrt as trt
import tensorrt.plugin as trtp
import torch
import torch_tensorrt
import triton
import triton.language as tl

trt_logger = trt.Logger(trt.Logger.VERBOSE)


@triton.jit
def add_one_kernel(x_ptr, n_elements, y_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    output = x + 1
    tl.store(y_ptr + offsets, output, mask=mask)


@torch.library.custom_op("my::add_one", mutates_args=())  # type: ignore[misc]
def add_one(X: torch.Tensor) -> torch.Tensor:
    # Ensure the tensors are on the GPU
    assert X.is_cuda

    # Create output tensor
    Y = torch.empty_like(X)

    # Define block size
    BLOCK_SIZE = 256

    # Grid of programs
    grid = lambda meta: (triton.cdiv(X.numel(), meta["BLOCK_SIZE"]),)

    # Launch the kernel
    add_one_kernel[grid](X, X.numel(), Y, BLOCK_SIZE=BLOCK_SIZE)

    return Y


@torch.library.register_fake("my::add_one")
def _(X: torch.Tensor) -> torch.Tensor:
    return X


@trtp.register("my::add_one")
def add_plugin_desc(X: trtp.TensorDesc) -> Tuple[trtp.TensorDesc]:
    return X.like()


@trtp.aot_impl("my::add_one")
def add_plugin_aot_impl(
    X: trtp.TensorDesc, outputs: Tuple[trtp.TensorDesc], tactic: int
) -> Tuple[
    Union[str, bytes], Union[str, bytes], trtp.KernelLaunchParams, trtp.SymExprs
]:
    type_str = "fp32" if X.dtype == trt.float32 else "fp16"

    block_size = 256
    src = triton.compiler.ASTSource(
        fn=add_one_kernel,
        signature={
            "x_ptr": f"*{type_str}",
            "n_elements": "i32",
            "y_ptr": f"*{type_str}",
            "BLOCK_SIZE": "constexpr",
        },
        constants={
            "BLOCK_SIZE": block_size,
        },
    )

    compiled_kernel = triton.compile(src)

    N = X.shape_expr.numel()
    launch_params = trtp.KernelLaunchParams()

    # grid dims
    launch_params.grid_x = trtp.cdiv(N, block_size)
    # block dims
    launch_params.block_x = compiled_kernel.metadata.num_warps * 32
    # shared memory
    launch_params.shared_mem = compiled_kernel.metadata.shared

    extra_args = trtp.SymIntExprs(1)
    extra_args[0] = trtp.SymInt32(N)

    return (
        compiled_kernel.metadata.name,
        compiled_kernel.asm["ptx"],
        launch_params,
        extra_args,
    )


torch_tensorrt.dynamo.conversion.plugins.generate_plugin_converter(
    "my::add_one",
    supports_dynamic_shapes=False,
    requires_output_allocator=False,
    use_aot_if_available=True,
)


class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        res = torch.ops.my.add_one.default(X)

        return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--aot", action="store_true", help="Try to use AOT compilation", default=False
    )
    args = parser.parse_args()

    my_model = MyModel().to("cuda")
    m = torch.full((64, 64), 2, device="cuda", dtype=torch.float)

    assert my_model(X=m)[0][0] == 3.0

    with torch_tensorrt.logging.debug():
        trt_inputs = [m]
        model_trt = torch_tensorrt.compile(
            my_model,
            inputs=trt_inputs,
            debug=True,
            min_block_size=1,
        )
        print("Model compiled successfully!")
        print("Running inference with compiled model...")
        for i in range(10):
            res = model_trt(m)
            assert torch.allclose(res, my_model(m)), "Results do not match!"

    print("Inference successful!")
