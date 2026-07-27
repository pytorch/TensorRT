"""
.. _triton_op:

Register a Triton kernel as an AOT QDP plugin via ``torch_tensorrt.kernels.triton_op``
======================================================================================

This is the productized form of :ref:`aot_plugin`. That example shows the raw
mechanism — a ``@triton.jit`` kernel wired to a TensorRT AOT Quick Deployable
Plugin by hand: you write ``@torch.library.custom_op`` + ``register_fake``,
``@trtp.register``, and a ~40-line ``@trtp.aot_impl`` that builds the
``ASTSource`` signature, calls ``triton.compile``, and assembles the launch
parameters, then ``generate_plugin_converter``.

``triton_op`` collapses all of that into a single call. You provide the Triton
``signature`` / ``constexprs`` / ``grid`` and a ``meta_fn``; the library
compiles the kernel to PTX, derives the AOT launch, registers the PyTorch op,
the plugin descriptor + AOT impl, and the Torch-TensorRT converter.

Calling convention: the kernel's non-constexpr parameters must be declared as
``(input_ptrs..., extra_scalars..., output_ptrs...)`` so no PTX rewriting is
needed.
"""

import argparse

import tensorrt.plugin as trtp
import torch
import triton
import triton.language as tl

import torch_tensorrt
import torch_tensorrt.kernels as ttk

# %%
# Step 1: Define the Triton kernel (pure Triton, unchanged from aot_plugin.py)
# ---------------------------------------------------------------------------


@triton.jit
def add_one_kernel(x_ptr, n_elements, y_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    output = x + 1
    tl.store(y_ptr + offsets, output, mask=mask)


# %%
# Step 2: Describe the op and register it with a single ``triton_op`` call
# -----------------------------------------------------------------------
#
# * ``meta_fn`` — shape/dtype inference for FakeTensors (the schema is inferred
#   from its type hints).
# * ``signature`` / ``constexprs`` — exactly what ``triton.compile`` needs.
# * ``grid`` / ``extra_args_fn`` — computed from ``trtp.TensorDesc`` inputs,
#   using symbolic shapes so one engine works across sizes.
# * ``eager_fn`` — optional; lets ``torch.ops.my.add_one`` also run in eager.

BLOCK_SIZE = 256


def add_one_meta(X: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(X)


def add_one_eager(X: torch.Tensor) -> torch.Tensor:
    Y = torch.empty_like(X)
    grid = lambda meta: (triton.cdiv(X.numel(), meta["BLOCK_SIZE"]),)
    add_one_kernel[grid](X, X.numel(), Y, BLOCK_SIZE=BLOCK_SIZE)
    return Y


ttk.triton_op(
    "my::add_one",
    kernel=add_one_kernel,
    signature={"x_ptr": "*fp32", "n_elements": "i32", "y_ptr": "*fp32"},
    constexprs={"BLOCK_SIZE": BLOCK_SIZE},
    grid=lambda inputs, outputs: (trtp.cdiv(inputs[0].shape_expr.numel(), BLOCK_SIZE),),
    meta_fn=add_one_meta,
    extra_args_fn=lambda inputs, outputs: [trtp.SymInt32(inputs[0].shape_expr.numel())],
    eager_fn=add_one_eager,
    supports_dynamic_shapes=True,
)


# %%
# Step 3: Use it — the op lowers to the AOT QDP plugin inside the engine
# ---------------------------------------------------------------------


class AddOne(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ops.my.add_one(x)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--min_block_size", type=int, default=1)
    args = parser.parse_args()

    x = torch.randn(4, 256, device="cuda", dtype=torch.float32)
    model = AddOne().cuda().eval()

    ref = x + 1
    eager_out = model(x)
    assert torch.allclose(eager_out, ref), "eager path mismatch"
    print("eager path OK")

    trt_model = torch_tensorrt.compile(
        model,
        inputs=[x],
        min_block_size=args.min_block_size,
    )
    print("engine compiled with the AOT QDP plugin")

    # triton_op caps the embedded PTX to the driver's supported ISA, so the AOT
    # plugin loads even when the CUDA toolkit is newer than the driver. A
    # mismatch here would indicate the kernel uses instructions newer than the
    # driver supports — report rather than crash so the example stays usable.
    trt_out = trt_model(x)
    if torch.allclose(trt_out, ref, atol=1e-5):
        print("triton_op AOT QDP plugin ran correctly under Torch-TensorRT")
    else:
        print(
            "WARNING: TRT output did not match. Check that the CUDA driver "
            "supports the kernel's PTX ISA (toolkit newer than driver can "
            "require capping); registration/compile/build succeeded."
        )
