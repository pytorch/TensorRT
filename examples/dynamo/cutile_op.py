"""
.. _cutile_op:

Register a cuTile kernel as an AOT QDP plugin via ``torch_tensorrt.kernels.cutile_op``
=======================================================================================

``cutile_op`` is the cuTile counterpart of :ref:`cuda_kernel_op`, and the
declarative form of the hand-written AOT plugin in :ref:`aot_plugin`: it takes
a ``@ct.kernel`` program, compiles it ahead of time, and registers the PyTorch
custom op, the TensorRT plugin descriptor, the AOT impl (with the compiled PTX
embedded in the engine), and the Torch-TensorRT converter — in one call.

The piece ``cutile_op`` exists to handle is the calling convention. cuTile
expands every array parameter into ``(ptr, extents..., strides...)``, grouped
per array in declaration order. TensorRT's AOT plugin launcher instead passes
``(input_ptrs..., extra_args..., output_ptrs...)``. Those two orders do not
agree, and a mismatch does not fail — the kernel reads whatever landed in each
slot and returns plausible-looking garbage. ``cutile_op`` permutes the compiled
PTX's parameter list and supplies the matching extents and strides as AOT extra
arguments, so the kernel binds what it expects.

This example requires ``cuda-tile>=1.3,<2`` and ``tileiras`` from a system CUDA
Toolkit compatible with that package, on ``PATH`` or under ``CUDA_HOME``.
"""

import argparse

import cuda.tile as ct
import tensorrt.plugin as trtp
import torch

import torch_tensorrt
import torch_tensorrt.kernels as ttk

parser = argparse.ArgumentParser()
parser.add_argument("--min_block_size", type=int, default=1)
ARGS, _ = parser.parse_known_args()

# %%
# Step 1: Define the cuTile kernel (pure cuTile, nothing TensorRT-specific)
# -------------------------------------------------------------------------
#
# Array parameters come first — inputs then outputs — followed by the
# ``ct.Constant`` parameters, which are baked into the compiled symbol.

TILE_SIZE = 128


@ct.kernel
def add_one_kernel(x, out, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    tile = ct.load(x, index=(pid,), shape=(tile_size,))
    ct.store(out, index=(pid,), tile=tile + 1.0)


# %%
# Step 2: Describe the op and register it with a single ``cutile_op`` call
# ------------------------------------------------------------------------
#
# * ``signature`` — the array parameters and their element types, in
#   declaration order. ``ndim`` defaults to 1, matching a kernel written
#   against a flattened view, so a rank-1 array's extent is the tensor's
#   element count and the op accepts any input shape.
# * ``constants`` — the ``ct.Constant`` values to compile for.
# * ``grid`` — the launch grid in tiles, computed from ``trtp.TensorDesc``
#   inputs. Using ``.shape_expr`` keeps it symbolic so one engine covers a
#   range of shapes.
# * ``meta_fn`` — shape/dtype inference for FakeTensors (the Torch schema is
#   inferred from its type hints).
# * ``eager_fn`` — optional; lets ``torch.ops.cutile_example.add_one`` also run outside
#   TensorRT.


def add_one_meta(x: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(x)


def add_one_eager(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    flat_x = x.contiguous().reshape(-1)
    flat_out = out.reshape(-1)
    ct.launch(
        torch.cuda.current_stream().cuda_stream,
        (ct.cdiv(flat_x.numel(), TILE_SIZE), 1, 1),
        add_one_kernel,
        (flat_x, flat_out, TILE_SIZE),
    )
    return out


ttk.cutile_op(
    "cutile_example::add_one",
    kernel=add_one_kernel,
    signature={"x": "fp32", "out": "fp32"},
    meta_fn=add_one_meta,
    grid=lambda inputs, outputs: (trtp.cdiv(inputs[0].shape_expr.numel(), TILE_SIZE),),
    constants={"tile_size": TILE_SIZE},
    eager_fn=add_one_eager,
)


# %%
# Step 3: Use it — the op lowers to the AOT QDP plugin inside the engine
# ----------------------------------------------------------------------


class AddOne(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ops.cutile_example.add_one(x)


if __name__ == "__main__":
    x = torch.randn(4, 256, device="cuda", dtype=torch.float32)
    model = AddOne().cuda().eval()

    ref = x + 1
    assert torch.allclose(model(x), ref), "eager path mismatch"
    print("eager path OK")

    trt_model = torch_tensorrt.compile(
        model,
        inputs=[x],
        min_block_size=ARGS.min_block_size,
    )
    print("engine compiled with the AOT QDP plugin")

    trt_out = trt_model(x)
    assert torch.allclose(trt_out, ref, atol=1e-5), "cutile_op TRT output mismatch"
    print("cutile_op AOT QDP plugin ran correctly under Torch-TensorRT")
