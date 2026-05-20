"""
.. _constant_duplication_example:

Inspecting ``constant_duplication`` with the TensorRT engine inspector
======================================================================

This example demonstrates the ``constant_duplication`` lowering pass and shows
how to check what TensorRT actually does with the duplicated constants by
dumping the per-layer engine info via the :class:`Debugger` context.

The pass clones constant subgraphs that have multiple users so subsequent
constant folding can fold each clone into its dedicated consumer, rather than
leaving a single shared constant feeding several ops. The motivating pattern
shows up in LLMs like Llama: a weight tensor is reused in multiple matmuls
with intermediate transposes/reshapes between the weight and its consumers.

The tradeoff in the lowered Python module is straightforward — each consumer
gets its own copy of the constant. Whether that translates to a TensorRT
engine difference depends on the engine inspector: if TensorRT can already
absorb the shared constant into per-consumer kernels (typical for matmul), the
engines come out identical; if not, duplication forces TensorRT to materialize
one private constant per consumer.
"""

import copy
import json
import os
import shutil
import tempfile

import torch
import torch.nn as nn
import torch_tensorrt
from torch_tensorrt.dynamo import Debugger
from torch_tensorrt.dynamo._settings import CompilationSettings
from torch_tensorrt.dynamo.lowering import post_lowering

# %% Model
#
# A small stand-in for the Llama tied-weight / shared-projection pattern. The
# intermediate ``w_t = self.weight.t().contiguous()`` is a *shared constant*:
# both matmuls consume the same FX node. This is the case ``constant_duplication``
# is designed for — without the flag, the standard folder leaves a single
# ``_frozen_param`` feeding both matmuls; with the flag, each matmul gets a
# private clone.


class SharedTransposedWeight(nn.Module):
    def __init__(self, vocab: int = 32000, dim: int = 4096):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(vocab, dim, dtype=torch.float16) * 0.02)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        w_t = self.weight.t().contiguous()  # shared intermediate constant
        return q @ w_t + k @ w_t


VOCAB, DIM, BATCH = 32000, 4096, 4
model = SharedTransposedWeight(VOCAB, DIM).cuda().half().eval()
inputs = (
    torch.randn(BATCH, DIM, device="cuda", dtype=torch.float16),
    torch.randn(BATCH, DIM, device="cuda", dtype=torch.float16),
)
exported = torch.export.export(model, inputs)


# %% FX graph and lowered parameter bytes
#
# Run the lowering passes manually (without engine build) with the flag off
# and on, so we can see exactly what the pass does to the graph.


def lowered_gm(flag: bool) -> torch.fx.GraphModule:
    gm = torch.export.export(model, inputs).module()
    return post_lowering(gm, CompilationSettings(constant_duplication=flag))


def print_graph(label: str, gm: torch.fx.GraphModule) -> None:
    print(f"\n--- {label} ---")
    for node in gm.graph.nodes:
        if node.op == "call_module":
            continue
        print(node.format_node())


def param_bytes(gm: torch.fx.GraphModule) -> int:
    return sum(p.numel() * p.element_size() for p in gm.parameters())


gm_off = lowered_gm(False)
gm_on = lowered_gm(True)
print_graph("constant_duplication = False", gm_off)
print_graph("constant_duplication = True", gm_on)
print(
    f"\nLowered GraphModule parameter bytes:"
    f"\n  off: {param_bytes(gm_off) / 1e6:>8.2f} MB"
    f"\n  on : {param_bytes(gm_on) / 1e6:>8.2f} MB"
)


# %% Compile and inspect the TensorRT engine
#
# Wrap each compile in :class:`torch_tensorrt.dynamo.Debugger` with
# ``save_layer_info=True``. The debugger raises TRT's profiling verbosity to
# ``DETAILED`` and writes the per-layer info to
# ``<logging_dir>/engine_layer_info.json`` after the engine has been built.
# We can then compare exactly what TensorRT did with each version.


def engine_size(mod: torch.nn.Module) -> int:
    return sum(
        len(getattr(sub, "serialized_engine", b"") or b"") for sub in mod.modules()
    )


def compile_and_inspect(label: str, *, constant_duplication: bool) -> None:
    workdir = tempfile.mkdtemp(prefix="trt_const_dup_")
    try:
        with Debugger(
            log_level="warning",
            logging_dir=workdir,
            save_layer_info=True,
            engine_builder_monitor=False,
        ):
            mod = torch_tensorrt.dynamo.compile(
                copy.deepcopy(exported),
                inputs,
                min_block_size=1,
                use_python_runtime=True,
                constant_duplication=constant_duplication,
            )
            # The layer info is written on first forward.
            _ = mod(*inputs)

        # Latency
        for _ in range(20):
            _ = mod(*inputs)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        iters = 500
        start.record()
        for _ in range(iters):
            out = mod(*inputs)
        end.record()
        torch.cuda.synchronize()
        us_per_iter = start.elapsed_time(end) / iters * 1000
        torch.testing.assert_close(out, model(*inputs), rtol=5e-2, atol=5e-2)

        # Engine layer info dumped by the debugger
        info_path = os.path.join(workdir, "engine_layer_info.json")
        with open(info_path) as f:
            data = json.load(f)
        layers = data.get("Layers", [])

        print(f"\n=== {label} ===")
        print(
            f"latency : {us_per_iter:7.1f} us/iter,  engine: "
            f"{engine_size(mod) / 1e6:.2f} MB,  {len(layers)} layers"
        )
        for L in layers:
            inputs_in = [i.get("Name") for i in L.get("Inputs", [])]
            outputs_out = [o.get("Name") for o in L.get("Outputs", [])]
            print(
                f"  {L.get('LayerType', '?'):8s} "
                f"in={inputs_in}  out={outputs_out}\n"
                f"           tactic={L.get('TacticName', '?')}"
            )
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


compile_and_inspect("constant_duplication=False", constant_duplication=False)
compile_and_inspect("constant_duplication=True ", constant_duplication=True)


# %% Reading the numbers
#
# Typical output on this fixture (8000 x 1024 fp16 weight, two matmul
# consumers — scale down ``VOCAB``/``DIM`` to fit your GPU):
#
# .. code-block:: text
#
#     Lowered GraphModule parameter bytes:
#       off:    524.29 MB
#       on :    786.43 MB
#
#     === constant_duplication=False ===
#     latency : 2130.4 us/iter,  engine: 262.16 MB,  2 layers
#       gemm     in=['k']  out=['output0']
#                tactic=sm80_xmma_gemm_f16f16_f16f32_f32_tn_n_tilesize64x96x32_...
#       gemm     in=['q']  out=['output0']
#                tactic=sm80_xmma_gemm_f16f16_f16f32_f32_tn_n_tilesize64x96x32_...
#
#     === constant_duplication=True  ===
#     latency : 2044.7 us/iter,  engine: 262.16 MB,  2 layers
#       gemm     in=['k']  out=['output0']
#                tactic=sm80_xmma_gemm_f16f16_f16f32_f32_tn_n_tilesize64x96x32_...
#       gemm     in=['q']  out=['output0']
#                tactic=sm80_xmma_gemm_f16f16_f16f32_f32_tn_n_tilesize64x96x32_...
#
# Observations:
#
# * **FX graph**: the pass clearly replaces the single shared ``_frozen_param``
#   with two private ``_frozen_param`` / ``_frozen_param_dup0`` get_attrs.
# * **Lowered GraphModule parameter bytes** grow ~1.5x with the flag on —
#   each cloned ``get_attr`` is backed by its own parameter copy. This is the
#   "model size" cost, and it is real before engine build and in any artifact
#   that serializes the GraphModule.
# * **TensorRT engine layers**: for a shared-constant-into-matmul pattern,
#   TensorRT already absorbs the constant into each gemm kernel — both
#   versions produce the *same* 2-gemm engine, the *same* tactic per gemm,
#   and the *same* engine bytes. The "size" the user paid for at the
#   GraphModule level was reclaimed by TRT's constant deduplication.
#
# When does duplication actually change the TRT engine? When TensorRT can't
# fold the shared constant into a per-consumer kernel — for example when the
# constant feeds an op that doesn't admit weight-absorption (some custom
# plugins, certain reduction patterns), or when downstream quantization/refit
# needs each consumer to own a private constant.  In those cases the
# ``engine_layer_info.json`` dump will show extra ``Constant`` layers and a
# different per-gemm tactic between the off and on configurations. For the
# vanilla shared-matmul-weight pattern shown here, leaving the flag off (the
# default) gives the smallest lowered module with no loss of engine quality.
