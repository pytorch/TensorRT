# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Qwen3 decode with whole decoder blocks as fused Triton kernels inside TensorRT.

:func:`patch_qwen3` replaces the forward of a loaded Hugging Face
``Qwen3ForCausalLM`` with an equivalent single-token decode forward in which
the chosen blocks run inside ``torch_tensorrt.region.fused_kernel`` scopes, so
:func:`compile_decode` (``torch_tensorrt.compile(..., ir="dynamo",
strict=True)``) compiles each block into the TensorRT engine as one Triton
kernel plugin (see ``qwen3_kernels.py``). Eager execution and the region
references compute what Hugging Face computes; the kernels accumulate in FP32
and round each output once.

Blocks (any subset):

* ``final``:    [residual add +] final RMSNorm + lm_head GEMV     -> logits
* ``attn_in``:  [residual add +] input RMSNorm + q/k/v GEMVs      -> q, k, v [, h]
* ``qk_rope``:  q/k per-head RMSNorm + rotary embedding (+ the softmax scale
  folded into q with ``FOLD_SCALE``)                              -> q, k
* ``attn_out``: o_proj GEMV + residual add                        -> h
* ``mlp_in``:   [residual add +] post-attention RMSNorm + gate/up GEMVs +
  SiLU(gate) * up                                                 -> act [, h]
* ``mlp_out``:  down GEMV + residual add                          -> h

A residual add belongs to the producer when that producer is a region
(``attn_out``/``mlp_out``); otherwise it is deferred into the next block (the
bracketed adds above), or left to TensorRT when that block is not a region.
The KV cache write and attention stay in TensorRT, where static_cache_v3
rewrites SDPA and adds the in-place cache update.

Region outputs follow production order in the region body, and every kernel
writes its outputs in that order (see ``qk_rope``).

The plugins are TensorRT 11.4 QDP AOT plugins, which reload their module on
every enqueue outside CUDA graphs: run the compiled decode step under CUDA
graphs. On an RTX 5090 (FP16, batch 1, CUDA graphs) the ``layers`` preset
decodes in about 1.02 ms/token against 1.18 ms/token without regions; the
``final`` block is neutral (TensorRT's lm_head GEMV already runs at DRAM peak).

Hybrid variants (``inductor``, one of :data:`INDUCTOR_VARIANTS`) hand
operations to Inductor's kernels; the rest of the step stays as above:

* ``attention-flex``: the kernels Inductor generates for FlexAttention decode
  (flex-decoding: one split-KV template kernel and two combine reductions)
  run inside the engine, each as a ``fused_kernel`` plugin; see
  :func:`inductor_flex_decoding`. No engine boundary.
* ``lm-head``: [residual add +] final RMSNorm + lm_head in one
  ``execute_in_torch(backend="inductor")`` region after the engine.
* ``attention``: each layer's decode attention in an
  ``execute_in_torch(backend="inductor")`` region, between two engines,
  running Inductor's flex-decoding. Export captures ``flex_attention`` as a
  higher-order operator, which region capture rejects inside a scope, so the
  region calls the opaque custom operator ``qwen3_regions::flex_decode``
  (:func:`flex_decode`), whose CUDA implementation runs ``flex_attention``
  compiled by Inductor (:func:`flex_decoding_function`).
* ``attention-math``: the same regions with plain attention math as their
  body (:func:`decode_attention`: two matmuls and a masked FP32 softmax),
  which Inductor compiles itself.

The attention variants read the caches inside the exported program, and a
region sees only tensors that exist at export, so their step takes the caches
as inputs (``static_cache_v3.cache_decode_step``) and writes them with a
functional ``index_copy``, which TensorRT lowers to its in-place cache
update; attention then reads the written caches and masks the slots after the
position. ``attention`` and ``attention-flex`` share one block mask per step
(:func:`flex_block_mask`), computed in the engine.
"""

from __future__ import annotations

import functools
import importlib
import sys
import types
from typing import Any, Iterable, Mapping, Optional

import torch
import torch.nn.functional as F

BLOCKS = ("final", "attn_in", "qk_rope", "attn_out", "mlp_in", "mlp_out")
# Operations handed to Inductor's kernels (kernels.inductor; module docstring),
# by the engine-region boundaries they add: none, one, one per attention.
INDUCTOR_VARIANTS = ("none", "attention-flex", "lm-head", "attention", "attention-math")
# Variants with execute_in_torch(backend="inductor") regions.
REGION_VARIANTS = ("lm-head", "attention", "attention-math")
# Variants whose attention reads the caches inside the exported program.
_EXPLICIT_CACHE_VARIANTS = ("attention", "attention-math", "attention-flex")
# Variants whose attention is flex-decoding over the step's block mask.
_FLEX_VARIANTS = ("attention", "attention-flex")

# Block sets offered to agentic optimization sessions (kernels.decode_blocks).
PRESETS: dict[str, tuple[str, ...]] = {
    "": (),
    "mlp": ("mlp_in", "mlp_out"),
    "mlp+attn": ("attn_in", "attn_out", "mlp_in", "mlp_out"),
    "layers": ("attn_in", "qk_rope", "attn_out", "mlp_in", "mlp_out"),
    "all": BLOCKS,
}

# Launch settings per block. BLOCK_N, BLOCK_K and num_warps come from an
# isolated, L2-defeated kernel sweep on an RTX 5090; EVICT and FOLD_SCALE were
# chosen from runs in the model. BLOCK_N: output rows per program (norm kernels
# keep the whole hidden row in registers); BLOCK_K: the K chunk gemv_add
# streams. EVICT=1 loads weights with an L2 evict_first policy so they do not
# evict the KV cache. FOLD_SCALE=1 multiplies q by the softmax scale in qk_rope
# and runs attention with scale 1 (no separate q-scaling kernel). BLOCK_N and
# BLOCK_K must be powers of two that divide each GEMV's rows and columns.
# EDITABLE-START
DEFAULT_LAUNCH: dict[str, dict[str, int]] = {
    "final": {"BLOCK_N": 1, "num_warps": 2, "EVICT": 1},
    "attn_in": {"BLOCK_N": 1, "num_warps": 4, "EVICT": 1},
    "qk_rope": {"num_warps": 1, "FOLD_SCALE": 1},
    "attn_out": {"BLOCK_N": 2, "BLOCK_K": 512, "num_warps": 1, "EVICT": 1},
    "mlp_in": {"BLOCK_N": 1, "num_warps": 4, "EVICT": 1},
    "mlp_out": {"BLOCK_N": 1, "BLOCK_K": 512, "num_warps": 1, "EVICT": 1},
}
# EDITABLE-END


# ----------------------------------------------------------------- factories
# fused_kernel needs module-level functions or partials (resolved outside
# tracing), not methods or closures over ``self``.
def _kernel(module: str, name: str) -> Any:
    return getattr(sys.modules[module], name)


def _grid(programs: int, inputs: Any, outputs: Any) -> tuple[int]:
    return (programs,)


def _fused(
    kernels: str,
    kernel_name: str,
    inputs: Iterable[torch.Tensor],
    *,
    programs: int,
    launch: Mapping[str, int],
    name: str,
    **constexprs: Any,
) -> Any:
    """A fused_kernel scope launching ``kernel_name`` from module ``kernels``."""
    from torch_tensorrt import region

    return region.fused_kernel(
        kernel=functools.partial(_kernel, kernels, kernel_name),
        inputs=list(inputs),
        name=name,
        constexprs=constexprs,
        grid=functools.partial(_grid, programs),
        num_warps=int(launch["num_warps"]),
    )


def _power_of_two(n: int) -> bool:
    return n > 0 and n & (n - 1) == 0


def _tile(name: str, what: str, size: int, *extents: int) -> int:
    """Check that a tile ``size`` is a power of two dividing every extent."""
    if not _power_of_two(size):
        raise ValueError(f"{name}: {what}={size} is not a power of two")
    for extent in extents:
        if extent % size:
            raise ValueError(f"{name}: {extent} is not a multiple of {what}={size}")
    return size


def _rows(
    launch: Mapping[str, int], n: int, name: str, *parts: int
) -> tuple[int, dict[str, int]]:
    """Programs and row-tiling constexprs for a GEMV over ``n`` output rows.

    ``parts`` are row ranges that programs must not straddle (q, k and v).
    """
    block_n = _tile(name, "BLOCK_N", int(launch["BLOCK_N"]), n, *parts)
    return n // block_n, {"BLOCK_N": block_n, "EVICT": int(launch.get("EVICT", 0))}


# ------------------------------------------------------- reference numerics
def _rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Hugging Face Qwen3RMSNorm."""
    dtype = x.dtype
    h = x.to(torch.float32)
    h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps)
    return weight * h.to(dtype)


def decode_attention(
    q: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    position_ids: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """One-token grouped-query attention over the cache slots up to the position.

    ``q`` is ``[1, heads, 1, head_dim]``, ``keys``/``values`` ``[1, kv_heads,
    slots, head_dim]`` caches whose slots after ``position_ids`` are ignored,
    whatever they hold. Returns ``[1, heads, 1, head_dim]``. Plain matmuls and
    a masked FP32 softmax: the body of the ``attention-math`` variant's
    Inductor regions. The scores leave the QK^T matmul in its input precision
    (FP16 for an FP16 model) before the FP32 softmax; flex-decoding keeps them
    in FP32.
    """
    batch, heads, seq, dim = q.shape
    kv_heads, slots = keys.shape[1], keys.shape[2]
    grouped = q.reshape(batch, kv_heads, heads // kv_heads * seq, dim)
    scores = torch.matmul(grouped, keys.transpose(-1, -2)).float() * scale
    visible = torch.arange(slots, device=q.device) <= position_ids.reshape(-1, 1, 1, 1)
    probs = torch.softmax(scores.masked_fill(~visible, float("-inf")), dim=-1)
    out = torch.matmul(probs.to(values.dtype), values)
    return out.reshape(batch, heads, seq, dim)


# ------------------------------------------- Inductor flex-decoding, hosted
# Inductor's flex-decoding code for a FlexAttention decode with the mask
# kv_idx <= position: a split-KV template kernel writes, per split, the row
# max M and sum L (log2 domain) and the unnormalized output ACC; two
# reductions combine the splits (global max and sum, then the output).
_LOG2E = 1.4426950408889634
# One KV block of the block mask (SPARSE_KV_BLOCK_SIZE).
FLEX_BLOCK = 128
_FLEX: dict[tuple, "FlexDecoding"] = {}


def flex_block_mask(
    position_ids: torch.Tensor, slots: int, block: int = FLEX_BLOCK
) -> tuple[torch.Tensor, ...]:
    """The block mask of ``kv_idx <= position``, and the position, for flex-decoding.

    Returns ``kv_num_blocks, kv_indices, full_kv_num_blocks, full_kv_indices,
    position`` (int32): the blocks before the position's block are full, and
    the position's block, unless it ends there, is the one partial block,
    where the kernel applies the mask. Plain integer ops, computed once per
    step and shared by every layer.
    """
    blocks = -(-slots // block)
    position = position_ids.reshape(-1).to(torch.int32)
    written = position + 1
    full = torch.div(written, block, rounding_mode="floor")
    kv_num = (written % block != 0).to(torch.int32).reshape(1, 1, 1)
    rank = torch.arange(blocks, device=position.device, dtype=torch.int32)
    kv_idx = ((rank + full) % blocks).reshape(1, 1, 1, blocks)
    full_num, full_idx = full.reshape(1, 1, 1), rank.reshape(1, 1, 1, blocks)
    return kv_num, kv_idx, full_num, full_idx, position


def _block_members(indices: torch.Tensor, count: torch.Tensor) -> torch.Tensor:
    """Which blocks the first ``count`` entries of ``indices`` list ([blocks] mask)."""
    blocks = indices.shape[-1]
    rank = torch.arange(blocks, device=indices.device)
    listed = (rank < count.reshape(1)).reshape(-1, 1)
    return ((indices.reshape(-1, 1) == rank.reshape(1, -1)) & listed).any(0)


def flex_split_reference(
    q: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    kv_num: torch.Tensor,
    kv_idx: torch.Tensor,
    full_num: torch.Tensor,
    full_idx: torch.Tensor,
    position: torch.Tensor,
    *,
    scale: float,
    splits: int,
    block: int = FLEX_BLOCK,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``M, L, ACC`` of a split-KV decode attention, as the template kernel writes them.

    ``M``/``L`` are ``[1, splits, heads, 1]`` FP32 (row max and sum of
    ``exp2`` in the log2 domain), ``ACC`` ``[1, splits, heads, 1, head_dim]``
    FP32, unnormalized. Slots are split contiguously here; the kernel assigns
    them differently, which changes the parts but not their combination.
    """
    batch, heads, _, dim = q.shape
    kv_heads, slots = keys.shape[1], keys.shape[2]
    group = heads // kv_heads
    slot = torch.arange(slots, device=q.device)
    owner = slot // block
    visible = _block_members(full_idx, full_num)[owner] | (
        _block_members(kv_idx, kv_num)[owner] & (slot <= position.reshape(1))
    )
    scores = torch.matmul(
        q.reshape(batch, kv_heads, group, dim).float(), keys.float().transpose(-1, -2)
    ) * (scale * _LOG2E)
    scores = scores.masked_fill(~visible, float("-inf"))
    tile = -(-slots // splits)
    pad = splits * tile - slots
    scores = F.pad(scores, (0, pad), value=float("-inf"))
    scores = scores.reshape(batch, kv_heads, group, splits, tile)
    # Outputs in production order: M, L, then ACC.
    m = scores.amax(-1)
    big_m = m.permute(0, 3, 1, 2).reshape(batch, splits, heads, 1).contiguous()
    shift = torch.where(m == float("-inf"), torch.zeros_like(m), m)
    p = torch.exp2(scores - shift.unsqueeze(-1))
    big_l = p.sum(-1).permute(0, 3, 1, 2).reshape(batch, splits, heads, 1).contiguous()
    v = F.pad(values.float(), (0, 0, 0, pad))
    v = v.reshape(batch, kv_heads, splits, tile, dim)
    acc = torch.einsum("bhgtn,bhtnd->bhgtd", p, v)
    big_acc = acc.permute(0, 3, 1, 2, 4).reshape(batch, splits, heads, 1, dim)
    return big_m, big_l, big_acc.contiguous()


def flex_lse_reference(
    big_m: torch.Tensor, big_l: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Global row max and sum over the splits: ``[1, heads, 1]`` each."""
    global_m = big_m.amax(1)
    shift = torch.where(
        global_m.unsqueeze(1) == float("-inf"),
        torch.zeros_like(big_m),
        big_m - global_m.unsqueeze(1),
    )
    return global_m, (big_l * torch.exp2(shift)).sum(1)


def flex_out_reference(
    big_acc: torch.Tensor,
    global_m: torch.Tensor,
    big_m: torch.Tensor,
    global_l: torch.Tensor,
    dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """The attention output ``[1, heads, 1, head_dim]`` from the splits."""
    shift = torch.where(
        global_m.unsqueeze(1) == float("-inf"),
        torch.zeros_like(big_m),
        big_m - global_m.unsqueeze(1),
    )
    out = (big_acc * torch.exp2(shift).unsqueeze(-1)).sum(1)
    denominator = torch.where(
        global_m == float("-inf"), torch.ones_like(global_l), global_l
    )
    return (out / denominator.unsqueeze(-1)).to(dtype)


class FlexDecoding:
    """Inductor's flex-decoding kernels for one attention shape, hosted as plugins.

    ``kernels`` maps ``split``/``lse``/``out`` to QDP-ABI wrapper kernels
    that call Inductor's generated kernels unchanged; ``launch`` holds each
    one's grid and the launch settings Inductor chose (``num_warps``,
    ``num_stages``, constexprs); ``sources`` the generated kernels' names and
    SHA-256 of their source.
    """

    def __init__(
        self,
        key: tuple,
        kernels: Mapping[str, Any],
        launch: Mapping[str, Mapping[str, Any]],
        splits: int,
        sources: Mapping[str, str],
    ) -> None:
        self.key = key
        self.kernels = dict(kernels)
        self.launch = {name: dict(value) for name, value in launch.items()}
        self.splits = splits
        self.sources = dict(sources)

    def describe(self) -> dict[str, Any]:
        return {
            "kernels": dict(self.sources),
            "launch": {name: dict(value) for name, value in self.launch.items()},
            "split_kv": self.splits,
        }


def _flex_kernel(key: tuple, role: str) -> Any:
    return _FLEX[key].kernels[role]


def _fixed_grid(grid: tuple[int, ...], inputs: Any, outputs: Any) -> tuple[int, ...]:
    return grid


def _flex_wrappers(
    split_fn: Any,
    lse_fn: Any,
    out_fn: Any,
    lse: Mapping[str, int],
    out: Mapping[str, int],
) -> dict[str, Any]:
    """QDP-ABI wrappers (inputs, then outputs) around Inductor's kernels.

    The plugin compiles without Triton's pointer-alignment specialization,
    which Inductor's own launch applies (16-byte ``tt.divisibility`` on every
    pointer). ``aligned`` restores it with a provably 16-byte-aligned copy of
    each pointer: the same address for the aligned tensors TensorRT binds, and
    the same machine code (pipelined async copies, 128-bit loads) as
    Inductor's launch. Inductor's kernels are called unchanged, with the
    sizes and block size of Inductor's launch (``lse``/``out``: ``XNUMEL``,
    ``RNUMEL``, ``XBLOCK``) as constants.
    """
    import triton
    import triton.language as tl

    sizes = ("XNUMEL", "RNUMEL", "XBLOCK")
    lse_x, lse_r, lse_block = (tl.constexpr(lse[k]) for k in sizes)
    out_x, out_r, out_block = (tl.constexpr(out[k]) for k in sizes)

    # Rounding down keeps an aligned address and silently moves any other, so
    # every binding must be 16-byte aligned: TensorRT's own buffers are (256
    # bytes), and the decode candidate checks the cache inputs it binds.
    @triton.jit
    def aligned(ptr):
        rounded = ptr.to(tl.uint64, bitcast=True) // 16 * 16
        return tl.multiple_of(rounded.to(ptr.dtype, bitcast=True), 16)

    @triton.jit
    def inductor_flex_split(Q, K, V, KVN, KVI, FKVN, FKVI, POS, M, L, ACC):
        split_fn(
            aligned(Q),
            aligned(K),
            aligned(V),
            aligned(M),
            aligned(L),
            aligned(KVN),
            aligned(KVI),
            aligned(FKVN),
            aligned(FKVI),
            aligned(POS),
            aligned(ACC),
        )

    @triton.jit
    def inductor_flex_lse(M, L, GM, GL):
        lse_fn(
            aligned(M), aligned(L), aligned(GM), aligned(GL), lse_x, lse_r, lse_block
        )

    @triton.jit
    def inductor_flex_out(ACC, GM, M, GL, O):
        out_fn(
            aligned(ACC),
            aligned(GM),
            aligned(M),
            aligned(GL),
            aligned(O),
            out_x,
            out_r,
            out_block,
        )

    return {
        "split": inductor_flex_split,
        "lse": inductor_flex_lse,
        "out": inductor_flex_out,
    }


# The template kernel's parameters: the flex inputs, the per-split buffers,
# one mask_mod buffer (the position) and the output.
_FLEX_TEMPLATE_PARAMS = (
    "arg_Q",
    "arg_K",
    "arg_V",
    "arg_M",
    "arg_L",
    "arg_KV_NUM_BLKS",
    "arg_KV_IDX",
    "arg_FULL_KV_NUM_BLKS",
    "arg_FULL_KV_IDX",
)


def parse_flex_decoding_code(source: str) -> dict[str, Any]:
    """Kernels, sources and dataflow of Inductor's flex-decoding output code.

    Expects exactly a template kernel writing ``M``, ``L`` and ``ACC``, then a
    reduction ``(M, L) -> (GM, GL)``, then a reduction ``(ACC, GM, M, GL) ->
    O``, and raises ``RuntimeError`` naming what differs. Returns the three
    kernel names (``split``, ``lse``, ``out``), their sources, the template
    grid and the reductions' ``xnumel``/``r0_numel``.
    """
    import re

    sources = dict(
        re.findall(
            r"async_compile\.triton\('(\w+)', '''(.*?)''', device_str", source, re.S
        )
    )
    runs = [
        (name, [arg.strip() for arg in args.split(",") if "stream=" not in arg])
        for name, args in re.findall(r"(triton_\w+)\.run\(([^)]*)\)", source)
    ]
    names = [name for name, _ in runs]
    if len(runs) != 3 or not names[0].startswith("triton_tem_"):
        raise RuntimeError(
            "Unexpected Inductor flex-decoding code: expected a template kernel "
            f"and two reductions, got {names}"
        )
    missing = [name for name in names if name not in sources]
    if missing:
        raise RuntimeError(f"Inductor flex-decoding code lacks the source of {missing}")
    template = re.search(r"^def " + names[0] + r"\(([^)]*)\)", sources[names[0]], re.M)
    params = tuple(p.strip() for p in template.group(1).split(",")) if template else ()
    if (
        len(params) != 11
        or params[:9] != _FLEX_TEMPLATE_PARAMS
        or not params[9].startswith("in_ptr")
        or not params[10].startswith("out_ptr")
    ):
        raise RuntimeError(
            "Unexpected flex-decoding template parameters (expected the flex "
            f"inputs, one mask buffer and one output): {params}"
        )
    (_, split_args), (_, lse_args), (_, out_args) = runs
    big_m, big_l, big_acc = split_args[3], split_args[4], split_args[10]
    if len(lse_args) != 6 or lse_args[:2] != [big_m, big_l]:
        raise RuntimeError(f"Unexpected flex-decoding max/sum reduction: {lse_args}")
    global_m, global_l = lse_args[2:4]
    if len(out_args) != 7 or out_args[:4] != [big_acc, global_m, big_m, global_l]:
        raise RuntimeError(f"Unexpected flex-decoding output reduction: {out_args}")
    return {
        "names": {"split": names[0], "lse": names[1], "out": names[2]},
        "sources": {name: sources[name] for name in names},
        "grid": tuple(int(value) for value in split_args[11:14]),
        "numel": {
            "lse": (int(lse_args[4]), int(lse_args[5])),
            "out": (int(out_args[5]), int(out_args[6])),
        },
    }


def _import_kernel(name: str, source: str, directory: str) -> Any:
    """Import Inductor's source of kernel ``name``; return its ``@triton.jit``."""
    import importlib.util
    import os

    path = os.path.join(directory, f"{name}.py")
    with open(path, "w") as stream:
        stream.write(source)
    spec = importlib.util.spec_from_file_location(f"qwen3_flex_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, name).fn


# The head-to-head's Inductor settings without Inductor's CUDA graphs (the
# decode step is captured whole).
FLEX_COMPILE_MODE = "max-autotune-no-cudagraphs"


def flex_attend(
    slots: int, scale: float, gqa: bool, compute_q_blocks: bool = True
) -> Any:
    """FlexAttention decode over ``slots`` cache slots up to the position.

    Returns ``attend(q, k, v, kv_num, kv_idx, full_num, full_idx, position)``
    taking the block mask and the position of :func:`flex_block_mask`: the
    function both flex-decoding variants compile. ``compute_q_blocks`` is
    ``BlockMask.from_kv_blocks``'s: the transposed (q-block) mask only serves
    the backward pass.
    """
    from torch.nn.attention.flex_attention import BlockMask, flex_attention

    def attend(q, k, v, kv_num, kv_idx, full_num, full_idx, last):  # type: ignore
        def visible(b, h, q_index, kv_index):  # type: ignore
            return kv_index <= last[0]

        block_mask = BlockMask.from_kv_blocks(
            kv_num,
            kv_idx,
            full_num,
            full_idx,
            BLOCK_SIZE=FLEX_BLOCK,
            mask_mod=visible,
            seq_lengths=(1, slots),
            compute_q_blocks=compute_q_blocks,
        )
        return flex_attention(
            q, k, v, block_mask=block_mask, scale=scale, enable_gqa=gqa
        )

    return attend


def inductor_flex_decoding(
    heads: int,
    kv_heads: int,
    head_dim: int,
    slots: int,
    scale: float,
    device: torch.device,
) -> FlexDecoding:
    """Generate Inductor's flex-decoding kernels for these shapes and host them.

    Compiles ``flex_attention`` of one FP16 query token over ``[1, kv_heads,
    slots, head_dim]`` caches with the mask ``kv_idx <= position`` (block mask
    from :func:`flex_block_mask`) with ``torch.compile(mode=
    "max-autotune-no-cudagraphs")``, the head-to-head's Inductor settings
    without Inductor's CUDA graphs, then imports the three generated kernels
    from their source and records the launch settings Inductor's autotuner
    chose. Cached per shape for the process.
    """
    import hashlib
    import tempfile

    from torch._inductor.codecache import PyCodeCache
    from torch._inductor.utils import run_and_get_code

    key = (heads, kv_heads, head_dim, slots, float(scale), FLEX_BLOCK, str(device))
    if key in _FLEX:
        return _FLEX[key]
    dtype = torch.float16
    q = torch.randn(1, heads, 1, head_dim, dtype=dtype, device=device)
    k = torch.randn(1, kv_heads, slots, head_dim, dtype=dtype, device=device)
    v = torch.randn(1, kv_heads, slots, head_dim, dtype=dtype, device=device)
    position = torch.full((1, 1), slots - 1, dtype=torch.int64, device=device)
    mask = flex_block_mask(position, slots)
    attend = flex_attend(slots, scale, heads != kv_heads)
    compiled = torch.compile(attend, mode=FLEX_COMPILE_MODE, fullgraph=True)
    with torch.no_grad():
        expected, code = run_and_get_code(compiled, q, k, v, *mask)
    parsed = parse_flex_decoding_code(code[0])
    names = parsed["names"]
    # The generated module Inductor just ran: its autotuners hold the chosen
    # launch settings (latest load first, should names repeat).
    loaded = next(
        (
            module
            for module in reversed(list(PyCodeCache.modules))
            if all(hasattr(module, name) for name in names.values())
        ),
        None,
    )
    if loaded is None:
        raise RuntimeError("Cannot find Inductor's loaded flex-decoding module")
    launch: dict[str, dict[str, Any]] = {}
    for role, name in names.items():
        launchers = getattr(getattr(loaded, name), "launchers", None) or []
        if not launchers:
            raise RuntimeError(f"Inductor kernel {name} has no launch configuration")
        config = launchers[0].config
        launch[role] = {
            "num_warps": int(config.num_warps),
            "num_stages": int(config.num_stages),
            **{str(k): int(v) for k, v in config.kwargs.items()},
        }
    launch["split"]["grid"] = parsed["grid"]
    for role in ("lse", "out"):
        xnumel, rnumel = parsed["numel"][role]
        launch[role].update(XNUMEL=xnumel, RNUMEL=rnumel)
        launch[role]["grid"] = (-(-xnumel // launch[role]["XBLOCK"]), 1, 1)
    directory = tempfile.mkdtemp(prefix="qwen3_flex_")
    functions = {
        role: _import_kernel(name, parsed["sources"][name], directory)
        for role, name in names.items()
    }
    kernels = _flex_wrappers(
        functions["split"],
        functions["lse"],
        functions["out"],
        launch["lse"],
        launch["out"],
    )
    sources = {
        name: hashlib.sha256(parsed["sources"][name].encode()).hexdigest()
        for name in names.values()
    }
    flex = FlexDecoding(key, kernels, launch, parsed["grid"][2], sources)
    # The hosted kernels must reproduce Inductor's own result on these inputs.
    actual = _run_flex(flex, q, k, v, mask)
    torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-3)
    _FLEX[key] = flex
    return flex


def _run_flex(
    flex: FlexDecoding,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: tuple[torch.Tensor, ...],
) -> torch.Tensor:
    """Launch the wrappers with Triton's runtime (a check outside TensorRT)."""
    heads, dim = q.shape[1], q.shape[3]
    splits = flex.splits
    f32 = dict(dtype=torch.float32, device=q.device)
    big_m = torch.empty(1, splits, heads, 1, **f32)
    big_l = torch.empty(1, splits, heads, 1, **f32)
    big_acc = torch.empty(1, splits, heads, 1, dim, **f32)
    global_m = torch.empty(1, heads, 1, **f32)
    global_l = torch.empty(1, heads, 1, **f32)
    out = torch.empty_like(q)
    split, lse, final = (flex.launch[role] for role in ("split", "lse", "out"))
    flex.kernels["split"][split["grid"]](
        q,
        k,
        v,
        *mask,
        big_m,
        big_l,
        big_acc,
        num_warps=split["num_warps"],
        num_stages=split["num_stages"],
    )
    flex.kernels["lse"][lse["grid"]](
        big_m,
        big_l,
        global_m,
        global_l,
        num_warps=lse["num_warps"],
        num_stages=lse["num_stages"],
    )
    flex.kernels["out"][final["grid"]](
        big_acc,
        global_m,
        big_m,
        global_l,
        out,
        num_warps=final["num_warps"],
        num_stages=final["num_stages"],
    )
    return out


# ----------------------------------------- Inductor flex-decoding, in a region
# Export captures flex_attention as a higher-order operator, which region
# capture rejects inside an execute_in_torch scope. An opaque custom operator
# is not one: the ``attention`` variant's regions call flex_decode, whose CUDA
# implementation runs flex_attention compiled by Inductor.
_FLEX_FUNCTIONS: dict[tuple, Any] = {}


def flex_decoding_function(
    heads: int,
    kv_heads: int,
    head_dim: int,
    slots: int,
    scale: float,
    device: torch.device,
) -> Any:
    """:func:`flex_attend` for these shapes, compiled by Inductor.

    ``torch.compile(mode=FLEX_COMPILE_MODE, fullgraph=True)``, as
    :func:`inductor_flex_decoding` compiles it; it compiles on its first call,
    inside the region's compiled code. There, Dynamo fails to trace the vmap
    that builds the q-block mask (FakeTensorMode meets a BatchedTensor:
    "Please convert all Tensors to FakeTensors first"), so it is not built:
    decode never reads it. Cached per shape for the process.
    """
    key = (heads, kv_heads, head_dim, slots, float(scale), FLEX_BLOCK, str(device))
    if key not in _FLEX_FUNCTIONS:
        _FLEX_FUNCTIONS[key] = torch.compile(
            flex_attend(slots, scale, heads != kv_heads, compute_q_blocks=False),
            mode=FLEX_COMPILE_MODE,
            fullgraph=True,
        )
    return _FLEX_FUNCTIONS[key]


def _flex_decode(
    q: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    kv_num: torch.Tensor,
    kv_idx: torch.Tensor,
    full_num: torch.Tensor,
    full_idx: torch.Tensor,
    position: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """One-token FlexAttention decode over the written cache slots.

    The implementation of ``torch.ops.qwen3_regions.flex_decode``. ``q`` is
    ``[1, heads, 1, head_dim]``, ``keys``/``values`` ``[1, kv_heads, slots,
    head_dim]`` caches whose slots after ``position`` are ignored, and the
    rest the block mask and position of :func:`flex_block_mask`. Returns
    ``[1, heads, 1, head_dim]``. CUDA tensors run
    :func:`flex_decoding_function` (Inductor's flex-decoding kernels), others
    FlexAttention's eager reference. Functional and opaque: export and region
    capture keep it as one operator.
    """
    heads, head_dim = q.shape[1], q.shape[3]
    kv_heads, slots = keys.shape[1], keys.shape[2]
    if q.is_cuda:
        attend = flex_decoding_function(
            heads, kv_heads, head_dim, slots, scale, q.device
        )
    else:
        # The reference: FlexAttention's eager implementation.
        attend = flex_attend(slots, scale, heads != kv_heads, compute_q_blocks=False)
    out = attend(q, keys, values, kv_num, kv_idx, full_num, full_idx, position)
    return out.contiguous()


def _define_flex_decode() -> None:
    """Register ``torch.ops.qwen3_regions.flex_decode`` once per process.

    A copy of this module (an edit.decode_launch edit) finds it registered and
    reuses it: the same code.
    """
    if hasattr(torch.ops.qwen3_regions, "flex_decode"):
        return
    op = torch.library.custom_op(
        "qwen3_regions::flex_decode",
        _flex_decode,
        mutates_args=(),
        schema="(Tensor q, Tensor keys, Tensor values, Tensor kv_num, "
        "Tensor kv_idx, Tensor full_num, Tensor full_idx, Tensor position, "
        "float scale) -> Tensor",
    )

    def fake(
        q: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        kv_num: torch.Tensor,
        kv_idx: torch.Tensor,
        full_num: torch.Tensor,
        full_idx: torch.Tensor,
        position: torch.Tensor,
        scale: float,
    ) -> torch.Tensor:
        return q.new_empty(q.shape)

    op.register_fake(fake)


_define_flex_decode()


class _Blocks:
    """The block implementations for one patched model."""

    def __init__(
        self,
        model: torch.nn.Module,
        blocks: Iterable[str],
        launch: Mapping[str, Mapping[str, int]],
        kernels: str,
        inductor: str = "none",
        generator: Any = None,
    ) -> None:
        self.blocks = frozenset(blocks)
        # A torch_tensorrt.region.KernelGenerator: generate each block's kernel
        # (PR 4914) instead of launching the hand-written one from ``kernels``.
        self.generator = generator
        unknown = self.blocks - set(BLOCKS)
        if unknown:
            raise ValueError(f"Unknown blocks {sorted(unknown)}; choose from {BLOCKS}")
        if inductor not in INDUCTOR_VARIANTS:
            raise ValueError(
                f"inductor must be one of {INDUCTOR_VARIANTS}; got {inductor!r}"
            )
        self.inductor = inductor
        # attention-flex: Inductor's kernels for this model's shapes (prepare()).
        self.flex: Optional[FlexDecoding] = None
        self.kernels = kernels
        self.launch = {
            key: dict(DEFAULT_LAUNCH[key], **dict(launch.get(key, {})))
            for key in BLOCKS
        }
        config = model.config
        self.eps = float(config.rms_norm_eps)
        self.hidden = config.hidden_size
        self.heads = config.num_attention_heads
        self.kv_heads = config.num_key_value_heads
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.q_size = self.heads * self.head_dim
        self.kv_size = self.kv_heads * self.head_dim
        # The norm kernels keep a whole hidden row in registers and qk_rope a
        # whole head: tl.arange needs powers of two.
        if self.blocks & {"final", "attn_in", "mlp_in"} and not _power_of_two(
            self.hidden
        ):
            raise ValueError(f"hidden_size={self.hidden} is not a power of two")
        if "qk_rope" in self.blocks and not _power_of_two(self.head_dim):
            raise ValueError(f"head_dim={self.head_dim} is not a power of two")
        if "attn_in" in self.blocks:
            for layer in model.model.layers:
                attn = layer.self_attn
                if any(
                    p.bias is not None for p in (attn.q_proj, attn.k_proj, attn.v_proj)
                ):
                    raise ValueError("attn_in does not support q/k/v biases")

    def _fused(self, kernel_name: str, inputs: Any, **kwargs: Any) -> Any:
        if self.generator is not None:
            from torch_tensorrt import region

            # The region body stays the eager reference; the generator writes,
            # validates and caches its kernel at compile time.
            return region.fused_kernel(
                generator=self.generator, inputs=list(inputs), name=kwargs["name"]
            )
        return _fused(self.kernels, kernel_name, inputs, **kwargs)

    # ------------------------------------------------------ Inductor routes
    @property
    def explicit_caches(self) -> bool:
        """Whether attention reads the caches in the exported program."""
        return self.inductor in _EXPLICIT_CACHE_VARIANTS

    def attention_scale(self, layer: Any) -> float:
        return 1.0 if self.fold_scale() else float(layer.self_attn.scaling)

    def prepare(self, model: Any, max_seq_len: int, device: torch.device) -> None:
        """Generate what the variant needs before export (attention-flex: kernels)."""
        if self.inductor != "attention-flex":
            return
        layer = model.model.layers[0]
        self.flex = inductor_flex_decoding(
            self.heads,
            self.kv_heads,
            self.head_dim,
            max_seq_len,
            self.attention_scale(layer),
            device,
        )

    def attend(
        self,
        index: int,
        layer: Any,
        q: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        position_ids: torch.Tensor,
        mask: Optional[tuple[torch.Tensor, ...]],
    ) -> torch.Tensor:
        """Attention over the written caches: ``[1, heads, 1, head_dim]``.

        ``attention`` and ``attention-math``: an Inductor region of
        flex-decoding (:func:`flex_decode`) or of plain math
        (:func:`decode_attention`). ``attention-flex``: Inductor's
        flex-decoding kernels as three plugins in the engine.
        """
        from torch_tensorrt import region

        scale = self.attention_scale(layer)
        if self.inductor == "attention-math":
            with region.execute_in_torch(name=f"attention_{index}", backend="inductor"):
                out = decode_attention(q, keys, values, position_ids, scale)
            return out
        if mask is None:
            raise RuntimeError(f"{self.inductor} needs the step's block mask")
        if self.inductor == "attention":
            with region.execute_in_torch(name=f"attention_{index}", backend="inductor"):
                out = torch.ops.qwen3_regions.flex_decode.default(
                    q, keys, values, *mask, scale
                )
            return out
        flex = self.flex
        if flex is None:
            raise RuntimeError(
                "attention-flex needs prepare(); compile with compile_decode"
            )
        split, lse, final = (flex.launch[role] for role in ("split", "lse", "out"))
        with region.fused_kernel(
            kernel=functools.partial(_flex_kernel, flex.key, "split"),
            inputs=[q, keys, values, *mask],
            name=f"flex_split_{index}",
            grid=functools.partial(_fixed_grid, tuple(split["grid"])),
            num_warps=split["num_warps"],
            num_stages=split["num_stages"],
        ):
            big_m, big_l, big_acc = flex_split_reference(
                q, keys, values, *mask, scale=scale, splits=flex.splits
            )
        with region.fused_kernel(
            kernel=functools.partial(_flex_kernel, flex.key, "lse"),
            inputs=[big_m, big_l],
            name=f"flex_lse_{index}",
            grid=functools.partial(_fixed_grid, tuple(lse["grid"])),
            num_warps=lse["num_warps"],
            num_stages=lse["num_stages"],
        ):
            global_m, global_l = flex_lse_reference(big_m, big_l)
        with region.fused_kernel(
            kernel=functools.partial(_flex_kernel, flex.key, "out"),
            inputs=[big_acc, global_m, big_m, global_l],
            name=f"flex_out_{index}",
            grid=functools.partial(_fixed_grid, tuple(final["grid"])),
            num_warps=final["num_warps"],
            num_stages=final["num_stages"],
        ):
            out = flex_out_reference(big_acc, global_m, big_m, global_l, q.dtype)
        return out

    # -------------------------------------------------------------- pieces
    def attn_in(self, layer: Any, h: torch.Tensor, pending: Optional[torch.Tensor]):
        """Return (h, q, k, v): the residual stream and the projections."""
        attn = layer.self_attn
        nw = layer.input_layernorm.weight
        if "attn_in" not in self.blocks:
            if pending is not None:
                h = h + pending
            x = layer.input_layernorm(h)
            return h, attn.q_proj(x), attn.k_proj(x), attn.v_proj(x)
        # Three outputs (q, k, v): TensorRT cannot fuse the split of one
        # plugin output into the q/k norms and would add slice copies.
        wq, wk, wv = attn.q_proj.weight, attn.k_proj.weight, attn.v_proj.weight
        launch = self.launch["attn_in"]
        programs, rows = _rows(
            launch,
            self.q_size + 2 * self.kv_size,
            "attn_in",
            self.q_size,
            self.kv_size,
        )
        shape = dict(NQ=self.q_size, NKV=self.kv_size, K=self.hidden, EPS=self.eps)
        if pending is None:
            with self._fused(
                "norm_qkv_kernel",
                [h, nw, wq, wk, wv],
                programs=programs,
                launch=launch,
                name="attn_in",
                **shape,
                **rows,
            ):
                x = _rmsnorm(h, nw, self.eps)
                q, k, v = F.linear(x, wq), F.linear(x, wk), F.linear(x, wv)
        else:
            with self._fused(
                "add_norm_qkv_kernel",
                [h, pending, nw, wq, wk, wv],
                programs=programs,
                launch=launch,
                name="attn_in_add",
                **shape,
                **rows,
            ):
                h = h + pending
                x = _rmsnorm(h, nw, self.eps)
                q, k, v = F.linear(x, wq), F.linear(x, wk), F.linear(x, wv)
        return h, q, k, v

    def fold_scale(self) -> bool:
        return "qk_rope" in self.blocks and bool(
            self.launch["qk_rope"].get("FOLD_SCALE")
        )

    def qk_rope(self, layer: Any, q, k, v, cos, sin):
        """Return q, k, v as [batch, heads, seq, head_dim], q/k normed and rotated
        (q pre-scaled by the softmax scale when fold_scale())."""
        from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

        attn = layer.self_attn
        batch, seq = q.shape[:2]
        hd = self.head_dim

        def reference(q, k, scale=1.0):
            # Region outputs follow production order, and the kernel writes q
            # before k: every q-side op must come before k is produced.
            q = attn.q_norm(q.view(batch, seq, -1, hd)).transpose(1, 2)
            if scale != 1.0:
                q = q * scale
            k = attn.k_norm(k.view(batch, seq, -1, hd)).transpose(1, 2)
            return apply_rotary_pos_emb(q, k, cos, sin)

        v = v.view(batch, seq, -1, hd).transpose(1, 2)
        if "qk_rope" not in self.blocks:
            q, k = reference(q, k)
            return q, k, v
        if batch * seq != 1:
            raise ValueError("qk_rope is a single-token decode kernel")
        scale = float(attn.scaling) if self.fold_scale() else 1.0
        with self._fused(
            "qk_norm_rope_kernel",
            [q, k, attn.q_norm.weight, attn.k_norm.weight, cos, sin],
            programs=self.heads + self.kv_heads,
            launch=self.launch["qk_rope"],
            name="qk_rope",
            HQ=self.heads,
            D=hd,
            EPS=self.eps,
            QSCALE=scale,
        ):
            q, k = reference(q, k, scale)
        return q, k, v

    def _gemv_add(self, block: str, a, w, h):
        launch = self.launch[block]
        programs, rows = _rows(launch, w.shape[0], block)
        block_k = _tile(block, "BLOCK_K", int(launch["BLOCK_K"]), w.shape[1])
        with self._fused(
            "gemv_add_kernel",
            [a, w, h],
            programs=programs,
            launch=launch,
            name=block,
            N=w.shape[0],
            K=w.shape[1],
            BLOCK_K=block_k,
            **rows,
        ):
            h = h + F.linear(a, w)
        return h

    def attn_out(self, layer: Any, o, h):
        """Return (h, pending)."""
        attn = layer.self_attn
        if "attn_out" in self.blocks:
            return self._gemv_add("attn_out", o, attn.o_proj.weight, h), None
        return h, attn.o_proj(o)

    def mlp_in(self, layer: Any, h, pending):
        """Return (h, act)."""
        mlp = layer.mlp
        nw = layer.post_attention_layernorm.weight
        if "mlp_in" not in self.blocks:
            if pending is not None:
                h = h + pending
            x = layer.post_attention_layernorm(h)
            return h, mlp.act_fn(mlp.gate_proj(x)) * mlp.up_proj(x)
        wg, wu = mlp.gate_proj.weight, mlp.up_proj.weight
        launch = self.launch["mlp_in"]
        programs, rows = _rows(launch, wg.shape[0], "mlp_in")
        shape = dict(N=wg.shape[0], K=self.hidden, EPS=self.eps)
        if pending is None:
            with self._fused(
                "norm_gateup_kernel",
                [h, nw, wg, wu],
                programs=programs,
                launch=launch,
                name="mlp_in",
                **shape,
                **rows,
            ):
                x = _rmsnorm(h, nw, self.eps)
                act = F.silu(F.linear(x, wg)) * F.linear(x, wu)
        else:
            with self._fused(
                "add_norm_gateup_kernel",
                [h, pending, nw, wg, wu],
                programs=programs,
                launch=launch,
                name="mlp_in_add",
                **shape,
                **rows,
            ):
                h = h + pending
                x = _rmsnorm(h, nw, self.eps)
                act = F.silu(F.linear(x, wg)) * F.linear(x, wu)
        return h, act

    def mlp_out(self, layer: Any, act, h):
        """Return (h, pending)."""
        mlp = layer.mlp
        if "mlp_out" in self.blocks:
            return self._gemv_add("mlp_out", act, mlp.down_proj.weight, h), None
        return h, mlp.down_proj(act)

    def final(self, model: Any, h, pending):
        norm, head = model.model.norm, model.lm_head
        if self.inductor == "lm-head":
            from torch_tensorrt import region

            with region.execute_in_torch(name="lm_head", backend="inductor"):
                if pending is not None:
                    h = h + pending
                logits = head(norm(h))
            return logits
        if "final" not in self.blocks or head.bias is not None:
            if pending is not None:
                h = h + pending
            return head(norm(h))
        w = head.weight
        launch = self.launch["final"]
        programs, rows = _rows(launch, w.shape[0], "final")
        shape = dict(N=w.shape[0], K=self.hidden, EPS=self.eps)
        if pending is None:
            with self._fused(
                "norm_gemv_kernel",
                [h, norm.weight, w],
                programs=programs,
                launch=launch,
                name="final",
                **shape,
                **rows,
            ):
                logits = F.linear(_rmsnorm(h, norm.weight, self.eps), w)
        else:
            with self._fused(
                "add_norm_gemv_out_kernel",
                [h, pending, norm.weight, w],
                programs=programs,
                launch=launch,
                name="final_add",
                **shape,
                **rows,
            ):
                logits = F.linear(_rmsnorm(h + pending, norm.weight, self.eps), w)
        return logits


def _decode_forward(self, input_ids=None, position_ids=None, caches=None, **kwargs):
    """Single-token decode forward of a patched Qwen3ForCausalLM.

    Without ``caches`` it has no KV cache: static_cache_v3 adds it during
    compilation. With ``caches`` (``key_cache_0, value_cache_0, ...``, the
    explicit-cache variants) every layer writes its key and value at the
    position and attends over the written caches, which it also returns
    (``.past_key_values``), as ``static_cache_v3.CacheDecodeStep`` expects.
    """
    from transformers.masking_utils import create_causal_mask
    from transformers.modeling_outputs import CausalLMOutputWithPast
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from transformers.models.qwen3.modeling_qwen3 import eager_attention_forward

    blocks: _Blocks = self._qwen3_blocks
    model = self.model
    config = self.config
    h = model.embed_tokens(input_ids)
    if position_ids is None:
        position_ids = torch.arange(h.shape[1], device=h.device).unsqueeze(0)
    if (caches is not None) != blocks.explicit_caches:
        raise ValueError(
            f"inductor={blocks.inductor!r} decodes "
            + ("with" if blocks.explicit_caches else "without")
            + " explicit caches"
        )
    mask = None
    if caches is None:
        mask = create_causal_mask(
            config=config,
            inputs_embeds=h,
            attention_mask=None,
            past_key_values=None,
            position_ids=position_ids,
        )
    elif blocks.inductor in _FLEX_VARIANTS:
        # One block mask per step, shared by every layer's flex-decoding.
        mask = flex_block_mask(position_ids, caches[0].shape[2])
    cos, sin = model.rotary_emb(h, position_ids)
    attention = ALL_ATTENTION_FUNCTIONS.get_interface(
        config._attn_implementation, eager_attention_forward
    )
    batch, seq = h.shape[:2]
    written: list[torch.Tensor] = []
    pending: Optional[torch.Tensor] = None
    for index, layer in enumerate(model.layers[: config.num_hidden_layers]):
        attn = layer.self_attn
        h, q, k, v = blocks.attn_in(layer, h, pending)
        q, k, v = blocks.qk_rope(layer, q, k, v, cos, sin)
        if caches is None:
            o, _ = attention(
                attn,
                q,
                k,
                v,
                mask,
                dropout=0.0,
                scaling=1.0 if blocks.fold_scale() else attn.scaling,
                sliding_window=attn.sliding_window,
            )
        else:
            import static_cache_v3

            # Written in TensorRT (in place); only the reads are in the region.
            keys = static_cache_v3.write(caches[2 * index], position_ids, k)
            values = static_cache_v3.write(caches[2 * index + 1], position_ids, v)
            written += [keys, values]
            o = blocks.attend(index, layer, q, keys, values, position_ids, mask)
            o = o.transpose(1, 2)
        o = o.reshape(batch, seq, -1).contiguous()
        h, pending = blocks.attn_out(layer, o, h)
        h, act = blocks.mlp_in(layer, h, pending)
        h, pending = blocks.mlp_out(layer, act, h)
    logits = blocks.final(self, h, pending)
    if caches is None:
        return CausalLMOutputWithPast(logits=logits)
    return CausalLMOutputWithPast(logits=logits, past_key_values=tuple(written))


def patch_qwen3(
    model: torch.nn.Module,
    blocks: Iterable[str] = PRESETS["layers"],
    launch: Optional[Mapping[str, Mapping[str, int]]] = None,
    kernels: str = "qwen3_kernels",
    inductor: str = "none",
    generator: Any = None,
) -> torch.nn.Module:
    """Make ``model`` (a loaded Qwen3ForCausalLM) run ``blocks`` as fused kernels.

    With ``generator`` (a ``torch_tensorrt.region.KernelGenerator``) each block's
    kernel is generated during compilation (PR 4914) instead of taken from
    ``kernels``; blocks of the same shape share one generated kernel.

    Replaces the instance's forward; ``model(input_ids, position_ids=...)``
    returns an object with ``.logits``, as ``static_cache_v3.DecodeStep``
    expects. With ``blocks=()`` it is Hugging Face's decode math without
    regions. ``launch`` overrides :data:`DEFAULT_LAUNCH` per block, and
    ``kernels`` names the module holding the Triton kernels (imported if it
    is not yet). ``inductor`` hands operations to Inductor's kernels (one of
    :data:`INDUCTOR_VARIANTS`, see the module docstring); ``lm-head`` replaces
    the ``final`` block, and the attention variants decode with explicit
    caches (``model(..., caches=[...])``).
    """
    if kernels not in sys.modules:
        importlib.import_module(kernels)
    model._qwen3_blocks = _Blocks(
        model, blocks, launch or {}, kernels, inductor, generator=generator
    )
    model.forward = types.MethodType(_decode_forward, model)
    return model


# -------------------------------------------------------------------- build
def compile_decode(
    model: torch.nn.Module,
    *,
    max_seq_len: int,
    device: torch.device,
    attention: str = "auto",
    **settings: Any,
) -> torch.nn.Module:
    """Strict-compile the static_cache_v3 decode step of a (patched) model.

    ``attention`` is static_cache_v3's attention: ``"auto"`` (a position mask
    up to 192 cache slots, IAttention ``key_value_lengths`` above),
    ``"mask"`` or ``"lengths"`` (see ``static_cache_v3.py``: TensorRT 11.4
    ignores the mask for long caches). ``settings`` are Torch-TensorRT
    compilation settings. Regions need an eval-mode module and
    ``offload_module_to_cpu=False``, which this sets.

    The explicit-cache variants (``inductor="attention"``,
    ``"attention-math"`` or ``"attention-flex"``) export
    ``static_cache_v3.cache_decode_step`` with ``max_seq_len``-slot caches,
    for the same calling convention; their attention ignores ``attention``.
    Inductor regions compile on the module's first call.
    """
    import static_cache_v3
    import torch_tensorrt

    static_cache_v3.configure(max_seq_len, attention=attention)
    ids = torch.zeros((1, 1), dtype=torch.int64, device=device)
    positions = torch.zeros((1, 1), dtype=torch.int64, device=device)
    options: dict[str, Any] = {
        "min_block_size": 1,
        "disable_tf32": True,
        "use_python_runtime": True,
        **settings,
        "offload_module_to_cpu": False,
        "device": device,
    }
    blocks: Optional[_Blocks] = getattr(model, "_qwen3_blocks", None)
    inputs: list[torch.Tensor] = [ids, positions]
    step: torch.nn.Module = static_cache_v3.DecodeStep(model)
    if blocks is not None:
        blocks.prepare(model, max_seq_len, device)
        if blocks.explicit_caches:
            dtype = next(model.parameters()).dtype
            inputs += static_cache_v3.cache_examples(
                model.config, max_seq_len, device, dtype
            )
            step = static_cache_v3.cache_decode_step(
                model, model.config.num_hidden_layers
            )
    with torch.no_grad():
        return torch_tensorrt.compile(
            step.eval(),
            ir="dynamo",
            arg_inputs=inputs,
            strict=True,
            **options,
        )
