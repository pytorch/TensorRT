# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Hugging Face encoder forward workload for ``torch_tensorrt.autotune`` sessions.

The measured unit is one batch-1 forward of a fixed natural-text sequence of
``length`` tokens through a Hugging Face encoder loaded with ``AutoModel``
(for ``answerdotai/ModernBERT-base``, ``ModernBertModel``). The output is
``last_hidden_state`` [1, length, hidden]; task heads (ModernBERT's masked-LM
head) are not part of the benchmark. The input is public-domain English prose
(Dickens, Melville, Austen and Lincoln; 973 ModernBERT tokens) repeated to
fill ``length`` tokens between the tokenizer's [CLS] and [SEP] tokens. Every
candidate is compared with an FP32 eager reference (Hugging Face SDPA
attention, no TF32) token by token (:func:`hidden_state_metrics`): a tight
check of the bulk of the tokens and a loose one of the few ill-conditioned
tokens that any perturbation moves, FP16 rounding included (see
:meth:`EncoderWorkload.accuracy_defaults`).

ModernBERT alternates one global attention layer with two sliding-window
("local") layers in which a token sees the tokens at most
``config.sliding_window`` (64) positions away. The model runs exactly as
Hugging Face runs it in eager mode: the global layers without a mask and the
local layers with a dense [length, length] band mask (:class:`_Encoder` passes
these masks explicitly). Exported, Hugging Face's own mask creation would also
materialize an all-true [length, length] mask for the global layers; TensorRT
11.4 crashes on it in FP16 (see :class:`EncoderWorkload`), and dropping it is
exact for an unpadded sequence. Workload knobs (offered for ModernBERT):

* ``workload.local_attention``: ``bandedB`` computes the local layers block by
  block: each block of B queries attends only to the ``B + 2 * window`` keys
  around it, with a small static mask, instead of all ``length`` keys
  (:func:`banded_attention`). At 8192 tokens it cuts the local layers'
  attention work 21-43x.
* ``graph.fold_attention_masks``: fold the attention masks into engine
  constants (Torch-TensorRT otherwise computes them in the engine on every
  call), which lets TensorRT analyze them while it builds the engine.
* ``kernels.inductor``: hybrid builds that hand each layer's QKV split and
  rotary embedding to Inductor, either as Inductor's own generated Triton
  kernel hosted inside the engine (``rope-kernel``, a
  ``torch_tensorrt.region.fused_kernel`` plugin per layer;
  ``rope-kernel-eager-numerics``, the kernel Inductor generates when it
  emulates eager's FP16 rounding) or as Inductor regions outside it
  (``inductor-regions``,
  ``torch_tensorrt.region.execute_in_torch(backend="inductor")``, two
  engine boundaries per layer); see ``modernbert_kernels.py``. All compile
  through strict export (``torch_tensorrt.compile(..., strict=True)``) and
  need CUDA graphs. Adding the knob changed the workload's knobs, so
  sessions created before it cannot be resumed (the session refuses a
  changed workload); start hybrid sessions in a new directory.
* ``kernels.generated``: ``split-rope`` runs the same per-layer subgraph (the
  ``split_rope_{layer}`` scope of ``modernbert_kernels.py``) as a kernel that
  PR 4914's ``torch_tensorrt.region.KernelGenerator`` writes during the build
  (``region_generation.py``), hosted in the engine like ``rope-kernel``. It
  needs CUDA graphs and ``kernels.inductor="none"``. Generated kernels are
  cached in ``kernel_cache`` (``region_generation.kernel_cache``), keyed by
  the engine options; a session prepares the value once, before its search.

plus precision (``engine.precision``, ``engine.use_fp32_acc``), Torch-TensorRT's
attention decomposition (``engine.decompose_attention``), TensorRT build routes
(``engine.build_route``) and the standard builder and runtime knobs.

Precision policy: FP16 TensorRT engines run their GEMMs with FP16
accumulation (``f16f16_f16f16_f16`` kernels), which TensorRT picks for FP16
by default. ``engine.use_fp32_acc`` makes every GEMM accumulate in FP32
(with casts that also unfuse a decomposed attention); the build routes
``-cask:fp16_acc=off`` (GEMMs outside fusions) and ``-cask:fp16_acc=off
-cask_fusion:fp16_acc=off`` (all GEMMs at 8192 tokens; at 512 one fused GEMM
per layer keeps FP16) do it without casts. PyTorch accumulates FP16 GEMMs in
FP32 unless ``bench.fp16_accumulation`` is set (below).

``EncoderWorkload.build`` also honors the head-to-head benchmark's
(``bench_compare.py``) ``bench.*`` values, which are not knobs:
``bench.backend`` (``tensorrt``, ``eager`` or
``inductor``), ``bench.mode`` (a ``torch.compile`` mode), ``bench.attn`` (the
Hugging Face attention implementation of PyTorch candidates: ``sdpa``,
``flex_attention`` or ``eager``), ``bench.inductor``
(``torch._inductor.config`` overrides) and this workload's
``bench.fp16_accumulation`` (FP16 only: sets
``torch.backends.cuda.matmul.allow_fp16_accumulation`` while the candidate
lives, so cuBLAS FP16 GEMMs accumulate in FP16 as TensorRT's do). Unknown
``bench.*`` keys, and PyTorch-only keys on a TensorRT build, are errors.
PyTorch candidates load the same model in the same precision, run the same
input, apply the same workload knobs and return the same tensor; they compile
and warm up inside ``build``, so its time includes compilation.
"""

from __future__ import annotations

import contextlib
import functools
import gc
import math
import sys
import time
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch_tensorrt.autotune import STANDARD_KNOBS, Candidate, Knob, Timing, Workload

# Public-domain prose: the openings of "A Tale of Two Cities" (Dickens, 1859),
# "Moby-Dick" (Melville, 1851) and "Pride and Prejudice" (Austen, 1813), and
# the Gettysburg Address (Lincoln, 1863).
TEXT = (
    "It was the best of times, it was the worst of times, it was the age of wisdom, "
    "it was the age of foolishness, it was the epoch of belief, it was the epoch of "
    "incredulity, it was the season of Light, it was the season of Darkness, it was "
    "the spring of hope, it was the winter of despair, we had everything before us, "
    "we had nothing before us, we were all going direct to Heaven, we were all going "
    "direct the other way - in short, the period was so far like the present period, "
    "that some of its noisiest authorities insisted on its being received, for good "
    "or for evil, in the superlative degree of comparison only. There were a king "
    "with a large jaw and a queen with a plain face, on the throne of England; there "
    "were a king with a large jaw and a queen with a fair face, on the throne of "
    "France. In both countries it was clearer than crystal to the lords of the State "
    "preserves of loaves and fishes, that things in general were settled for ever.\n"
    "Call me Ishmael. Some years ago - never mind how long precisely - having little "
    "or no money in my purse, and nothing particular to interest me on shore, I "
    "thought I would sail about a little and see the watery part of the world. It is "
    "a way I have of driving off the spleen and regulating the circulation. Whenever "
    "I find myself growing grim about the mouth; whenever it is a damp, drizzly "
    "November in my soul; whenever I find myself involuntarily pausing before coffin "
    "warehouses, and bringing up the rear of every funeral I meet; and especially "
    "whenever my hypos get such an upper hand of me, that it requires a strong moral "
    "principle to prevent me from deliberately stepping into the street, and "
    "methodically knocking people's hats off - then, I account it high time to get "
    "to sea as soon as I can. This is my substitute for pistol and ball. With a "
    "philosophical flourish Cato throws himself upon his sword; I quietly take to "
    "the ship. There is nothing surprising in this. If they but knew it, almost all "
    "men in their degree, some time or other, cherish very nearly the same feelings "
    "towards the ocean with me.\n"
    "It is a truth universally acknowledged, that a single man in possession of a "
    "good fortune, must be in want of a wife. However little known the feelings or "
    "views of such a man may be on his first entering a neighbourhood, this truth is "
    "so well fixed in the minds of the surrounding families, that he is considered "
    'the rightful property of some one or other of their daughters. "My dear Mr. '
    'Bennet," said his lady to him one day, "have you heard that Netherfield Park '
    'is let at last?" Mr. Bennet replied that he had not. "But it is," returned '
    'she; "for Mrs. Long has just been here, and she told me all about it." Mr. '
    'Bennet made no answer. "Do you not want to know who has taken it?" cried his '
    'wife impatiently. "You want to tell me, and I have no objection to hearing '
    'it." This was invitation enough.\n'
    "Four score and seven years ago our fathers brought forth on this continent, a "
    "new nation, conceived in Liberty, and dedicated to the proposition that all men "
    "are created equal. Now we are engaged in a great civil war, testing whether "
    "that nation, or any nation so conceived and so dedicated, can long endure. We "
    "are met on a great battle-field of that war. We have come to dedicate a portion "
    "of that field, as a final resting place for those who here gave their lives "
    "that that nation might live. It is altogether fitting and proper that we should "
    "do this. But, in a larger sense, we can not dedicate - we can not consecrate - "
    "we can not hallow - this ground. The brave men, living and dead, who struggled "
    "here, have consecrated it, far above our poor power to add or detract. The "
    "world will little note, nor long remember what we say here, but it can never "
    "forget what they did here. It is for us the living, rather, to be dedicated "
    "here to the unfinished work which they who fought here have thus far so nobly "
    "advanced. It is rather for us to be here dedicated to the great task remaining "
    "before us - that from these honored dead we take increased devotion to that "
    "cause for which they gave the last full measure of devotion - that we here "
    "highly resolve that these dead shall not have died in vain - that this nation, "
    "under God, shall have a new birth of freedom - and that government of the "
    "people, by the people, for the people, shall not perish from the earth.\n"
)

_DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
# Block-banded local attention: the query block size B of each option.
LOCAL_ATTENTION = {"dense": None, "banded128": 128, "banded64": 64, "banded256": 256}
# TensorRT build routes that change ModernBERT's FP16 engines (RTX 5090,
# TensorRT 11.4). "-cask:fp16_acc=off" forbids FP16-accumulation GEMM kernels
# outside fusions: 88 of 110 GEMM launches per forward then accumulate in FP32
# at 512 tokens, 44 of 109 at 8192; adding "-cask_fusion:fp16_acc=off" (alone
# a no-op) moves all 109 to FP32 at 8192 and changes nothing at 512 (one fused
# 64x64 GEMM per layer keeps FP16 accumulation under both). Unlike
# use_fp32_acc they insert no casts, so attention stays fused.
# "-tactic:splitk=off" drops split-K GEMM tactics (other GEMM tiles in the
# dense baseline engine; decompose_attention engines came out bitwise
# unchanged). They need TensorRT with Global Performance Tuning (11.1+, not
# TensorRT-RTX or Windows). Left out:
# "-cask_fusion:fp16_acc=off" alone, "-late_h_fusion=off" and
# "-mha_fusion:v2:force_flash_tiled=on" built the same kernels with bitwise
# equal outputs (with and without decompose_attention);
# "-kgen:codegen:cuda_tile=2" fails to build (a Myelin check in
# copy_vtx_analyzer).
BUILD_ROUTES = (
    "",
    "-cask:fp16_acc=off",
    "-cask:fp16_acc=off -cask_fusion:fp16_acc=off",
    "-tactic:splitk=off",
)
# Standard knobs offered. engine.tiling_optimization_level is left out: "fast"
# crashes the TensorRT 11.4 builder (segmentation fault) on ModernBERT.
_STANDARD = (
    "runtime.cudagraphs",
    "runtime.pre_allocated_outputs",
    "engine.optimization_level",
)
# Head-to-head keys (see the module docstring) and those only PyTorch uses.
_BENCH_KEYS = (
    "bench.backend",
    "bench.mode",
    "bench.attn",
    "bench.inductor",
    "bench.fp16_accumulation",
)
_TORCH_BENCH_KEYS = _BENCH_KEYS[1:]
# torch.compile modes that replay CUDA graphs.
_CUDAGRAPH_MODES = ("reduce-overhead", "max-autotune")
_ATTENTION_NAME = "agentic_encoder"
# kernels.inductor values (modernbert_kernels.VARIANTS, without importing it).
HYBRID_VARIANTS = (
    "none",
    "rope-kernel",
    "inductor-regions",
    "rope-kernel-eager-numerics",
)
# The kernels.inductor values that build execute_in_torch(backend="inductor")
# regions (the others host a generated kernel inside the engine).
REGION_VARIANTS = ("inductor-regions",)
# kernels.generated values: regions whose kernel PR 4914's KernelGenerator
# writes (modernbert_kernels.GENERATED, without importing it).
GENERATED_KERNELS = ("none", "split-rope")
# Accuracy check (hidden_state_metrics): the bulk check leaves out this
# fraction of the tokens (the largest relative errors, at least one token),
# the local check takes the median token cosine of blocks of this many
# consecutive tokens, and tokens below this cosine count as outliers.
TRIM_FRACTION = 0.01
BLOCK_TOKENS = 64
OUTLIER_COSINE = 0.9


def create_case(
    model: str = "answerdotai/ModernBERT-base",
    length: int = 512,
    baseline: Mapping[str, Any] | None = None,
    kernel_cache: str | None = None,
) -> "EncoderWorkload":
    """Workload factory: ``WorkloadFactory("tools/llm/agentic_encoder.py:create_case", {...})``."""
    return EncoderWorkload(
        model, length=length, baseline=baseline, kernel_cache=kernel_cache
    )


# ------------------------------------------------------------------ helpers
def sequence_ids(
    text_ids: Sequence[int], length: int, first: int | None, last: int | None
) -> list[int]:
    """``length`` token ids: ``first``, ``text_ids`` repeated, then ``last``.

    ``first`` and ``last`` are the tokenizer's [CLS] and [SEP] ids (``None``
    leaves either out).
    """
    if not text_ids:
        raise ValueError("The text has no tokens")
    ends = [token for token in (first, last) if token is not None]
    body = length - len(ends)
    if body < 1:
        raise ValueError(f"length={length} leaves no room for text tokens")
    repeats = -(-body // len(text_ids))
    filled = (list(text_ids) * repeats)[:body]
    head = [first] if first is not None else []
    tail = [last] if last is not None else []
    return head + filled + tail


def band_block_mask(
    length: int, window: int, block: int, device: torch.device | str = "cpu"
) -> torch.Tensor:
    """Visibility of :func:`banded_attention`'s key spans: [blocks, 1, block, span].

    Query ``n * block + r`` sees span position ``c`` (key ``n * block + c -
    window``) when ``|r + window - c| <= window`` and the key lies inside the
    sequence. Built from ``arange``, so an exported graph computes it.
    """
    blocks = -(-length // block)
    span = block + 2 * window
    rows = torch.arange(block, device=device).unsqueeze(1)
    cols = torch.arange(span, device=device).unsqueeze(0)
    band = (cols >= rows) & (cols <= rows + 2 * window)
    keys = (
        torch.arange(blocks, device=device).view(blocks, 1, 1) * block
        - window
        + cols.unsqueeze(0)
    )
    inside = (keys >= 0) & (keys < length)
    return (band.unsqueeze(0) & inside).unsqueeze(1)


def banded_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    window: int,
    block: int,
    scale: float | None = None,
) -> torch.Tensor:
    """Bidirectional sliding-window attention, computed block by block.

    Key ``j`` is visible to query ``i`` when ``|i - j| <= window``, as in
    Hugging Face's bidirectional sliding-window mask. Queries are split into
    blocks of ``block`` tokens, and each block attends only to the ``block + 2
    * window`` keys around it with a small static mask, so the work is
    ``length * (block + 2 * window)`` scores instead of ``length ** 2``.
    Inputs are [batch, heads, length, head_dim]; the output is [batch, length,
    heads, head_dim], the layout Hugging Face attention functions return.
    """
    if window < 0 or block < 1:
        raise ValueError("banded_attention needs window >= 0 and block >= 1")
    batch, heads, length, dim = query.shape
    blocks = -(-length // block)
    padded = blocks * block
    span = block + 2 * window
    if padded != length:
        query = F.pad(query, (0, 0, 0, padded - length))
    # Keys and values with `window` zero rows before the sequence and enough
    # after it that block n's span is rows [n * block, n * block + span).
    pad = (0, 0, window, window + padded - length)
    starts = torch.arange(blocks, device=query.device).unsqueeze(1) * block
    index = (starts + torch.arange(span, device=query.device)).flatten()
    key = F.pad(key, pad).index_select(2, index).view(batch, heads, blocks, span, dim)
    value = (
        F.pad(value, pad).index_select(2, index).view(batch, heads, blocks, span, dim)
    )
    # Blocks become the batch: [batch * blocks, heads, rows, dim].
    query = query.view(batch, heads, blocks, block, dim).transpose(1, 2)
    query = query.reshape(batch * blocks, heads, block, dim)
    key = key.transpose(1, 2).reshape(batch * blocks, heads, span, dim)
    value = value.transpose(1, 2).reshape(batch * blocks, heads, span, dim)
    mask = band_block_mask(length, window, block, query.device)
    mask = mask.expand(batch, blocks, 1, block, span).reshape(
        batch * blocks, 1, block, span
    )
    out = F.scaled_dot_product_attention(query, key, value, attn_mask=mask, scale=scale)
    out = out.view(batch, blocks, heads, block, dim).transpose(2, 3)
    return out.reshape(batch, padded, heads, dim)[:, :length]


def _lowered_attention(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float = 0.0,
    scaling: float | None = None,
    sliding_window: int | None = None,
    *,
    block: int | None = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, None]:
    """A Hugging Face attention function: SDPA, or banded SDPA for local layers.

    Hugging Face passes ``sliding_window`` (window + 1, flash-attention's
    convention) only to the sliding-window layers.
    """
    if sliding_window is not None and block is not None:
        out = banded_attention(
            query,
            key,
            value,
            window=int(module.config.sliding_window),
            block=block,
            scale=scaling,
        )
    else:
        out = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, scale=scaling
        ).transpose(1, 2)
    return out.contiguous(), None


def band_mask(length: int, window: int, device: Any = "cpu") -> torch.Tensor:
    """Hugging Face's dense SDPA mask of a bidirectional sliding-window layer.

    [1, 1, length, length], true where ``|i - j| <= window``; built by
    ``transformers.masking_utils.sdpa_mask`` from ``arange``, exactly as the
    model's own mask creation builds it.
    """
    from transformers.masking_utils import (
        sdpa_mask,
        sliding_window_bidirectional_mask_function,
    )

    return sdpa_mask(
        batch_size=1,
        q_length=length,
        kv_length=length,
        mask_function=sliding_window_bidirectional_mask_function(window),
        allow_is_causal_skip=False,
        allow_is_bidirectional_skip=False,
        device=device,
    )


class _Encoder(torch.nn.Module):
    """``input_ids`` -> ``last_hidden_state``.

    With ``window`` (the half width of ModernBERT's sliding-window layers) it
    passes the per-layer-type masks Hugging Face's eager path uses: none for
    the global layers, the dense band for the local ones. Exported, the
    model's own mask creation would add an all-true [length, length] mask to
    the global layers. ``local_attention="bandedB"`` registers
    :func:`_lowered_attention` as the model's attention implementation, which
    computes the local layers with :func:`banded_attention` instead. Without
    ``window`` it is the plain Hugging Face forward.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        window: int | None,
        local_attention: str = "dense",
    ) -> None:
        super().__init__()
        self.model = model
        self.window = window
        self.block = LOCAL_ATTENTION[local_attention]
        if self.block is not None:
            if window is None:
                raise ValueError(f"{local_attention} needs sliding-window layers")
            from transformers import AttentionInterface

            name = f"{_ATTENTION_NAME}_{local_attention}"
            AttentionInterface.register(
                name, functools.partial(_lowered_attention, block=self.block)
            )
            model.set_attn_implementation(name)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if self.window is None:
            return self.model(input_ids=input_ids).last_hidden_state
        local = None
        if self.block is None:
            local = band_mask(input_ids.shape[1], self.window, input_ids.device)
        masks = {"full_attention": None, "sliding_attention": local}
        return self.model(input_ids=input_ids, attention_mask=masks).last_hidden_state


def _blocks(values: torch.Tensor, size: int) -> list[torch.Tensor]:
    """Consecutive blocks of ``size``; a remainder under half a block joins the last."""
    blocks = list(torch.split(values, size))
    if len(blocks) > 1 and blocks[-1].numel() < size // 2:
        blocks[-2:] = [torch.cat(blocks[-2:])]
    return blocks


def hidden_state_metrics(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    trim: float = TRIM_FRACTION,
    block: int = BLOCK_TOKENS,
) -> dict[str, float]:
    """Token-level accuracy of hidden states [..., tokens, hidden] in float64.

    Over all tokens: ``token_cosine_min`` and ``token_cosine_mean`` (the
    smallest and the mean cosine similarity of a token's hidden vectors),
    ``rel_l2_error`` (L2 norm of the difference over the reference's) and
    ``max_abs_error``; ``outlier_token_fraction``, the share of tokens with
    a cosine below ``OUTLIER_COSINE``. Bulk: ``rel_l2_error_trimmed``, the
    relative L2 error of the tokens left after dropping the ``trim`` fraction
    (at least one) with the largest relative error. Local:
    ``block_cosine_median_min``, the smallest median token cosine of
    ``block`` consecutive tokens, which sees errors confined to too few
    tokens for the bulk check. Non-finite outputs give cosines of -1, an
    outlier fraction of 1 and infinite errors.
    """
    if tuple(actual.shape) != tuple(expected.shape):
        raise ValueError(
            f"Hidden states {tuple(actual.shape)} != reference {tuple(expected.shape)}"
        )
    a = actual.detach().to("cpu", torch.float64).reshape(-1, actual.shape[-1])
    b = expected.detach().to("cpu", torch.float64).reshape(-1, expected.shape[-1])
    if not bool(torch.isfinite(a).all()):
        return {
            "token_cosine_min": -1.0,
            "token_cosine_mean": -1.0,
            "outlier_token_fraction": 1.0,
            "rel_l2_error": float("inf"),
            "rel_l2_error_trimmed": float("inf"),
            "block_cosine_median_min": -1.0,
            "max_abs_error": float("inf"),
        }
    cosine = F.cosine_similarity(a, b, dim=-1)
    difference = a - b
    error = difference.norm(dim=-1)
    norm = b.norm(dim=-1)
    tokens = cosine.numel()
    dropped = max(1, int(tokens * trim))
    keep = (error / norm.clamp_min(1e-30)).argsort()[: tokens - dropped]
    if keep.numel() == 0:  # a single token: nothing is left to trim to
        keep = torch.arange(tokens)
    return {
        "token_cosine_min": cosine.min().item(),
        "token_cosine_mean": cosine.mean().item(),
        "outlier_token_fraction": (cosine < OUTLIER_COSINE).double().mean().item(),
        "rel_l2_error": (difference.norm() / b.norm()).item(),
        "rel_l2_error_trimmed": (error[keep].norm() / norm[keep].norm()).item(),
        "block_cosine_median_min": min(
            part.median().item() for part in _blocks(cosine, block)
        ),
        "max_abs_error": difference.abs().max().item(),
    }


# Calibrated accuracy limits (see EncoderWorkload.accuracy_defaults) at 512 and
# 8192 tokens; the bulk limit is interpolated in log2(length) in between.
_BULK_LIMIT = {512: 0.0075, 8192: 0.0095}


def accuracy_bounds(length: int) -> dict[str, tuple[str, float]]:
    """Accuracy limits for ``length`` tokens; see ``EncoderWorkload.accuracy_defaults``.

    Only the bulk limit depends on the length. It was calibrated at 512 and
    8192 tokens; other lengths get it interpolated in ``log2(length)``
    (clamped outside), an untested guess.
    """
    (short, low), (long, high) = sorted(_BULK_LIMIT.items())
    position = (math.log2(length) - math.log2(short)) / (
        math.log2(long) - math.log2(short)
    )
    bulk = low + (high - low) * min(max(position, 0.0), 1.0)
    return {
        # Bulk and local: tight.
        "rel_l2_error_trimmed": ("<=", round(bulk, 5)),
        "block_cosine_median_min": (">=", 0.9997),
        # Outliers: loose, and at most as many as the bulk check leaves out.
        "outlier_token_fraction": ("<=", TRIM_FRACTION),
        "token_cosine_min": (">=", 0.0),
        "rel_l2_error": ("<=", 0.10),
    }


def attention_gflop(
    length: int,
    hidden: int,
    layer_types: Sequence[str],
    window: int | None,
    *,
    block: int | None,
) -> dict[str, float]:
    """Attention GFLOP per forward: as computed (dense or banded) and as needed.

    Each layer computes ``4 * length * keys * hidden`` FLOPs (Q K^T and P V)
    for ``keys`` keys per query: ``length`` when dense, ``block + 2 * window``
    when banded, and ``2 * window + 1`` at least for a local layer.
    """
    computed = needed = 0.0
    for kind in layer_types:
        local = kind == "sliding_attention" and window is not None
        keys = length
        least = min(length, 2 * window + 1) if local else length
        if local and block is not None:
            keys = min(length, block + 2 * window)
        computed += 4.0 * length * keys * hidden
        needed += 4.0 * length * least * hidden
    return {"attention_gflop": computed / 1e9, "attention_gflop_needed": needed / 1e9}


@functools.lru_cache(maxsize=None)
def build_routes_supported() -> bool:
    """Whether this TensorRT has Global Performance Tuning build routes.

    Decided from the TensorRT Python API alone (no builder, no CUDA context),
    so every process, CPU-only ones included, offers the same knobs; a
    route the installation cannot build then fails, and that trial records
    why.
    """
    try:
        import tensorrt as trt
    except ImportError:
        return False
    config = getattr(trt, "IBuilderConfig", None)
    return (
        sys.platform != "win32"
        and getattr(trt, "_package_name", "") != "tensorrt_rtx"
        and hasattr(config, "build_route")
        and hasattr(config, "all_build_routes")
    )


@contextlib.contextmanager
def _fp16_accumulation(enabled: bool) -> Iterator[None]:
    """Set cuBLAS FP16 GEMMs to accumulate in FP16 (``enabled``), then restore."""
    matmul = torch.backends.cuda.matmul
    previous = matmul.allow_fp16_accumulation
    matmul.allow_fp16_accumulation = enabled
    try:
        yield
    finally:
        matmul.allow_fp16_accumulation = previous


def check_bench_values(values: Mapping[str, Any]) -> str:
    """Validate the ``bench.*`` values; return the backend.

    Unknown keys are errors, and so are the PyTorch-only keys (``bench.mode``,
    ``bench.attn``, ``bench.inductor``, ``bench.fp16_accumulation``) on a
    TensorRT build and the Inductor-only ones on an eager one.
    """
    unknown = sorted(
        key for key in values if key.startswith("bench.") and key not in _BENCH_KEYS
    )
    if unknown:
        raise ValueError(f"Unknown bench keys {unknown}; known: {list(_BENCH_KEYS)}")
    backend = values.get("bench.backend", "tensorrt")
    if backend not in ("tensorrt", "eager", "inductor"):
        raise ValueError(f"Unknown bench.backend {backend!r}")
    ignored = {
        "tensorrt": _TORCH_BENCH_KEYS,
        "eager": ("bench.mode", "bench.inductor"),
        "inductor": (),
    }[backend]
    misplaced = [key for key in ignored if key in values]
    if misplaced:
        raise ValueError(f"{misplaced} do not apply to bench.backend {backend!r}")
    precision = values.get("engine.precision")
    if values.get("bench.fp16_accumulation") and precision != "fp16":
        raise ValueError("bench.fp16_accumulation needs engine.precision 'fp16'")
    return backend


def check_hybrid_values(values: Mapping[str, Any]) -> str:
    """Validate ``kernels.inductor``; return its value (``"none"`` when unset).

    Its variants need CUDA graphs (the knob's ``requires``, checked here too
    for direct builds that bypass the session's checks).
    """
    hybrid = values.get("kernels.inductor", "none")
    if hybrid not in HYBRID_VARIANTS:
        raise ValueError(
            f"Unknown kernels.inductor {hybrid!r}; choose one of {HYBRID_VARIANTS}"
        )
    if hybrid != "none" and not values.get("runtime.cudagraphs"):
        raise ValueError(
            f"kernels.inductor={hybrid!r} needs runtime.cudagraphs=True: TensorRT "
            "11.4 reloads plugin modules on every enqueue outside CUDA graph "
            "replay, and regions are captured as one whole-model CUDA graph"
        )
    return hybrid


def check_generated_values(values: Mapping[str, Any]) -> str:
    """Validate ``kernels.generated``; return its value (``"none"`` when unset).

    ``region_generation.check_generated``: a generated value needs CUDA graphs
    and ``kernels.inductor="none"``.
    """
    import region_generation

    return region_generation.check_generated(
        values,
        GENERATED_KERNELS,
        conflict="both replace each layer's QKV split and rotary embedding",
    )


def _patch_flex_block_mask() -> None:
    """Let Dynamo trace Hugging Face's FlexAttention mask creation.

    transformers 5.18 calls ``create_block_mask(_compile=True)``, which this
    torch nightly deprecates with a warning that Dynamo cannot trace. The
    forward is compiled as a whole already, so drop the flag, as the warning
    recommends.
    """
    import transformers.masking_utils as masking_utils
    from torch.nn.attention.flex_attention import create_block_mask

    def create_block_mask_without_compile(
        *args: Any, _compile: bool = False, **kwargs: Any
    ) -> Any:
        return create_block_mask(*args, **kwargs)

    masking_utils.create_block_mask = create_block_mask_without_compile


# ----------------------------------------------------------------- workload
class EncoderWorkload(Workload):
    """Batch-1 forward of a Hugging Face encoder on ``length`` tokens.

    The baseline is the straightforward Torch-TensorRT compile of the model in
    FP16 with Hugging Face's default SDPA attention: ``torch.export`` and
    ``torch_tensorrt.dynamo.compile`` with default settings (``min_block_size``
    5, builder optimization level 3, attention as TensorRT attention layers,
    GEMMs with TensorRT's default FP16 accumulation), no CUDA graphs. FP16 is
    the precision of a typical deployment and of the PyTorch baselines; FP32
    remains a knob value, BF16 is not offered (it misses the accuracy limits).

    One deviation from a plain ``torch_tensorrt.compile(model)``: the global
    layers get no mask (see :class:`_Encoder`). With the all-true [length,
    length] mask that Hugging Face's mask creation exports for them, the FP16
    engine builds but crashes at its first run (segmentation fault in
    ``enqueueV3``, TensorRT 11.4, RTX 5090) as soon as two attention layers
    read that mask; FP32 runs. Dropping it is exact and is what Hugging Face
    does in eager mode.

    ``kernel_cache`` is the KernelGenerator cache that
    ``kernels.generated`` takes its kernels from
    (``region_generation.kernel_cache``).
    """

    def __init__(
        self,
        model: str,
        *,
        length: int,
        baseline: Mapping[str, Any] | None = None,
        kernel_cache: str | None = None,
    ) -> None:
        if length < 8:
            raise ValueError("EncoderWorkload needs length >= 8")
        import region_generation

        self.model_name = model
        self.length = int(length)
        self.name = f"{model.rsplit('/', 1)[-1]}-encoder-{length}"
        self._baseline = dict(baseline or {})
        # Where kernels.generated takes its kernels from (and caches new ones).
        self.kernel_cache = region_generation.kernel_cache(kernel_cache)
        self._config: Any = None
        self._ids: torch.Tensor | None = None

    def config(self) -> Any:
        if self._config is None:
            from transformers import AutoConfig

            self._config = AutoConfig.from_pretrained(self.model_name)
        return self._config

    def _local_window(self) -> int | None:
        """Half width of the sliding-window layers, or None without such layers.

        The attention knobs rewrite Hugging Face's bidirectional
        sliding-window attention; they are offered for ModernBERT, whose
        local layers see the keys at most ``config.sliding_window`` away.
        """
        config = self.config()
        if config.model_type != "modernbert":
            return None
        if "sliding_attention" not in (getattr(config, "layer_types", None) or ()):
            return None
        return int(config.sliding_window)

    # ------------------------------------------------------------ contract

    def knobs(self) -> Sequence[Knob]:
        knobs = [
            Knob(
                "engine.precision",
                ("fp16", "fp32"),
                "fp16",
                "Model weight and activation precision (fp32 runs without TF32: "
                "exact, at FP32 GEMM rates). bf16 is not offered: its 3 fewer "
                "mantissa bits raise the error of every token (bulk relative L2 "
                "0.017-0.019, over the accuracy limit).",
                ("precision", "dominant_kernel"),
            ),
            Knob(
                "engine.use_fp32_acc",
                (False, True),
                False,
                "With fp16, accumulate matrix multiplications in FP32 by inserting "
                "FP32 casts around every matmul (TensorRT's FP16 GEMMs otherwise "
                "accumulate in FP16). With decompose_attention the attention "
                "matmuls are cast too, so TensorRT can no longer fuse the "
                "attention and runs it unfused in FP32. The build route "
                "-cask:fp16_acc=off gets FP32 GEMM accumulation without the casts.",
                ("precision",),
            ),
            Knob(
                "engine.decompose_attention",
                (False, True),
                False,
                "Torch-TensorRT decomposes scaled_dot_product_attention into "
                "matmul, masked softmax and matmul instead of a TensorRT attention "
                "layer; TensorRT may fuse the pattern again. Changes the attention "
                "kernels and their rounding.",
                ("dominant_kernel",),
            ),
        ]
        window = self._local_window()
        if window is not None:
            knobs.append(
                Knob(
                    "workload.local_attention",
                    tuple(LOCAL_ATTENTION),
                    "dense",
                    "Sliding-window layers (a token sees keys at most "
                    f"{window} positions away). 'dense' masks a full [length, "
                    "length] attention; 'bandedB' splits the queries into blocks "
                    f"of B tokens that attend only to the B + {2 * window} keys "
                    "around them with a static mask (exact): local attention work "
                    f"drops from length^2 to length * (B + {2 * window}). "
                    "Smaller blocks compute fewer masked scores but run smaller "
                    "attention batches.",
                    ("dominant_kernel",),
                )
            )
            knobs.append(
                Knob(
                    "graph.fold_attention_masks",
                    (False, True),
                    False,
                    "Fold the attention masks into engine constants (Torch-TensorRT "
                    "keeps the arange ops that build them in the engine by "
                    "default), so TensorRT sees the mask pattern while it builds "
                    "the engine; the dense band mask then costs length^2 bytes of "
                    "engine weights.",
                    ("dominant_kernel",),
                )
            )
            layers = int(self.config().num_hidden_layers)
            knobs.append(
                Knob(
                    "kernels.inductor",
                    HYBRID_VARIANTS,
                    "none",
                    "Hand each layer's QKV split and rotary embedding to Inductor: "
                    "query and key unbound from the Wqkv output, the cos/sin "
                    "tables, the RoPE products in FP32 and one rounding back. "
                    "Query and key leave the subgraph token-major; the value, the "
                    "transposes for the attention, the attention and every GEMM "
                    "stay in TensorRT. 'rope-kernel': at build time torch.compile "
                    "(max-autotune-no-cudagraphs) compiles that subgraph, and the "
                    "single Triton kernel Inductor generates, with the launch "
                    "configuration its autotuner chose, runs inside the TensorRT "
                    "engine as a torch_tensorrt.region.fused_kernel plugin: one "
                    f"plugin layer per transformer layer ({layers}), no engine "
                    "split. TensorRT copies each plugin output into the layout its "
                    "attention reads, and the plugin can change how TensorRT fuses "
                    "the neighbouring layers. With a cold cache Inductor's "
                    "autotuner can pick another launch configuration from build to "
                    "build, which changes the kernel's speed but not its results. "
                    "'rope-kernel-eager-numerics': the same, with the kernel "
                    "generated under Inductor's emulate_precision_casts, which "
                    "keeps eager's FP16 rounding of the cos/sin tables: query and "
                    "key are then Hugging Face's, and the engine keeps the "
                    "numerics of the build without this knob. "
                    "'inductor-regions': execute_in_torch(backend='inductor') "
                    "around the same subgraph in every layer, compiled by Inductor "
                    "at the first call in its default mode outside TensorRT: "
                    f"{layers} regions, {2 * layers} TensorRT/PyTorch boundaries "
                    f"and {layers + 1} engines, captured as one whole-model CUDA "
                    "graph; the separate engines and the whole-graph capture need "
                    "more GPU memory than one engine, the more so the longer the "
                    "sequence; and such a build cannot be saved "
                    "(torch_tensorrt.save refuses execute_in_torch regions whose "
                    "backend is 'inductor'), so a session that keeps one writes "
                    "its config but no artifact. All build through strict export "
                    "(torch_tensorrt.compile(strict=True)) without module "
                    "offloading, and need CUDA graphs: TensorRT 11.4 reloads "
                    "plugin modules on every enqueue outside CUDA graph replay, "
                    "and each boundary costs host work. Numerics: by default "
                    "Inductor computes the cos/sin tables in FP32 where Hugging "
                    "Face and the TensorRT build round them to FP16, so with "
                    "'rope-kernel' and 'inductor-regions' query and key differ in "
                    "the last FP16 bit for part of the elements. That moves the "
                    "few ill-conditioned tokens like any other change of rounding, "
                    "for better on some inputs and for worse on others: on the "
                    "512-token bench input 'rope-kernel' loses ill-conditioned "
                    "tokens that the plain build keeps, and its bulk error comes "
                    "close to the accuracy limit.",
                    ("dominant_kernel", "launch_overhead"),
                    {"runtime.cudagraphs": True},
                )
            )
            knobs.append(self._generated_knob(layers))
        knobs += self._build_route_knobs()
        knobs += [knob for knob in STANDARD_KNOBS if knob.name in _STANDARD]
        return tuple(knobs)

    @staticmethod
    def _generated_knob(layers: int) -> Knob:
        import region_generation

        return Knob(
            "kernels.generated",
            GENERATED_KERNELS,
            "none",
            region_generation.GENERATED_INTRO + " 'none': no generated "
            "kernels. 'split-rope': each layer's QKV split and rotary embedding "
            "(the subgraph kernels.inductor hands to Inductor: query and key "
            "from the Wqkv output and the layer type's RoPE inv_freq, the "
            "cos/sin tables, the RoPE products and one rounding back) runs as a "
            f"generated kernel, one plugin per transformer layer ({layers}), "
            "no engine split; identical layers share one kernel. The value, "
            "the transposes for the attention, the attention and every GEMM "
            "stay in TensorRT, which copies each plugin output into the layout "
            "its attention reads. Requires runtime.cudagraphs=True (TensorRT "
            "11.4 reloads plugin modules on every enqueue outside CUDA graph "
            "replay) and kernels.inductor='none' (both replace the same "
            "subgraph). Builds through strict export "
            "(torch_tensorrt.compile(strict=True)) without module offloading. "
            + region_generation.CACHE_NOTE,
            ("dominant_kernel", "launch_overhead"),
            {**region_generation.PLUGIN_REQUIRES, "kernels.inductor": "none"},
            prepare=GENERATED_KERNELS[1:],
        )

    @staticmethod
    def _build_route_knobs() -> tuple[Knob, ...]:
        if not build_routes_supported():
            return ()
        return (
            Knob(
                "engine.build_route",
                BUILD_ROUTES,
                "",
                "TensorRT Global Performance Tuning build route. "
                "'-cask:fp16_acc=off' forbids FP16-accumulation GEMM kernels "
                "outside fusions (FP32 accumulation for 88 of 110 GEMM launches "
                "at 512 tokens, 44 of 109 at 8192); adding "
                "'-cask_fusion:fp16_acc=off' covers the fused GEMMs too (all "
                "109 at 8192; no change at 512). Both keep attention fused, "
                "unlike use_fp32_acc, and are more accurate on the bulk of the "
                "tokens. '-tactic:splitk=off' drops split-K GEMM tactics (other "
                "GEMM tiles in the dense baseline engine; engines built with "
                "decompose_attention come out bitwise unchanged).",
                ("precision", "dominant_kernel"),
            ),
        )

    def baseline(self) -> dict[str, Any]:
        return dict(self._baseline)

    def timing(self) -> Timing:
        calls = 2 if self.length >= 4096 else 10
        return Timing(warmup=3, rounds=15, calls=calls, profile_calls=2)

    def accuracy_defaults(self) -> dict[str, tuple[str, float]]:
        """A tight check of the bulk of the tokens, a loose one of a few outliers.

        The final hidden state of a few tokens (" the", ","; 2 of 512, up to
        17 of 8192) is ill-conditioned: any perturbation of the forward moves
        it to a token cosine between 0.17 and 0.9, be it FP16 rounding, FP16
        instead of FP32 GEMM accumulation, other attention kernels or, in
        FP32, a local window one token wider. (They are not the tokens whose
        residual stream reaches |x| ~ 3.7e4: the worst one at 512 tokens peaks
        at 239.) Which of them a correct FP16 build loses changes with every
        kernel choice: Inductor recompiles of
        one config gave a smallest cosine of 0.77 to 0.997 at 512 tokens. So,
        at 512 / 8192 tokens (RTX 5090 measurements):

        * Bulk, tight: ``rel_l2_error_trimmed`` (the worst 1% of the tokens
          left out) <= 0.0075 / 0.0095. Correct FP16 builds 0.0019-0.0062 /
          0.0019-0.0082 (the largest: FP16 accumulation at 512, the dense
          TensorRT engines at 8192); FP32 with the local window one token
          narrower or wider 0.0087-0.0090 / 0.0109-0.0123; BF16 eager 0.0165 /
          0.0186.
        * Local, tight: ``block_cosine_median_min`` (64-token blocks) >=
          0.9997. Correct FP16 builds >= 0.99995 / 0.99987; FP32 with one
          64x64 tile of the band mask dropped in every local layer 0.9982-
          0.9989 / 0.9987-0.9994, which the bulk check misses at 8192 tokens
          (0.0035-0.0071).
        * Outliers, loose: at most 1% of the tokens below cosine 0.9
          (``outlier_token_fraction``; FP16 builds <= 0.4% / 0.21%), none
          below 0 (FP16 builds >= 0.17), and ``rel_l2_error`` over all tokens
          <= 0.10 (FP16 builds <= 0.070 / 0.040), which also bounds the
          magnitude of the outliers' error.

        Broken models miss by far: no local mask gives a relative L2 error of
        0.62 / 0.98, the global RoPE base in the local layers 0.30 / 0.38.

        Hybrid builds that host Inductor's RoPE kernel
        (``kernels.inductor=rope-kernel``) pass at 512 tokens but outside the
        FP16 ranges above (bulk 0.0062-0.0068, ``rel_l2_error`` 0.070,
        smallest cosine 0.16-0.18): they lose the two ill-conditioned tokens
        (" the" at 271 and 287) that FP16 GEMM accumulation in PyTorch eager
        loses as well (0.0062, 0.17). Inductor's FP32 cos/sin tables alone
        cause it: with Hugging Face's FP16 tables
        (``rope-kernel-eager-numerics``) the same engines are bitwise equal to
        the config's. On seven other 512-token windows of the text the
        rope-kernel builds gave a bulk error of 0.0038-0.0059 and the configs
        without it 0.0039-0.0063, with a smallest cosine down to 0.16 too.
        """
        return accuracy_bounds(self.length)

    def input_ids(self) -> torch.Tensor:
        """The fixed input: [CLS], ``TEXT`` repeated, [SEP]; ``length`` ids."""
        if self._ids is None:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            text = tokenizer(TEXT, add_special_tokens=False)["input_ids"]
            first = tokenizer.cls_token_id
            if first is None:
                first = tokenizer.bos_token_id
            last = tokenizer.sep_token_id
            if last is None:
                last = tokenizer.eos_token_id
            ids = sequence_ids(text, self.length, first, last)
            self._ids = torch.tensor(ids, dtype=torch.int64)
        return self._ids

    def reference(self) -> Any:
        """FP32 eager ``last_hidden_state`` (SDPA attention, no TF32).

        Computed on the GPU with Hugging Face's own forward and masks; returned
        on the CPU.
        """
        from transformers import AutoModel

        previous = torch.get_float32_matmul_precision()
        torch.set_float32_matmul_precision("highest")
        try:
            with torch.inference_mode():
                model = (
                    AutoModel.from_pretrained(
                        self.model_name, attn_implementation="sdpa", dtype=torch.float32
                    )
                    .eval()
                    .cuda()
                )
                ids = self.input_ids().cuda().unsqueeze(0)
                hidden = model(input_ids=ids).last_hidden_state.float().cpu()
        finally:
            torch.set_float32_matmul_precision(previous)
        del model
        torch.cuda.empty_cache()
        return hidden

    def compare(self, outputs: Any, reference: Any) -> dict[str, float]:
        return hidden_state_metrics(outputs, reference)

    # -------------------------------------------------------------- build

    def _load(self, precision: str, attn: str, device: torch.device) -> Any:
        from transformers import AutoModel

        return (
            AutoModel.from_pretrained(
                self.model_name, attn_implementation=attn, dtype=_DTYPES[precision]
            )
            .eval()
            .to(device)
        )

    def _details(self, model: Any, values: Mapping[str, Any]) -> dict[str, Any]:
        """FLOP accounting and the attention lowering, for describe()."""
        config = model.config
        linear = sum(
            m.in_features * m.out_features
            for m in model.modules()
            if isinstance(m, torch.nn.Linear)
        )
        window = self._local_window()
        block = LOCAL_ATTENTION[values.get("workload.local_attention", "dense")]
        layer_types = getattr(config, "layer_types", None) or (
            ["full_attention"] * config.num_hidden_layers
        )
        flops = attention_gflop(
            self.length, config.hidden_size, layer_types, window, block=block
        )
        return {
            "length": self.length,
            "linear_gflop": round(2.0 * self.length * linear / 1e9, 3),
            **{key: round(value, 3) for key, value in flops.items()},
            "local_attention": values.get("workload.local_attention", "dense"),
        }

    def _wrapper(
        self, model: Any, values: Mapping[str, Any], *, masks: bool = True
    ) -> _Encoder:
        """The exported or compiled module; ``masks=False`` keeps the model's own."""
        return _Encoder(
            model,
            window=self._local_window() if masks else None,
            local_attention=values.get("workload.local_attention", "dense"),
        )

    def build(self, values: Mapping[str, Any]) -> Candidate:
        backend = check_bench_values(values)
        if backend == "tensorrt":
            return self._build_tensorrt(values)
        return self._build_torch(values, backend)

    def _build_tensorrt(self, values: Mapping[str, Any]) -> Candidate:
        import torch_tensorrt

        precision = values["engine.precision"]
        hybrid = check_hybrid_values(values)
        generated = check_generated_values(values)
        # Regions and generated kernels are captured only by strict export.
        strict = hybrid != "none" or generated != "none"
        device = torch.device("cuda:0")
        model = self._load(precision, "sdpa", device)
        details = self._details(model, values)
        if hybrid != "none":
            details.update(self._install_hybrid(model, hybrid))
        elif generated != "none":
            details.update(self._install_generated(model))
        wrapper = self._wrapper(model, values).eval()
        ids = self.input_ids().to(device).unsqueeze(0)
        settings: dict[str, Any] = {
            "device": device,
            "disable_tf32": precision == "fp32",
            "use_fp32_acc": precision == "fp16" and values["engine.use_fp32_acc"],
            "optimization_level": values["engine.optimization_level"],
            "tiling_optimization_level": values.get(
                "engine.tiling_optimization_level", "none"
            ),
            # Regions build without offloading (strict export keeps references
            # to the parameters); the strict path deallocates them below.
            "offload_module_to_cpu": not strict,
        }
        if values.get("engine.decompose_attention"):
            settings["decompose_attention"] = True
        if values.get("engine.build_route"):
            settings["build_route"] = values["engine.build_route"]
        if values.get("graph.fold_attention_masks"):
            settings["disabled_constant_fold_exclusions"] = {"attention_mask_arange"}
        if not strict:
            with torch.no_grad():
                program = torch.export.export(wrapper, (ids,), strict=False)
                module = torch_tensorrt.dynamo.compile(
                    program, arg_inputs=[ids], **settings
                )
            del program, wrapper, model
        else:
            from torch_tensorrt.dynamo.utils import deallocate_module

            with torch.no_grad():
                # Regions are captured only by strict export through
                # torch_tensorrt.compile.
                module = torch_tensorrt.compile(
                    wrapper, ir="dynamo", arg_inputs=[ids], strict=True, **settings
                )
            deallocate_module(wrapper)
            del wrapper, model
        gc.collect()
        torch.cuda.empty_cache()
        if hybrid == "inductor-regions":
            # Inductor compiles the regions at their first call: do it here so
            # build_s includes it, as it includes the engine build.
            start = time.perf_counter()
            with torch.inference_mode():
                module(ids)
            torch.cuda.synchronize()
            details["region_compile_s"] = round(time.perf_counter() - start, 2)
        return _EncoderCandidate(
            module,
            ids,
            backend="tensorrt",
            details=details,
            cudagraphs=values["runtime.cudagraphs"],
            pre_allocated_outputs=values["runtime.pre_allocated_outputs"],
        )

    def kernel_generator(self) -> Any:
        """The KernelGenerator of ``kernels.generated``: the shared settings
        (``region_generation.make_generator``) with the region's notes, the
        same in every build."""
        import modernbert_kernels
        import region_generation

        return region_generation.make_generator(
            self.kernel_cache, modernbert_kernels.GENERATION_NOTES
        )

    def _install_generated(self, model: Any) -> dict[str, Any]:
        """Patch ``model`` for ``kernels.generated="split-rope"``; return what the build reports.

        Every layer's ``split_rope_{layer}`` scope gets :meth:`kernel_generator`,
        which writes, checks and caches its kernel when the model compiles, or
        reuses the cached one.
        """
        import modernbert_kernels
        import region_generation

        generator = self.kernel_generator()
        layers = modernbert_kernels.install(
            model, modernbert_kernels.GENERATED, generator=generator
        )
        return {
            "kernels_generated": modernbert_kernels.GENERATED,
            "generated_layers": layers,
            **region_generation.describe(generator),
        }

    def _install_hybrid(self, model: Any, hybrid: str) -> dict[str, Any]:
        """Patch ``model`` for ``kernels.inductor``; return what the build reports."""
        import modernbert_kernels

        details: dict[str, Any] = {"inductor_kernels": hybrid}
        kernel = None
        if hybrid in modernbert_kernels.KERNEL_VARIANTS:
            config = model.config
            heads = int(config.num_attention_heads)
            # One kernel serves every layer (inv_freq is a plugin input, the
            # RoPE scaling is compiled in): the layer types must agree.
            rope = modernbert_kernels.layer_rope(model)
            if len({(scaling, freq.dtype) for scaling, freq in rope.values()}) > 1:
                raise ValueError(
                    f"kernels.inductor={hybrid} hosts one kernel for every layer, "
                    "but the layer types' RoPE scaling or inv_freq dtype differ: "
                    f"{ {key: (s, f.dtype) for key, (s, f) in rope.items()} }"
                )
            scaling, inv_freq = rope[model.layers[0].attention_type]
            start = time.perf_counter()
            kernel = modernbert_kernels.generate_rope_kernel(
                self.length,
                heads,
                int(config.hidden_size) // heads,
                scaling=scaling,
                dtype=next(model.parameters()).dtype,
                inv_freq_dtype=inv_freq.dtype,
                eager_numerics=modernbert_kernels.KERNEL_VARIANTS[hybrid],
            )
            details["inductor_generate_s"] = round(time.perf_counter() - start, 2)
            details.update(kernel.describe())
            details["plugin_kernel"] = modernbert_kernels.check_resources(kernel)
        details["hybrid_layers"] = modernbert_kernels.install(model, hybrid, kernel)
        return details

    def _build_torch(self, values: Mapping[str, Any], backend: str) -> Candidate:
        for knob in ("kernels.inductor", "kernels.generated"):
            if values.get(knob, "none") != "none":
                raise ValueError(
                    f"{knob} builds TensorRT engines with other kernels; use it "
                    "with bench.backend='tensorrt'"
                )
        mode = values.get("bench.mode", "default")
        attn = values.get("bench.attn", "sdpa")
        fp16_accumulation = bool(values.get("bench.fp16_accumulation", False))
        banded = values.get("workload.local_attention", "dense") != "dense"
        if banded and attn != "sdpa":
            raise ValueError(
                "workload.local_attention replaces the attention implementation; "
                "use it with bench.attn='sdpa'"
            )
        if backend == "inductor":
            import torch._inductor.config as inductor_config

            for key, value in dict(values.get("bench.inductor") or {}).items():
                setattr(inductor_config, key, value)
        if attn == "flex_attention":
            _patch_flex_block_mask()
        device = torch.device("cuda:0")
        # The flag must hold while Inductor autotunes and while the candidate
        # runs; the candidate restores it when it closes.
        stack = contextlib.ExitStack()
        stack.enter_context(_fp16_accumulation(fp16_accumulation))
        try:
            model = self._load(values["engine.precision"], attn, device)
            details = self._details(model, values)
            # FlexAttention needs block masks and eager attention additive
            # masks: the model builds those itself (and, not exported, no
            # global mask).
            wrapper = self._wrapper(model, values, masks=attn == "sdpa").eval()
            run: Any = wrapper
            if backend == "inductor":
                run = torch.compile(wrapper, mode=mode, fullgraph=True, dynamic=False)
            candidate = _EncoderCandidate(
                run,
                self.input_ids().to(device).unsqueeze(0),
                backend=backend,
                details={
                    **details,
                    "mode": mode if backend == "inductor" else None,
                    "attn": attn,
                    "fp16_accumulation": fp16_accumulation,
                },
                mark_step=backend == "inductor" and mode in _CUDAGRAPH_MODES,
                stack=stack,
            )
            # Compile (and record CUDA graphs) here, so build_s includes it as
            # it includes the TensorRT engine build.
            start = time.perf_counter()
            for _ in range(3):
                candidate()
            torch.cuda.synchronize()
        except BaseException:
            stack.close()
            raise
        candidate.details["compile_s"] = round(time.perf_counter() - start, 2)
        return candidate


class _EncoderCandidate(Candidate):
    def __init__(
        self,
        module: Any,
        input_ids: torch.Tensor,
        *,
        backend: str,
        details: dict[str, Any],
        cudagraphs: bool = False,
        pre_allocated_outputs: bool = False,
        mark_step: bool = False,
        stack: contextlib.ExitStack | None = None,
    ) -> None:
        self.module = module
        self.input_ids = input_ids
        self.backend = backend
        self.details = details
        self.cudagraphs = cudagraphs
        self.pre_allocated_outputs = pre_allocated_outputs
        self.mark_step = mark_step
        self.units = 1
        self.items = int(input_ids.shape[0])
        # Global settings the build changed (FP16 accumulation); close()
        # restores them after the runtime contexts entered in setup().
        self._stack = stack if stack is not None else contextlib.ExitStack()
        self._run: Any = module

    def setup(self) -> None:
        if self.backend != "tensorrt":
            return
        import torch_tensorrt

        if self.pre_allocated_outputs:
            self._stack.enter_context(
                torch_tensorrt.runtime.enable_pre_allocated_outputs(self.module)
            )
        if self.cudagraphs:
            self._run = self._stack.enter_context(
                torch_tensorrt.runtime.enable_cudagraphs(self.module)
            )

    def __call__(self) -> torch.Tensor:
        if self.backend == "tensorrt":
            with torch.inference_mode():
                return self._run(self.input_ids)
        with torch.no_grad():
            if self.mark_step:
                torch.compiler.cudagraph_mark_step_begin()
            return self._run(self.input_ids)

    def describe(self) -> dict[str, Any]:
        if self.backend == "tensorrt":
            from torch_tensorrt.autotune import describe_module

            return {**describe_module(self.module), **self.details}
        return {
            "trt_engines": 0,
            "torch_partitions": 0,
            "plugin_regions": 0,
            "backend": self.backend,
            **self.details,
        }

    def export(self, directory: Path) -> list[str]:
        """Save the compiled encoder as ``encoder.ep``.

        ``kernels.inductor`` builds with Inductor regions
        (:data:`REGION_VARIANTS`) cannot be saved: the exporter rejects
        ``execute_in_torch`` regions whose backend is not ``eager``, before it
        writes anything. That rejection is reported as ``NotImplementedError``,
        the contract's "cannot export", naming the value; other errors
        propagate unchanged.
        """
        if self.backend != "tensorrt":
            return []
        import torch_tensorrt
        from torch_tensorrt import region

        path = Path(directory) / "encoder.ep"
        try:
            torch_tensorrt.save(
                self.module, str(path), arg_inputs=[self.input_ids], retrace=False
            )
        except region.RegionError as error:
            hybrid = self.details.get("inductor_kernels")
            if hybrid not in REGION_VARIANTS:
                raise
            raise NotImplementedError(
                f"kernels.inductor={hybrid!r} builds cannot be saved: their "
                "execute_in_torch(backend='inductor') regions run code that "
                "Inductor compiles at run time, and torch_tensorrt.save cannot "
                "preserve that backend. Rebuild the config instead (the "
                "rope-kernel values host Inductor's kernel in the engine and add "
                "no region)."
            ) from error
        return [path.name]

    def close(self) -> None:
        self._stack.close()
        self._run = self.module
