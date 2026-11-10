# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Hugging Face causal-LM prefill workload for ``torch_tensorrt.autotune`` sessions.

The measured unit is one batch-1 prefill: a forward pass over a fixed
``length``-token prompt that returns the logits at ``positions`` evenly spaced
positions, the last one included (the time-to-first-token computation; the
LM head runs on those rows only, through Hugging Face's ``logits_to_keep``
index tensor). The logits are compared with an FP32 eager reference at the
same positions: top-1 agreement, mean and largest KL divergence, and the
largest logit error. The default limits (:meth:`PrefillWorkload.accuracy_defaults`)
gate the top-1 agreement, the mean KL and the largest KL at any one position,
so an error confined to a few positions (the last one, say) cannot hide in
the mean.

Prompt (``prompt``): ``python-docs`` (default) is the start of the Python
language reference that ships with CPython (``pydoc_data.topics``, all topics
in sorted order, each paragraph once): long, natural English that never
repeats, so attention that drops or wraps distant context changes the
logits. ``dickens`` repeats the decode workload's paragraph, which makes most
tokens copyable from the previous repeat and the check far less sensitive.
A Gemma tokenizer adds its BOS token; the prompt is the first ``length``
tokens.

Workload knobs (Hugging Face attention lowering for export; see
:func:`prefill_attention`):

* ``workload.masks``: ``hf`` exports Hugging Face's own masks. Under
  ``torch.export`` they always materialize, and they AND the causal (and
  sliding-window) pattern with a packed-sequence term computed from the
  position ids (a cumsum), which TensorRT's attention mask analysis cannot
  read: every attention layer then runs dense over the full ``length x length``
  square. ``positions`` builds the masks from position ranges only; TensorRT
  recognizes them and skips the fully masked tiles.
* ``workload.global_attention`` (full-attention layers): ``mask`` passes the
  mask; ``causal`` calls SDPA with ``is_causal=True`` and native GQA (no mask
  tensor, keys and values not repeated); ``torch-flash`` runs PyTorch's
  flash-attention kernel (causal, native GQA) in a custom operator outside
  TensorRT, which splits the engine at every such layer.
* ``workload.local_attention`` (sliding-window layers, offered when the
  prompt is longer than the window): ``mask`` passes the mask; ``banded-B``
  splits the queries into chunks of ``B`` that attend to the window of keys
  before them as one batched attention (``B + window`` keys per chunk, no
  ``length x length`` mask); ``torch-flash`` runs PyTorch's flash kernel with
  its sliding window outside TensorRT.
  When neither mode uses a mask, ``workload.masks`` has no effect: Hugging
  Face then builds no mask at all (:meth:`PrefillWorkload.attention`).
* ``engine.precision``: ``bf16`` or ``fp16`` weights and activations.
  TensorRT builds FP16 GEMMs with FP16 accumulation, about 1.5x the BF16
  (FP32-accumulation) throughput on GeForce GPUs. FP16's range is +-65504:
  Gemma 3's residual stream peaks at about 61,000 at the BOS token (the
  same for every prompt: position 0 attends only to itself) and at about
  48,800 on the content tokens of these prompts, so FP16 must pass the
  accuracy limits to be kept (an overflow shows as an infinite KL).
* ``engine.build_route`` (TensorRT with Global Performance Tuning; FP16
  only): ``-cask:fp16_acc=off`` forbids the FP16-accumulation GEMM kernels,
  so an FP16 engine accumulates in FP32 (slower, more accurate). The other
  one-knob routes tried here built the same kernels with bit-identical
  outputs on an RTX 5090 with TensorRT 11.4 and are not offered.
* ``kernels.inductor`` (TensorRT only; default ``none``): hybrid builds that
  hand the attention of some layers (``-global``: the full-attention layers;
  otherwise every layer) to the kernels Inductor runs for it, with the rest
  of the config unchanged. Inductor's own builds of this workload run
  attention as PyTorch's flash-attention kernel (FA2) through the
  ``torch-flash`` lowering, as an extern call. Every value replaces
  ``workload.global_attention``/``workload.local_attention`` for the layers
  it moves (their values are ignored there). ``attention-fa2-*`` set the
  ``torch-flash`` modes for those layers: they build the same model as the
  configs with ``workload.global_attention="torch-flash"`` (``-global``) or
  both attention knobs at ``"torch-flash"`` (``-all``), under another name.
  ``attention-inductor*`` put the same operator in a
  ``torch_tensorrt.region.execute_in_torch(backend="inductor")`` region per
  layer, which Inductor compiles; ``attention-rope-inductor*`` widen that
  region to the q/k RMSNorms and the rotary embedding before the attention,
  for which Inductor generates Triton kernels, and run the q/k/v projections
  as one GEMM whose output the region reads (:func:`install_attention_regions`;
  models with q/k norms only). ``rope-plugin`` moves no attention and keeps
  the engine whole: Inductor's own generated q/k-norm + RoPE kernels
  (:func:`inductor_rope_kernels`) run inside the engine as
  ``torch_tensorrt.region.fused_kernel`` QDP plugins after the same fused
  q/k/v GEMM, and the attention stays in TensorRT under the config's modes.
  The attention values split the TensorRT engine at every moved layer (two
  boundaries per layer), and TensorRT 11.4 reloads a QDP plugin's module on
  every enqueue outside a CUDA graph, so all values require
  ``runtime.cudagraphs``, which captures the whole graph (engines and PyTorch
  parts) in one CUDA graph. With ``-global`` values the window layers keep
  their masks, and Torch-TensorRT's partitioner builds all of those masks in
  the first engine and passes them to the later engines as ``[1, 1, S, S]``
  inputs. Regions and plugins need strict export: those builds compile with
  ``torch_tensorrt.compile(..., ir="dynamo", strict=True)`` (see
  :func:`release_strict_export_traces`) and run the first call (Inductor's
  compilation) during the build. Region builds cannot be saved:
  ``torch_tensorrt.save`` rejects ``execute_in_torch`` regions whose backend
  is not ``eager``, so their candidates' ``export`` raises
  (:meth:`_TensorRTPrefill.export`) and a session that keeps one writes its
  config but no artifact; the ``attention-fa2-*`` and ``rope-plugin`` builds
  save as usual.
* ``kernels.generated`` (TensorRT only, models with q/k norms; default
  ``none``): ``qk-norm-rope`` runs the q and k RMSNorm + rotary embedding
  regions of every layer (the ``fused_kernel`` scopes ``rope-plugin`` fills
  with Inductor's kernels, after the same fused q/k/v GEMM) with kernels that
  PR 4914's ``torch_tensorrt.region.KernelGenerator`` writes during the build
  (``region_generation.py``; see :func:`_attention_plugin_forward`). It needs
  ``runtime.cudagraphs`` and ``kernels.inductor="none"``. Generated kernels
  are cached in ``kernel_cache`` (``region_generation.kernel_cache``), keyed
  by the engine options; a session prepares the value once, before its search.

plus the standard builder and runtime knobs. The baseline is the documented
Torch-TensorRT usage: ``torch.export.export`` of the Hugging Face model with
SDPA attention in its native precision, compiled by
``torch_tensorrt.dynamo.compile(program, arg_inputs=[input_ids])`` with
default settings.

Head-to-head benchmarks (``bench_compare.py``) reuse the workload through
``bench.*`` values, which are not knobs (a session never sets them):
``bench.backend`` ``"tensorrt"``
(default) builds the config as above; ``"eager"`` and ``"inductor"`` run the
same Hugging Face forward eagerly or compiled by ``torch.compile(mode=
bench.mode, fullgraph=True, dynamic=False)`` (``bench.mode``: a torch.compile
mode, default ``"default"``). ``bench.attn`` is the Hugging Face
``attn_implementation`` (``"sdpa"``, ``"flex_attention"`` for FlexAttention's
block-sparse kernels, ``"eager"``); without it the config's attention knobs
apply, as for TensorRT (``"sdpa"`` for their defaults), so PyTorch can use
the same lowerings (``torch-flash``: flash attention with a sliding window).
``bench.inductor`` is a dict of ``torch._inductor.config`` overrides, such as
``{"coordinate_descent_tuning": True}``. ``bench.fp16_accumulation`` (this
workload's own key, FP16 only) sets
``torch.backends.cuda.matmul.allow_fp16_accumulation``, so FP16 cuBLAS GEMMs
accumulate in FP16 as TensorRT's FP16 GEMM tactics do. They use the same
tokens, precision (``engine.precision``), outputs and reference as the
TensorRT candidates; compilation and CUDA-graph recording happen in
``build``. With ``bench.attn="flex_attention"``, eager keeps Hugging Face's
block-mask creation as is (``create_block_mask(_compile=True)`` compiles the
mask builder); Inductor drops that flag, which Dynamo cannot trace inside the
compiled forward, and compiles the mask creation with the forward. Unknown
``bench.*`` keys, and keys that do not apply to the chosen backend, are
errors (:func:`bench_settings`). Global settings a build changes (FP16
accumulation, Inductor options, the block-mask patch) are restored when the
candidate closes.

Prefill is compute-bound, so candidates do not declare ``bytes_per_unit``
(the memory-bandwidth roofline would always fire); their structure reports
``gflop_per_unit`` (linear layers, causal and windowed attention without
masked work, and the LM head rows) and ``compute_floor_ms`` at the GPU's
measured dense GEMM throughput instead: BF16 with FP32 accumulation, and
``compute_floor_ms_fp16_acc`` for FP16 GEMMs that accumulate in FP16 (which
TensorRT picks for FP16 engines; about 1.5x the BF16 rate on GeForce GPUs).
"""

from __future__ import annotations

import contextlib
import dataclasses
import functools
import gc
import hashlib
import sys
import time
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch_tensorrt import region
from torch_tensorrt.autotune import STANDARD_KNOBS, Candidate, Knob, Timing, Workload

PROMPTS = ("python-docs", "dickens")
BACKENDS = ("tensorrt", "eager", "inductor")
TORCH_COMPILE_MODES = (
    "default",
    "reduce-overhead",
    "max-autotune",
    "max-autotune-no-cudagraphs",
)
MASKS = ("hf", "positions")
GLOBAL_ATTENTION = ("mask", "causal", "torch-flash")
LOCAL_ATTENTION = ("mask", "banded-128", "banded-512", "torch-flash")
# kernels.inductor values -> (layers, route); see InductorPlacement.
# Ordered by route (the operator between engines, the operator in an Inductor
# region, q/k-norm + RoPE + the operator in one; then Inductor's q/k-norm +
# RoPE kernels inside the engine), then by layer set (the full-attention
# layers, every layer).
INDUCTOR_KERNELS = {
    "none": ("", ""),
    "attention-fa2-global": ("global", "torch-flash"),
    "attention-fa2-all": ("all", "torch-flash"),
    "attention-inductor-global": ("global", "region"),
    "attention-inductor": ("all", "region"),
    "attention-rope-inductor-global": ("global", "region-rope"),
    "attention-rope-inductor": ("all", "region-rope"),
    "rope-plugin": ("all", "plugin"),
}
# Hugging Face model types whose attention install_attention_regions rewrites.
_ROPE_REGION_MODELS = ("gemma3_text", "qwen3")
# kernels.generated values: "qk-norm-rope" generates the kernels of the q and
# k RMSNorm + rotary embedding regions (_attention_plugin_forward).
GENERATED_KERNELS = ("none", "qk-norm-rope")
# torch.compile mode that generates the rope-plugin kernels: the head-to-head's
# Inductor setting, without Inductor's CUDA graphs (TensorRT replays them).
ROPE_PLUGIN_COMPILE_MODE = "max-autotune-no-cudagraphs"
# torch.compile modes that replay the forward from CUDA graphs.
_CUDAGRAPH_MODES = ("reduce-overhead", "max-autotune")
_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
# The head-to-head keys build() understands (see bench_settings).
_BENCH_KEYS = (
    "bench.backend",
    "bench.mode",
    "bench.attn",
    "bench.inductor",
    "bench.fp16_accumulation",
)
# TensorRT Global Performance Tuning routes (11.1+, not TensorRT-RTX) that
# change this workload's engines. Tried on Gemma 3 1B prefill (RTX 5090,
# TensorRT 11.4) and left out: "-mha_fusion:v2:force_ws=on",
# "-mha_fusion:v2:force_flash_tiled=on", "-mha_fusion:hide_load_latency=on"
# and "-tactic:splitk=off" built the same attention kernels with
# bit-identical logits and latency within 1%; "-cask_fusion:fp16_acc=off"
# changed nothing; "-kgen:codegen:cuda_tile=2" fails to build (a Myelin CHECK
# in copy_vtx_analyzer).
BUILD_ROUTES = ("", "-cask:fp16_acc=off")
_STANDARD = (
    "runtime.cudagraphs",
    "runtime.pre_allocated_outputs",
    "engine.optimization_level",
    "engine.tiling_optimization_level",
)
# Name prefix of the attention implementations registered with transformers.
_ATTENTION = "agentic_prefill"


def create_case(
    model: str = "Qwen/Qwen3-0.6B",
    length: int = 2048,
    positions: int = 64,
    prompt: str = "python-docs",
    baseline: Mapping[str, Any] | None = None,
    kernel_cache: str | None = None,
) -> "PrefillWorkload":
    """Workload factory: ``WorkloadFactory("tools/llm/agentic_prefill.py:create_case", {...})``."""
    return PrefillWorkload(
        model,
        length=length,
        positions=positions,
        prompt=prompt,
        baseline=baseline,
        kernel_cache=kernel_cache,
    )


# ------------------------------------------------------------------ inputs
def prompt_text(prompt: str, characters: int) -> str:
    """At least ``characters`` characters of the named prompt text."""
    if prompt == "python-docs":
        from pydoc_data.topics import topics

        # Some topics repeat sections of others: keep each paragraph once.
        paragraphs = dict.fromkeys(
            paragraph
            for key in sorted(topics)
            for paragraph in topics[key].split("\n\n")
            if paragraph.strip()
        )
        text = "\n\n".join(paragraphs)
    elif prompt == "dickens":
        from agentic_decode import TEXT

        text = TEXT
    else:
        raise ValueError(f"prompt must be one of {PROMPTS}; got {prompt!r}")
    if len(text) < characters:
        text = text * -(-characters // len(text))
    return text[:characters] if prompt == "python-docs" else text


def logit_positions(length: int, count: int) -> torch.Tensor:
    """``count`` evenly spaced positions in ``[0, length)``, ending at the last."""
    if not 1 <= count <= length:
        raise ValueError(f"positions must be in [1, {length}]; got {count}")
    return (torch.arange(1, count + 1) * length) // count - 1


def layer_windows(config: Any) -> list[int | None]:
    """The sliding window of every decoder layer (None for full attention)."""
    config = getattr(config, "text_config", None) or config
    layers = config.num_hidden_layers
    types = getattr(config, "layer_types", None) or ["full_attention"] * layers
    window = getattr(config, "sliding_window", None)
    return [
        window if kind == "sliding_attention" and window else None
        for kind in types[:layers]
    ]


def chunk_size(local: str) -> int:
    """The query chunk of a ``banded-B`` local attention."""
    if not local.startswith("banded-"):
        raise ValueError(f"{local!r} is not a banded local attention")
    return int(local.partition("-")[2])


# ---------------------------------------------------------------- attention
def position_mask(
    q_length: int, kv_length: int, window: int | None, device: torch.device
) -> torch.Tensor:
    """Causal (and sliding-window) mask from position ranges, ``[1, 1, q, kv]``.

    Queries are the last ``q_length`` of ``kv_length`` positions. A key is
    visible when it is not in the future and, with a window, fewer than
    ``window`` positions back (Hugging Face's sliding-window semantics).
    """
    query = torch.arange(q_length, device=device)[:, None] + (kv_length - q_length)
    key = torch.arange(kv_length, device=device)[None, :]
    mask = key <= query
    if window is not None:
        mask = mask & (query - key < window)
    return mask[None, None]


def banded_mask(
    chunks: int, chunk: int, window: int, device: torch.device
) -> torch.Tensor:
    """Masks of :func:`banded_attention`: ``[chunks, 1, chunk, window + chunk]``.

    Query ``i`` of chunk ``c`` is position ``c * chunk + i``; key ``j`` of its
    band is position ``c * chunk - window + j`` (negative ones are padding).
    """
    query = torch.arange(chunk, device=device)[:, None]
    key = torch.arange(window + chunk, device=device)[None, :]
    start = torch.arange(chunks, device=device)[:, None, None] * chunk - window
    distance = window + query - key
    mask = (distance >= 0) & (distance < window) & (start + key >= 0)
    return mask[:, None]


def _bands(x: torch.Tensor, chunk: int, window: int) -> torch.Tensor:
    """``[1, heads, length, dim]`` -> ``[chunks, heads, window + chunk, dim]``.

    Band ``c`` holds positions ``c * chunk - window`` to ``(c + 1) * chunk - 1``,
    zero-padded before position 0.
    """
    _, heads, length, dim = x.shape
    steps = window // chunk
    padded = F.pad(x, (0, 0, window, 0))
    parts = torch.stack(
        [padded[:, :, s * chunk : s * chunk + length] for s in range(steps + 1)], 2
    )
    parts = parts.view(1, heads, steps + 1, length // chunk, chunk, dim)
    return parts.permute(3, 0, 1, 2, 4, 5).reshape(
        length // chunk, heads, window + chunk, dim
    )


def banded_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    window: int,
    chunk: int,
    scale: float | None,
) -> torch.Tensor:
    """Sliding-window causal attention as one batched attention over chunks.

    Each chunk of ``chunk`` queries attends to the ``window + chunk`` keys
    ending at its last position, so the work grows linearly with the length
    and no ``length x length`` mask exists. Inputs are ``[1, heads, length,
    dim]`` (keys may have fewer heads); the output matches SDPA's.
    """
    batch, heads, length, dim = query.shape
    if batch != 1 or length % chunk or window % chunk:
        raise ValueError(
            f"banded attention needs batch 1 and a chunk ({chunk}) dividing the "
            f"length ({length}) and the window ({window})"
        )
    chunks = length // chunk
    queries = query.view(1, heads, chunks, chunk, dim).transpose(0, 2)
    queries = queries.reshape(chunks, heads, chunk, dim)
    out = F.scaled_dot_product_attention(
        queries,
        _bands(key, chunk, window),
        _bands(value, chunk, window),
        attn_mask=banded_mask(chunks, chunk, window, query.device),
        scale=scale,
        enable_gqa=key.shape[1] != heads,
    )
    return (
        out.view(chunks, heads, 1, chunk, dim)
        .transpose(0, 2)
        .reshape(1, heads, length, dim)
    )


def flash_attention_reference(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
    window: int,
) -> torch.Tensor:
    """What ``agentic_prefill::flash_attention`` computes, in ``[B, S, H, D]``.

    Inputs are ``[B, H, S, D]``; ``window`` 0 means full causal attention.
    """
    mask = position_mask(query.shape[2], key.shape[2], window or None, query.device)
    return (
        F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=mask,
            scale=scale,
            enable_gqa=key.shape[1] != query.shape[1],
        )
        .transpose(1, 2)
        .contiguous()
    )


_FLASH_ATTENTION_OP: Any = None


def _flash_attention_op() -> Any:
    """``agentic_prefill::flash_attention``: causal attention kept in PyTorch.

    TensorRT has no converter for it, so Torch-TensorRT runs it between
    engines. On CUDA it calls PyTorch's flash-attention kernel with native
    GQA and its sliding window (``window_size_left``); elsewhere,
    :func:`flash_attention_reference`. Registered on first use, which must
    happen outside ``torch.compile`` tracing (:func:`attention_implementation`
    does it).
    """
    global _FLASH_ATTENTION_OP
    if _FLASH_ATTENTION_OP is not None:
        return _FLASH_ATTENTION_OP
    namespace = torch.ops.agentic_prefill
    if not hasattr(namespace, "flash_attention"):

        @torch.library.custom_op("agentic_prefill::flash_attention", mutates_args=())
        def flash_attention(
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            scale: float,
            window: int,
        ) -> torch.Tensor:
            if query.is_cuda and query.dtype in (torch.float16, torch.bfloat16):
                length = query.shape[2]
                out = torch.ops.aten._flash_attention_forward(
                    query.transpose(1, 2),
                    key.transpose(1, 2),
                    value.transpose(1, 2),
                    None,
                    None,
                    length,
                    key.shape[2],
                    0.0,
                    True,
                    False,
                    scale=scale,
                    window_size_left=window - 1 if window else None,
                    window_size_right=0 if window else None,
                )[0]
                return out.contiguous()
            return flash_attention_reference(query, key, value, scale, window)

        @flash_attention.register_fake
        def _(query, key, value, scale, window):  # noqa: ANN001, ANN202
            batch, heads, length, dim = query.shape
            return query.new_empty((batch, length, heads, dim))

    _FLASH_ATTENTION_OP = namespace.flash_attention.default
    return _FLASH_ATTENTION_OP


def prefill_attention(
    global_mode: str,
    local_mode: str,
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float = 0.0,
    scaling: float | None = None,
    sliding_window: int | None = None,
    softcap: float | None = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, None]:
    """A Hugging Face attention function with selectable prefill lowerings.

    Registered (partially applied with the two modes) as an
    ``attn_implementation``. A ``mask`` mode uses Hugging Face's mask when it
    created one (``workload.masks=hf``) and :func:`position_mask` otherwise.
    The ``inductor`` mode (``kernels.inductor``, not a workload value) runs
    the ``torch-flash`` operator inside an
    ``execute_in_torch(backend="inductor")`` region, which strict
    Torch-TensorRT compilation keeps out of TensorRT and compiles with
    Inductor; elsewhere the region is ordinary code.
    """
    from transformers.integrations.sdpa_attention import sdpa_attention_forward

    if softcap is not None or dropout:
        raise ValueError("prefill attention supports neither softcapping nor dropout")
    mode = local_mode if sliding_window else global_mode
    gqa = key.shape[1] != query.shape[1]
    if mode == "mask" and attention_mask is not None:
        return sdpa_attention_forward(
            module, query, key, value, attention_mask, scaling=scaling, **kwargs
        )
    if mode == "torch-flash":
        if scaling is None:
            scaling = query.shape[-1] ** -0.5
        op = _FLASH_ATTENTION_OP or _flash_attention_op()
        out = op(query, key, value, float(scaling), int(sliding_window or 0))
        return out, None
    if mode == "inductor":
        if scaling is None:
            scaling = query.shape[-1] ** -0.5
        op = _FLASH_ATTENTION_OP or _flash_attention_op()
        with region.execute_in_torch(name=_region_name(module), backend="inductor"):
            out = op(query, key, value, float(scaling), int(sliding_window or 0))
        return out, None
    if mode == "mask":
        mask = position_mask(query.shape[2], key.shape[2], sliding_window, query.device)
        out = F.scaled_dot_product_attention(
            query, key, value, attn_mask=mask, scale=scaling, enable_gqa=gqa
        )
    elif mode == "causal":
        out = F.scaled_dot_product_attention(
            query, key, value, is_causal=True, scale=scaling, enable_gqa=gqa
        )
    elif mode.startswith("banded-"):
        out = banded_attention(
            query, key, value, sliding_window, chunk_size(mode), scaling
        )
    else:
        raise ValueError(f"Unknown prefill attention mode {mode!r}")
    return out.transpose(1, 2).contiguous(), None


def attention_implementation(masks: str, global_mode: str, local_mode: str) -> str:
    """Register (once) and name the attention implementation for these modes.

    ``hf`` masks keep Hugging Face's SDPA mask creation; ``positions`` masks
    register no mask function, so Hugging Face passes no mask and
    :func:`prefill_attention` builds its own.
    """
    if masks == "hf" and global_mode == "mask" and local_mode == "mask":
        return "sdpa"
    from transformers import AttentionInterface, AttentionMaskInterface
    from transformers.masking_utils import sdpa_mask

    # Hugging Face reads a name that contains "flash" as a hub flash kernel.
    modes = "_".join((masks, global_mode, local_mode)).replace("torch-flash", "pt")
    name = f"{_ATTENTION}_{modes}".replace("-", "")
    if {"torch-flash", "inductor"} & {global_mode, local_mode}:
        _flash_attention_op()
    AttentionInterface.register(
        name, functools.partial(prefill_attention, global_mode, local_mode)
    )
    if masks == "hf":
        AttentionMaskInterface.register(name, sdpa_mask)
    return name


# ------------------------------------------------------ Inductor's attention
@dataclasses.dataclass(frozen=True)
class InductorPlacement:
    """What a ``kernels.inductor`` value moves out of TensorRT, and how.

    ``layers`` is ``""`` (nothing), ``"global"`` (the full-attention layers)
    or ``"all"``. ``route`` is ``"torch-flash"`` (the flash-attention custom
    operator between engines), ``"region"`` (that operator in an Inductor
    region), ``"region-rope"`` (the q/k RMSNorms, the rotary embedding and
    the operator in one Inductor region) or ``"plugin"`` (the q/k RMSNorms
    and the rotary embedding as Inductor's kernels inside the engine; the
    attention stays in TensorRT).
    """

    layers: str = ""
    route: str = ""

    @property
    def regions(self) -> bool:
        """Whether the build needs strict region capture."""
        return self.route.startswith("region") or self.route == "plugin"

    @property
    def moves_attention(self) -> bool:
        """Whether the moved layers' attention leaves TensorRT."""
        return bool(self.layers) and self.route != "plugin"

    def modes(self, global_mode: str, local_mode: str) -> tuple[str, str]:
        """The global and local attention modes after the move.

        Moved layers run ``torch-flash``, or ``inductor`` for a narrow region;
        ``region-rope`` layers call ``torch-flash`` inside the region that
        :func:`install_attention_regions` opens around it. ``plugin`` layers
        keep the config's modes.
        """
        if not self.moves_attention:
            return global_mode, local_mode
        mode = "inductor" if self.route == "region" else "torch-flash"
        return mode, mode if self.layers == "all" else local_mode

    def moved(self, windows: Sequence[int | None]) -> list[int]:
        """Indices of the layers the placement moves, given their windows."""
        if not self.layers:
            return []
        return [
            index
            for index, window in enumerate(windows)
            if self.layers == "all" or window is None
        ]


def inductor_placement(value: str) -> InductorPlacement:
    """The :class:`InductorPlacement` of a ``kernels.inductor`` value."""
    if value not in INDUCTOR_KERNELS:
        raise ValueError(
            f"kernels.inductor must be one of {list(INDUCTOR_KERNELS)}; got {value!r}"
        )
    return InductorPlacement(*INDUCTOR_KERNELS[value])


def _region_name(module: torch.nn.Module) -> str:
    index = getattr(module, "layer_idx", None)
    return "attention" if index is None else f"attention.{index}"


def _attention_region_forward(
    self: torch.nn.Module,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
    attention_mask: torch.Tensor | None = None,
    past_key_values: Any = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, None]:
    """Gemma 3 / Qwen 3 attention with q/k-norm, RoPE and attention in a region.

    The same operations as Hugging Face's forward (projections, per-head
    RMSNorms, rotary embedding, the configured attention function, output
    projection). The q, k and v projections run as one GEMM over their
    concatenated weights (:func:`install_attention_regions`), as TensorRT
    fuses them in a single engine; everything between that GEMM and the
    output projection runs in one ``execute_in_torch(backend="inductor")``
    region, which reads the GEMM's output.
    """
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    if past_key_values is not None:
        raise ValueError("attention regions run a prefill without a cache")
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)
    qkv = F.linear(hidden_states, self.qkv_weight, self.qkv_bias)
    cos, sin = position_embeddings
    attention = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
    with region.execute_in_torch(name=_region_name(self), backend="inductor"):
        query, key, value = qkv.split(self.qkv_sizes, dim=-1)
        query = self.q_norm(query.view(hidden_shape).transpose(1, 2))
        key = self.k_norm(key.view(hidden_shape).transpose(1, 2))
        value = value.view(hidden_shape).transpose(1, 2)
        query, key = self._apply_rotary(query, key, cos, sin)
        out, _ = attention(
            self,
            query,
            key,
            value,
            attention_mask,
            dropout=0.0,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )
    out = out.reshape(*input_shape, -1).contiguous()
    return self.o_proj(out), None


def _norm_rope(
    norm: Any,
    rotary: Any,
    sizes: Sequence[int],
    head_dim: int,
    role: str,
    qkv: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """The per-head RMSNorm and rotary embedding of q (``role="q"``) or k.

    The operations of Hugging Face's attention forward on the q or k columns
    of the fused q/k/v GEMM output (``sizes``: the q, k and v widths);
    returns ``[B, heads, S, head_dim]``, contiguous: the layout the
    ``rope-plugin`` kernels write and TensorRT's attention reads.
    """
    start, width = (0, sizes[0]) if role == "q" else (sizes[0], sizes[1])
    x = qkv[..., start : start + width]
    x = norm(x.view(*qkv.shape[:-1], -1, head_dim).transpose(1, 2))
    return rotary(x, x, cos, sin)[0].contiguous()


def _attention_plugin_forward(
    self: torch.nn.Module,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
    attention_mask: torch.Tensor | None = None,
    past_key_values: Any = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, None]:
    """Gemma 3 / Qwen 3 attention with q/k-norm + RoPE kernels in the engine.

    The operations of Hugging Face's forward, with the q, k and v projections
    as one GEMM (:func:`_fuse_qkv`) and the q and k RMSNorm + rotary
    embedding each in a ``torch_tensorrt.region.fused_kernel`` scope (inputs
    ``qkv``, the norm's weight, ``cos`` and ``sin``) that strict
    Torch-TensorRT compilation replaces with a kernel hosted in the engine as
    a QDP plugin: Inductor's generated kernel for it
    (:func:`inductor_rope_kernels`; ``rope_plugin``), or with
    ``rope_generator`` (a ``torch_tensorrt.region.KernelGenerator``) the one
    the generator writes for the scope. The attention itself is the config's
    (TensorRT). Elsewhere the scopes run their body (:func:`_norm_rope`) as
    ordinary code.
    """
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    if past_key_values is not None:
        raise ValueError("attention regions run a prefill without a cache")
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)
    qkv = F.linear(hidden_states, self.qkv_weight, self.qkv_bias)
    cos, sin = position_embeddings
    generator = getattr(self, "rope_generator", None)
    rotated = []
    for role, norm in (("q", self.q_norm), ("k", self.k_norm)):
        name = f"{role}_norm_rope.{getattr(self, 'layer_idx', 0)}"
        inputs = [qkv, norm.weight, cos, sin]
        if generator is not None:
            # The body stays the eager reference; the generator writes,
            # checks and caches the kernel that replaces it at compile time.
            scope = region.fused_kernel(generator=generator, inputs=inputs, name=name)
        else:
            hosted = _ROPE_KERNELS[self.rope_plugin][role]
            scope = region.fused_kernel(
                kernel=functools.partial(_rope_kernel, self.rope_plugin, role),
                inputs=inputs,
                name=name,
                grid=functools.partial(_fixed_grid, hosted.grid),
                num_warps=hosted.num_warps,
                num_stages=hosted.num_stages,
            )
        with scope:
            out = _norm_rope(
                norm,
                self._apply_rotary,
                self.qkv_sizes,
                self.head_dim,
                role,
                qkv,
                cos,
                sin,
            )
        rotated.append(out)
    query, key = rotated
    value = qkv[..., self.qkv_sizes[0] + self.qkv_sizes[1] :]
    value = value.view(hidden_shape).transpose(1, 2)
    attention = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
    out, _ = attention(
        self,
        query,
        key,
        value,
        attention_mask,
        dropout=0.0,
        scaling=self.scaling,
        sliding_window=self.sliding_window,
        **kwargs,
    )
    out = out.reshape(*input_shape, -1).contiguous()
    return self.o_proj(out), None


@dataclasses.dataclass(frozen=True)
class HostedKernel:
    """One of Inductor's generated kernels, wrapped for a QDP plugin.

    ``kernel`` is a ``@triton.jit`` wrapper with the plugin ABI (inputs, then
    the output) that calls Inductor's kernel unchanged with the arguments of
    Inductor's own launch; ``grid``, ``num_warps``, ``num_stages`` and
    ``constexprs`` are that launch's. ``name`` and ``sha256`` identify
    Inductor's generated source; ``matches_inductor`` is ``"bitwise"`` when
    the wrapper, launched by Triton, reproduced Inductor's output exactly
    (``"close"`` otherwise, with ``max_abs_diff``).
    """

    kernel: Any
    grid: tuple[int, ...]
    num_warps: int
    num_stages: int
    constexprs: Mapping[str, int]
    name: str
    sha256: str
    source_path: str
    matches_inductor: str
    max_abs_diff: float = 0.0

    def describe(self) -> dict[str, Any]:
        return {
            "inductor_kernel": self.name,
            "plugin_kernel": getattr(self.kernel, "__name__", None),
            "sha256": self.sha256,
            "grid": list(self.grid),
            "num_warps": self.num_warps,
            "num_stages": self.num_stages,
            "constexprs": dict(self.constexprs),
            "matches_inductor": self.matches_inductor,
            "max_abs_diff": self.max_abs_diff,
        }


# rope-plugin kernels per attention shape (inductor_rope_kernels), resolved by
# the fused_kernel factories below outside tracing.
_ROPE_KERNELS: dict[tuple, dict[str, HostedKernel]] = {}


def _rope_kernel(key: tuple, role: str) -> Any:
    return _ROPE_KERNELS[key][role].kernel


def _fixed_grid(grid: tuple[int, ...], inputs: Any, outputs: Any) -> tuple[int, ...]:
    return grid


def parse_inductor_kernel(source: str) -> tuple[str, str, list[str]]:
    """The one Triton kernel of Inductor's output code ``source``.

    Returns its name, its source and its parameter names. Raises
    ``RuntimeError`` unless the code launches exactly one Triton kernel whose
    source it contains.
    """
    import re

    sources = dict(
        re.findall(
            r"async_compile\.triton\('(\w+)', '''(.*?)''', device_str", source, re.S
        )
    )
    launched = re.findall(r"(triton_\w+)\.run\(", source)
    if len(launched) != 1 or launched[0] not in sources:
        raise RuntimeError(
            "Expected Inductor code that launches one Triton kernel; it launches "
            f"{launched} with sources for {sorted(sources)}"
        )
    name = launched[0]
    header = re.search(r"^def " + name + r"\(([^)]*)\)", sources[name], re.M)
    if header is None:
        raise RuntimeError(f"Inductor's source of {name} has no kernel definition")
    params = [p.split(":")[0].strip() for p in header.group(1).split(",")]
    return name, sources[name], [p for p in params if p]


def plugin_call_arguments(
    params: Sequence[str],
    run_args: Sequence[Any],
    roles: Mapping[int, str],
    output_ptr: int,
    constexprs: Mapping[str, int],
) -> list[str]:
    """Inductor's kernel call, written over the plugin wrapper's parameters.

    ``run_args`` are the positional arguments of Inductor's launch (tensors
    and integers), ``roles`` maps the data pointers of the wrapper's inputs to
    their parameter names and ``output_ptr`` is the output's; ``constexprs``
    are the launch's constexpr values. Every pointer is passed through
    ``_aligned`` (see :func:`inductor_rope_kernels`). Raises ``RuntimeError``
    for an argument that is none of these.
    """
    arguments = []
    for index, param in enumerate(params):
        if index < len(run_args):
            value = run_args[index]
            if isinstance(value, torch.Tensor):
                pointer = value.data_ptr()
                if pointer == output_ptr:
                    arguments.append("_aligned(OUT)")
                elif pointer in roles:
                    arguments.append(f"_aligned({roles[pointer]})")
                else:
                    raise RuntimeError(
                        f"Inductor passes {param} a tensor that is none of the "
                        "plugin's inputs or its output"
                    )
            elif isinstance(value, int):
                arguments.append(str(value))
            else:
                raise RuntimeError(f"Unexpected Inductor argument {param}={value!r}")
        elif param in constexprs:
            arguments.append(str(int(constexprs[param])))
        else:
            raise RuntimeError(f"Inductor's launch gives no value for {param}")
    return arguments


_ROPE_WRAPPER = """

_inductor_kernel = {name}.fn


@triton.jit
def _aligned(ptr):
    rounded = ptr.to(tl.uint64, bitcast=True) // 16 * 16
    return tl.multiple_of(rounded.to(ptr.dtype, bitcast=True), 16)


@triton.jit
def {wrapper}(QKV, WEIGHT, COS, SIN, OUT):
    _inductor_kernel({arguments})
"""


def inductor_rope_kernels(
    attention: torch.nn.Module, length: int, directory: str | None = None
) -> tuple:
    """Generate Inductor's q/k-norm + RoPE kernels for ``attention``; host them.

    For q and k separately, compiles :func:`_norm_rope` (the module's own
    RMSNorm and rotary embedding, on a ``[1, length, q + k + v]`` q/k/v GEMM
    output) with ``torch.compile(mode=ROPE_PLUGIN_COMPILE_MODE)``, requires
    the generated code to launch exactly one Triton kernel, records that
    launch (its tensors, identified by data pointer, its integers and the
    launch settings Inductor's autotuner chose) and writes a plugin wrapper
    that calls Inductor's kernel unchanged with the launch's arguments. The
    plugin compiles without the 16-byte pointer specialization Inductor's
    launch applies; ``_aligned`` restores it (TensorRT's buffers are 256-byte
    aligned). The wrapper, launched by Triton, must reproduce Inductor's
    output. Returns the key of :data:`_ROPE_KERNELS`; cached per shape.
    """
    import hashlib
    import importlib.util
    import os
    import tempfile

    from torch._inductor.runtime.triton_heuristics import CachingAutotuner
    from torch._inductor.utils import run_and_get_code

    weight = attention.q_proj.weight
    sizes = tuple(
        projection.weight.shape[0]
        for projection in (attention.q_proj, attention.k_proj, attention.v_proj)
    )
    rotary = sys.modules[type(attention).__module__].apply_rotary_pos_emb
    norms = {"q": attention.q_norm, "k": attention.k_norm}
    eps = tuple(
        getattr(norm, "eps", getattr(norm, "variance_epsilon", None))
        for norm in norms.values()
    )
    key = (
        type(attention).__name__,
        length,
        sizes,
        attention.head_dim,
        eps,
        str(weight.dtype),
        str(weight.device),
        ROPE_PLUGIN_COMPILE_MODE,
    )
    if key in _ROPE_KERNELS:
        return key
    directory = directory or tempfile.mkdtemp(prefix="agentic_prefill_rope_")
    generator = torch.Generator(device=weight.device).manual_seed(0)
    options = dict(device=weight.device, dtype=weight.dtype)
    qkv = torch.randn(1, length, sum(sizes), generator=generator, **options)
    angles = torch.rand(1, length, attention.head_dim, generator=generator, **options)
    cos, sin = angles.cos(), angles.sin()
    hosted = {}
    for role, norm in norms.items():

        def function(
            qkv: torch.Tensor,
            cos: torch.Tensor,
            sin: torch.Tensor,
            norm: Any = norm,
            role: str = role,
        ) -> torch.Tensor:
            return _norm_rope(
                norm, rotary, sizes, attention.head_dim, role, qkv, cos, sin
            )

        compiled = torch.compile(
            function, mode=ROPE_PLUGIN_COMPILE_MODE, fullgraph=True, dynamic=False
        )
        with torch.no_grad():
            expected, code = run_and_get_code(compiled, qkv, cos, sin)
        name, source, params = parse_inductor_kernel(code[0])
        if "tl.load(in_out_ptr" in source:
            raise RuntimeError(f"Inductor's {name} reads its output buffer")
        # Record the launch of a second call: its tensors and integers, and
        # the autotuner that ran it (which holds the chosen launch settings).
        launches: list[tuple[Any, tuple]] = []
        original_run = CachingAutotuner.run

        def record(tuner: Any, *args: Any, **kwargs: Any) -> Any:
            if getattr(tuner.fn, "__name__", None) == name:
                launches.append((tuner, args))
            return original_run(tuner, *args, **kwargs)

        CachingAutotuner.run = record
        try:
            with torch.no_grad():
                again = compiled(qkv, cos, sin)
            torch.cuda.synchronize()
        finally:
            CachingAutotuner.run = original_run
        if len(launches) != 1:
            raise RuntimeError(f"Inductor launched {name} {len(launches)} times")
        autotuner, run_args = launches[0]
        launchers = getattr(autotuner, "launchers", None) or []
        if len(launchers) != 1:
            raise RuntimeError(f"Inductor kept {len(launchers)} launchers for {name}")
        config = launchers[0].config
        constexprs = {str(k): int(v) for k, v in config.kwargs.items()}
        inputs = {
            qkv.data_ptr(): "QKV",
            norm.weight.data_ptr(): "WEIGHT",
            cos.data_ptr(): "COS",
            sin.data_ptr(): "SIN",
        }
        arguments = plugin_call_arguments(
            params, run_args, inputs, again.data_ptr(), constexprs
        )
        rows = next((a for p, a in zip(params, run_args) if p == "xnumel"), None)
        if not isinstance(rows, int) or "XBLOCK" not in constexprs:
            raise RuntimeError(f"Expected a 1-D Inductor grid for {name}")
        wrapper = f"{name}_{role}"
        path = os.path.join(directory, f"{wrapper}.py")
        with open(path, "w") as stream:
            stream.write(source)
            stream.write(
                _ROPE_WRAPPER.format(
                    name=name, wrapper=wrapper, arguments=", ".join(arguments)
                )
            )
        spec = importlib.util.spec_from_file_location(f"agentic_rope_{wrapper}", path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        kernel = getattr(module, wrapper)
        grid = (-(-rows // constexprs["XBLOCK"]), 1, 1)
        actual = torch.empty_like(expected)
        kernel[grid](
            qkv,
            norm.weight,
            cos,
            sin,
            actual,
            num_warps=int(config.num_warps),
            num_stages=int(config.num_stages),
        )
        torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-3)
        hosted[role] = HostedKernel(
            kernel=kernel,
            grid=grid,
            num_warps=int(config.num_warps),
            num_stages=int(config.num_stages),
            constexprs=constexprs,
            name=name,
            sha256=hashlib.sha256(source.encode()).hexdigest(),
            source_path=path,
            matches_inductor="bitwise" if torch.equal(actual, expected) else "close",
            max_abs_diff=(actual.float() - expected.float()).abs().max().item(),
        )
    _ROPE_KERNELS[key] = hosted
    return key


_REGION_ATTENTION_CLASSES: dict[tuple[type, str], type] = {}
_REGION_FORWARDS = {
    "region": _attention_region_forward,
    "plugin": _attention_plugin_forward,
}


def _fuse_qkv(attention: torch.nn.Module) -> None:
    """Give ``attention`` one q/k/v weight (and bias) that its projections view.

    ``qkv_weight`` is the concatenation of the three weights; the projections'
    own weights become views of it, so no memory is duplicated and the
    module computes exactly what it did.
    """
    projections = (attention.q_proj, attention.k_proj, attention.v_proj)
    with torch.no_grad():
        weight = torch.cat([p.weight for p in projections])
        biases = [p.bias for p in projections]
        bias = None if biases[0] is None else torch.cat(biases)
        start = 0
        for projection in projections:
            rows = projection.weight.shape[0]
            projection.weight = torch.nn.Parameter(
                weight[start : start + rows], requires_grad=False
            )
            if bias is not None:
                projection.bias = torch.nn.Parameter(
                    bias[start : start + rows], requires_grad=False
                )
            start += rows
    attention.qkv_weight = torch.nn.Parameter(weight, requires_grad=False)
    attention.qkv_bias = (
        None if bias is None else torch.nn.Parameter(bias, requires_grad=False)
    )
    attention.qkv_sizes = tuple(p.weight.shape[0] for p in projections)


def install_attention_regions(
    model: torch.nn.Module,
    layers: Sequence[int],
    plugin: tuple | None = None,
    *,
    generator: Any = None,
) -> int:
    """Make the attention of ``layers`` run :func:`_attention_region_forward`.

    Each attention module becomes an instance of a subclass of its class
    (made once per class and forward) whose forward opens the region, and
    gets the fused q/k/v weight that forward uses (:func:`_fuse_qkv`). With
    ``plugin`` (a key of :func:`inductor_rope_kernels`), the forward is
    :func:`_attention_plugin_forward` instead, hosting those kernels; with
    ``generator`` (a ``torch_tensorrt.region.KernelGenerator``) the same
    forward, whose q/k-norm + RoPE scopes run the kernels the generator
    writes. Returns the number of modules changed. Needs Hugging Face's
    Gemma 3 / Qwen 3 attention layout: q/k/v/o projections,
    ``q_norm``/``k_norm`` and the module's ``apply_rotary_pos_emb``.
    """
    if plugin is not None and generator is not None:
        raise ValueError("Pass hosted kernels (plugin) or a generator, not both")
    kind = "region" if plugin is None and generator is None else "plugin"
    changed = 0
    for index in layers:
        attention = model.model.layers[index].self_attn
        base = type(attention)
        if base in _REGION_ATTENTION_CLASSES.values():
            continue
        rotary = getattr(sys.modules[base.__module__], "apply_rotary_pos_emb", None)
        needed = ("q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm")
        if rotary is None or not all(hasattr(attention, n) for n in needed):
            raise ValueError(
                f"{base.__name__} lacks the q/k-norm attention layout that "
                "attention regions rewrite"
            )
        cls = _REGION_ATTENTION_CLASSES.get((base, kind))
        if cls is None:
            cls = type(
                f"{kind.capitalize()}{base.__name__}",
                (base,),
                {
                    "forward": _REGION_FORWARDS[kind],
                    "_apply_rotary": staticmethod(rotary),
                },
            )
            _REGION_ATTENTION_CLASSES[(base, kind)] = cls
        _fuse_qkv(attention)
        if plugin is not None:
            attention.rope_plugin = plugin
        if generator is not None:
            attention.rope_generator = generator
        attention.__class__ = cls
        changed += 1
    return changed


# ------------------------------------------------------------------- roofline
_PEAK_TFLOPS: dict[bool, float] = {}


def peak_tflops(fp16_accumulation: bool = False) -> float | None:
    """Dense GEMM throughput of this GPU (TFLOP/s), measured once per process.

    BF16 inputs with FP32 accumulation by default; with ``fp16_accumulation``,
    FP16 inputs accumulated in FP16 (cuBLAS), as TensorRT's FP16 tactics do.
    """
    if fp16_accumulation not in _PEAK_TFLOPS and torch.cuda.is_available():
        dtype = torch.float16 if fp16_accumulation else torch.bfloat16
        matmul = torch.backends.cuda.matmul
        allowed = matmul.allow_fp16_accumulation
        try:
            a = torch.randn(8192, 8192, device="cuda", dtype=dtype)
            b = torch.randn(8192, 8192, device="cuda", dtype=dtype)
        except RuntimeError:  # not enough free memory
            return None
        best_ms = float("inf")
        try:
            matmul.allow_fp16_accumulation = fp16_accumulation
            for attempt in range(8):
                start, end = torch.cuda.Event(True), torch.cuda.Event(True)
                start.record()
                a @ b
                end.record()
                end.synchronize()
                if attempt >= 2:
                    best_ms = min(best_ms, start.elapsed_time(end))
        finally:
            matmul.allow_fp16_accumulation = allowed
        _PEAK_TFLOPS[fp16_accumulation] = 2 * 8192**3 / best_ms / 1e9
        del a, b
        torch.cuda.empty_cache()
    return _PEAK_TFLOPS.get(fp16_accumulation)


def prefill_flops(model: torch.nn.Module, length: int, kept: int) -> dict[str, float]:
    """FLOPs of one prefill without masked work, in GFLOP.

    Linear layers run on every position, the LM head on the ``kept`` rows;
    attention counts the (query, key) pairs a causal or sliding-window layer
    needs (4 FLOPs per pair, head and dimension: two matmuls).
    """
    config = model.config
    heads = config.num_attention_heads
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // heads
    layers = model.model.layers
    linear = sum(
        m.weight.numel()
        for layer in layers
        for m in layer.modules()
        if isinstance(m, torch.nn.Linear)
    )
    head = model.get_output_embeddings().weight.numel()
    pairs = 0
    for window in layer_windows(config)[: len(layers)]:
        if window is None or window >= length:
            pairs += length * (length + 1) // 2
        else:
            pairs += window * (window + 1) // 2 + (length - window) * window
    attention = 4 * heads * head_dim * pairs
    dense = 4 * heads * head_dim * length * length * len(layers)
    total = 2 * length * linear + 2 * kept * head + attention
    return {
        "gflop_per_unit": total / 1e9,
        "linear_gflop": 2 * length * linear / 1e9,
        "attention_gflop": attention / 1e9,
        "attention_gflop_dense": dense / 1e9,
        "lm_head_gflop": 2 * kept * head / 1e9,
    }


# ------------------------------------------------------------------- workload
class _Prefill(torch.nn.Module):
    """``input_ids -> logits`` at fixed positions (the exported forward)."""

    def __init__(self, model: torch.nn.Module, positions: torch.Tensor) -> None:
        super().__init__()
        self.model = model
        self.register_buffer("positions", positions, persistent=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model(
            input_ids=input_ids, logits_to_keep=self.positions, use_cache=False
        ).logits


class PrefillWorkload(Workload):
    """Batch-1 prefill of a Hugging Face causal LM, logits at fixed positions.

    The baseline is the documented Torch-TensorRT compile of the exported
    model (SDPA attention, native precision, default settings).
    ``kernel_cache`` is the KernelGenerator cache that ``kernels.generated``
    takes its kernels from (``region_generation.kernel_cache``).
    """

    def __init__(
        self,
        model: str,
        *,
        length: int,
        positions: int = 64,
        prompt: str = "python-docs",
        baseline: Mapping[str, Any] | None = None,
        kernel_cache: str | None = None,
    ) -> None:
        if length < 2:
            raise ValueError("PrefillWorkload needs a prompt of at least 2 tokens")
        if prompt not in PROMPTS:
            raise ValueError(f"prompt must be one of {PROMPTS}; got {prompt!r}")
        logit_positions(length, positions)
        import region_generation

        self.model_name = model
        self.name = f"{model.rsplit('/', 1)[-1]}-prefill-{length}"
        self.length, self.kept, self.prompt = length, positions, prompt
        self._baseline = dict(baseline or {})
        self.kernel_cache = region_generation.kernel_cache(kernel_cache)
        self._tokens: torch.Tensor | None = None
        self._config: Any = None

    def config(self) -> Any:
        if self._config is None:
            from transformers import AutoConfig

            self._config = AutoConfig.from_pretrained(self.model_name)
        return self._config

    def native_precision(self) -> str:
        dtype = getattr(self.config(), "dtype", None) or getattr(
            self.config(), "torch_dtype", None
        )
        return "fp16" if dtype in (torch.float16, "float16") else "bf16"

    def local_modes(self) -> tuple[str, ...]:
        """``workload.local_attention`` values that fit this model and length."""
        windows = {w for w in layer_windows(self.config()) if w is not None}
        if not windows or min(windows) >= self.length:
            return ()
        return (
            ("mask",)
            + tuple(
                mode
                for mode in LOCAL_ATTENTION
                if mode.startswith("banded-")
                and all(not w % chunk_size(mode) for w in windows)
                and not self.length % chunk_size(mode)
            )
            + ("torch-flash",)
        )

    # ------------------------------------------------------------ contract
    def knobs(self) -> Sequence[Knob]:
        native = self.native_precision()
        windows = layer_windows(self.config())
        knobs = [
            Knob(
                "workload.masks",
                MASKS,
                "hf",
                "Attention masks in the exported graph. 'hf': Hugging Face's own; "
                "under export they AND the causal (and sliding-window) pattern "
                "with a packed-sequence term computed from the position ids, which "
                "TensorRT's attention mask analysis cannot read, so every "
                "attention layer runs dense over the whole length x length square. "
                "'positions': causal and window masks from position ranges only, "
                "which TensorRT recognizes, skipping fully masked tiles. "
                "Matters for the layers whose global/local mode is 'mask'.",
                ("dominant_kernel",),
            ),
        ]
        if any(window is None for window in windows):
            knobs.append(
                Knob(
                    "workload.global_attention",
                    GLOBAL_ATTENTION,
                    "mask",
                    "Full-attention layers. 'mask': SDPA with the explicit mask "
                    "(workload.masks); 'causal': SDPA is_causal with native GQA "
                    "(TensorRT causal attention, no mask tensor, keys and values "
                    "not repeated); 'torch-flash': PyTorch's flash-attention "
                    "kernel (causal, native GQA) in a custom operator that runs "
                    "outside TensorRT, which splits the engine at every such "
                    "layer.",
                    ("dominant_kernel",),
                )
            )
        local = self.local_modes()
        if len(local) > 1:
            knobs.append(
                Knob(
                    "workload.local_attention",
                    local,
                    "mask",
                    "Sliding-window layers. 'mask': SDPA with the explicit "
                    "length x length window mask (workload.masks); 'banded-B': "
                    "queries in chunks of B attend to the window of keys before "
                    "them as one batched attention (B + window keys per chunk, "
                    "no length x length mask); smaller B wastes fewer masked "
                    "pairs but makes smaller attention problems; 'torch-flash': "
                    "PyTorch's flash-attention kernel with its sliding window, "
                    "outside TensorRT (one engine boundary per layer).",
                    ("dominant_kernel",),
                )
            )
        knobs.append(
            Knob(
                "engine.precision",
                (native,) + tuple(p for p in ("bf16", "fp16") if p != native),
                native,
                "Weight and activation precision. BF16 GEMMs accumulate in "
                "FP32; TensorRT builds FP16 GEMMs with FP16 accumulation "
                "(about 1.5x the BF16 GEMM rate; engine.build_route "
                "'-cask:fp16_acc=off' keeps FP32 accumulation). FP16 has 3 "
                "more mantissa bits but a +-65504 range: Gemma 3 1B's "
                "residual stream reaches about 61,000 at the BOS token (for "
                "every prompt) and about 48,800 on these prompts' content "
                "tokens; an overflow shows as an infinite KL.",
                ("precision", "dominant_kernel"),
            )
        )
        knobs.extend(self._build_route_knobs())
        knobs.append(self._inductor_knob())
        if self.rope_regions():
            knobs.append(self._generated_knob())
        knobs.extend(knob for knob in STANDARD_KNOBS if knob.name in _STANDARD)
        return tuple(knobs)

    def rope_regions(self) -> bool:
        """Whether :func:`install_attention_regions` can rewrite this model's attention."""
        config = getattr(self.config(), "text_config", None) or self.config()
        return getattr(config, "model_type", None) in _ROPE_REGION_MODELS

    def inductor_kernels(self) -> tuple[str, ...]:
        """``kernels.inductor`` values that fit this model and length.

        ``-global`` values need both full-attention layers and sliding-window
        layers shorter than the prompt (otherwise they equal the all-layer
        ones); ``attention-rope-inductor*`` and ``rope-plugin`` need a model
        whose attention :func:`install_attention_regions` can rewrite.
        """
        windows = layer_windows(self.config())
        split = None in windows and len(self.local_modes()) > 1
        rope = self.rope_regions()
        return tuple(
            value
            for value, (layers, route) in INDUCTOR_KERNELS.items()
            if (layers != "global" or split)
            and (route not in ("region-rope", "plugin") or rope)
        )

    def _inductor_knob(self) -> Knob:
        windows = layer_windows(self.config())
        moved = {
            layers: 2 * len(InductorPlacement(layers, "region").moved(windows))
            for layers in ("global", "all")
        }
        return Knob(
            "kernels.inductor",
            self.inductor_kernels(),
            "none",
            "Hybrid TensorRT + Inductor kernels: part of each named layer runs "
            "on the kernels Inductor's own builds of this workload run for it; "
            "everything else stays in TensorRT under the rest of the config. "
            "The 'attention-*' values move the attention of the layers they "
            "name ('-global': the full-attention layers; otherwise every "
            "layer) to PyTorch's flash attention (FA2: causal, native GQA, the "
            "sliding window inside the kernel, no mask), the kernel Inductor "
            "calls for it. For those layers the value replaces "
            "workload.global_attention/local_attention, whose values are "
            "ignored there: configs that differ only in them build the same "
            "model (workload.masks is moot once no layer keeps a mask). "
            "'attention-fa2-*': the torch-flash custom operator between "
            "engines; they build the same model as "
            "workload.global_attention='torch-flash' ('-global') or both "
            "attention knobs at 'torch-flash' ('-all'), i.e. aliases of those "
            "configs. 'attention-inductor*': the same operator inside one "
            "execute_in_torch(backend='inductor') region per layer, which "
            "Inductor (default mode) compiles on the first call, during the "
            "build. 'attention-rope-inductor*': the region also holds the q/k "
            "RMSNorms and the rotary embedding, which Inductor fuses into its "
            "own Triton kernels; the q/k/v projections run as one GEMM whose "
            "output the region reads. The attention values split the engine "
            "at each moved layer, 2 "
            f"boundaries per layer ({moved['global']} for the global layers, "
            f"{moved['all']} for all). Inside the whole-graph CUDA graph a "
            "boundary adds no per-call host work; it ends TensorRT's fusion "
            "there, so the tensors crossing it go through memory, and the "
            "operator copies its output to [B, S, H, D] unless its query "
            "already has that layout. With '-global' values the window layers "
            "keep their masks: the partitioner builds them all in the first "
            "engine and passes them to the later engines as [1, 1, S, S] "
            "inputs, which those engines materialize instead of reading "
            "position ranges. 'rope-plugin': no attention moves and the "
            "engine stays whole; in every layer the q and k RMSNorm + rotary "
            "embedding run as Inductor's own generated Triton kernels (one "
            "for q, one for k, generated by torch.compile at the prompt length "
            "during the build and checked against Inductor's output), called "
            "unchanged inside fused_kernel QDP plugins in the TensorRT engine "
            "after one q/k/v GEMM; the attention stays in TensorRT under the "
            "config's attention knobs, and TensorRT cannot fuse across a "
            "plugin. Region and plugin values compile with strict export "
            "(torch_tensorrt.compile(..., strict=True)). Region values cannot "
            "be saved: torch_tensorrt.save rejects execute_in_torch regions "
            "with backend='inductor', so a session that keeps one exports no "
            "artifact (its config still rebuilds it); the 'attention-fa2-*' "
            "and 'rope-plugin' builds save normally. Requires "
            "runtime.cudagraphs, which captures the whole graph (engines and "
            "PyTorch parts) in one CUDA graph; TensorRT 11.4 also needs it for "
            "QDP plugins, whose module it otherwise reloads on every enqueue.",
            ("dominant_kernel",),
            requires={"runtime.cudagraphs": True},
        )

    def _generated_knob(self) -> Knob:
        import region_generation

        layers = len(layer_windows(self.config()))
        return Knob(
            "kernels.generated",
            GENERATED_KERNELS,
            "none",
            region_generation.GENERATED_INTRO + " 'none': no generated "
            "kernels. 'qk-norm-rope': in every layer the q and k RMSNorm + "
            "rotary embedding (the two regions kernels.inductor='rope-plugin' "
            "fills with Inductor's kernels, each reading the output of one "
            "q/k/v GEMM, the norm's weight and the cos/sin tables) run as "
            f"generated kernels: two plugins per layer ({2 * layers} in all), "
            "no engine split; identical regions share one kernel (one for q, "
            "one for k). The attention stays in TensorRT under the config's "
            "attention knobs, and TensorRT cannot fuse across a plugin. "
            "Requires runtime.cudagraphs=True (TensorRT 11.4 reloads plugin "
            "modules on every enqueue outside CUDA graph replay) and "
            "kernels.inductor='none' (its values rewrite the same attention). "
            "Builds through strict export (torch_tensorrt.compile(strict=True)). "
            + region_generation.CACHE_NOTE,
            ("dominant_kernel",),
            {**region_generation.PLUGIN_REQUIRES, "kernels.inductor": "none"},
            prepare=GENERATED_KERNELS[1:],
        )

    @staticmethod
    def _build_route_knobs() -> tuple[Knob, ...]:
        try:
            from torch_tensorrt.dynamo import is_global_perf_tuning_available
        except ImportError:
            return ()
        if not is_global_perf_tuning_available():
            return ()
        return (
            Knob(
                "engine.build_route",
                BUILD_ROUTES,
                "",
                "TensorRT Global Performance Tuning build route (a Myelin "
                "knob); needs engine.precision 'fp16'. '-cask:fp16_acc=off' "
                "forbids the FP16-accumulation GEMM kernels TensorRT picks for "
                "FP16 engines, so the GEMMs accumulate in FP32: slower, more "
                "accurate. (Attention and split-K routes built identical "
                "engines on this workload and are not offered.)",
                ("precision", "dominant_kernel"),
                requires={"engine.precision": "fp16"},
            ),
        )

    def baseline(self) -> dict[str, Any]:
        return dict(self._baseline)

    def timing(self) -> Timing:
        return Timing(warmup=3, rounds=10, calls=2, profile_calls=1)

    def accuracy_defaults(self) -> dict[str, tuple[str, float]]:
        """Limits against the FP32 reference at the kept positions.

        On Gemma 3 1B (2k and 8k tokens) correct builds measure top-1
        0.95-1.0 (BF16 flips near-ties), mean KL up to 1.2e-3 and largest KL
        at one position up to 0.035; dropping or doubling the sliding window
        or limiting global layers to a window measures top-1 0.17-0.88, mean
        KL 0.2-4.4 and largest KL 6-14. The largest-KL bound (about 3x the
        worst correct build) catches errors confined to a few positions,
        which the mean hides. A window off by one token stays within BF16
        noise end to end; the unit tests check the windows instead.
        """
        return {
            "top1_agreement": (">=", 0.9),
            "mean_kl": ("<=", 0.005),
            "max_kl": ("<=", 0.1),
        }

    def tokens(self) -> torch.Tensor:
        """The ``length`` prompt token ids (BOS first if the tokenizer adds one)."""
        if self._tokens is None:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            # Eight characters per token is more than any English text needs.
            text = prompt_text(self.prompt, 8 * self.length)
            ids = tokenizer(text, return_tensors="pt")["input_ids"][0]
            if ids.numel() < self.length:
                raise ValueError(
                    f"The {self.prompt} prompt has only {ids.numel()} tokens"
                )
            self._tokens = ids[: self.length].clone()
        return self._tokens

    def positions(self) -> torch.Tensor:
        return logit_positions(self.length, self.kept)

    def reference(self) -> Any:
        """FP32 eager logits (SDPA, no TF32) at the kept positions."""
        from transformers import AutoModelForCausalLM

        matmul = torch.backends.cuda.matmul
        allow_tf32, matmul.allow_tf32 = matmul.allow_tf32, False
        try:
            with torch.inference_mode():
                model = (
                    AutoModelForCausalLM.from_pretrained(
                        self.model_name,
                        attn_implementation="sdpa",
                        dtype=torch.float32,
                    )
                    .eval()
                    .cuda()
                )
                logits = model(
                    self.tokens().cuda().unsqueeze(0),
                    logits_to_keep=self.positions().cuda(),
                    use_cache=False,
                ).logits[0]
                logits = logits.float().cpu()
        finally:
            matmul.allow_tf32 = allow_tf32
        del model
        gc.collect()
        torch.cuda.empty_cache()
        return logits

    def compare(self, outputs: Any, reference: Any) -> dict[str, float]:
        actual = outputs.detach().double().cpu().reshape(reference.shape)
        expected = reference.double()
        kl = torch.nn.functional.kl_div(
            actual.log_softmax(-1),
            expected.log_softmax(-1),
            log_target=True,
            reduction="none",
        ).sum(-1)
        if not torch.isfinite(actual).all():
            kl = torch.full_like(kl, float("inf"))
        return {
            "top1_agreement": (actual.argmax(-1) == expected.argmax(-1))
            .double()
            .mean()
            .item(),
            "mean_kl": kl.mean().item(),
            "max_kl": kl.max().item(),
            "max_abs_logit_error": (actual - expected).abs().max().item(),
        }

    # -------------------------------------------------------------- build
    def attention_modes(self, values: Mapping[str, Any]) -> tuple[str, str, str]:
        """The masks, global and local attention modes of resolved ``values``.

        ``kernels.inductor`` overrides the modes of the layers it moves
        (:meth:`InductorPlacement.modes`). ``workload.masks`` matters only
        when some layer runs in ``mask`` mode; otherwise Hugging Face builds
        no mask (``positions``), so configs that differ only in an unused
        masks value build the same model.
        """
        placement = inductor_placement(values.get("kernels.inductor", "none"))
        global_mode, local_mode = placement.modes(
            values.get("workload.global_attention", "mask"),
            values.get("workload.local_attention", "mask"),
        )
        windows = layer_windows(self.config())
        masked = (global_mode == "mask" and None in windows) or (
            local_mode == "mask" and any(w is not None for w in windows)
        )
        masks = values["workload.masks"] if masked else "positions"
        return masks, global_mode, local_mode

    def attention(self, values: Mapping[str, Any]) -> str:
        """Register and name the ``attn_implementation`` of resolved ``values``."""
        return attention_implementation(*self.attention_modes(values))

    def _load(self, precision: str, attn: str) -> torch.nn.Module:
        from transformers import AutoModelForCausalLM

        return (
            AutoModelForCausalLM.from_pretrained(
                self.model_name, attn_implementation=attn, dtype=_DTYPES[precision]
            )
            .eval()
            .cuda()
        )

    def _structure(self, model: torch.nn.Module) -> dict[str, Any]:
        flops = prefill_flops(model, self.length, self.kept)
        structure: dict[str, Any] = {
            "prompt_sha256": hashlib.sha256(
                self.tokens().numpy().tobytes()
            ).hexdigest()[:16],
            **{key: round(value, 2) for key, value in flops.items()},
        }
        for suffix, fp16_accumulation in (("", False), ("_fp16_acc", True)):
            peak = peak_tflops(fp16_accumulation)
            if peak:
                structure[f"peak_tflops{suffix}"] = round(peak, 1)
                structure[f"compute_floor_ms{suffix}"] = round(
                    flops["gflop_per_unit"] / peak, 3
                )
        return structure

    def build(self, values: Mapping[str, Any]) -> Candidate:
        settings = bench_settings(values)
        if settings.backend != "tensorrt":
            return self._build_torch(values, settings)
        import torch_tensorrt

        kernels = values.get("kernels.inductor", "none")
        placement = inductor_placement(kernels)
        if placement.layers and not values.get("runtime.cudagraphs"):
            raise ValueError(
                f"kernels.inductor={kernels!r} needs runtime.cudagraphs=True: "
                "each moved layer splits the engine"
            )
        generated = self.generated_kernels(values)
        attn = self.attention(values)
        model = self._load(values["engine.precision"], attn)
        structure = {"attn_implementation": attn, **self._structure(model)}
        if generated != "none":
            structure.update(self._install_generated(model))
        if placement.layers:
            moved = placement.moved(layer_windows(self.config()))
            if placement.route == "region-rope":
                install_attention_regions(model, moved)
            elif placement.route == "plugin":
                key = inductor_rope_kernels(
                    model.model.layers[moved[0]].self_attn, self.length
                )
                install_attention_regions(model, moved, plugin=key)
                structure["rope_plugin"] = {
                    role: hosted.describe()
                    for role, hosted in _ROPE_KERNELS[key].items()
                }
                structure["rope_plugin_layers"] = len(moved)
            structure["kernels_inductor"] = kernels
            structure["moved_attention_layers"] = (
                len(moved) if placement.moves_attention else 0
            )
        ids = self.tokens().cuda().unsqueeze(0)
        options: dict[str, Any] = {
            "arg_inputs": [ids],
            "optimization_level": values["engine.optimization_level"],
            "tiling_optimization_level": values["engine.tiling_optimization_level"],
        }
        if values.get("engine.build_route"):
            options["build_route"] = values["engine.build_route"]
        prefill = _Prefill(model, self.positions().cuda())
        if placement.regions or generated != "none":
            # Region capture needs strict export through torch_tensorrt.compile,
            # of a module entirely in eval mode (the wrapper included).
            with torch.no_grad():
                module = torch_tensorrt.compile(
                    prefill.eval(), ir="dynamo", strict=True, **options
                )
            release_strict_export_traces()
        else:
            with torch.no_grad():
                program = torch.export.export(prefill, (ids,), strict=False)
            module = torch_tensorrt.dynamo.compile(program, **options)
            del program
        del prefill, model
        gc.collect()
        torch.cuda.empty_cache()
        if placement.regions:
            structure.update(_compile_regions(module, ids))
        return _TensorRTPrefill(
            module,
            ids,
            cudagraphs=values["runtime.cudagraphs"],
            pre_allocated_outputs=values["runtime.pre_allocated_outputs"],
            structure=structure,
        )

    def generated_kernels(self, values: Mapping[str, Any]) -> str:
        """Validate ``kernels.generated``; return its value (``"none"`` when unset).

        Its value needs a model with q/k norms, CUDA graphs and
        ``kernels.inductor="none"`` (the knob's ``requires``, checked here
        too for direct builds).
        """
        import region_generation

        generated = values.get("kernels.generated", "none")
        if generated in GENERATED_KERNELS[1:] and not self.rope_regions():
            raise ValueError(
                f"kernels.generated={generated!r} needs a model whose attention "
                f"has q/k norms ({', '.join(_ROPE_REGION_MODELS)})"
            )
        return region_generation.check_generated(
            values, GENERATED_KERNELS, conflict="its values rewrite the same attention"
        )

    def kernel_generator(self) -> Any:
        """The KernelGenerator of ``kernels.generated``: the shared settings
        (``region_generation.make_generator``), the same in every build. No
        notes: no measured fact about these regions applies."""
        import region_generation

        return region_generation.make_generator(self.kernel_cache)

    def _install_generated(self, model: torch.nn.Module) -> dict[str, Any]:
        """Patch every attention for ``kernels.generated="qk-norm-rope"``.

        Each layer's q and k RMSNorm + rotary embedding scope gets
        :meth:`kernel_generator`, which writes, checks and caches its kernel
        when the model compiles, or reuses the cached one. Returns what the
        build reports.
        """
        import region_generation

        generator = self.kernel_generator()
        layers = list(range(len(model.model.layers)))
        install_attention_regions(model, layers, generator=generator)
        return {
            "kernels_generated": "qk-norm-rope",
            "generated_regions": 2 * len(layers),
            **region_generation.describe(generator),
        }

    def _build_torch(
        self, values: Mapping[str, Any], settings: "BenchSettings"
    ) -> Candidate:
        """The Hugging Face forward run eagerly or compiled by Inductor.

        Compilation (and CUDA-graph recording) happens in three warm-up calls
        here, so the build time includes it.
        """
        kernels = values.get("kernels.inductor", "none")
        if kernels != "none":
            raise ValueError(
                f"kernels.inductor={kernels!r} places attention around TensorRT "
                f"engines; bench.backend {settings.backend!r} cannot honor it"
            )
        generated = values.get("kernels.generated", "none")
        if generated != "none":
            raise ValueError(
                f"kernels.generated={generated!r} runs kernels inside TensorRT "
                f"engines; bench.backend {settings.backend!r} cannot honor it"
            )
        # An explicit Hugging Face attn_implementation wins; otherwise the
        # workload's attention knobs apply ("sdpa" for their defaults).
        attn = settings.attn or self.attention(values)
        stack = torch_backend_context(settings, attn)
        try:
            model = self._load(values["engine.precision"], attn)
            structure = {
                "trt_engines": 0,
                "torch_partitions": 0,
                "plugin_regions": 0,
                "backend": settings.backend,
                "mode": settings.mode if settings.backend == "inductor" else None,
                "attn": attn,
                "inductor": dict(settings.inductor),
                "fp16_accumulation": settings.fp16_accumulation,
                **self._structure(model),
            }
            forward = _Prefill(model, self.positions().cuda())
            if settings.backend == "inductor":
                forward = torch.compile(
                    forward, mode=settings.mode, fullgraph=True, dynamic=False
                )
            candidate = _TorchPrefill(
                forward,
                self.tokens().cuda().unsqueeze(0),
                cudagraphs=settings.cudagraphs,
                structure=structure,
                stack=stack,
            )
            start = time.perf_counter()
            for _ in range(3):
                candidate()
            torch.cuda.synchronize()
        except BaseException:
            stack.close()
            raise
        structure["warmup_s"] = round(time.perf_counter() - start, 2)
        return candidate


# ----------------------------------------------------------------- benchmarks
@dataclasses.dataclass(frozen=True)
class BenchSettings:
    """The head-to-head ``bench.*`` values of one build (module docstring)."""

    backend: str = "tensorrt"
    mode: str = "default"
    attn: str | None = None
    inductor: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    fp16_accumulation: bool = False

    @property
    def cudagraphs(self) -> bool:
        """Whether torch.compile replays the forward from CUDA graphs."""
        return self.backend == "inductor" and self.mode in _CUDAGRAPH_MODES


def bench_settings(values: Mapping[str, Any]) -> BenchSettings:
    """Validate and collect the ``bench.*`` values of resolved ``values``.

    Unknown keys, and keys that do not apply to the backend (a torch.compile
    mode for eager, say), raise ``ValueError`` instead of being ignored.
    """
    bench = {key: value for key, value in values.items() if key.startswith("bench.")}
    unknown = sorted(set(bench) - set(_BENCH_KEYS))
    if unknown:
        raise ValueError(f"Unknown bench values {unknown}; known: {list(_BENCH_KEYS)}")
    overrides = bench.get("bench.inductor") or {}
    if not isinstance(overrides, Mapping):
        raise ValueError(f"bench.inductor must be a mapping; got {overrides!r}")
    settings = BenchSettings(
        backend=bench.get("bench.backend") or "tensorrt",
        mode=bench.get("bench.mode") or "default",
        attn=bench.get("bench.attn") or None,
        inductor=dict(overrides),
        fp16_accumulation=bool(bench.get("bench.fp16_accumulation")),
    )
    if settings.backend not in BACKENDS:
        raise ValueError(
            f"bench.backend must be one of {BACKENDS}; got {settings.backend!r}"
        )
    if settings.mode not in TORCH_COMPILE_MODES:
        raise ValueError(
            f"bench.mode must be one of {TORCH_COMPILE_MODES}; got {settings.mode!r}"
        )
    applicable = {
        "tensorrt": ("bench.backend",),
        "eager": ("bench.backend", "bench.attn", "bench.fp16_accumulation"),
        "inductor": _BENCH_KEYS,
    }[settings.backend]
    given = sorted(
        key
        for key, value in bench.items()
        if value and key not in applicable and (key, value) != ("bench.mode", "default")
    )
    if given:
        raise ValueError(f"{given} do not apply to bench.backend {settings.backend!r}")
    if settings.fp16_accumulation and values.get("engine.precision") != "fp16":
        raise ValueError("bench.fp16_accumulation needs engine.precision 'fp16'")
    return settings


def torch_backend_context(settings: BenchSettings, attn: str) -> contextlib.ExitStack:
    """The global settings an eager or Inductor build runs under, entered.

    FP16 accumulation in cuBLAS, the Inductor option overrides and (Inductor
    with FlexAttention) the traceable block masks of
    :func:`_traceable_block_masks`. Closing the stack restores them all; the
    candidate does so when it closes, after its last call.
    """
    stack = contextlib.ExitStack()
    try:
        if settings.fp16_accumulation:
            stack.enter_context(_fp16_accumulation())
        if settings.backend == "inductor":
            import torch._inductor.config as inductor_config

            for key in settings.inductor:
                if not hasattr(inductor_config, key):
                    raise ValueError(f"Unknown torch._inductor.config option {key!r}")
            stack.enter_context(inductor_config.patch(dict(settings.inductor)))
            if attn == "flex_attention":
                stack.enter_context(_traceable_block_masks())
    except BaseException:
        stack.close()
        raise
    return stack


@contextlib.contextmanager
def _fp16_accumulation() -> Iterator[None]:
    """Let FP16 cuBLAS GEMMs accumulate in FP16, as TensorRT's FP16 tactics do."""
    matmul = torch.backends.cuda.matmul
    allowed = matmul.allow_fp16_accumulation
    matmul.allow_fp16_accumulation = True
    try:
        yield
    finally:
        matmul.allow_fp16_accumulation = allowed


@contextlib.contextmanager
def _traceable_block_masks() -> Iterator[None]:
    """Let Dynamo trace transformers' FlexAttention block masks (Inductor only).

    transformers 5.18 calls ``create_block_mask(_compile=True)``, which this
    torch nightly deprecates with a warning that Dynamo cannot trace inside a
    compiled forward. Inside ``torch.compile`` the mask creation is compiled
    with the forward, so drop the flag, as the warning itself recommends.
    Eager keeps the flag: it compiles the mask builder there.
    """
    import transformers.masking_utils as masking_utils
    from torch.nn.attention.flex_attention import create_block_mask

    def create_block_mask_without_compile_flag(
        *args: Any, _compile: bool = False, **kwargs: Any
    ) -> Any:
        return create_block_mask(*args, **kwargs)

    original = masking_utils.create_block_mask
    masking_utils.create_block_mask = create_block_mask_without_compile_flag
    try:
        yield
    finally:
        masking_utils.create_block_mask = original


# ----------------------------------------------------------------- candidates
def release_strict_export_traces() -> int:
    """Drop the traced modules strict export leaves behind; returns their number.

    With this PyTorch, ``torch.export.export(strict=True)`` installs Dynamo's
    compiled function, which holds the traced module and through it every
    parameter of the exported model, as a ``__compiled_fn_*`` global of
    ``torch._dynamo.functional_export``, and nothing removes it: after a
    strict build the model's weights would stay on the GPU next to the
    engines' own copies. Nothing calls these globals after the export.
    """
    module = sys.modules.get("torch._dynamo.functional_export")
    if module is None:
        return 0
    names = [name for name in vars(module) if name.startswith("__compiled_fn")]
    for name in names:
        delattr(module, name)
    return len(names)


def _compile_regions(module: torch.nn.Module, ids: torch.Tensor) -> dict[str, Any]:
    """Run ``module`` once, so Inductor compiles its regions during the build.

    Torch-TensorRT compiles ``execute_in_torch(backend="inductor")`` regions
    on their first execution. Returns the number of Inductor regions and the
    seconds the first call took.
    """
    from torch_tensorrt.dynamo.region_compilation import get_region_records

    regions = sum(
        1 for record in get_region_records(module) if record.backend == "inductor"
    )
    start = time.perf_counter()
    with torch.inference_mode():
        module(ids)
    torch.cuda.synchronize()
    return {
        "inductor_regions": regions,
        "region_first_call_s": round(time.perf_counter() - start, 2),
    }


class _TensorRTPrefill(Candidate):
    def __init__(
        self,
        module: torch.nn.Module,
        ids: torch.Tensor,
        *,
        cudagraphs: bool,
        pre_allocated_outputs: bool,
        structure: Mapping[str, Any],
    ) -> None:
        self.module = module
        self.ids = ids
        self.cudagraphs = cudagraphs
        self.pre_allocated_outputs = pre_allocated_outputs
        self.structure = dict(structure)
        self._stack = contextlib.ExitStack()
        self._run: Any = module

    def setup(self) -> None:
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
        with torch.inference_mode():
            return self._run(self.ids)

    def describe(self) -> dict[str, Any]:
        from torch_tensorrt.autotune import describe_module

        return {**describe_module(self.module), **self.structure}

    def export(self, directory: Path) -> list[str]:
        """Save the compiled program as ``prefill.ep``.

        ``kernels.inductor`` region builds cannot be saved: the exporter
        rejects ``execute_in_torch`` regions whose backend is not ``eager``
        (it validates before writing anything). That rejection is reported as
        ``NotImplementedError``, the contract's "cannot export", naming the
        value; other errors propagate unchanged.
        """
        import torch_tensorrt

        path = Path(directory) / "prefill.ep"
        try:
            torch_tensorrt.save(self.module, str(path), retrace=False)
        except region.RegionError as error:
            kernels = self.structure.get("kernels_inductor")
            if not self.structure.get("inductor_regions"):
                raise
            raise NotImplementedError(
                f"kernels.inductor={kernels!r} builds cannot be saved: their "
                "execute_in_torch(backend='inductor') regions run code that "
                "Inductor compiles at run time, and torch_tensorrt.save cannot "
                "preserve that backend. Rebuild the config instead; the "
                "attention-fa2-* values run the same attention operator "
                "without regions and save normally."
            ) from error
        return [path.name]

    def close(self) -> None:
        self._stack.close()
        self._run = self.module


class _TorchPrefill(Candidate):
    def __init__(
        self,
        forward: Any,
        ids: torch.Tensor,
        *,
        cudagraphs: bool,
        structure: Mapping[str, Any],
        stack: contextlib.ExitStack | None = None,
    ) -> None:
        self.forward = forward
        self.ids = ids
        self.cudagraphs = cudagraphs
        self.structure = structure
        # The global settings of the build (torch_backend_context).
        self._stack = stack or contextlib.ExitStack()

    def __call__(self) -> torch.Tensor:
        with torch.inference_mode():
            if self.cudagraphs:
                torch.compiler.cudagraph_mark_step_begin()
            return self.forward(self.ids)

    def describe(self) -> dict[str, Any]:
        return dict(self.structure)

    def export(self, directory: Path) -> list[str]:
        return []

    def close(self) -> None:
        self._stack.close()
