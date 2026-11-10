# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Qwen3.5 batch-1 prefill workload for ``torch_tensorrt.autotune`` sessions.

The measured unit is one prefill: a forward pass of the text causal LM of
``Qwen/Qwen3.5-0.8B`` over a fixed ``length``-token prompt from an empty
state, returning the logits at ``positions`` evenly spaced positions, the
last one included (``[positions, vocab]`` in the model's dtype; the LM head
runs on those rows only). They are checked against the FP32 eager Hugging
Face model on the same tokens: top-1 agreement, mean and largest KL
divergence and the largest logit error, with the default limits of
:meth:`PrefillWorkload.accuracy_defaults`. The prompt is the start of the
Python language reference that ships with CPython (``pydoc_data.topics``,
each paragraph once): long English that never repeats, as in
``agentic_prefill.py``.

Qwen3.5 interleaves 18 Gated DeltaNet linear-attention layers with 6 gated
full-attention layers (8 query and 2 key/value heads of dimension 256).
Without ``fla``, Hugging Face computes Gated DeltaNet with a PyTorch
fallback (a chunk loop with in-place writes, a triangular solve in eager
mode and a 63-step forward substitution under export) that TensorRT has no
layer for; ``qwen35_gdn.py`` computes the same math as a functional graph.

Workload knobs:

* ``workload.gdn``: ``"hf"`` exports Hugging Face's fallback as it is (the
  baseline: it exports and builds, in about five minutes); ``"functional"``
  swaps every Gated DeltaNet layer's forward for
  ``qwen35_gdn.functional_forward`` before export.
* ``workload.chunk_size`` and ``workload.conv`` (functional only): the chunk
  length, and the causal convolution: Hugging Face's channel-major
  ``F.conv1d`` or token-major shifted multiply-adds.
* ``kernels.gdn_scan`` (functional only): the inter-chunk state recurrence
  as one Triton kernel per layer inside the engine
  (``torch_tensorrt.region.fused_kernel``, strict export) with IEEE-FP32 or
  TF32 dots, instead of an unrolled loop of small matrix products. Requires
  CUDA graphs: TensorRT 11.4 reloads plugin modules on every enqueue
  outside them. The build compiles the kernel first and refuses it if the
  plugin launch could not run it (``qwen35_gdn.check_scan_kernel``).
  ``generated`` runs the same region with a kernel that PR 4914's
  ``torch_tensorrt.region.KernelGenerator`` writes during the build
  (``region_generation.py``), held to the error of the region's FP32
  products with TF32 (as ``triton-tf32``). Generated kernels are cached in
  ``kernel_cache`` (``region_generation.kernel_cache``), keyed by the engine
  options; a session prepares the value once, before its search.
* ``workload.attention``: the 6 full-attention layers with Hugging Face's
  mask, or SDPA ``is_causal`` with native GQA. Under ``torch.export`` and
  ``torch.compile`` Hugging Face cannot run its data-dependent
  packed-sequence check, so it always materializes a dense ``length x
  length`` mask (eager mode skips it and calls SDPA ``is_causal``).
* ``kernels.inductor``: hybrid builds that hand chosen operations to PyTorch
  Inductor inside the TensorRT model (``torch_tensorrt.region.
  execute_in_torch(backend="inductor")``, strict export): ``"attention"``
  the attention core of the 6 full-attention layers
  (:func:`inductor_causal_attention`), ``"prologue"`` the prologue of the 18
  Gated DeltaNet layers (``qwen35_gdn.prologue_region_forward``),
  ``"prologue-decay"`` the prologue with the gates and the chunk rule's
  elementwise head, or the attention core with either prologue.
  Every region ends one engine and starts the next; the build keeps the
  regions' weights on the GPU, and the candidate runs the regions under the
  config's matrix precision policy and replays the whole model from one CUDA
  graph (``runtime.cudagraphs``). GPU memory grows with the number of
  engines and regions (the knob's description says why). The prologue
  values refuse a ``gdn`` edit of the functions they do not run
  (:func:`prologue_edit_conflicts`). Builds with regions are not exported:
  ``torch_tensorrt.save`` refuses regions whose backend is ``inductor``.
* ``engine.precision``, ``engine.use_fp32_acc`` (FP16 only),
  ``engine.disable_tf32``, ``engine.decompose_attention`` plus the standard
  builder and runtime knobs.

An editable file, ``gdn``, lets an LLM planner edit the functional Gated
DeltaNet's numerics, the Triton scan kernel and its launch settings
(``qwen35_gdn.py`` inside its ``EDITABLE`` fences).

The TensorRT baseline is the straightforward compile of the BF16 model (its
native dtype) with Hugging Face's own Gated DeltaNet and attention:
``torch.export.export`` and ``torch_tensorrt.dynamo.compile`` with the
``tools/llm`` settings (``min_block_size=1``, ``disable_tf32=True`` so FP32
stays FP32 as in Hugging Face, ``use_fp32_acc`` with FP16, Python runtime,
module offloaded to the CPU), no CUDA graphs.

Head-to-head benchmarks (``bench_compare.py``) reuse the workload through
``bench.*`` values, which are not knobs (a session never sets them):
``bench.backend`` ``"tensorrt"``
(default) builds the config; ``"eager"`` and ``"inductor"`` run the same
forward eagerly or compiled by ``torch.compile(mode=bench.mode,
fullgraph=True, dynamic=False)`` (``bench.mode``: a torch.compile mode,
default ``"default"``). ``bench.inductor`` is a dict of
``torch._inductor.config`` overrides, such as
``{"coordinate_descent_tuning": True}``. ``bench.attn`` is an explicit
Hugging Face ``attn_implementation`` (``"sdpa"``, ``"eager"``; transformers
5.18 offers no FlexAttention for Qwen3.5); without it ``workload.attention``
applies. Eager and Inductor honor the config's formulation and precision
policy as TensorRT does, so the same config on another backend is a matched
pair:

* ``workload.gdn``, ``workload.conv``, ``workload.chunk_size``: the Gated
  DeltaNet formulation (Hugging Face's fallback by default);
* ``workload.attention``: ``causal`` registers :func:`causal_attention`;
* ``engine.precision``: the weight and activation dtype;
* ``engine.use_fp32_acc=False`` (FP16): sets
  ``torch.backends.cuda.matmul.allow_fp16_accumulation``, so FP16 cuBLAS
  GEMMs may accumulate in FP16 as TensorRT's FP16 GEMM tactics do;
* ``engine.disable_tf32=False``: allows TF32 in FP32 matrix products and
  convolutions (``allow_tf32``).

Config values only TensorRT can honor (``kernels.gdn_scan``,
``kernels.inductor``, ``engine.decompose_attention``, the runtime and other
builder knobs) are refused at a non-default value, as are unknown
``bench.*`` keys and keys that do not apply to the backend
(:func:`bench_settings`). Inductor compiles and records its CUDA graphs in
``build``; the global settings a build changes are restored when the
candidate closes.

Prefill is compute-bound, so candidates do not declare ``bytes_per_unit``;
their structure reports ``gflop_per_unit`` and ``compute_floor_ms`` at the
GPU's measured dense BF16 GEMM throughput (FP32 accumulation) instead, and
``compute_floor_ms_fp16_acc`` at the FP16-accumulation GEMM throughput.
"""

from __future__ import annotations

import contextlib
import dataclasses
import gc
import time
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch_tensorrt.autotune import (
    STANDARD_KNOBS,
    Candidate,
    EditableFile,
    Knob,
    Timing,
    Workload,
    load_module,
)

_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
_STANDARD = (
    "runtime.cudagraphs",
    "runtime.pre_allocated_outputs",
    "engine.optimization_level",
    "engine.tiling_optimization_level",
)
BACKENDS = ("tensorrt", "eager", "inductor")
TORCH_COMPILE_MODES = (
    "default",
    "reduce-overhead",
    "max-autotune",
    "max-autotune-no-cudagraphs",
)
ATTENTION = ("mask", "causal")
# kernels.gdn_scan values: the scan (qwen35_gdn.GDNSettings.scan) and the
# Triton scan's dot precision.
SCANS = {
    "loop": ("loop", False),
    "triton": ("triton", False),
    "triton-tf32": ("triton", True),
    "generated": ("generated", False),
}
# How exact the generated scan kernel's FP32 matrix products must be
# (KernelGenerator.fp32_matmul_precision): "high" also accepts the error the
# region's own products have with TF32, the precision triton-tf32 runs at.
# One setting for every build, so the build that generates the kernel and
# every later build ask for the same kernel.
GENERATED_SCAN_PRECISION = "high"
# kernels.inductor values: (whether the attention core runs in Inductor, where
# the Gated DeltaNet prologue runs: qwen35_gdn.GDNSettings.prologue).
INDUCTOR_REGIONS = {
    "none": (False, "tensorrt"),
    "attention": (True, "tensorrt"),
    "prologue": (False, "inductor"),
    "prologue-decay": (False, "inductor-decay"),
    "attention+prologue": (True, "inductor"),
    "attention+prologue-decay": (True, "inductor-decay"),
}
# torch.compile modes that replay the forward from CUDA graphs.
_CUDAGRAPH_MODES = ("reduce-overhead", "max-autotune")
_CAUSAL = "qwen35_prefill_causal"
_CAUSAL_INDUCTOR = "qwen35_prefill_causal_inductor"
# The head-to-head keys build() understands (see bench_settings).
_BENCH_KEYS = ("bench.backend", "bench.mode", "bench.attn", "bench.inductor")
# Knobs eager and Inductor cannot honor: refused at non-default values.
_TENSORRT_ONLY = (
    "kernels.gdn_scan",
    "kernels.inductor",
    "engine.decompose_attention",
    *_STANDARD,
)


def create_case(
    model: str = "Qwen/Qwen3.5-0.8B",
    length: int = 2048,
    positions: int = 64,
    baseline: Mapping[str, Any] | None = None,
    kernel_cache: str | None = None,
) -> "PrefillWorkload":
    """Workload factory for ``tools/llm/qwen35_prefill.py:create_case``."""
    return PrefillWorkload(
        model,
        length=length,
        positions=positions,
        baseline=baseline,
        kernel_cache=kernel_cache,
    )


# ------------------------------------------------------------------ inputs
def prompt_text(characters: int) -> str:
    """At least ``characters`` characters of CPython's language reference.

    ``pydoc_data.topics``, all topics in sorted order, each paragraph once
    (some topics repeat sections of others).
    """
    from pydoc_data.topics import topics

    paragraphs = dict.fromkeys(
        paragraph
        for key in sorted(topics)
        for paragraph in topics[key].split("\n\n")
        if paragraph.strip()
    )
    text = "\n\n".join(paragraphs)
    if len(text) < characters:
        text = text * -(-characters // len(text))
    return text[:characters]


def logit_positions(length: int, count: int) -> list[int]:
    """``count`` evenly spaced prompt positions ending at the last one."""
    if not 1 <= count <= length:
        raise ValueError(f"Need 1 <= positions <= length; got {count} and {length}")
    return [(i + 1) * length // count - 1 for i in range(count)]


# ---------------------------------------------------------------- roofline
_PEAK_TFLOPS: dict[bool, float] = {}


def peak_tflops(fp16_accumulation: bool = False) -> float | None:
    """Dense GEMM throughput of this GPU (TFLOP/s), measured once per process.

    BF16 inputs with FP32 accumulation by default; with ``fp16_accumulation``,
    FP16 inputs with FP16 accumulation (what TensorRT picks for FP16 engines
    without ``use_fp32_acc``). 8192^3 cuBLAS GEMMs, best of six.
    """
    if fp16_accumulation not in _PEAK_TFLOPS and torch.cuda.is_available():
        dtype = torch.float16 if fp16_accumulation else torch.bfloat16
        try:
            a = torch.randn(8192, 8192, device="cuda", dtype=dtype)
            b = torch.randn(8192, 8192, device="cuda", dtype=dtype)
        except RuntimeError:  # not enough free memory
            return None
        with _torch_matmul_flags(tf32=False, fp16_accumulation=fp16_accumulation):
            best_ms = float("inf")
            for attempt in range(8):
                start, end = torch.cuda.Event(True), torch.cuda.Event(True)
                start.record()
                a @ b
                end.record()
                end.synchronize()
                if attempt >= 2:
                    best_ms = min(best_ms, start.elapsed_time(end))
        _PEAK_TFLOPS[fp16_accumulation] = 2 * 8192**3 / best_ms / 1e9
        del a, b
        torch.cuda.empty_cache()
    return _PEAK_TFLOPS.get(fp16_accumulation)


@contextlib.contextmanager
def _torch_matmul_flags(*, tf32: bool, fp16_accumulation: bool) -> Iterator[None]:
    """PyTorch's TF32 (matrix products and cuDNN) and FP16-accumulation flags, restored.

    Uses the legacy ``allow_tf32`` flags throughout, as :meth:`PrefillWorkload.
    reference` does: this torch refuses to read them after a mix of the
    legacy and the ``fp32_precision`` API. Inductor follows them.
    """
    matmul, cudnn = torch.backends.cuda.matmul, torch.backends.cudnn
    saved = (matmul.allow_tf32, cudnn.allow_tf32, matmul.allow_fp16_accumulation)
    matmul.allow_tf32 = tf32
    _set_cudnn_tf32(tf32)
    matmul.allow_fp16_accumulation = fp16_accumulation
    try:
        yield
    finally:
        matmul.allow_tf32 = saved[0]
        _set_cudnn_tf32(saved[1])
        matmul.allow_fp16_accumulation = saved[2]


def matmul_policy(values: Mapping[str, Any]) -> dict[str, bool]:
    """The config's matrix precision policy as :func:`_torch_matmul_flags` arguments.

    TF32 with ``engine.disable_tf32=False``; FP16 accumulation with
    ``engine.precision=fp16`` and ``engine.use_fp32_acc=False``. Eager and
    Inductor builds, and the Inductor regions of ``kernels.inductor``, run
    under it.
    """
    precision = values.get("engine.precision", "bf16")
    return {
        "tf32": not values.get("engine.disable_tf32", True),
        "fp16_accumulation": precision == "fp16"
        and not values.get("engine.use_fp32_acc", True),
    }


def _set_cudnn_tf32(value: bool) -> None:
    """Set cuDNN's TF32 flag, also once ``torch.backends.disable_global_flags()``
    has frozen the attribute (PyTorch's test utilities call it on import)."""
    torch.backends.cudnn.set_flags(_allow_tf32=value, _fp32_precision=None)


def prefill_flops(config: Any, length: int, positions: int, chunk_size: int) -> float:
    """Matrix-product FLOPs of one prefill (projections, attention, GDN, lm_head).

    Causal attention counts the lower triangle only; Gated DeltaNet counts
    the functional formulation's chunk math (``qwen35_gdn.flops_per_layer``);
    elementwise work and the depthwise convolution are left out.
    """
    from qwen35_gdn import flops_per_layer

    hidden, inter = config.hidden_size, config.intermediate_size
    heads, kv_heads = config.num_attention_heads, config.num_key_value_heads
    head_dim = getattr(config, "head_dim", None) or hidden // heads
    lk, lv = config.linear_key_head_dim, config.linear_value_head_dim
    lkh, lvh = config.linear_num_key_heads, config.linear_num_value_heads
    # in_proj_qkv, in_proj_z, in_proj_b, in_proj_a and out_proj.
    gdn_proj = hidden * (2 * lk * lkh + 2 * lv * lvh + 2 * lvh) + lv * lvh * hidden
    # q_proj (with its output gate), k_proj, v_proj and o_proj.
    attn_proj = hidden * (2 * heads * head_dim + 2 * kv_heads * head_dim)
    attn_proj += heads * head_dim * hidden
    mlp = 3 * hidden * inter
    types = list(config.layer_types)
    gdn, full = types.count("linear_attention"), types.count("full_attention")
    flops = 2.0 * length * (mlp * len(types) + gdn_proj * gdn + attn_proj * full)
    flops += full * 2.0 * 2 * heads * head_dim * length * (length + 1) / 2
    flops += gdn * flops_per_layer(length, lvh, lk, lv, chunk_size)
    flops += 2.0 * positions * hidden * config.vocab_size
    return flops


# --------------------------------------------------------------- attention
def causal_attention(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float = 0.0,
    scaling: float | None = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, None]:
    """A Hugging Face attention function: SDPA ``is_causal`` with native GQA.

    For a prefill from position 0 (queries and keys of equal length) it
    computes what the materialized causal mask does, without a ``length x
    length`` mask tensor and without repeating keys and values per query
    head.
    """
    if dropout or query.shape[2] != key.shape[2]:
        raise ValueError("causal_attention is for prefill without dropout")
    out = F.scaled_dot_product_attention(
        query,
        key,
        value,
        is_causal=True,
        scale=scaling,
        enable_gqa=key.shape[1] != query.shape[1],
    )
    return out.transpose(1, 2).contiguous(), None


def inductor_causal_attention(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float = 0.0,
    scaling: float | None = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, None]:
    """:func:`causal_attention` in an Inductor region (``kernels.inductor``).

    The SDPA call and its output transpose run inside
    ``torch_tensorrt.region.execute_in_torch(name="attn_core",
    backend="inductor")``: under ``torch_tensorrt.compile(..., strict=True)``
    TensorRT's engine ends at the rotated queries and the keys and values,
    Inductor compiles the scope (it calls PyTorch's fused SDPA kernels for
    it), and the next engine starts at the token-major output. Elsewhere it
    computes exactly what :func:`causal_attention` computes.
    """
    from torch_tensorrt import region

    with region.execute_in_torch(name="attn_core", backend="inductor"):
        out, _ = causal_attention(
            module, query, key, value, attention_mask, dropout, scaling, **kwargs
        )
    return out, None


def attention_implementation(attention: str, inductor: bool = False) -> str:
    """Register (once) and name the implementation of ``workload.attention``.

    ``inductor`` (``kernels.inductor`` with the attention core) names the
    causal implementation that runs SDPA in an Inductor region
    (:func:`inductor_causal_attention`); it needs ``attention="causal"``.
    """
    if attention == "mask" and not inductor:
        return "sdpa"
    if attention != "causal":
        raise ValueError(
            f"workload.attention={attention!r}; choose one of {ATTENTION}"
            + (" (the Inductor attention region needs 'causal')" if inductor else "")
        )
    from transformers import AttentionInterface, AttentionMaskInterface
    from transformers.masking_utils import sdpa_mask

    name, function = (
        (_CAUSAL_INDUCTOR, inductor_causal_attention)
        if inductor
        else (_CAUSAL, causal_attention)
    )
    AttentionInterface.register(name, function)
    AttentionMaskInterface.register(name, sdpa_mask)
    return name


def inductor_regions(value: str) -> tuple[bool, str]:
    """``kernels.inductor``: (attention core in Inductor, GDN prologue setting)."""
    if value not in INDUCTOR_REGIONS:
        raise ValueError(
            f"kernels.inductor={value!r}; choose one of {tuple(INDUCTOR_REGIONS)}"
        )
    return INDUCTOR_REGIONS[value]


# Fenced qwen35_gdn functions that the prologue values of kernels.inductor do
# not run as written: prologue_region_forward (outside the fences) replaces
# functional_forward and applies the q/k L2 norms itself, and the decay value
# runs its own copy of the chunk rule (chunk_rule_head, chunk_rule_body).
_PROLOGUE_BYPASSED = ("functional_forward", "chunk_gated_delta_rule")


def prologue_edit_conflicts(gdn: Any, prologue: str) -> list[str]:
    """The ``edit.gdn`` changes that a Gated DeltaNet prologue region would leave out.

    ``gdn`` is the module :meth:`PrefillWorkload.gdn_module` returns and
    ``prologue`` the ``qwen35_gdn.GDNSettings.prologue`` value. Returns the
    functions of :data:`_PROLOGUE_BYPASSED` whose source in ``gdn`` differs
    from the shipped ``qwen35_gdn`` (and ``prologue_region_forward`` when the
    edited file predates it); empty for the shipped module or when the prologue
    stays in TensorRT. Edits elsewhere in the fences (the convolution, the
    inverse, the scan and its kernel) reach the prologue builds.
    """
    import inspect

    import qwen35_gdn

    if prologue == "tensorrt" or gdn is qwen35_gdn:
        return []
    if not hasattr(gdn, "prologue_region_forward"):
        return ["prologue_region_forward (missing: the edited file predates it)"]
    return [
        name
        for name in _PROLOGUE_BYPASSED
        if inspect.getsource(getattr(gdn, name))
        != inspect.getsource(getattr(qwen35_gdn, name))
    ]


class PrefillLogits(torch.nn.Module):
    """``input_ids [1, length]`` -> logits at fixed positions ``[positions, vocab]``."""

    def __init__(self, model: torch.nn.Module, positions: Sequence[int]) -> None:
        super().__init__()
        self.model = model
        device = next(model.parameters()).device
        self.register_buffer(
            "positions",
            torch.tensor(list(positions), dtype=torch.int64, device=device),
            persistent=False,
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.model.model(input_ids=input_ids, use_cache=False)
        selected = hidden.last_hidden_state[0].index_select(0, self.positions)
        return self.model.lm_head(selected)


# ---------------------------------------------------------------- workload
class PrefillWorkload(Workload):
    """Batch-1 prefill of Qwen3.5's text model, logits at fixed positions.

    ``kernel_cache`` is the KernelGenerator cache that
    ``kernels.gdn_scan="generated"`` takes its kernel from
    (``region_generation.kernel_cache``).
    """

    def __init__(
        self,
        model: str,
        *,
        length: int,
        positions: int,
        baseline: Mapping[str, Any] | None = None,
        kernel_cache: str | None = None,
    ) -> None:
        if length < 2:
            raise ValueError("PrefillWorkload needs a prompt of at least 2 tokens")
        import region_generation

        self.kernel_cache = region_generation.kernel_cache(kernel_cache)
        self.model_name = model
        self.length = length
        self.positions = logit_positions(length, positions)
        self.name = f"{model.rsplit('/', 1)[-1]}-prefill-{length}"
        self._baseline = dict(baseline or {})
        self._tokens: torch.Tensor | None = None
        self._config: Any = None

    def config(self) -> Any:
        """The text config (``qwen3_5_text``) of the model."""
        if self._config is None:
            from transformers import AutoConfig

            config = AutoConfig.from_pretrained(self.model_name)
            self._config = getattr(config, "text_config", None) or config
        return self._config

    # ------------------------------------------------------------ contract

    def knobs(self) -> Sequence[Knob]:
        import qwen35_gdn
        import region_generation

        functional = {"workload.gdn": "functional"}
        return (
            Knob(
                "workload.gdn",
                ("hf", "functional"),
                "hf",
                "Gated DeltaNet formulation exported into the engine (18 of 24 "
                "layers). 'hf': Hugging Face's PyTorch fallback as exported, a "
                "63-step forward substitution plus an in-place loop over 64-token "
                "chunks per layer (about 6,700 kernels per prefill; five-minute "
                "build). 'functional' (qwen35_gdn.py, same math in FP32): every "
                "chunk's decays, UT system and scores as batched matrix products, "
                "the UT inverse by block-doubling forward substitution, and only "
                "the inter-chunk state recurrence sequential (two small matrix "
                "products per chunk); about 4,300 kernels, 80-second build.",
                ("dominant_kernel", "launch_overhead"),
            ),
            Knob(
                "workload.conv",
                tuple(qwen35_gdn.CONVS),
                "conv1d",
                "Causal depthwise convolution (width 4) of the functional Gated "
                "DeltaNet. 'conv1d' is Hugging Face's: transpose to channel-major, "
                "grouped conv1d, transpose back, after which TensorRT fuses the "
                "q/k/v split and L2 norms into one kernel that reads across the "
                "strided layout (about 0.9 ms per layer at 2k tokens). 'shifts' "
                "stays token-major: four shifted multiply-adds in FP32 fused with "
                "the SiLU.",
                ("dominant_kernel",),
                functional,
            ),
            Knob(
                "workload.chunk_size",
                tuple(qwen35_gdn.CHUNK_SIZES),
                64,
                "Chunk length of the functional Gated DeltaNet. Shorter chunks make "
                "the intra-chunk systems and the UT inverse cheaper but the "
                "sequential state recurrence longer (length/chunk steps per "
                "layer); longer chunks the reverse.",
                ("dominant_kernel", "launch_overhead"),
                functional,
            ),
            Knob(
                "kernels.gdn_scan",
                tuple(SCANS),
                "loop",
                "Inter-chunk state recurrence of the functional Gated DeltaNet. "
                "'loop': unrolled in TensorRT, two small FP32 matrix products and "
                "elementwise work per chunk (about 0.3 ms per layer at 2k tokens). "
                "'triton': one Triton kernel per layer inside the engine "
                "(torch_tensorrt.region.fused_kernel, strict export) that keeps "
                "16-column FP32 state blocks on chip across all chunks, IEEE-FP32 "
                "dots (about 0.2 ms per layer in isolation); 'triton-tf32': TF32 "
                "tensor-core dots (about 0.14 ms, 10-bit mantissa products). "
                "'generated': the same region (inputs w, u, k_decayed and gamma "
                "of every layer) runs a kernel that PR 4914's KernelGenerator "
                "writes during the build instead of gdn_scan_kernel: an LLM "
                "author (Claude) proposes Triton kernels, and the generator keeps "
                "one only if it computes what the region's PyTorch loop computes "
                "and runs at least as fast as TensorRT's own build of the region; "
                "the layers share one kernel, and the editable file's kernel and "
                "SCAN_LAUNCH do not apply to it. TF32 applies to the region: its "
                "products are FP32 matrix products, and the kernel may err as "
                "much as they do on TF32 tensor cores (the precision triton-tf32 "
                "runs at), whatever engine.disable_tf32 says. "
                + region_generation.CACHE_NOTE
                + " Every value but 'loop' requires CUDA graphs: TensorRT "
                "11.4 reloads plugin modules on every enqueue outside them. "
                "Strict export changes how TensorRT fuses the elementwise work "
                "around the region (see the profile).",
                ("launch_overhead", "dominant_kernel"),
                {**functional, **region_generation.PLUGIN_REQUIRES},
                prepare=("generated",),
            ),
            Knob(
                "workload.attention",
                ATTENTION,
                "mask",
                "The 6 full-attention layers (head_dim 256, 8 query and 2 key/value "
                "heads). 'mask': Hugging Face's SDPA with its causal mask, which "
                "export materializes and TensorRT's attention layer reads as a "
                "dense length x length bias. 'causal': SDPA is_causal with native "
                "GQA (no mask tensor, keys and values not repeated), so TensorRT "
                "can skip the masked tiles.",
                ("dominant_kernel",),
            ),
            Knob(
                "engine.precision",
                ("bf16", "fp16", "fp32"),
                "bf16",
                "Weight and activation precision (bf16 is native; FP32 "
                "accumulation in BF16/FP16 matrix products). Gated DeltaNet's "
                "chunk math and the norms stay FP32 in every case, as in Hugging "
                "Face. fp32 runs the whole model in FP32 (without TF32 unless "
                "engine.disable_tf32=False).",
                ("precision", "dominant_kernel"),
            ),
            Knob(
                "engine.use_fp32_acc",
                (True, False),
                True,
                "With fp16, accumulate matrix products in FP32 by inserting casts "
                "(the documented tools/llm setting). False lets TensorRT pick "
                "FP16-accumulation GEMMs, which run at twice the tensor-core rate "
                "of FP32 accumulation on GeForce GPUs, with less accurate sums.",
                ("dominant_kernel", "precision"),
                {"engine.precision": "fp16"},
            ),
            Knob(
                "engine.disable_tf32",
                (True, False),
                True,
                "Disable TF32 in FP32 matrix products (Gated DeltaNet's chunk "
                "math). False lets TensorRT run them on TF32 tensor cores: "
                "faster, with 10-bit-mantissa products.",
                ("dominant_kernel",),
            ),
            Knob(
                "engine.decompose_attention",
                (False, True),
                False,
                "Decompose scaled_dot_product_attention into matrix products and a "
                "softmax instead of TensorRT's attention layer.",
                ("dominant_kernel",),
            ),
            Knob(
                "kernels.inductor",
                tuple(INDUCTOR_REGIONS),
                "none",
                "Hybrid builds that hand chosen operations to PyTorch Inductor "
                "inside the TensorRT model (torch_tensorrt.region.execute_in_torch("
                "backend='inductor'), strict export). 'attention': the attention "
                "core of the 6 full-attention layers (causal SDPA with native GQA "
                "and its output transpose) runs as Inductor compiles it, which calls "
                "PyTorch's fused SDPA kernels (FlashAttention-2), instead of "
                "TensorRT's attention kernel: 6 regions, 7 engines, 12 engine "
                "boundaries. 'prologue': each Gated DeltaNet layer's in_proj_qkv "
                "projection, causal convolution and SiLU, q/k/v split and FP32 L2 "
                "norms run as Inductor builds them (a cuBLAS GEMM and generated "
                "Triton kernels): 18 regions, 19 engines, 36 boundaries; the gates "
                "and the chunk rule's elementwise work on q, k and v (scale, "
                "decays, beta products) stay in TensorRT, which reads the region's "
                "FP32 q and k and its v back. 'prologue-decay': the same regions "
                "extended through the gates and that elementwise head (cumulative "
                "decays, pairwise decay mask, k_beta, v_beta, the w input, decayed "
                "q and k; the workload's own copy of that part of the chunk rule), "
                "so TensorRT keeps the chunk matrix products, the inverse, the scan "
                "and the outputs, and the region writes all of their inputs (about "
                "120 MB per layer at 2k tokens). With either prologue the in_proj_qkv "
                "GEMM leaves TensorRT's merged input projection (in_proj_z/b/a stay "
                "in TensorRT). 'attention+prologue' and 'attention+prologue-decay': "
                "both kinds of region (24 regions, 25 engines, 48 boundaries). Every "
                "boundary also cuts TensorRT's fusion where it sits. The regions run "
                "under the config's matrix precision policy "
                "(FP16 accumulation, TF32), and Inductor's fused kernels may skip "
                "intermediate roundings to FP16 (its emulate_precision_casts option "
                "is off); the first call compiles them. GPU memory grows with the "
                "number of engines and regions: each engine reserves its own "
                "TensorRT activation memory, the first and the last engine both "
                "store the tied embedding and LM-head weight, and the tensors "
                "crossing the boundaries stay allocated between calls (each engine "
                "keeps its inputs until its next call), so the CUDA graph's memory "
                "pool holds all of them. Requires "
                "runtime.cudagraphs=True, which replays the whole model (engines "
                "and regions) from one CUDA graph; without it every boundary adds "
                "host launches. The values share one set of requirements, so each "
                "also needs workload.gdn=functional and workload.attention=causal. "
                "With edit.gdn, the prologue values refuse an edit of "
                "functional_forward or chunk_gated_delta_rule, which their builds "
                "do not run as written (edits to the convolution, the inverse, the "
                "scan and its kernel reach them). Builds with regions are not "
                "exported: torch_tensorrt.save refuses execute_in_torch regions "
                "whose backend is 'inductor', so a session that keeps one writes "
                "its config but no artifact.",
                ("dominant_kernel",),
                {
                    **functional,
                    "workload.attention": "causal",
                    "runtime.cudagraphs": True,
                },
            ),
            *(knob for knob in STANDARD_KNOBS if knob.name in _STANDARD),
        )

    def editable_files(self) -> Sequence[EditableFile]:
        here = Path(__file__).resolve().parent
        return (
            EditableFile(
                "gdn",
                str(here / "qwen35_gdn.py"),
                "The functional Gated DeltaNet (workload.gdn=functional): FP32 chunk "
                "math in chunk_gated_delta_rule (decays, the UT system and its "
                "inverse by block-doubling forward substitution in "
                "unit_lower_inverse, u/w, intra-chunk scores, outputs), the "
                "inter-chunk state loop scan_states, the Triton state-scan kernel "
                "gdn_scan_kernel (kernels.gdn_scan=triton*) and its launch settings "
                "SCAN_LAUNCH, and the layer forward around them "
                "(causal_conv1d_silu, functional_forward: conv, q/k/v split, "
                "gates), whose layout sets how TensorRT fuses the elementwise work "
                "around the chunk rule and the scan region. With "
                "kernels.gdn_scan=generated, scan_states is also the body whose "
                "kernel the generator writes: editing it changes that region, "
                "whose kernel is then generated again. Shapes at 2k tokens: 16 "
                "heads, key and value head dim 128, 32 chunks of 64 per layer, 18 "
                "layers. Keep function names and "
                "signatures and the kernel's parameter order (the plugin ABI: input "
                "pointers w, u, k, gamma, then outputs h, v, then constexprs) and "
                "its tl.multiple_of(..., 16) hints on dereferenced pointers. The "
                "TensorRT 11.4 plugin launch gives the kernel at most 48 KB of "
                "shared memory (Triton's metadata.shared) and null pointers for "
                "Triton's global and profile scratch buffers, so the build compiles "
                "the kernel first and refuses one that needs more shared memory or "
                "any scratch (check_scan_kernel). kernels.inductor's prologue "
                "values bind their own layer forward (prologue_region_forward, "
                "outside the fences), and its prologue-decay values also their own "
                "copy of the chunk rule's elementwise head and the rest of the rule "
                "(chunk_rule_head/chunk_rule_body): those builds refuse an edit "
                "of functional_forward or chunk_gated_delta_rule, which "
                "they do not run as written; edits to the convolution, the "
                "inverse, the scan and its kernel reach them.",
            ),
        )

    def baseline(self) -> dict[str, Any]:
        return dict(self._baseline)

    def timing(self) -> Timing:
        return Timing(warmup=3, rounds=12, calls=3, profile_calls=2)

    def accuracy_defaults(self) -> dict[str, tuple[str, float]]:
        """Limits against the FP32 reference at the kept positions.

        On Qwen3.5-0.8B at 2k tokens, correct builds (BF16 or FP16, either
        accumulation, TF32, chunk 32, 64 or 128, the Triton scan, BF16
        operands in the chunk products) measure top-1 0.98-1.0, mean KL at
        most 4.9e-4 and largest KL at one position at most 2.2e-3; FP32
        builds 3e-11. Subtle errors measure mean KL 1.6e-3-0.1 with largest
        KL 0.012-0.68 (one Gated DeltaNet layer without inter-chunk state,
        the last chunk's state correction skipped, one attention layer
        seeing future tokens, a UT inverse truncated to 32-wide blocks), and
        gross ones (no gating, no inverse, reversed conv taps) KL 5-11 or
        infinite. The mean-KL and largest-KL bounds sit about 4x above the
        worst correct build. Errors confined to the last Gated DeltaNet
        layer can stay within them (largest KL 2.5e-3-7.6e-3), so the unit
        tests check the formulation exactly. Top-1 stays at 0.95: six of
        the 64 reference positions have a top-2 logit gap of 0.08-0.32,
        within BF16's 0.2-0.3 logit error, and every wrong variant measured
        fails a KL bound before top-1.
        """
        return {
            "top1_agreement": (">=", 0.95),
            "mean_kl": ("<=", 0.002),
            "max_kl": ("<=", 0.01),
        }

    def tokens(self) -> torch.Tensor:
        """The fixed prompt: the first ``length`` tokens of :func:`prompt_text`."""
        if self._tokens is None:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            # Eight characters per token is more than any English text needs.
            ids = tokenizer(prompt_text(8 * self.length), return_tensors="pt")
            ids = ids["input_ids"][0]
            if ids.numel() < self.length:
                raise ValueError(f"The prompt has only {ids.numel()} tokens")
            self._tokens = ids[: self.length].clone()
        return self._tokens

    def load_model(
        self, dtype: torch.dtype, attn_implementation: str = "sdpa"
    ) -> torch.nn.Module:
        """The text causal LM (no vision tower) on the GPU in ``dtype``."""
        from transformers import AutoModelForCausalLM

        return (
            AutoModelForCausalLM.from_pretrained(
                self.model_name,
                dtype=dtype,
                attn_implementation=attn_implementation,
            )
            .eval()
            .cuda()
        )

    def reference(self) -> Any:
        """FP32 eager logits of Hugging Face's model (SDPA, fallbacks, no TF32)."""
        matmul = torch.backends.cuda.matmul
        allow_tf32, matmul.allow_tf32 = matmul.allow_tf32, False
        try:
            with torch.inference_mode():
                model = self.load_model(torch.float32)
                ids = self.tokens().cuda().unsqueeze(0)
                logits = PrefillLogits(model, self.positions)(ids).float().cpu()
        finally:
            matmul.allow_tf32 = allow_tf32
        del model
        gc.collect()
        torch.cuda.empty_cache()
        return logits

    def compare(self, outputs: Any, reference: Any) -> dict[str, float]:
        actual = outputs.detach().double().cpu()
        expected = reference.double()
        if actual.shape != expected.shape:
            raise ValueError(
                f"Logits shape {tuple(actual.shape)} != {tuple(expected.shape)}"
            )
        if not torch.isfinite(actual).all():
            return {
                "top1_agreement": 0.0,
                "mean_kl": float("inf"),
                "max_kl": float("inf"),
                "max_abs_logit_error": float("inf"),
            }
        kl = torch.nn.functional.kl_div(
            actual.log_softmax(-1),
            expected.log_softmax(-1),
            log_target=True,
            reduction="none",
        ).sum(-1)
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

    @staticmethod
    def gdn_module(values: Mapping[str, Any]) -> Any:
        """``qwen35_gdn``, or its edited version that ``edit.gdn`` selects."""
        if values.get("edit.gdn"):
            return load_module(values["edit.gdn"], "qwen35_gdn_edited")
        import qwen35_gdn

        return qwen35_gdn

    @staticmethod
    def gdn_settings(
        values: Mapping[str, Any], gdn: Any = None, generator: Any = None
    ) -> Any:
        """The functional Gated DeltaNet settings that ``values`` select.

        ``generator`` is the KernelGenerator of ``kernels.gdn_scan="generated"``
        (:meth:`kernel_generator`), which that value needs.
        """
        if gdn is None:
            import qwen35_gdn as gdn

        scan, tf32 = SCANS[values.get("kernels.gdn_scan", "loop")]
        _, prologue = inductor_regions(values.get("kernels.inductor", "none"))
        # Only when chosen: an edited file made before the option has no field.
        options: dict[str, Any] = {}
        if prologue != "tensorrt":
            options["prologue"] = prologue
        if scan == "generated":
            options["generator"] = generator
        return gdn.GDNSettings(
            chunk_size=int(values.get("workload.chunk_size", 64)),
            conv=values.get("workload.conv", "conv1d"),
            scan=scan,
            scan_tf32=tf32,
            **options,
        )

    def kernel_generator(self) -> Any:
        """The KernelGenerator of ``kernels.gdn_scan="generated"``.

        ``region_generation.make_generator`` with the scan's notes and
        :data:`GENERATED_SCAN_PRECISION`: the same settings in every build.
        """
        import qwen35_gdn
        import region_generation

        return region_generation.make_generator(
            self.kernel_cache,
            qwen35_gdn.GENERATION_NOTES,
            fp32_matmul_precision=GENERATED_SCAN_PRECISION,
        )

    def _structure(self, values: Mapping[str, Any]) -> dict[str, Any]:
        structure: dict[str, Any] = {}
        try:
            flops = prefill_flops(
                self.config(),
                self.length,
                len(self.positions),
                int(values.get("workload.chunk_size", 64)),
            )
        except Exception:  # noqa: BLE001 - a diagnostic must not fail the build
            return structure
        structure["gflop_per_unit"] = round(flops / 1e9, 1)
        for suffix, fp16_accumulation in (("", False), ("_fp16_acc", True)):
            try:
                peak = peak_tflops(fp16_accumulation)
            except Exception:  # noqa: BLE001 - a diagnostic must not fail the build
                peak = None
            if peak:
                structure[f"peak_tflops{suffix}"] = round(peak, 1)
                structure[f"compute_floor_ms{suffix}"] = round(flops / 1e9 / peak, 3)
        return structure

    def build(self, values: Mapping[str, Any]) -> Candidate:
        settings = bench_settings(values)
        if settings.backend != "tensorrt":
            return self._build_torch(values, settings)
        return self._build_tensorrt(values)

    def _build_tensorrt(self, values: Mapping[str, Any]) -> Candidate:
        import torch_tensorrt

        gdn = values.get("workload.gdn", "hf")
        inductor = values.get("kernels.inductor", "none")
        inductor_attention, prologue = inductor_regions(inductor)
        hybrid = inductor != "none"
        if hybrid and (
            gdn != "functional"
            or values.get("workload.attention", "mask") != "causal"
            or not values.get("runtime.cudagraphs")
        ):
            raise ValueError(
                "kernels.inductor needs workload.gdn=functional, "
                "workload.attention=causal and runtime.cudagraphs=True (one CUDA "
                "graph replays the engines and the Inductor regions)"
            )
        module_gdn = self.gdn_module(values)
        conflicts = prologue_edit_conflicts(module_gdn, prologue)
        if conflicts:
            # Otherwise the candidate would pass as the edit without running it.
            raise ValueError(
                f"kernels.inductor={inductor!r} does not run edit.gdn's changes to "
                f"{conflicts}: its builds bind prologue_region_forward instead of "
                "functional_forward and apply the q/k L2 norms there (the decay "
                "values also run their own copy of the chunk rule). Drop the edit, "
                "or use kernels.inductor 'none' or 'attention'."
            )
        generator = None
        if values.get("kernels.gdn_scan") == "generated":
            generator = self.kernel_generator()
        settings_gdn = self.gdn_settings(values, module_gdn, generator)
        scan_plugin = settings_gdn.scan != "loop"
        if scan_plugin and gdn != "functional":
            raise ValueError("kernels.gdn_scan needs workload.gdn=functional")
        if scan_plugin and not values.get("runtime.cudagraphs"):
            raise ValueError(
                "kernels.gdn_scan needs runtime.cudagraphs=True: TensorRT 11.4 "
                "reloads plugin modules on every enqueue outside CUDA graphs"
            )
        # Plugin and Inductor regions are captured only by strict export.
        regions = scan_plugin or hybrid
        scan_resources: dict[str, int] = {}
        if settings_gdn.scan == "triton":
            # Fail here, not at the first enqueue, if the plugin cannot launch it.
            config = self.config()
            chunk = settings_gdn.chunk_size
            scan_resources = module_gdn.check_scan_kernel(
                -(-self.length // chunk),
                chunk,
                config.linear_key_head_dim,
                config.linear_value_head_dim,
                settings_gdn.scan_tf32,
            )
        device = torch.device("cuda:0")
        precision = values.get("engine.precision", "bf16")
        attention = attention_implementation(
            values.get("workload.attention", "mask"), inductor=inductor_attention
        )
        model = self.load_model(_DTYPES[precision], attention)
        if gdn == "functional":
            module_gdn.install(model, settings_gdn)
        elif gdn != "hf":
            raise ValueError(f"workload.gdn={gdn!r}; choose 'hf' or 'functional'")
        structure = {"attn_implementation": attention, **self._structure(values)}
        if scan_resources:
            structure["scan_kernel_shared_bytes"] = scan_resources["shared"]
        if generator is not None:
            import region_generation

            structure["kernels_generated"] = "gdn_scan"
            structure.update(region_generation.describe(generator))
        # The Inductor regions follow the config's matrix precision policy, as
        # bench.backend=inductor does (the candidate sets it while it runs).
        region_flags: dict[str, bool] | None = None
        if hybrid:
            region_flags = matmul_policy(values)
            structure["inductor_regions"] = inductor
            structure["region_matmul_flags"] = dict(region_flags)
        wrapper = PrefillLogits(model, self.positions).eval()
        ids = self.tokens().to(device).unsqueeze(0)
        settings: dict[str, Any] = {
            "min_block_size": 1,
            "use_fp32_acc": precision == "fp16"
            and bool(values.get("engine.use_fp32_acc", True)),
            "disable_tf32": bool(values.get("engine.disable_tf32", True)),
            "use_python_runtime": True,
            # Regions build without offloading (strict export keeps references
            # to the parameters); they are deallocated below instead.
            "offload_module_to_cpu": not regions,
            "device": device,
            "decompose_attention": bool(values.get("engine.decompose_attention")),
            "optimization_level": values.get("engine.optimization_level"),
            "tiling_optimization_level": values.get(
                "engine.tiling_optimization_level", "none"
            ),
        }
        with torch.no_grad():
            if regions:
                # Region capture needs strict export through torch_tensorrt.compile.
                module = torch_tensorrt.compile(
                    wrapper, ir="dynamo", arg_inputs=[ids], strict=True, **settings
                )
            else:
                program = torch.export.export(wrapper, (ids,), strict=False)
                module = torch_tensorrt.dynamo.compile(
                    program, arg_inputs=[ids], **settings
                )
                del program
        if regions:
            from torch_tensorrt.dynamo.utils import deallocate_module

            deallocate_module(wrapper)
            if hybrid:
                from torch_tensorrt.dynamo.utils import pin_torch_executed_state

                # The Inductor regions read the model's own parameters (the
                # prologue's in_proj_qkv and conv weights), which the line above
                # moved to the CPU with the rest: bring those back.
                pin_torch_executed_state(module, device)
        del model, wrapper
        gc.collect()
        torch.cuda.empty_cache()
        return _TensorRTPrefill(
            module,
            ids,
            cudagraphs=bool(values.get("runtime.cudagraphs")),
            pre_allocated_outputs=bool(values.get("runtime.pre_allocated_outputs")),
            structure=structure,
            region_flags=region_flags,
        )

    def _build_torch(
        self, values: Mapping[str, Any], settings: "BenchSettings"
    ) -> Candidate:
        """The same forward run eagerly or compiled by Inductor.

        It honors the config's formulation, attention and precision policy
        (module docstring) and refuses TensorRT-only knobs. Compilation (and
        CUDA-graph recording) happens in three warm-up calls here, so the
        build time includes it.
        """
        from torch_tensorrt.autotune._knobs import knobs_by_name

        knobs = knobs_by_name(self.knobs())
        given = {
            name: values[name]
            for name in _TENSORRT_ONLY
            if name in values and values[name] != knobs[name].default
        }
        if given:
            raise ValueError(
                f"{given} only apply to TensorRT; bench.backend {settings.backend!r} "
                "cannot honor them"
            )
        precision = values.get("engine.precision", "bf16")
        policy = matmul_policy(values)
        fp16_accumulation, tf32 = policy["fp16_accumulation"], policy["tf32"]
        attention = values.get("workload.attention", "mask")
        if settings.attn and attention != "mask":
            raise ValueError(
                f"bench.attn={settings.attn!r} conflicts with "
                f"workload.attention={attention!r}; set one of them"
            )
        attn = settings.attn or attention_implementation(attention)
        gdn = values.get("workload.gdn", "hf")
        if gdn not in ("hf", "functional"):
            raise ValueError(f"workload.gdn={gdn!r}; choose 'hf' or 'functional'")
        stack = contextlib.ExitStack()
        try:
            stack.enter_context(
                _torch_matmul_flags(tf32=tf32, fp16_accumulation=fp16_accumulation)
            )
            if settings.backend == "inductor":
                import torch._inductor.config as inductor_config

                for key in settings.inductor:
                    if not hasattr(inductor_config, key):
                        raise ValueError(
                            f"Unknown torch._inductor.config option {key!r}"
                        )
                stack.enter_context(inductor_config.patch(dict(settings.inductor)))
            model = self.load_model(_DTYPES[precision], attn)
            if gdn == "functional":
                module_gdn = self.gdn_module(values)
                module_gdn.install(model, self.gdn_settings(values, module_gdn))
            structure = {
                "trt_engines": 0,
                "torch_partitions": 0,
                "plugin_regions": 0,
                "backend": settings.backend,
                "mode": settings.mode if settings.backend == "inductor" else None,
                "attn": attn,
                "gdn": gdn,
                "precision": precision,
                "fp16_accumulation": fp16_accumulation,
                "tf32": tf32,
                "inductor": dict(settings.inductor),
                **self._structure(values),
            }
            forward: Any = PrefillLogits(model, self.positions).eval()
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
        candidate.structure["warmup_s"] = round(time.perf_counter() - start, 2)
        return candidate


# ----------------------------------------------------------------- benchmarks
@dataclasses.dataclass(frozen=True)
class BenchSettings:
    """The head-to-head ``bench.*`` values of one build (module docstring)."""

    backend: str = "tensorrt"
    mode: str = "default"
    attn: str | None = None
    inductor: Mapping[str, Any] = dataclasses.field(default_factory=dict)

    @property
    def cudagraphs(self) -> bool:
        """Whether torch.compile replays the forward from CUDA graphs."""
        return self.backend == "inductor" and self.mode in _CUDAGRAPH_MODES


def bench_settings(values: Mapping[str, Any]) -> BenchSettings:
    """Validate and collect the ``bench.*`` values of ``values``.

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
    )
    if settings.backend not in BACKENDS:
        raise ValueError(
            f"bench.backend={settings.backend!r}; choose one of {BACKENDS}"
        )
    if settings.mode not in TORCH_COMPILE_MODES:
        raise ValueError(
            f"bench.mode={settings.mode!r}; choose one of {TORCH_COMPILE_MODES}"
        )
    applicable = {
        "tensorrt": ("bench.backend",),
        "eager": ("bench.backend", "bench.attn"),
        "inductor": _BENCH_KEYS,
    }[settings.backend]
    given = sorted(
        key
        for key, value in bench.items()
        if value and key not in applicable and (key, value) != ("bench.mode", "default")
    )
    if given:
        raise ValueError(f"{given} do not apply to bench.backend {settings.backend!r}")
    return settings


# -------------------------------------------------------------- candidates
class _TensorRTPrefill(Candidate):
    def __init__(
        self,
        module: torch.nn.Module,
        ids: torch.Tensor,
        *,
        cudagraphs: bool,
        pre_allocated_outputs: bool,
        structure: Mapping[str, Any],
        region_flags: Mapping[str, bool] | None = None,
    ) -> None:
        self.module = module
        self.ids = ids
        self.cudagraphs = cudagraphs
        self.pre_allocated_outputs = pre_allocated_outputs
        self.structure = dict(structure)
        # PyTorch matmul flags (_torch_matmul_flags) for Inductor regions:
        # set from setup to close, so they also hold when the regions compile.
        self.region_flags = dict(region_flags) if region_flags else None
        self._stack = contextlib.ExitStack()
        self._run: Any = module

    def setup(self) -> None:
        import torch_tensorrt

        if self.region_flags:
            self._stack.enter_context(_torch_matmul_flags(**self.region_flags))
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

        ``kernels.inductor`` builds are refused with ``NotImplementedError``,
        the contract's "cannot export" (the session then records the export
        as unsupported): ``torch_tensorrt.save`` rejects ``execute_in_torch``
        regions whose backend is not ``eager``.
        """
        import torch_tensorrt

        inductor = self.structure.get("inductor_regions")
        if inductor:
            raise NotImplementedError(
                f"kernels.inductor={inductor!r} builds are not exported: their "
                "execute_in_torch(backend='inductor') regions run code that "
                "Inductor compiles at run time, and torch_tensorrt.save cannot "
                "preserve that backend (it refuses such regions). Rebuild the "
                "config instead."
            )
        path = Path(directory) / "prefill.ep"
        torch_tensorrt.save(self.module, str(path), retrace=False)
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
        self.structure = dict(structure)
        # Global settings (TF32, FP16 accumulation, Inductor options) the
        # forward runs under; restored on close.
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
