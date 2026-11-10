# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Nemotron-H (Mamba-2 hybrid) batch-1 decode and prefill workloads for ``torch_tensorrt.autotune``.

``nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16`` interleaves 21 Mamba-2 mixers, 17
squared-ReLU MLPs and 4 grouped-query attention layers (no positional
encoding). The measured unit is one decoded token at batch 1: a fixed
128-token prompt is prefilled once, untimed, and a timed call then decodes 64
tokens teacher-forced on a fixed public-domain text (:data:`TEXT`), so one
different token cannot derail the rest. Every step's logits are checked
against an FP32 eager reference at the same position: the Hugging Face model
in FP32 prefilling the prompt and decoding token by token with its own cache
(top-1 agreement, KL divergence and the largest logit error). The reference
stores the token ids it was computed for, and ``compare`` refuses a reference
whose tokens differ from the workload's.

The TensorRT candidates compile :class:`NemotronDecodeStep`, a shape-static,
functional decode step written in plain PyTorch ops over the Hugging Face
model's weights:

* each attention layer's key and value caches are inputs of shape
  ``[1, kv_heads, isl + osl, head_dim]``; the new key and value are written at
  ``position_ids`` with ``index_copy``, which TensorRT lowers to an in-place
  ``IKVCacheUpdateLayer``, and attention runs over the whole cache with the
  mask ``arange(max_seq_len) <= position_ids``;
* each Mamba-2 layer's convolution state ``[1, conv_dim, conv_kernel]`` and
  SSM state ``[1, heads, head_dim, state_size]`` are inputs; the single-token
  update (conv-state shift and depthwise convolution, ``h = exp(dt*A)*h +
  dt*B*x``, ``y = C.h + D*x``, gated RMSNorm) is pure PyTorch, computed in
  FP32, and the new states are outputs.

The engine has no host synchronization and can be captured in a CUDA graph.
The prompt is prefilled by the Hugging Face model in the candidate's
precision (its chunked scan), and its caches are copied into the static
buffers; the Hugging Face decode step from those caches reproduces
:class:`NemotronDecodeStep` (see the tests).

Workload knobs:

* ``engine.precision``: BF16 (the checkpoint's) or FP16 weights and
  activations; ``engine.use_fp32_acc`` adds FP32 accumulation for FP16.
* ``workload.ssm_state``: the SSM state's storage precision, FP32 (as Hugging
  Face keeps it) or BF16, which halves the state traffic.
* ``workload.state_update``: ``functional`` returns the new conv and SSM
  states as engine outputs, which a CUDA graph copies in and out of its
  staging buffers every step; ``inplace`` writes them into the input buffers
  through the same ``IKVCacheUpdateLayer`` as the KV cache (a one-slot cache
  with the whole state as its slot), with no copies.
* ``engine.build_route``: the TensorRT Global Performance Tuning routes that
  change this engine (:data:`DECODE_ROUTES`), where the feature is available.
* ``runtime.cudagraphs``.

Builder options that measured as no-ops on this engine are not offered
(optimization levels 4 and 5, and the build routes named in the comment on
:data:`DECODE_ROUTES`), nor is TensorRT's tiling optimization: TensorRT 11.4
fails to build this engine at the ``fast``, ``moderate`` and ``full`` tiling
levels, because Myelin's tactic timing does not bind the aliased KV-cache
inputs and outputs to the same memory (``validateIOTensorAliasing``). Nor is
the CUDA Tile build route, which fails a Myelin check on the same KV-cache
update copies.

The editable file ``decode_step`` lets an LLM planner rewrite the RMSNorm and
Mamba-2 step formulations inside their ``EDITABLE`` fences in this file: the
PyTorch ops Myelin fuses into the non-GEMM kernels. Such a rewrite must not
change the math, and BF16 noise can hide a small formulation error from the
accuracy limits, so an edited build first passes an FP32 gate
(:func:`check_formulation`): the edited and the original step decode
:data:`FORMULATION_STEPS` tokens eagerly in FP32 from the same FP32 prefill,
and the build fails unless their logits and every cache and state output
agree within :data:`FORMULATION_LIMITS`.

The second workload, :func:`create_prefill_case`, times batch-1 prefill: one
forward pass over a ``length``-token prompt (2048 by default, of the
non-repeating :data:`PREFILL_TEXT`) through Hugging Face's chunked SSD scan,
compiled whole (``PrefillLogits``: the model's full-sequence forward with
logits at 64 evenly spaced positions, checked against the FP32 model like the
decode logits). Its knobs are the precision, FP32 accumulation, TF32 for the
scan's FP32 einsums (``engine.tf32``, applied to every backend), the scan's
formulation (``workload.ssd``: Hugging Face's, or :func:`ssd_chunk_scan` with
segment sums from one cumulative sum and per-group B and C) and chunk length
(``workload.chunk_size``). CUDA graphs and optimization level 5 measured as
no-ops on this one-engine, 140 ms forward, so neither they nor level 4 are
offered.

Both workloads also offer ``kernels.inductor`` (default ``"none"``): hybrid
variants of the TensorRT build that hand chosen operations to Inductor inside
the compiled program, through ``torch_tensorrt.region.execute_in_torch(
backend="inductor")`` scopes. Only strict export captures such scopes, so
these variants compile with ``torch_tensorrt.compile(..., ir="dynamo",
strict=True)`` and keep the model on the GPU during the build (regions
refuse ``offload_module_to_cpu``); afterwards every weight that no region
reads is freed, since the engines hold their own copies. Each region is a
separate graph that Inductor compiles on its first call, during the build,
in the head-to-head's Inductor mode, max-autotune, without Inductor's own
CUDA graphs (:data:`REGION_INDUCTOR_MODE`). Its coordinate-descent tuning is
what lets Inductor lower a batch-1 matrix multiply to a fused Triton
reduction (a GEMV) instead of a cuBLAS call. Each region splits the
TensorRT graph, and every split adds an engine boundary:

* decode ``lm-head``: the final RMSNorm and the LM-head GEMV as one region
  after the last engine (one engine, one boundary);
* decode ``mlp-inductor``: each of the 17 MLP blocks (RMSNorm, up GEMV,
  ReLU², down GEMV and the residual add) as its own region (18 engines, 34
  boundaries);
* prefill ``mamba-inductor``: each of the 21 Mamba-2 mixers' interior, from
  the in_proj output through the split, the causal depthwise convolution and
  SiLU, the chunked SSD scan and the gated RMSNorm (which also combines the
  scan's intra- and inter-chunk outputs) to the tensor out_proj reads, as its
  own region (:func:`install_mamba_regions`; 22 engines, 42 boundaries).
  in_proj and out_proj stay in TensorRT, so a boundary carries only tensors
  the one-engine build also writes. ``engine.tf32`` applies to the region's
  FP32 einsums through ``torch.backends`` as for the other PyTorch backends.
  The host calls the 43 engines and regions one by one;
  ``mamba-inductor-graph`` builds the same program and records the whole
  forward as one CUDA graph (``torch_tensorrt.runtime.enable_cudagraphs``,
  correct here because no input is updated in place).

The decode variants need in-place state updates and CUDA graphs: the whole
decode step, engines and regions, is recorded as one CUDA graph over the
static cache and state buffers (:class:`_TensorRTCandidate`), so a boundary
costs a gap between kernels, not a host-side engine or region call. The
library's whole-graph wrapper (``torch_tensorrt.runtime.enable_cudagraphs``)
is not used: it copies every input into its own static buffers on each
replay, so in-place cache and state writes would land in its copies. The
variants of both workloads cannot be saved (``torch_tensorrt.save`` refuses
Inductor regions), so their export raises ``NotImplementedError``
(:func:`save_program`).

The decode workload also offers ``kernels.generated`` (default ``"none"``):
``"mlp"`` runs each MLP block of the decode step as two
``torch_tensorrt.region.fused_kernel`` regions whose kernels PR 4914's
``torch_tensorrt.region.KernelGenerator`` writes during the build
(``region_generation.py``; :meth:`NemotronDecodeStep._generated_mlp`):
``mlp_in_<layer>``, the block RMSNorm, up GEMV and ReLU², and
``mlp_out_<layer>``, the down GEMV and the residual add. The kernels run
inside the one TensorRT engine as QDP plugins (no engine split); identical
blocks share one kernel per region. Plugins need strict export
(``torch_tensorrt.compile(..., strict=True)``, the model kept on the GPU
during the build, then every weight the engine copied is freed) and CUDA
graphs: TensorRT 11.4 reloads plugin modules on every enqueue outside them.
Generated kernels are cached in ``kernel_cache`` (``region_generation.kernel_cache``),
keyed by the engine options; a session prepares the value once, before its search.

``Workload.build`` of both workloads also honors the head-to-head driver's
(``bench_compare.py``) ``bench.*`` values:
``bench.backend`` ``"eager"`` or ``"inductor"`` decodes with the Hugging Face
model (decode: with its hybrid ``StaticCache``, from the same prefilled
state and with the same teacher-forced protocol; prefill: ``PrefillLogits``),
with ``bench.mode`` the ``torch.compile`` mode, ``bench.attn`` the attention
implementation and ``bench.inductor`` Inductor config overrides. The config's
``engine.precision`` (and for prefill ``engine.tf32``, ``workload.ssd`` and
``workload.chunk_size``) apply to those backends too. With
``bench.attn="flex_attention"`` the decode mask is corrected first (see
:func:`fix_flex_attention_masks`): transformers 5.18 lets every
FlexAttention decode query see the next, unwritten cache slot. The prefill
workload's own key ``bench.fp16_accumulation`` (eager and Inductor, FP16
only) sets ``torch.backends.cuda.matmul.allow_fp16_accumulation``, so cuBLAS
FP16 GEMMs accumulate in FP16 as TensorRT's plain FP16 tactics do. The
decode workload's own key ``bench.whole_graph`` (TensorRT, with in-place
states and CUDA graphs) captures the whole step as one CUDA graph without
any region, as the ``kernels.inductor`` variants do: a control for that
capture.
"""

from __future__ import annotations

import contextlib
import gc
import re
import time
import types
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

MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16"
_DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
# torch.compile modes that capture CUDA graphs.
_CUDAGRAPH_MODES = ("reduce-overhead", "max-autotune")

# Opening of "A Tale of Two Cities" (Charles Dickens, 1859; public domain), as
# agentic_decode.py has it: a frozen copy, because the stored decode
# references depend on these exact tokens.
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
    "preserves of loaves and fishes, that things in general were settled for ever. "
)

# Prose written for the prefill workload (no model has memorized it), long
# enough for 2048 tokens without repeating: every checked position predicts
# new text, so top-1 agreement and KL divergence both carry information.
PREFILL_TEXT = (
    "The harbour at Wrenmouth wakes before the town does. By half past four the "
    "first trawlers are already nosing out past the breakwater, their deck lights "
    "swinging in the swell, and the gulls that roost on the fish market roof lift "
    "off in ones and twos to follow them. Ada Morrow, who has kept the tide office "
    "on the north quay for nineteen years, unlocks the door, lights the paraffin "
    "heater that the harbour board keeps promising to replace, and copies the "
    "overnight readings from the gauge into a ledger whose spine has been mended "
    "with sailcloth tape. Nobody has asked her to keep the ledger since the board "
    "installed an automatic recorder in the spring, but she keeps it anyway, "
    "because the recorder has twice lost a week of data and the ledger never has.\n\n"
    "To make the bread her grandmother called harvest loaf, begin the night before. "
    "Stir a spoonful of the old dough into a bowl of lukewarm water with two "
    "handfuls of coarse rye flour, cover it with a damp cloth, and leave it near "
    "the stove. In the morning the mixture should smell sour and faintly of "
    "apples, and its surface should be pocked with small bubbles. Add white flour, "
    "salt, a little honey and a cup of cracked barley that has soaked overnight, "
    "then knead for longer than seems reasonable, until the dough stops tearing "
    "and begins to stretch. Shape it into a round, dust the top with bran, and let "
    "it rise in a basket lined with linen. Bake it hot at first, then lower the "
    "heat, and do not cut it until it has cooled, however strongly the kitchen "
    "insists otherwise.\n\n"
    "The engineers who inspected the Calder footbridge last autumn found that two "
    "of its suspender cables had lost nearly a fifth of their cross section to "
    "corrosion where they passed through the deck. Replacing a suspender on a "
    "bridge that stays in place is a delicate sequence. First a temporary hanger "
    "is clamped to the main cable beside the damaged one and tensioned with a "
    "hydraulic jack until it carries the load; strain gauges on the deck beams "
    "confirm that the forces have shifted as predicted. Only then is the old "
    "suspender cut free, lowered on a winch and replaced with a new galvanized "
    "strand, which is tensioned in small increments while the temporary hanger is "
    "released. The whole operation took eleven nights, because the work could "
    "only be done between midnight and five, when the bridge was closed to "
    "walkers and the river traffic had stopped.\n\n"
    "'You are certain it was the eastern road?' the magistrate asked, without "
    "looking up from his papers.\n"
    "'As certain as I am of my own name,' said the carter. 'I had the mill on my "
    "left and the sun in my eyes the whole way, and I passed the stone marker with "
    "the broken top an hour before noon.'\n"
    "'Then explain to me how your wagon came to be found in the ford below "
    "Hollin, which is south of the town and nowhere near the eastern road.'\n"
    "The carter turned his hat around in his hands several times before he "
    "answered. 'I cannot explain it, sir. I can only tell you that I unhitched the "
    "horses at the inn that evening, and that in the morning the wagon was "
    "gone.'\n\n"
    "Every few years the river eels make their way downstream on a single "
    "moonless night in October, and the people who live along the lower valley "
    "still speak of those nights as if they were weather. The eels have spent ten "
    "or fifteen years in the ponds and ditches of the upper catchment, growing "
    "slowly on insects and snails, and then something in the shortening days and "
    "the cooling water changes them. Their eyes grow larger, their bellies turn "
    "silver, and they stop eating. When the first heavy rain of autumn swells the "
    "river, they leave together, slipping over weirs and through culverts and "
    "even across wet grass, on the first stage of a journey of several thousand "
    "kilometres to the sea where they were born and where they will spawn and "
    "die.\n\n"
    "Dear Mr Okafor, thank you for your letter of the ninth and for the sample of "
    "glazed tiles, which arrived intact despite the efforts of the carrier. We "
    "have shown them to the client, who likes the green very much but is less "
    "sure about the blue, which she feels will look cold in a room that faces "
    "north. Would it be possible to produce a trial batch with a little more iron "
    "in the glaze, so that the blue leans toward slate? We would need about forty "
    "tiles to make a fair comparison on the wall itself. As to the schedule, the "
    "plasterers expect to finish by the end of the month, so the tiling could "
    "begin in the second week of next month if the order is confirmed soon. I "
    "enclose the measured drawings of the fireplace surround and the window "
    "reveals, with every dimension checked twice on site.\n\n"
    "The Lowther Lending Library was founded in 1871 by a subscription of forty "
    "shillings each from twenty-six members, most of them shopkeepers and "
    "schoolteachers, who wanted a place in the town where books could be borrowed "
    "without the expense of buying them. For its first ten years it occupied a "
    "single room above a saddler's shop, and the librarian, a retired clerk named "
    "Josiah Venn, kept the catalogue in his head. When the collection outgrew the "
    "room, the members raised enough money to rent the old corn exchange, and the "
    "library moved there with its nine hundred volumes carried through the "
    "streets in handcarts on a Saturday afternoon. A photograph of the procession "
    "still hangs in the entrance hall, though the corn exchange itself was pulled "
    "down in the nineteen sixties to make room for a car park.\n\n"
    "Each player begins with three tokens on the outer ring and five cards drawn "
    "from the shuffled deck. On your turn you may move one token clockwise by the "
    "number shown on any card in your hand, or you may discard two cards and draw "
    "three. A token that ends its move on a space occupied by an opponent sends "
    "that token back to the outer ring, unless the space is marked with a "
    "lantern, where any number of tokens may rest safely. The first player to "
    "bring all three tokens to the centre wins, but a token may only enter the "
    "centre by an exact count, so the last few turns often depend on careful "
    "management of the cards that remain.\n\n"
    "We left the hut at first light, when the snow was still hard enough to carry "
    "us without crampons, and followed the line of cairns up the shoulder of the "
    "ridge. By nine the sun had reached the slope and the surface had begun to "
    "soften, so that every step broke through a thin crust into wet granular snow "
    "beneath. At the col the wind came at us out of the west with a force that "
    "made talking pointless, and the cloud that had been sitting quietly on the "
    "summit all morning began to pour over the ridge like water over a dam. We "
    "sheltered for twenty minutes behind a boulder, ate the last of the dried "
    "apricots, and agreed without much discussion to turn back.\n\n"
    "A sundial tells solar time, not clock time, and the difference between them "
    "can be surprisingly large. Because the orbit of the Earth is an ellipse and "
    "its axis is tilted, the sun does not cross the meridian at the same clock "
    "time every day: in early November it arrives about sixteen minutes early, "
    "and in February about fourteen minutes late. Old sundials often carried a "
    "small table, called the equation of time, that let the owner correct the "
    "reading, and some clockmakers built the correction into elaborate mechanical "
    "clocks. Add the offset of the local meridian from the centre of the time "
    "zone, and a dial in the western part of a zone may disagree with a "
    "wristwatch by the better part of an hour.\n\n"
    "On Thursdays the square fills with stalls before eight. The cheese sellers "
    "come down from the hill farms with wheels of hard white cheese wrapped in "
    "muslin, the fishmonger arrives from the coast with a van full of ice, and an "
    "old man who sells nothing but knives and scissors sets up a grinding wheel by "
    "the fountain and sharpens anything that is brought to him for a few coins. "
    "By noon the best of the vegetables are gone, and by two the stallholders are "
    "folding their awnings, sweeping cabbage leaves into the gutters and counting "
    "the takings in the cabs of their vans.\n\n"
    "The clock had stopped, according to its owner, on the night her father died, "
    "and nobody had wound it since. When the movement was lifted out of the case "
    "it was clear that the stoppage had a more ordinary cause: a pivot on the "
    "third wheel had worn a slot in its bearing, so that the wheel could tilt just "
    "enough for its teeth to bind against the pinion of the escape wheel. The "
    "repair meant bushing the plate, turning a new pivot, cleaning every part in "
    "solvent, and reassembling the train with fresh oil on each bearing. After a "
    "week on the test stand the clock was keeping time to within a minute a week, "
    "and its owner, who had expected to be sad, found herself laughing when it "
    "struck twelve in her kitchen for the first time in thirty years.\n\n"
    "The lock at Felling Cross is worked by hand, as it has been since the canal "
    "opened. A boat coming downhill waits above the top gates while the chamber "
    "fills; when the levels match, the gates swing open with a push on the "
    "balance beams, the boat glides in, and the gates are closed behind it. Then "
    "the paddles in the bottom gates are wound up with a windlass, and the water "
    "drains away in a brown rush until the boat sits nine feet lower, in the "
    "shadow of wet brick walls hung with ferns. The keeper's cottage beside the "
    "lock sells ice creams in summer and kindling in winter, and on the "
    "windowsill a cat watches every boat go through with the same expression of "
    "mild disapproval.\n\n"
    "A heron who fished the same stretch of river every morning grew tired of "
    "waiting for the fish to come to her. She asked the otter how he caught so "
    "many, and the otter told her that he simply chased them until they were too "
    "tired to swim. The next day the heron tried to chase the fish, splashing up "
    "and down the shallows with her wings spread, and by evening she had caught "
    "nothing and frightened every fish for half a mile. The otter, passing by, "
    "asked her why she looked so unhappy. 'Your way does not suit me,' said the "
    "heron. 'No,' said the otter, 'but you had to try it to find that out, and "
    "tomorrow you will stand still with more patience than before.'\n"
)

# TensorRT Global Performance Tuning build routes that change this decode
# engine (each measured in agentic_bench/briefs/nemotron-decode.md). Left out:
# routes that measured as no-ops on it ('-cask_fusion:fp16_acc=off',
# '-peep:match_gemv_mha=on', '-mha_fusion:gemv_mha_v2=off', '-late_h_fusion=off',
# and split-K off on top of kgen, whose GEMVs have no split-K), and
# '-kgen:codegen:cuda_tile=2', which TensorRT 11.4 cannot build it with
# (copy_vtx_analyzer.cpp: CHECK(vtx_->input_operands().size() == 1) on the
# KV-cache update copies).
DECODE_ROUTES = ("", "-kgen:lower_fc:all=on", "-tactic:splitk=off")

# The FP32 gate of an edited decode step (see check_formulation): decode
# steps, and the largest logit difference and relative cache or state
# difference allowed against the original step. A correct rewrite differs by
# FP32 rounding only (about 1e-5 in the logits); a broken state update, cache
# write or norm differs by orders of magnitude more.
FORMULATION_STEPS = 8
FORMULATION_LIMITS = {"max_abs_logit_error": 1e-3, "max_rel_state_error": 1e-4}

# kernels.inductor: the hybrid variants (operations handed to Inductor regions)
# of each workload's TensorRT build; "none" is the plain build.
DECODE_INDUCTOR = ("none", "lm-head", "mlp-inductor")
PREFILL_INDUCTOR = ("none", "mamba-inductor", "mamba-inductor-graph")
# kernels.generated (decode): the blocks whose regions run kernels PR 4914's
# KernelGenerator writes; "mlp" is every MLP block (mlp_in and mlp_out).
DECODE_GENERATED = ("none", "mlp")
# The generator's notes for those regions: measured facts about batch-1 decode
# GEMV kernels on this GPU. Part of the kernel cache key; editing the file
# starts a new cache.
GENERATION_NOTES = "nemotron_decode_kernel_notes.md"
# The torch.compile mode whose Inductor options the regions compile with: the
# head-to-head's max-autotune, without Inductor's CUDA graphs, which the
# region runtime turns off (a decode step is captured whole instead).
REGION_INDUCTOR_MODE = "max-autotune-no-cudagraphs"


def create_case(
    model: str = MODEL,
    isl: int = 128,
    osl: int = 64,
    baseline: Mapping[str, Any] | None = None,
    kernel_cache: str | None = None,
) -> "NemotronDecodeWorkload":
    """Workload factory: ``WorkloadFactory("tools/llm/nemotron_decode.py:create_case", {...})``."""
    return NemotronDecodeWorkload(
        model, isl=isl, osl=osl, baseline=baseline, kernel_cache=kernel_cache
    )


def create_prefill_case(
    model: str = MODEL,
    length: int = 2048,
    positions: int = 64,
    baseline: Mapping[str, Any] | None = None,
) -> "NemotronPrefillWorkload":
    """Workload factory of the prefill benchmark (see :class:`NemotronPrefillWorkload`)."""
    return NemotronPrefillWorkload(
        model, length=length, positions=positions, baseline=baseline
    )


# ---------------------------------------------------------------- decode step


def layer_kinds(config: Any) -> list[str]:
    """Each layer's mixer: ``linear_attention`` (Mamba-2), ``full_attention`` or ``mlp``."""
    kinds = list(config.layers_block_type)
    unknown = set(kinds) - {"linear_attention", "full_attention", "mlp"}
    if unknown:
        raise ValueError(f"Unsupported Nemotron-H layer types {sorted(unknown)}")
    return kinds


def state_shapes(
    config: Any, max_seq_len: int, *, inplace: bool = False
) -> dict[str, tuple[int, ...]]:
    """Shapes of one layer's decode state tensors.

    ``inplace`` lays each Mamba state out as a one-slot cache ``[1, rows, 1,
    cols]`` (the same memory), which ``IKVCacheUpdateLayer`` can overwrite.
    """
    heads, head_dim = config.mamba_num_heads, config.mamba_head_dim
    conv_dim = heads * head_dim + 2 * config.n_groups * config.ssm_state_size
    kv_head_dim = getattr(config, "head_dim", None) or (
        config.hidden_size // config.num_attention_heads
    )
    shapes = {
        "kv": (1, config.num_key_value_heads, max_seq_len, kv_head_dim),
        "conv": (1, conv_dim, config.conv_kernel),
        "ssm": (1, heads, head_dim, config.ssm_state_size),
    }
    if inplace:
        shapes["conv"] = (1, conv_dim, 1, config.conv_kernel)
        shapes["ssm"] = (1, heads * head_dim, 1, config.ssm_state_size)
    return shapes


# EDITABLE-START
def _rms_norm(norm: torch.nn.Module, hidden: torch.Tensor) -> torch.Tensor:
    """Nemotron-H's RMSNorm: FP32 statistics and weight multiply, input dtype out."""
    dtype = hidden.dtype
    hidden = hidden.float()
    hidden = hidden * torch.rsqrt(
        hidden.pow(2).mean(-1, keepdim=True) + norm.variance_epsilon
    )
    return (norm.weight.float() * hidden).to(dtype)


# EDITABLE-END


class NemotronDecodeStep(torch.nn.Module):
    """One shape-static decode step of a Hugging Face Nemotron-H causal LM.

    ``forward(input_ids, position_ids, *kv_caches, *conv_states, *ssm_states)``
    takes ``[1, 1]`` token and position tensors, then the key and value cache
    of every attention layer (in layer order, key before value), the conv
    state of every Mamba layer and the SSM state of every Mamba layer, with the
    shapes of :func:`state_shapes`. It returns ``(logits, *kv_caches,
    *conv_states, *ssm_states)`` with FP32 logits of shape ``[1, vocab]``.

    The caches are updated with ``index_copy`` at ``position_ids`` (in place
    once TensorRT lowers it). The states are returned as new tensors, or, with
    ``inplace``, written into the inputs as one-slot caches. The SSM update
    and the convolution run in FP32 whatever the model and state dtypes.

    ``inductor`` (a :data:`DECODE_INDUCTOR` value) wraps operations in
    ``execute_in_torch(backend="inductor")`` scopes: ``"lm-head"`` the final
    RMSNorm and LM head, ``"mlp-inductor"`` every MLP block with its residual
    add. ``generator`` (a ``torch_tensorrt.region.KernelGenerator``) runs
    every MLP block as two ``fused_kernel`` regions whose kernels it writes
    (:meth:`_generated_mlp`). Run eagerly, or traced by anything but strict
    ``torch_tensorrt`` compilation, the scopes change nothing.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        max_seq_len: int,
        *,
        inplace: bool = False,
        inductor: str = "none",
        generator: Any = None,
    ) -> None:
        super().__init__()
        if inductor not in DECODE_INDUCTOR:
            raise ValueError(
                f"Unknown kernels.inductor {inductor!r}; choose from {DECODE_INDUCTOR}"
            )
        if generator is not None and inductor != "none":
            raise ValueError(
                "Generated MLP kernels and kernels.inductor regions cannot be "
                f"combined; got kernels.inductor={inductor!r}"
            )
        self.model = model
        config = model.config
        self.kinds = layer_kinds(config)
        self.max_seq_len = max_seq_len
        self.inplace = inplace
        self.inductor = inductor
        # Created before tracing: fused_kernel takes it as a constant.
        self.generator = generator
        self.attention_layers = [
            i for i, kind in enumerate(self.kinds) if kind == "full_attention"
        ]
        self.mamba_layers = [
            i for i, kind in enumerate(self.kinds) if kind == "linear_attention"
        ]

    def forward(
        self, input_ids: torch.Tensor, position_ids: torch.Tensor, *states: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        n_attention, n_mamba = len(self.attention_layers), len(self.mamba_layers)
        if len(states) != 2 * n_attention + 2 * n_mamba:
            raise ValueError(
                f"Expected {2 * n_attention + 2 * n_mamba} cache and state tensors; "
                f"got {len(states)}"
            )
        kv = states[: 2 * n_attention]
        conv = states[2 * n_attention : 2 * n_attention + n_mamba]
        ssm = states[2 * n_attention + n_mamba :]
        backbone = self.model.model
        hidden = backbone.embeddings(input_ids)
        write_index = position_ids.reshape(-1)
        key_positions = torch.arange(
            self.max_seq_len, dtype=position_ids.dtype, device=position_ids.device
        )
        mask = key_positions <= position_ids.reshape(-1, 1, 1, 1)
        slot = torch.zeros(1, dtype=torch.int64, device=position_ids.device)
        new_kv: list[torch.Tensor] = []
        new_conv: list[torch.Tensor] = []
        new_ssm: list[torch.Tensor] = []
        attention_index = mamba_index = 0
        for index, (block, kind) in enumerate(zip(backbone.layers, self.kinds)):
            residual = hidden
            if kind == "mlp" and self.inductor == "mlp-inductor":
                # The whole block, residual add included: the region reads and
                # writes only the residual stream.
                with _inductor_scope(f"mlp_{index}"):
                    hidden = residual + block.mixer(_rms_norm(block.norm, residual))
                continue
            if kind == "mlp" and self.generator is not None:
                hidden = self._generated_mlp(index, block, residual)
                continue
            hidden = _rms_norm(block.norm, hidden)
            if kind == "linear_attention":
                hidden, conv_state, ssm_state = self._mamba(
                    block.mixer, hidden, conv[mamba_index], ssm[mamba_index], slot
                )
                new_conv.append(conv_state)
                new_ssm.append(ssm_state)
                mamba_index += 1
            elif kind == "full_attention":
                hidden, keys, values = self._attention(
                    block.mixer,
                    hidden,
                    kv[2 * attention_index],
                    kv[2 * attention_index + 1],
                    write_index,
                    mask,
                )
                new_kv += [keys, values]
                attention_index += 1
            else:
                hidden = block.mixer(hidden)
            hidden = residual + hidden
        if self.inductor == "lm-head":
            with _inductor_scope("lm_head"):
                logits = self._logits(hidden)
        else:
            logits = self._logits(hidden)
        return (logits, *new_kv, *new_conv, *new_ssm)

    def _generated_mlp(
        self, index: int, block: torch.nn.Module, residual: torch.Tensor
    ) -> torch.Tensor:
        """One MLP block as two regions whose kernels the generator writes.

        ``mlp_in_<index>`` reads the residual stream, the block norm's weight
        and the up projection and returns the activation: the RMSNorm, the
        up GEMV and the MLP's activation (ReLU²). ``mlp_out_<index>`` reads
        the activation, the down projection and the residual stream and
        returns the next residual stream: the down GEMV and the residual add.
        The operations, their order and their dtypes are those of the plain
        step's ``residual + mixer(_rms_norm(norm, residual))``, the mixer
        being Hugging Face's ``down_proj(act_fn(up_proj(x)))``; outside strict
        ``torch_tensorrt`` compilation the scopes change nothing.
        """
        from torch_tensorrt import region

        mlp = block.mixer
        up, down = mlp.up_proj, mlp.down_proj
        inputs = [residual, block.norm.weight, *_linear_tensors(up)]
        with region.fused_kernel(
            generator=self.generator, inputs=inputs, name=f"mlp_in_{index}"
        ):
            act = mlp.act_fn(up(_rms_norm(block.norm, residual)))
        inputs = [act, *_linear_tensors(down), residual]
        with region.fused_kernel(
            generator=self.generator, inputs=inputs, name=f"mlp_out_{index}"
        ):
            hidden = residual + down(act)
        return hidden

    def _logits(self, hidden: torch.Tensor) -> torch.Tensor:
        """The final RMSNorm and the LM head: FP32 logits ``[1, vocab]``."""
        hidden = _rms_norm(self.model.model.norm_f, hidden)
        return self.model.lm_head(hidden).float().reshape(1, -1)

    @staticmethod
    def _attention(
        attention: torch.nn.Module,
        hidden: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        write_index: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        head_dim = attention.head_dim
        query = attention.q_proj(hidden).view(1, 1, -1, head_dim).transpose(1, 2)
        key = attention.k_proj(hidden).view(1, 1, -1, head_dim).transpose(1, 2)
        value = attention.v_proj(hidden).view(1, 1, -1, head_dim).transpose(1, 2)
        keys = torch.index_copy(key_cache, 2, write_index, key.to(key_cache.dtype))
        values = torch.index_copy(
            value_cache, 2, write_index, value.to(value_cache.dtype)
        )
        out = F.scaled_dot_product_attention(
            query,
            keys,
            values,
            attn_mask=mask,
            scale=attention.scaling,
            enable_gqa=True,
        )
        out = out.transpose(1, 2).reshape(1, 1, -1)
        return attention.o_proj(out), keys, values

    # EDITABLE-START
    def _mamba(
        self,
        mixer: torch.nn.Module,
        hidden: torch.Tensor,
        conv_state: torch.Tensor,
        ssm_state: torch.Tensor,
        slot: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dtype = hidden.dtype
        heads, head_dim = mixer.num_heads, mixer.head_dim
        groups, state_size = mixer.n_groups, mixer.ssm_state_size
        inner, conv_dim = mixer.intermediate_size, mixer.conv_dim
        projected = mixer.in_proj(hidden)
        gate, xbc, dt = projected.split([inner, conv_dim, heads], dim=-1)

        # Depthwise causal convolution over the last conv_kernel inputs.
        previous = conv_state.reshape(1, conv_dim, -1)
        window = torch.cat(
            [previous[:, :, 1:], xbc.reshape(1, conv_dim, 1).to(previous.dtype)], dim=-1
        )
        weight = mixer.conv1d.weight.reshape(conv_dim, -1).float()
        xbc = (window.float() * weight).sum(-1)
        if mixer.conv1d.bias is not None:
            xbc = xbc + mixer.conv1d.bias.float()
        xbc = F.silu(xbc)
        x, b, c = xbc.split([inner, groups * state_size, groups * state_size], dim=-1)

        # Selective state update, h = exp(dt*A)*h + dt*B*x and y = C.h + D*x.
        dt = F.softplus(dt.reshape(1, heads).float() + mixer.dt_bias.float())
        decay = torch.exp(dt * -torch.exp(mixer.A_log.float()))
        x = x.reshape(1, heads, head_dim)
        repeat = heads // groups
        b = b.reshape(1, groups, 1, state_size).expand(1, groups, repeat, state_size)
        c = c.reshape(1, groups, 1, state_size).expand(1, groups, repeat, state_size)
        b = b.reshape(1, heads, 1, state_size)
        c = c.reshape(1, heads, 1, state_size)
        state = ssm_state.reshape(1, heads, head_dim, state_size).float()
        state = (
            state * decay.reshape(1, heads, 1, 1)
            + (dt.reshape(1, heads, 1, 1) * x.unsqueeze(-1)) * b
        )
        y = (state * c).sum(-1) + mixer.D.float().reshape(1, heads, 1) * x

        # Gated RMSNorm per group, as Hugging Face's Zamba2RMSNormGated.
        y = y.reshape(1, 1, inner) * F.silu(gate.float())
        norm = mixer.norm
        grouped = y.reshape(1, 1, -1, norm.group_size)
        grouped = grouped * torch.rsqrt(
            grouped.pow(2).mean(-1, keepdim=True) + norm.variance_epsilon
        )
        y = norm.weight * grouped.reshape(1, 1, inner).to(dtype)
        out = mixer.out_proj(y.to(dtype))

        new_conv = window.reshape(conv_state.shape)
        new_ssm = state.to(ssm_state.dtype).reshape(ssm_state.shape)
        if self.inplace:
            new_conv = torch.index_copy(conv_state, 2, slot, new_conv)
            new_ssm = torch.index_copy(ssm_state, 2, slot, new_ssm)
        return out, new_conv, new_ssm

    # EDITABLE-END


def _linear_tensors(linear: torch.nn.Module) -> list[torch.Tensor]:
    """A linear layer's weight, and its bias if it has one (region inputs)."""
    bias = getattr(linear, "bias", None)
    return [linear.weight] if bias is None else [linear.weight, bias]


# ------------------------------------------------------------ prefill helpers


def prefill(model: torch.nn.Module, prompt: torch.Tensor, max_seq_len: int) -> Any:
    """Prefill ``prompt`` ([1, isl]) eagerly into a new Hugging Face ``StaticCache``."""
    from transformers import StaticCache

    cache = StaticCache(config=model.config, max_cache_len=max_seq_len)
    with torch.inference_mode():
        model(input_ids=prompt, past_key_values=cache, use_cache=True, logits_to_keep=1)
    return cache


def static_states(
    cache: Any,
    kinds: Sequence[str],
    *,
    ssm_dtype: torch.dtype = torch.float32,
    inplace: bool = False,
) -> list[torch.Tensor]:
    """The decode step's cache and state inputs, copied from a prefilled ``StaticCache``."""
    kv: list[torch.Tensor] = []
    conv: list[torch.Tensor] = []
    ssm: list[torch.Tensor] = []
    for index, kind in enumerate(kinds):
        layer = cache.layers[index]
        if kind == "full_attention":
            kv += [layer.keys.clone(), layer.values.clone()]
        elif kind == "linear_attention":
            conv_state = layer.conv_states[0].clone()
            ssm_state = layer.recurrent_states[0].to(ssm_dtype, copy=True)
            if inplace:
                conv_state = conv_state.unsqueeze(2)
                ssm_state = ssm_state.reshape(
                    ssm_state.shape[0], -1, 1, ssm_state.shape[-1]
                )
            conv.append(conv_state.contiguous())
            ssm.append(ssm_state.contiguous())
    return kv + conv + ssm


def decode_bytes_per_token(
    config: Any,
    *,
    weight_bytes: float,
    embedding_bytes: float,
    max_seq_len: int,
    dtype_size: int,
    ssm_dtype_size: int,
) -> float:
    """Bytes one decode step must move: weights, the whole static KV cache, states.

    The untied input embedding contributes one row and is left out; the LM
    head is read in full. Attention reads all ``max_seq_len`` cache slots.
    The conv and SSM states are read and written once each.
    """
    shapes = state_shapes(config, max_seq_len)
    kinds = layer_kinds(config)
    n_attention = kinds.count("full_attention")
    n_mamba = kinds.count("linear_attention")
    kv = 2 * n_attention * _numel(shapes["kv"]) * dtype_size
    conv = 2 * n_mamba * _numel(shapes["conv"]) * dtype_size
    ssm = 2 * n_mamba * _numel(shapes["ssm"]) * ssm_dtype_size
    return float(weight_bytes - embedding_bytes + kv + conv + ssm)


def _numel(shape: Sequence[int]) -> int:
    count = 1
    for size in shape:
        count *= size
    return count


# ------------------------------------------------------ the formulation gate


def check_formulation(
    model: torch.nn.Module,
    tokens: torch.Tensor,
    isl: int,
    max_seq_len: int,
    step_class: type,
    *,
    inplace: bool = False,
    steps: int = FORMULATION_STEPS,
) -> dict[str, float]:
    """Compare ``step_class`` with :class:`NemotronDecodeStep` on ``model`` as it is.

    Both steps decode ``steps`` teacher-forced tokens (at most as many as
    ``tokens`` and ``max_seq_len`` hold) eagerly from the same prefill of
    ``tokens[:isl]``, each threading its own outputs. Returns the largest
    absolute logit difference and the largest difference of any cache or
    state output relative to that tensor's largest magnitude. Run it with the
    model in FP32 (see :func:`formulation_gate`), where a correct rewrite
    differs by rounding only.
    """
    steps = min(steps, max_seq_len - isl, tokens.numel() - isl)
    kinds = layer_kinds(model.config)
    cache = prefill(model, tokens[:isl].view(1, -1), max_seq_len)
    start = static_states(cache, kinds, inplace=inplace)
    del cache
    original = NemotronDecodeStep(model, max_seq_len, inplace=inplace).eval()
    candidate = step_class(model, max_seq_len, inplace=inplace).eval()
    expected_states = [state.clone() for state in start]
    actual_states = [state.clone() for state in start]
    logit_error = state_error = 0.0
    with torch.inference_mode():
        for j in range(steps):
            ids = tokens[isl + j].view(1, 1)
            position = torch.full(
                (1, 1), isl + j, dtype=torch.int64, device=tokens.device
            )
            expected = original(ids, position, *expected_states)
            actual = candidate(ids, position, *actual_states)
            if len(actual) != len(expected):
                raise ValueError(
                    f"The step returns {len(actual)} tensors; expected {len(expected)}"
                )
            logit_error = max(logit_error, _max_difference(actual[0], expected[0]))
            for got, want in zip(actual[1:], expected[1:]):
                if got.shape != want.shape or got.dtype != want.dtype:
                    raise ValueError(
                        f"A cache or state output is {got.dtype} "
                        f"{tuple(got.shape)}; expected {want.dtype} "
                        f"{tuple(want.shape)}"
                    )
                scale = max(want.abs().max().item(), 1e-30)
                state_error = max(state_error, _max_difference(got, want) / scale)
            # Eagerly, index_copy returns new tensors: thread every output.
            expected_states, actual_states = list(expected[1:]), list(actual[1:])
    return {
        "max_abs_logit_error": logit_error,
        "max_rel_state_error": state_error,
        "steps": float(steps),
    }


def formulation_failures(report: Mapping[str, float]) -> list[str]:
    """The :data:`FORMULATION_LIMITS` that a :func:`check_formulation` report breaks."""
    return [
        f"{name} {report[name]:.3g} > {limit:g}"
        for name, limit in FORMULATION_LIMITS.items()
        if not report[name] <= limit
    ]


def formulation_gate(
    model: torch.nn.Module,
    tokens: torch.Tensor,
    isl: int,
    max_seq_len: int,
    step_class: type,
    *,
    inplace: bool = False,
) -> dict[str, float]:
    """Run :func:`check_formulation` in FP32 (no TF32); raise if it breaks the limits.

    The model's floating-point tensors are cast to FP32 for the check and
    back to their own dtypes after it, which restores them exactly.
    """
    with _float32(model), _without_tf32():
        report = check_formulation(
            model, tokens, isl, max_seq_len, step_class, inplace=inplace
        )
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    failures = formulation_failures(report)
    if failures:
        raise ValueError(
            "edit.decode_step changes the decode step's FP32 results, so it is "
            "not a reformulation of the same math: "
            + "; ".join(failures)
            + f" over {report['steps']:.0f} steps against the original step"
        )
    return report


def _max_difference(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return (actual.double() - expected.double()).abs().max().item()


@contextlib.contextmanager
def _float32(model: torch.nn.Module) -> Iterator[None]:
    dtypes = {key: tensor.dtype for key, tensor in _floating_tensors(model)}
    _cast_floating(model, lambda key: torch.float32)
    try:
        yield
    finally:
        _cast_floating(model, dtypes.__getitem__)


def _floating_tensors(model: torch.nn.Module) -> Iterator[tuple[str, torch.Tensor]]:
    for prefix, module in model.named_modules():
        for store in (module._parameters, module._buffers):
            for name, tensor in store.items():
                if tensor is not None and tensor.is_floating_point():
                    yield f"{prefix}.{name}", tensor


def _cast_floating(model: torch.nn.Module, dtype_of: Any) -> None:
    with torch.no_grad():
        for prefix, module in model.named_modules():
            for store in (module._parameters, module._buffers):
                for name, tensor in store.items():
                    if tensor is None or not tensor.is_floating_point():
                        continue
                    dtype = dtype_of(f"{prefix}.{name}")
                    if tensor.dtype == dtype:
                        continue
                    if isinstance(tensor, torch.nn.Parameter):
                        tensor.data = tensor.data.to(dtype)
                    else:
                        store[name] = tensor.to(dtype)


# ----------------------------------------------------------- Inductor regions


def _inductor_scope(name: str) -> Any:
    """A scope strict ``torch_tensorrt`` compilation keeps out of TensorRT and
    compiles with Inductor; anywhere else it runs its body unchanged."""
    from torch_tensorrt import region

    return region.execute_in_torch(name=name, backend="inductor")


def named_inputs(module: torch.nn.Module, count: int) -> torch.nn.Module:
    """``module(*inputs)`` behind a forward with one named parameter per input.

    ``torch_tensorrt.compile`` names each input's ``dynamic_shapes`` entry
    after a parameter of ``forward``; a ``*states`` parameter takes one name
    for many inputs, and ``torch.export`` then rejects the structure. The
    wrapper's forward runs ``module`` with the same inputs, but its signature
    names ``count`` positional parameters ``input_0``, ``input_1``, ...
    """
    import inspect

    def forward(self: torch.nn.Module, *inputs: Any) -> Any:
        return self.wrapped(*inputs)

    forward.__signature__ = inspect.Signature(  # type: ignore[attr-defined]
        [
            inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD)
            for name in ("self", *(f"input_{i}" for i in range(count)))
        ]
    )
    wrapper = type("NamedInputs", (torch.nn.Module,), {"forward": forward})()
    wrapper.wrapped = module
    return wrapper.eval()


def region_compilation(tf32: bool | None = None) -> contextlib.ExitStack:
    """Global settings Inductor regions compile and run under, until closed.

    The region runtime compiles each region with ``torch.compile`` options
    that only turn Inductor's CUDA graphs off, so the rest of the Inductor
    configuration is global: this sets the options of
    :data:`REGION_INDUCTOR_MODE`. ``tf32``, if given, sets PyTorch's TF32
    flags (matmul and cuDNN), which the regions' FP32 matrix multiplications
    and convolutions read when they run.
    """
    import torch._inductor
    import torch._inductor.config as inductor_config

    stack = contextlib.ExitStack()
    try:
        stack.enter_context(
            inductor_config.patch(
                torch._inductor.list_mode_options(REGION_INDUCTOR_MODE)
            )
        )
        if tf32 is not None:
            matmul = torch.backends.cuda.matmul.allow_tf32
            cudnn = torch.backends.cudnn.allow_tf32
            stack.callback(_set_cudnn_tf32, cudnn)
            stack.callback(setattr, torch.backends.cuda.matmul, "allow_tf32", matmul)
            torch.backends.cuda.matmul.allow_tf32 = tf32
            _set_cudnn_tf32(tf32)
    except BaseException:
        stack.close()
        raise
    return stack


def release_unused_weights(model: torch.nn.Module, compiled: torch.nn.Module) -> int:
    """Free the memory of ``model``'s weights that ``compiled`` does not hold.

    Region builds cannot offload the model to the CPU, and the regions run on
    some of its tensors; the engines hold copies of the rest. Every parameter
    and buffer whose storage ``compiled`` does not reference (as a parameter,
    buffer, tensor attribute or ``get_attr`` target) is replaced by an empty
    tensor. Returns the bytes released.
    """
    keep: set[int] = set()

    def hold(value: Any) -> None:
        if isinstance(value, torch.Tensor):
            keep.add(value.untyped_storage().data_ptr())

    for module in compiled.modules():
        for value in (
            *module.parameters(recurse=False),
            *module.buffers(recurse=False),
        ):
            hold(value)
        for value in vars(module).values():
            hold(value)
        if isinstance(module, torch.fx.GraphModule):
            for node in module.graph.nodes:
                if node.op == "get_attr":
                    value: Any = module
                    with contextlib.suppress(AttributeError):
                        for part in str(node.target).split("."):
                            value = getattr(value, part)
                        hold(value)
    released: dict[int, int] = {}
    with torch.no_grad():
        for tensor in (*model.parameters(), *model.buffers()):
            storage = tensor.untyped_storage()
            if storage.data_ptr() not in keep and storage.nbytes():
                released[storage.data_ptr()] = storage.nbytes()
                tensor.data = torch.empty(0, dtype=tensor.dtype, device=tensor.device)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return sum(released.values())


def inductor_regions(module: torch.nn.Module) -> int:
    """How many ``execute_in_torch(backend="inductor")`` regions ``module`` runs."""
    from torch_tensorrt.dynamo.region_compilation import get_region_records

    return sum(
        1
        for record in get_region_records(module)
        if record.route == "torch" and record.backend == "inductor"
    )


def inductor_kernels() -> list[str]:
    """The kernels in the Inductor output code this process has loaded.

    Triton kernels by name and library calls as ``extern_kernels.<op>``,
    read from the generated wrapper modules: a record of what the regions
    run.
    """
    from torch._inductor.codecache import PyCodeCache

    names: set[str] = set()
    for module in list(PyCodeCache.modules):
        try:
            source = Path(module.__file__).read_text()
        except (OSError, TypeError, AttributeError):
            continue
        if "async_compile" not in source or "def call(" not in source:
            continue
        names.update(
            re.findall(r"^(triton_\w+) = async_compile\.triton\(", source, re.M)
        )
        names.update(
            f"extern_kernels.{op}"
            for op in re.findall(r"extern_kernels\.(\w+)\(", source)
        )
    return sorted(names)


def check_generated_values(values: Mapping[str, Any]) -> str:
    """Validate the decode workload's ``kernels.generated``; return its value.

    ``region_generation.check_generated``: a generated value needs CUDA graphs
    and ``kernels.inductor="none"``.
    """
    import region_generation

    return region_generation.check_generated(values, DECODE_GENERATED)


def save_program(
    module: torch.nn.Module, path: Path, inductor: str | None = None
) -> list[str]:
    """Save a compiled TensorRT program with ``torch_tensorrt.save``; return its name.

    The engine (about 8 GB of weights) is one pickled constant, which pickle
    protocol 2, the default, cannot serialize above 4 GiB: protocol 4 is used.
    A ``kernels.inductor`` build (``inductor``, the knob's value) cannot be
    saved: the exporter rejects ``execute_in_torch`` regions whose backend is
    not ``eager``, before it writes anything. That rejection is reported as
    ``NotImplementedError``, the contract's "cannot export", naming the value;
    other errors propagate unchanged.
    """
    import torch_tensorrt
    from torch_tensorrt import region

    try:
        torch_tensorrt.save(module, str(path), retrace=False, pickle_protocol=4)
    except region.RegionError as error:
        if inductor in (None, "none"):
            raise
        raise NotImplementedError(
            f"kernels.inductor={inductor!r} builds cannot be saved: their "
            "execute_in_torch(backend='inductor') regions run code that Inductor "
            "compiles at run time, and torch_tensorrt.save cannot preserve that "
            "backend. Rebuild the config instead."
        ) from error
    return [path.name]


def _set_cudnn_tf32(value: bool) -> None:
    """Set cuDNN's TF32 flag, also once ``torch.backends.disable_global_flags()``
    has frozen the attribute (PyTorch's test utilities call it on import)."""
    torch.backends.cudnn.set_flags(_allow_tf32=value, _fp32_precision=None)


@contextlib.contextmanager
def _without_tf32() -> Iterator[None]:
    matmul = torch.backends.cuda.matmul.allow_tf32
    cudnn = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    _set_cudnn_tf32(False)
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = matmul
        _set_cudnn_tf32(cudnn)


# ----------------------------------------------------------------- workloads


class NemotronDecodeWorkload(Workload):
    """Batch-1 greedy decode of a Nemotron-H model with static caches and states.

    The baseline is the straightforward Torch-TensorRT flow for such a model:
    the functional decode step in the checkpoint's BF16, exported with
    ``torch.export`` and compiled by ``torch_tensorrt.dynamo.compile`` with
    default builder settings (optimization level 3), FP32 SSM state, no CUDA
    graphs. ``kernel_cache`` is the KernelGenerator cache that
    ``kernels.generated`` takes its kernels from
    (``region_generation.kernel_cache``).
    """

    def __init__(
        self,
        model: str = MODEL,
        *,
        isl: int = 128,
        osl: int = 64,
        baseline: Mapping[str, Any] | None = None,
        kernel_cache: str | None = None,
    ) -> None:
        if isl < 1 or osl < 2:
            raise ValueError("NemotronDecodeWorkload needs isl >= 1 and osl >= 2")
        import region_generation

        self.kernel_cache = region_generation.kernel_cache(kernel_cache)
        self.model_name = model
        self.name = f"{model.rsplit('/', 1)[-1]}-decode-isl{isl}-osl{osl}"
        self.isl, self.osl = isl, osl
        self.max_seq = isl + osl
        self._baseline = dict(baseline or {})
        self._tokens: torch.Tensor | None = None

    # ------------------------------------------------------------ contract

    def knobs(self) -> Sequence[Knob]:
        import region_generation

        return (
            Knob(
                "engine.precision",
                ("bf16", "fp16"),
                "bf16",
                "Model weight and activation precision. bf16 is the checkpoint's; "
                "fp16 has more mantissa bits but a narrower range (the residual "
                "stream must stay below 65504). The SSM update and convolution "
                "run in FP32 either way.",
                ("precision",),
            ),
            Knob(
                "engine.use_fp32_acc",
                (False, True),
                False,
                "With fp16, accumulate matrix multiplications in FP32 by inserting "
                "casts (bf16 matrix multiplications accumulate in FP32 already).",
                ("precision", "dominant_kernel"),
                {"engine.precision": "fp16"},
            ),
            Knob(
                "workload.ssm_state",
                ("fp32", "bf16"),
                "fp32",
                "Storage precision of the Mamba-2 SSM states (21 x [96, 80, 128]; "
                "82.6 MB in fp32), read and written every step. bf16 halves that "
                "traffic; the update itself stays FP32.",
                ("roofline_gap", "copy_overhead"),
            ),
            Knob(
                "workload.state_update",
                ("functional", "inplace"),
                "functional",
                "How the engine updates the Mamba conv and SSM states. functional "
                "returns them as new outputs: CUDA graphs then copy every state "
                "into and out of their staging buffers each step (about 4x the "
                "state bytes of extra traffic). inplace writes them into the input "
                "buffers through IKVCacheUpdateLayer, as the KV cache is, with no "
                "copies.",
                ("copy_overhead", "roofline_gap"),
            ),
            *self._build_route_knobs(),
            *(knob for knob in STANDARD_KNOBS if knob.name == "runtime.cudagraphs"),
            Knob(
                "kernels.inductor",
                DECODE_INDUCTOR,
                "none",
                "Hybrid build: hand operations to Inductor-compiled regions inside "
                "the compiled decode step (torch_tensorrt.region.execute_in_torch, "
                "backend inductor). Regions need strict export, so these builds use "
                "torch_tensorrt.compile(strict=True) and keep the model on the GPU "
                "during the build. Each region compiles during the build in "
                "Inductor's max-autotune mode, whose coordinate-descent tuning "
                "lowers a batch-1 matrix multiply to a Triton reduction kernel with "
                "the preceding RMSNorm fused into it. lm-head: the final RMSNorm and "
                "the LM-head GEMV as one region after the single engine (1 engine "
                "boundary). mlp-inductor: each of the 17 MLP blocks (RMSNorm, up "
                "GEMV, ReLU^2, down GEMV, residual add) as its own region, which "
                "splits the step into 18 engines (34 boundaries). Requires in-place "
                "state updates and CUDA graphs: the whole step, engines and regions, "
                "is captured as one CUDA graph, so a boundary costs a gap between "
                "kernels rather than a host-side call. Costs: longer builds "
                "(Inductor autotuning) and a kernel gap per boundary; the moved "
                "work runs on Inductor's kernels instead of TensorRT's. These "
                "builds cannot be saved: torch_tensorrt.save refuses "
                "execute_in_torch regions whose backend is 'inductor', so a "
                "session that keeps one writes its config but no artifact.",
                ("dominant_kernel", "roofline_gap"),
                {"runtime.cudagraphs": True, "workload.state_update": "inplace"},
            ),
            Knob(
                "kernels.generated",
                DECODE_GENERATED,
                "none",
                region_generation.GENERATED_INTRO + " 'none': no generated "
                "kernels. 'mlp': each "
                "of the 17 MLP blocks as two regions, the block RMSNorm + up "
                "GEMV + ReLU^2 (mlp_in) and the down GEMV + residual add "
                "(mlp_out), 34 plugins in the one engine (no engine split); "
                "identical blocks share one kernel per region. Mamba-2 and "
                "attention layers, the LM head and the KV-cache and state "
                "updates stay in TensorRT. Requires runtime.cudagraphs=True "
                "(TensorRT 11.4 reloads plugin modules on every enqueue outside "
                "CUDA graphs; each engine replays its own graph) and "
                "kernels.inductor='none'. Plugins need strict export, so the "
                "build uses torch_tensorrt.compile(strict=True) and keeps the "
                "model on the GPU while it builds. " + region_generation.CACHE_NOTE,
                ("dominant_kernel", "roofline_gap", "launch_overhead"),
                {**region_generation.PLUGIN_REQUIRES, "kernels.inductor": "none"},
                prepare=DECODE_GENERATED[1:],
            ),
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
                DECODE_ROUTES,
                "",
                "TensorRT Global Performance Tuning build route (Myelin knobs). "
                "'-kgen:lower_fc:all=on' generates every GEMV with kgen, which "
                "fuses the RMSNorms into them, instead of cuBLAS-style xmma "
                "kernels; '-tactic:splitk=off' forbids split-K GEMV tactics and "
                "their reduction kernels (kgen GEMVs have none).",
                ("dominant_kernel",),
            ),
        )

    def editable_files(self) -> Sequence[EditableFile]:
        return (
            EditableFile(
                "decode_step",
                str(Path(__file__).resolve()),
                "The PyTorch formulation of the TensorRT decode step's non-GEMM "
                "work: _rms_norm (the 42 block norms and the final norm) and "
                "NemotronDecodeStep._mamba (in_proj split, depthwise conv over "
                "the conv-state window, softplus/exp decay, the SSM state update "
                "h = exp(dt*A)*h + dt*B*x with y = C.h + D*x, the gated group "
                "RMSNorm, out_proj, and the state outputs, written in place with "
                "index_copy when inplace). Myelin fuses these ops into its own "
                "kernels, so op order, casts, layouts and reductions decide how "
                "many kernels run and how fast (in functional mode Myelin's "
                "fused Mamba-step kernel takes ~22 us per layer, ~0.46 ms per "
                "token; in place, ~4.7 us per layer plus one ~0.09 ms state "
                "scatter per token; each RMSNorm is a ~4.5 us latency-bound "
                "kernel). Keep the math (FP32 statistics and SSM update), every "
                "input and output with its shape and dtype, and the index_copy "
                "writes of the in-place mode (TensorRT lowers them to in-place "
                "IKVCacheUpdateLayer writes). An edited build first runs an FP32 "
                "gate (about 1 s): the edited and the original step decode 8 "
                "tokens eagerly in FP32, and the build fails unless the logits "
                "agree within 1e-3 and every cache and state within 1e-4 of its "
                "magnitude. Matters for the TensorRT backend only.",
            ),
        )

    def baseline(self) -> dict[str, Any]:
        return dict(self._baseline)

    def timing(self) -> Timing:
        return Timing(warmup=2, rounds=10, calls=1, profile_calls=1)

    def accuracy_defaults(self) -> dict[str, tuple[str, float]]:
        # Correct BF16 engines measure a mean KL of 1e-4 to 7.4e-4 here and
        # FP16 ones 4e-6 to 2e-5; 2e-3 still passes them with a 2.7x margin.
        return {"top1_agreement": (">=", 0.97), "mean_kl": ("<=", 2e-3)}

    def tokens(self) -> torch.Tensor:
        """The fixed token sequence: ``isl`` prompt tokens, then ``osl`` decode inputs."""
        if self._tokens is None:
            self._tokens = text_tokens(self.model_name, self.max_seq, TEXT)
        return self._tokens

    def reference(self) -> Any:
        """FP32 eager logits at every decode position, and the tokens they are for.

        The Hugging Face model in FP32 prefills the prompt with its chunked
        scan and then decodes token by token with its own cache, the
        recurrent path generation takes.
        """
        model = _load(self.model_name, torch.float32, torch.device("cuda:0"))
        tokens = self.tokens()
        try:
            logits = _hf_decode_logits(model, tokens.cuda(), self.isl, self.osl)
            return {"logits": logits.cpu(), "tokens": tokens.clone()}
        finally:
            del model
            gc.collect()
            torch.cuda.empty_cache()

    def compare(self, outputs: Any, reference: Any) -> dict[str, float]:
        return compare_logits(
            outputs, reference_logits(reference, tokens=self.tokens())
        )

    # -------------------------------------------------------------- build

    def kernel_generator(self) -> Any:
        """The KernelGenerator of ``kernels.generated``: the shared settings
        (``region_generation.make_generator``) with the MLP regions' notes,
        the same in every build."""
        import region_generation

        return region_generation.make_generator(self.kernel_cache, GENERATION_NOTES)

    def build(self, values: Mapping[str, Any]) -> Candidate:
        backend = values.get("bench.backend", "tensorrt")
        if backend in ("eager", "inductor"):
            return self._build_torch(values, backend)
        if backend != "tensorrt":
            raise ValueError(f"Unknown bench.backend {backend!r}")
        return self._build_tensorrt(values)

    def _build_tensorrt(self, values: Mapping[str, Any]) -> Candidate:
        import torch_tensorrt

        precision = values["engine.precision"]
        dtype = _DTYPES[precision]
        ssm_dtype = _DTYPES[values["workload.ssm_state"]]
        inplace = values["workload.state_update"] == "inplace"
        inductor = values.get("kernels.inductor", "none")
        if inductor not in DECODE_INDUCTOR:
            raise ValueError(
                f"Unknown kernels.inductor {inductor!r}; choose from {DECODE_INDUCTOR}"
            )
        generated = check_generated_values(values)
        generator = self.kernel_generator() if generated != "none" else None
        hybrid = inductor != "none"
        # bench.whole_graph: the region builds' whole-step capture without
        # regions, a head-to-head control for that capture.
        whole_graph = hybrid or bool(values.get("bench.whole_graph", False))
        if whole_graph and not (inplace and values["runtime.cudagraphs"]):
            raise ValueError(
                f"kernels.inductor={inductor!r} (or bench.whole_graph) needs "
                "workload.state_update='inplace' and runtime.cudagraphs=True: "
                "the step, engines and Inductor regions, is captured as one "
                "CUDA graph over static state buffers"
            )
        device = torch.device("cuda:0")
        model = _load(self.model_name, dtype, device)
        kinds = layer_kinds(model.config)
        tokens = self.tokens().to(device)
        step_class = NemotronDecodeStep
        details: dict[str, Any] = {
            "precision": precision,
            "ssm_state": values["workload.ssm_state"],
            "state_update": values["workload.state_update"],
        }
        if values.get("edit.decode_step"):
            edited = load_module(values["edit.decode_step"], "nemotron_decode_edited")
            step_class = edited.NemotronDecodeStep
            start_s = time.perf_counter()
            report = formulation_gate(
                model, tokens, self.isl, self.max_seq, step_class, inplace=inplace
            )
            details["formulation_check"] = {
                **report,
                "seconds": time.perf_counter() - start_s,
            }
        cache = prefill(model, tokens[: self.isl].view(1, -1), self.max_seq)
        start = static_states(cache, kinds, ssm_dtype=ssm_dtype, inplace=inplace)
        del cache
        bytes_per_unit = _bytes_per_unit(model, self.max_seq, ssm_dtype)
        if hybrid:
            step = step_class(
                model, self.max_seq, inplace=inplace, inductor=inductor
            ).eval()
        elif generator is not None:
            import region_generation

            step = step_class(
                model, self.max_seq, inplace=inplace, generator=generator
            ).eval()
            details["kernels_generated"] = generated
            details["generated_regions"] = 2 * kinds.count("mlp")
            details.update(region_generation.describe(generator))
        else:
            step = step_class(model, self.max_seq, inplace=inplace).eval()
        ids = tokens[self.isl].view(1, 1)
        position = torch.full((1, 1), self.isl, dtype=torch.int64, device=device)
        example = [ids, position, *(state.clone() for state in start)]
        settings = {
            "use_fp32_acc": precision == "fp16" and values["engine.use_fp32_acc"],
            "disable_tf32": True,
            "device": device,
            "offload_module_to_cpu": True,
            "min_block_size": 1,
            "optimization_level": 3,
            "build_route": values.get("engine.build_route") or None,
        }
        # Region builds: Inductor's options stay set while the candidate lives
        # (the regions compile on their first call), restored on close().
        stack = region_compilation() if hybrid else contextlib.ExitStack()
        try:
            if hybrid or generator is not None:
                # Only strict export captures regions, and regions refuse
                # offloading the model to the CPU.
                settings["offload_module_to_cpu"] = False
                with torch.no_grad():
                    module = torch_tensorrt.compile(
                        named_inputs(step, len(example)),
                        ir="dynamo",
                        arg_inputs=example,
                        strict=True,
                        **settings,
                    )
                if hybrid:
                    details["inductor"] = inductor
                    details["inductor_regions"] = inductor_regions(module)
                details["released_mb"] = release_unused_weights(model, module) / 2**20
            else:
                with torch.inference_mode():
                    program = torch.export.export(step, tuple(example), strict=False)
                    module = torch_tensorrt.dynamo.compile(
                        program, arg_inputs=example, **settings
                    )
                del program
        except BaseException:
            stack.close()
            raise
        del example, step, model
        gc.collect()
        torch.cuda.empty_cache()
        candidate = _TensorRTCandidate(
            module,
            tokens=tokens,
            isl=self.isl,
            osl=self.osl,
            start=start,
            n_kv=2 * kinds.count("full_attention"),
            inplace=inplace,
            cudagraphs=values["runtime.cudagraphs"],
            whole_graph=whole_graph,
            stack=stack,
        )
        if whole_graph:
            details["cudagraph"] = "whole step"
        candidate.bytes_per_unit = bytes_per_unit
        candidate.details = details
        if hybrid:
            # Compile (and autotune) the regions here, so build_s includes it.
            try:
                details["compile_s"] = candidate.warm_up()
            except BaseException:
                candidate.close()
                raise
            details["inductor_kernels"] = inductor_kernels()
        return candidate

    def _build_torch(self, values: Mapping[str, Any], backend: str) -> Candidate:
        """The Hugging Face decode step with its hybrid StaticCache, eager or compiled."""
        import torch._inductor.config as inductor_config

        if values.get("kernels.generated", "none") != "none":
            raise ValueError(
                "kernels.generated runs kernels inside the TensorRT engine; use "
                "it with bench.backend='tensorrt'"
            )
        precision = values.get("engine.precision", "bf16")
        attn = values.get("bench.attn") or "sdpa"
        mode = values.get("bench.mode") or "default"
        for key, value in dict(values.get("bench.inductor") or {}).items():
            setattr(inductor_config, key, value)
        if attn == "flex_attention":
            fix_flex_attention_masks()
        device = torch.device("cuda:0")
        model = _load(self.model_name, _DTYPES[precision], device, attn=attn)
        tokens = self.tokens().to(device)
        cache = prefill(model, tokens[: self.isl].view(1, -1), self.max_seq)

        def decode_step(input_ids: torch.Tensor, position_ids: torch.Tensor) -> Any:
            return model(
                input_ids=input_ids,
                position_ids=position_ids,
                past_key_values=cache,
                use_cache=True,
            ).logits

        step = (
            decode_step
            if backend == "eager"
            else torch.compile(decode_step, mode=mode, fullgraph=True)
        )
        candidate = _TorchCandidate(
            step,
            cache,
            kinds=layer_kinds(model.config),
            tokens=tokens,
            isl=self.isl,
            osl=self.osl,
            cudagraphs=backend == "inductor" and mode in _CUDAGRAPH_MODES,
        )
        candidate.bytes_per_unit = _bytes_per_unit(model, self.max_seq, torch.float32)
        candidate.details = {"backend": backend, "mode": mode, "attn": attn}
        # Compile (and record CUDA graphs) here, so build_s includes it as it
        # includes the TensorRT engine build.
        start = time.perf_counter()
        for _ in range(3):
            candidate()
        torch.cuda.synchronize()
        candidate.details["compile_s"] = time.perf_counter() - start
        return candidate


# ------------------------------------------------------------------ prefill


class PrefillLogits(torch.nn.Module):
    """A Nemotron-H causal LM's full-sequence forward, logits at fixed positions.

    ``forward(input_ids [1, length])`` runs Hugging Face's own forward without
    a cache (the Mamba layers take the chunked SSD scan) and returns FP32
    logits ``[len(positions), vocab]`` of the selected positions only.
    """

    def __init__(self, model: torch.nn.Module, positions: torch.Tensor) -> None:
        super().__init__()
        self.model = model
        self.register_buffer("positions", positions, persistent=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.model.model(input_ids=input_ids, use_cache=False)
        hidden = hidden.last_hidden_state.index_select(1, self.positions)
        return self.model.lm_head(hidden).float()[0]


def prefill_positions(length: int, count: int) -> torch.Tensor:
    """``count`` evenly spaced positions of a ``length``-token prompt, ending at the last."""
    if not 1 <= count <= length:
        raise ValueError("Need 1 <= positions <= length")
    return torch.arange(1, count + 1) * length // count - 1


def ssd_chunk_scan(
    hidden_states: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    chunk_size: int,
    D: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    initial_states: torch.Tensor | None = None,
    dt_softplus: bool = False,
    dt_limit: tuple[float, float] = (0.0, float("inf")),
    return_final_states: bool = False,
    *,
    grouped: bool = True,
    **kwargs: Any,
) -> Any:
    """Mamba-2's chunked SSD scan, a drop-in for Hugging Face's ``mamba2_chunk_scan``.

    The same FP32 math with two rewrites, as mamba_ssm's kernels compute it:
    the within-chunk segment sums ``cumsum[i] - cumsum[j]`` come from one
    cumulative sum per chunk instead of a cumulative sum over an expanded,
    masked ``[chunk, chunk]`` tensor; and with ``grouped`` the ``C.B`` scores,
    chunk states and state outputs use B and C per group (shared by
    ``heads / groups`` heads) instead of repeating them per head.
    """
    batch, length, heads, head_dim = hidden_states.shape
    groups, state_size = B.shape[2], B.shape[3]
    repeat = heads // groups
    if dt_bias is not None:
        dt = dt + dt_bias.to(dt.dtype)
    if dt_softplus:
        dt = F.softplus(dt)
    dt = torch.clamp(dt, min=dt_limit[0], max=dt_limit[1]).float()
    x = hidden_states.float()
    B, C = B.float(), C.float()
    pad = (chunk_size - length % chunk_size) % chunk_size
    residual = None
    if D is not None:
        residual = D[..., None] * F.pad(x, (0, 0, 0, 0, 0, pad))
    x = x * dt[..., None]
    A = A.to(x.dtype) * dt
    chunks = (length + pad) // chunk_size

    def chunked(tensor: torch.Tensor) -> torch.Tensor:
        widths = [0, 0] * (tensor.dim() - 2) + [0, pad]
        tensor = F.pad(tensor, widths)
        return tensor.reshape(batch, chunks, chunk_size, *tensor.shape[2:])

    x, A, B, C = chunked(x), chunked(A), chunked(B), chunked(C)
    A = A.permute(0, 3, 1, 2)  # [b, h, c, l]
    A_cumsum = torch.cumsum(A, dim=-1)
    causal = torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=x.device)
    causal = torch.tril(causal)
    segment = A_cumsum[..., :, None] - A_cumsum[..., None, :]
    decay = torch.exp(segment.masked_fill(~causal, -torch.inf))  # [b, h, c, l, s]
    decay_states = torch.exp(A_cumsum[..., -1:] - A_cumsum)  # [b, h, c, l]
    x_decay = x * decay_states.permute(0, 2, 3, 1)[..., None]  # [b, c, l, h, p]
    if grouped:
        scores = torch.einsum("bclgn,bcsgn->bcgls", C, B)
        scores = scores[:, :, :, None].expand(-1, -1, -1, repeat, -1, -1)
        scores = scores.reshape(batch, chunks, heads, chunk_size, chunk_size)
        y_diag = torch.einsum(
            "bchls,bcshp->bclhp", scores * decay.permute(0, 2, 1, 3, 4), x
        )
        states = torch.einsum(
            "bclgn,bclgrp->bcgrpn",
            B,
            x_decay.reshape(batch, chunks, chunk_size, groups, repeat, head_dim),
        ).reshape(batch, chunks, heads, head_dim, state_size)
    else:
        B = B.repeat_interleave(repeat, dim=3)
        C = C.repeat_interleave(repeat, dim=3)
        scores = torch.einsum("bclhn,bcshn->bclsh", C, B)
        y_diag = torch.einsum(
            "bclsh,bcshp->bclhp", scores * decay.permute(0, 2, 3, 4, 1), x
        )
        states = torch.einsum("bclhn,bclhp->bchpn", B, x_decay)
    if initial_states is not None:
        previous = initial_states[:, None].to(dtype=states.dtype, device=states.device)
    else:
        previous = torch.zeros_like(states[:, :1])
    states = torch.cat([previous, states], dim=1)
    # Decay between chunk boundaries: segment sums of the per-chunk totals.
    ends = torch.cumsum(F.pad(A_cumsum[..., -1], (1, 0)), dim=-1)  # [b, h, c + 1]
    steps = torch.tril(
        torch.ones(chunks + 1, chunks + 1, dtype=torch.bool, device=x.device)
    )
    chunk_decay = torch.exp(
        (ends[..., :, None] - ends[..., None, :]).masked_fill(~steps, -torch.inf)
    ).transpose(
        1, 3
    )  # [b, z, c, h]
    new_states = torch.einsum("bzch,bzhpn->bchpn", chunk_decay, states)
    states, final_state = new_states[:, :-1], new_states[:, -1]
    out_decay = torch.exp(A_cumsum).permute(0, 2, 3, 1)[..., None]  # [b, c, l, h, 1]
    if grouped:
        y_off = torch.einsum(
            "bclgn,bcgrpn->bclgrp",
            C,
            states.reshape(batch, chunks, groups, repeat, head_dim, state_size),
        ).reshape(batch, chunks, chunk_size, heads, head_dim)
    else:
        y_off = torch.einsum("bclhn,bchpn->bclhp", C, states)
    y = (y_diag + y_off * out_decay).reshape(batch, -1, heads, head_dim)
    if residual is not None:
        y = y + residual
    if pad:
        y = y[:, :length]
    return (y, final_state) if return_final_states else y


def use_ssd(variant: str) -> None:
    """Select the chunked SSD scan of every Nemotron-H Mamba layer in this process.

    ``"hf"`` is Hugging Face's ``mamba2_chunk_scan``; ``"segsum"`` and
    ``"grouped"`` are :func:`ssd_chunk_scan` without and with per-group B and C.
    """
    import functools

    from transformers.models.nemotron_h import modeling_nemotron_h as modeling

    original = getattr(modeling, "_hf_mamba2_chunk_scan", None)
    if original is None:
        original = modeling._hf_mamba2_chunk_scan = modeling.mamba2_chunk_scan
    if variant == "hf":
        modeling.mamba2_chunk_scan = original
    elif variant in ("segsum", "grouped"):
        modeling.mamba2_chunk_scan = functools.partial(
            ssd_chunk_scan, grouped=variant == "grouped"
        )
    else:
        raise ValueError(f"Unknown SSD scan {variant!r}")


def set_chunk_size(model: torch.nn.Module, chunk_size: int) -> None:
    """Set the chunk length of every Mamba-2 mixer's chunked SSD scan."""
    for module in model.modules():
        if hasattr(module, "chunk_size") and hasattr(module, "A_log"):
            module.chunk_size = int(chunk_size)


def mamba_mixers(model: torch.nn.Module) -> list[torch.nn.Module]:
    """The model's Mamba-2 mixers, in layer order."""
    return [
        module
        for module in model.modules()
        if hasattr(module, "A_log") and hasattr(module, "in_proj")
    ]


def install_mamba_regions(model: torch.nn.Module) -> int:
    """Run each Mamba-2 mixer's interior as an Inductor region (``mamba-inductor``).

    Binds :func:`mamba_region_forward` to every mixer of ``model`` (in place)
    and returns how many it patched.
    """
    mixers = mamba_mixers(model)
    for mixer in mixers:
        mixer.forward = types.MethodType(mamba_region_forward, mixer)
    return len(mixers)


def mamba_region_forward(
    self: torch.nn.Module,
    hidden_states: torch.Tensor,
    cache_params: Any = None,
    attention_mask: torch.Tensor | None = None,
    **kwargs: Any,
) -> torch.Tensor:
    """``NemotronHMamba2Mixer.forward`` without a cache, its interior in a region.

    in_proj and out_proj stay outside: the region reads the in_proj output
    and returns the tensor out_proj reads (:func:`mamba_interior`). A call
    with a cache or a padding mask takes Hugging Face's own forward.
    """
    if cache_params is not None or attention_mask is not None or self.training:
        return type(self).forward(
            self,
            hidden_states,
            cache_params=cache_params,
            attention_mask=attention_mask,
            **kwargs,
        )
    projected = self.in_proj(hidden_states)
    with _inductor_scope("mamba_scan"):
        mixed = mamba_interior(self, projected, hidden_states.dtype, **kwargs)
    return self.out_proj(mixed)


def mamba_interior(
    mixer: torch.nn.Module, projected: torch.Tensor, dtype: torch.dtype, **kwargs: Any
) -> torch.Tensor:
    """A Mamba-2 mixer's work between in_proj and out_proj, as Hugging Face does it.

    The split of the in_proj output, the causal depthwise convolution and
    SiLU, the chunked SSD scan (whichever :func:`use_ssd` selected) with its
    D skip, and the gated group RMSNorm, cast to ``dtype`` for out_proj.
    """
    from transformers.models.nemotron_h import modeling_nemotron_h as modeling

    batch, length, _ = projected.shape
    heads, head_dim = mixer.num_heads, mixer.head_dim
    groups, state_size = mixer.n_groups, mixer.ssm_state_size
    A = -torch.exp(mixer.A_log.float())
    gate, xbc, dt = projected.split(
        [mixer.intermediate_size, mixer.conv_dim, heads], dim=-1
    )
    xbc = modeling.causal_conv1d_fn(
        xbc.transpose(1, 2),
        mixer.conv1d.weight.squeeze(1),
        mixer.conv1d.bias,
        activation=mixer.activation,
        **kwargs,
    ).transpose(1, 2)
    x, B, C = torch.split(
        xbc, [mixer.intermediate_size, groups * state_size, groups * state_size], dim=-1
    )
    y = modeling.mamba2_chunk_scan(
        x.view(batch, length, heads, head_dim),
        dt,
        A,
        B.view(batch, length, groups, state_size),
        C.view(batch, length, groups, state_size),
        chunk_size=mixer.chunk_size,
        D=mixer.D,
        z=None,
        return_final_states=False,
        dt_bias=mixer.dt_bias,
        dt_softplus=True,
        initial_states=None,
        dt_limit=mixer.time_step_limit,
        **kwargs,
    )
    y = mixer.norm(y.reshape(batch, length, -1), gate)
    return y.to(dtype)


class NemotronPrefillWorkload(Workload):
    """Batch-1 prefill of a Nemotron-H model: one forward over ``length`` tokens.

    The unit is one prefill (time to first token) of the first ``length``
    tokens of :data:`PREFILL_TEXT`. Logits at ``positions`` evenly spaced
    positions are checked against the FP32 eager model, with the decode
    workload's metrics. The baseline is the straightforward Torch-TensorRT
    flow: :class:`PrefillLogits` in the checkpoint's BF16, exported with
    ``torch.export`` and compiled by ``torch_tensorrt.dynamo.compile`` into
    one engine, the scan's FP32 einsums without TF32 (as Hugging Face runs
    them), chunk length 256, optimization level 3, no CUDA graphs.
    """

    def __init__(
        self,
        model: str = MODEL,
        *,
        length: int = 2048,
        positions: int = 64,
        baseline: Mapping[str, Any] | None = None,
    ) -> None:
        self.model_name = model
        self.length = length
        self.positions = prefill_positions(length, positions)
        self.name = f"{model.rsplit('/', 1)[-1]}-prefill-{length}"
        self._baseline = dict(baseline or {})
        self._tokens: torch.Tensor | None = None

    def knobs(self) -> Sequence[Knob]:
        return (
            Knob(
                "engine.precision",
                ("bf16", "fp16"),
                "bf16",
                "Model weight and activation precision. bf16 is the checkpoint's; "
                "fp16 has more mantissa bits but a narrower range. The SSD scan "
                "runs in FP32 either way (Hugging Face upcasts it).",
                ("precision", "dominant_kernel"),
            ),
            Knob(
                "engine.use_fp32_acc",
                (False, True),
                False,
                "With fp16, accumulate matrix multiplications in FP32 by inserting "
                "casts (bf16 matrix multiplications accumulate in FP32 already).",
                ("precision", "dominant_kernel"),
                {"engine.precision": "fp16"},
            ),
            Knob(
                "engine.tf32",
                (False, True),
                False,
                "Run FP32 matrix multiplications, the chunked SSD scan's einsums "
                "(C.B scores, the intra-chunk output, chunk states, state "
                "passing), on TF32 tensor cores instead of FP32 CUDA cores. "
                "Applies to every backend (torch.backends allow_tf32 for "
                "eager and Inductor).",
                ("dominant_kernel", "precision"),
            ),
            Knob(
                "workload.ssd",
                ("hf", "segsum", "grouped"),
                "hf",
                "Formulation of the chunked SSD scan (exact in real arithmetic; "
                "applies to every backend). hf: Hugging Face's PyTorch fallback, "
                "whose stable segment sums take a cumulative sum over an "
                "expanded, masked [chunk, chunk] tensor per head and chunk, and "
                "which repeats B and C for every head. segsum: segment sums as "
                "differences of one cumulative sum per chunk (as mamba_ssm's "
                "kernels compute them). grouped: segsum plus the C.B scores, "
                "chunk states and state outputs per group of 12 heads, 12x "
                "fewer score FLOPs.",
                ("dominant_kernel",),
            ),
            Knob(
                "workload.chunk_size",
                (256, 128, 64),
                256,
                "Chunk length of Mamba-2's chunked SSD scan (the checkpoint's 256). "
                "Mathematically equivalent: smaller chunks shrink the quadratic "
                "intra-chunk tensors ([heads, chunk, chunk] per chunk; 201 MB "
                "per layer in FP32 at 256) and their FLOPs, at the cost of more "
                "inter-chunk state passing. Applies to every backend.",
                ("dominant_kernel",),
            ),
            Knob(
                "kernels.inductor",
                PREFILL_INDUCTOR,
                "none",
                "Hybrid build: hand operations to Inductor-compiled regions inside "
                "the compiled forward (torch_tensorrt.region.execute_in_torch, "
                "backend inductor). mamba-inductor: each of the 21 Mamba-2 mixers' "
                "interior is one region, from the in_proj output through the split, "
                "the causal depthwise conv1d + SiLU, the chunked SSD scan (as "
                "workload.ssd and workload.chunk_size select it) and the gated "
                "RMSNorm, which also combines the scan's intra- and inter-chunk "
                "outputs, to the tensor out_proj reads. in_proj, out_proj, the "
                "block norms, MLPs, attention and the LM head stay in TensorRT: "
                "22 engines and 42 engine boundaries, each carrying only a tensor "
                "the one-engine build also writes. Regions need strict export, so "
                "the build uses torch_tensorrt.compile(strict=True) and keeps the "
                "model on the GPU while building; each region compiles during the "
                "build in Inductor's max-autotune mode, with engine.tf32 deciding "
                "TF32 for its FP32 einsums. mamba-inductor runs the 43 engine and "
                "region calls from the host, each boundary a host-side call and a "
                "GPU gap; mamba-inductor-graph builds the same program and records "
                "the whole forward as one CUDA graph "
                "(torch_tensorrt.runtime.enable_cudagraphs), so a boundary costs a "
                "gap between kernels, at the price of an input copy and an output "
                "copy per call. Costs: longer builds (Inductor autotuning), GPU "
                "memory for 22 engines' activations, and the scan runs on "
                "Inductor's kernels instead of TensorRT's. These builds cannot "
                "be saved: torch_tensorrt.save refuses execute_in_torch regions "
                "whose backend is 'inductor', so a session that keeps one writes "
                "its config but no artifact.",
                ("dominant_kernel", "launch_overhead"),
            ),
        )

    def baseline(self) -> dict[str, Any]:
        return dict(self._baseline)

    def timing(self) -> Timing:
        return Timing(warmup=3, rounds=10, calls=1, profile_calls=1)

    def accuracy_defaults(self) -> dict[str, tuple[str, float]]:
        # The prompt is new prose: the FP32 model is unsure at most checked
        # positions (median top-1 probability 0.43; 13 of 64 positions have
        # their top two tokens within 0.1 nats), so correct engines flip one
        # to five top-1 tokens of 64 (BF16 eager 0.922; even FP16 with FP32
        # accumulation, mean KL 8e-6, flips one), at a mean KL of at most
        # 5.5e-4. Mean KL is the decisive check: SSD scan errors measured
        # 0.31 (no state passing between chunks) to 8.2; top-1 only catches
        # gross errors.
        return {"top1_agreement": (">=", 0.85), "mean_kl": ("<=", 2e-3)}

    def tokens(self) -> torch.Tensor:
        if self._tokens is None:
            self._tokens = text_tokens(self.model_name, self.length, PREFILL_TEXT)
        return self._tokens

    def reference(self) -> Any:
        """FP32 eager logits at the checked positions (TF32 off), with their inputs."""
        device = torch.device("cuda:0")
        model = _load(self.model_name, torch.float32, device)
        tokens = self.tokens()
        try:
            with torch.inference_mode():
                module = PrefillLogits(model, self.positions.to(device))
                logits = module(tokens.to(device).view(1, -1)).cpu()
            return {
                "logits": logits,
                "tokens": tokens.clone(),
                "positions": self.positions.clone(),
            }
        finally:
            del model
            gc.collect()
            torch.cuda.empty_cache()

    def compare(self, outputs: Any, reference: Any) -> dict[str, float]:
        expected = reference_logits(
            reference, tokens=self.tokens(), positions=self.positions
        )
        return compare_logits(outputs, expected)

    def build(self, values: Mapping[str, Any]) -> Candidate:
        import torch_tensorrt

        backend = values.get("bench.backend", "tensorrt")
        if backend not in ("tensorrt", "eager", "inductor"):
            raise ValueError(f"Unknown bench.backend {backend!r}")
        precision = values.get("engine.precision", "bf16")
        tf32 = bool(values.get("engine.tf32", False))
        attn = values.get("bench.attn") or "sdpa"
        mode = values.get("bench.mode") or "default"
        fp16_accumulation = bool(values.get("bench.fp16_accumulation", False))
        if fp16_accumulation and (backend == "tensorrt" or precision != "fp16"):
            raise ValueError(
                "bench.fp16_accumulation applies to the eager and inductor "
                "backends with engine.precision='fp16'"
            )
        inductor = values.get("kernels.inductor", "none")
        if inductor not in PREFILL_INDUCTOR:
            raise ValueError(
                f"Unknown kernels.inductor {inductor!r}; choose from {PREFILL_INDUCTOR}"
            )
        hybrid = inductor != "none"
        if hybrid and backend != "tensorrt":
            raise ValueError(
                "kernels.inductor builds hybrid TensorRT programs; "
                f"bench.backend {backend!r} cannot honor it"
            )
        device = torch.device("cuda:0")
        if backend != "tensorrt":
            import torch._inductor.config as inductor_config

            torch.backends.cuda.matmul.allow_tf32 = tf32
            _set_cudnn_tf32(tf32)
            # TensorRT's plain fp16 GEMM tactics accumulate in FP16 (twice the
            # FP32-accumulation rate on GeForce GPUs); this lets cuBLAS do so.
            torch.backends.cuda.matmul.allow_fp16_accumulation = fp16_accumulation
            for key, value in dict(values.get("bench.inductor") or {}).items():
                setattr(inductor_config, key, value)
            if attn == "flex_attention":
                fix_flex_attention_masks()
        use_ssd(values.get("workload.ssd", "hf"))
        model = _load(self.model_name, _DTYPES[precision], device, attn=attn)
        set_chunk_size(model, values.get("workload.chunk_size", 256))
        module = PrefillLogits(model, self.positions.to(device)).eval()
        ids = self.tokens().to(device).view(1, -1)
        details: dict[str, Any] = {
            "precision": precision,
            "tf32": tf32,
            "chunk_size": values.get("workload.chunk_size", 256),
            "ssd": values.get("workload.ssd", "hf"),
        }
        if backend == "tensorrt" and hybrid:
            candidate = self._build_hybrid(values, model, module, ids, details)
        elif backend == "tensorrt":
            with torch.inference_mode():
                program = torch.export.export(module, (ids,), strict=False)
                compiled = torch_tensorrt.dynamo.compile(
                    program,
                    arg_inputs=[ids],
                    use_fp32_acc=precision == "fp16" and values["engine.use_fp32_acc"],
                    disable_tf32=not tf32,
                    device=device,
                    offload_module_to_cpu=True,
                    min_block_size=1,
                    optimization_level=3,
                )
            del program, module, model
            gc.collect()
            torch.cuda.empty_cache()
            candidate = _PrefillCandidate(compiled, ids, backend=backend)
        else:
            run = (
                module
                if backend == "eager"
                else torch.compile(module, mode=mode, fullgraph=True)
            )
            candidate = _PrefillCandidate(
                run,
                ids,
                backend=backend,
                cudagraphs=backend == "inductor" and mode in _CUDAGRAPH_MODES,
            )
            details.update(
                backend=backend,
                mode=mode,
                attn=attn,
                fp16_accumulation=fp16_accumulation,
            )
            # Compile (and record CUDA graphs) here, so build_s includes it.
            start = time.perf_counter()
            for _ in range(3):
                candidate()
            torch.cuda.synchronize()
            details["compile_s"] = time.perf_counter() - start
        candidate.details = details
        return candidate

    def _build_hybrid(
        self,
        values: Mapping[str, Any],
        model: torch.nn.Module,
        module: torch.nn.Module,
        ids: torch.Tensor,
        details: dict[str, Any],
    ) -> "_PrefillCandidate":
        """The TensorRT build with every Mamba-2 mixer's interior an Inductor region."""
        import torch_tensorrt

        precision = values.get("engine.precision", "bf16")
        tf32 = bool(values.get("engine.tf32", False))
        details["inductor"] = values["kernels.inductor"]
        details["mamba_regions"] = install_mamba_regions(model)
        # Inductor's options and the TF32 flags stay set while the candidate
        # lives (the regions compile on their first call and read the flags
        # when they run), restored on close().
        stack = region_compilation(tf32)
        try:
            with torch.no_grad():
                # Only strict export captures regions, and regions refuse
                # offloading the model to the CPU.
                compiled = torch_tensorrt.compile(
                    module,
                    ir="dynamo",
                    arg_inputs=[ids],
                    strict=True,
                    use_fp32_acc=precision == "fp16" and values["engine.use_fp32_acc"],
                    disable_tf32=not tf32,
                    device=ids.device,
                    offload_module_to_cpu=False,
                    min_block_size=1,
                    optimization_level=3,
                )
            details["inductor_regions"] = inductor_regions(compiled)
            details["released_mb"] = release_unused_weights(model, compiled) / 2**20
        except BaseException:
            stack.close()
            raise
        whole_graph = values["kernels.inductor"] == "mamba-inductor-graph"
        if whole_graph:
            details["cudagraph"] = "whole forward"
        candidate = _PrefillCandidate(
            compiled, ids, backend="tensorrt", stack=stack, whole_graph=whole_graph
        )
        # Compile (and autotune) the regions here, so build_s includes it.
        start = time.perf_counter()
        try:
            candidate()
            torch.cuda.synchronize()
        except BaseException:
            candidate.close()
            raise
        details["compile_s"] = time.perf_counter() - start
        details["inductor_kernels"] = inductor_kernels()
        return candidate


class _PrefillCandidate(Candidate):
    """One prefill per call: the compiled engine, or the eager or compiled model.

    ``cudagraphs`` marks an Inductor module that replays CUDA graphs.
    ``whole_graph`` records a compiled program with regions as one CUDA graph
    (``torch_tensorrt.runtime.enable_cudagraphs``, entered in :meth:`setup`).
    ``stack`` holds global settings the module runs under (the Inductor
    regions' options and TF32 flags), restored on close.
    """

    def __init__(
        self,
        module: Any,
        input_ids: torch.Tensor,
        *,
        backend: str,
        cudagraphs: bool = False,
        stack: contextlib.ExitStack | None = None,
        whole_graph: bool = False,
    ) -> None:
        self.module, self.input_ids = module, input_ids
        self.backend, self.cudagraphs = backend, cudagraphs
        self.whole_graph = whole_graph
        self.units, self.items = 1, 1
        self.details: dict[str, Any] = {}
        self._stack = stack or contextlib.ExitStack()
        self._run: Any = module

    def setup(self) -> None:
        if self.whole_graph:
            import torch_tensorrt

            # The first call warms the regions up and records the graph.
            self._run = self._stack.enter_context(
                torch_tensorrt.runtime.enable_cudagraphs(self.module)
            )

    def __call__(self) -> torch.Tensor:
        with torch.inference_mode():
            if self.backend == "inductor" and self.cudagraphs:
                torch.compiler.cudagraph_mark_step_begin()
            out = self._run(self.input_ids)
            if isinstance(out, (tuple, list)):
                out = out[0]
            return out.to(torch.float32, copy=True)

    def describe(self) -> dict[str, Any]:
        if self.backend != "tensorrt":
            return {
                "trt_engines": 0,
                "torch_partitions": 0,
                "plugin_regions": 0,
                **self.details,
            }
        from torch_tensorrt.autotune import describe_module

        return {**describe_module(self.module), **self.details}

    def export(self, directory: Path) -> list[str]:
        if self.backend != "tensorrt":
            return []
        return save_program(
            self.module, Path(directory) / "prefill.ep", self.details.get("inductor")
        )

    def close(self) -> None:
        self._stack.close()
        self._run = self.module


# ------------------------------------------------------------------ helpers


def text_tokens(model_name: str, length: int, text: str) -> torch.Tensor:
    """The first ``length`` tokens of ``text``, repeated if it is shorter.

    The text is tokenized without special tokens; the tokenizer's BOS token,
    if it has one, starts the sequence.
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
    bos = [tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else []
    repeats = -(-length // ids.numel())
    sequence = torch.cat([torch.tensor(bos, dtype=ids.dtype), ids.repeat(repeats)])
    return sequence[:length].clone()


def reference_logits(reference: Any, **inputs: torch.Tensor) -> torch.Tensor:
    """The reference's logits, once its recorded inputs match ``inputs``.

    A reference computed for other tokens (an edited prompt text, another
    tokenizer) would grade every candidate against the wrong logits.
    """
    if not isinstance(reference, Mapping) or "logits" not in reference:
        raise ValueError(
            "The reference records no inputs (computed by an older version of "
            "this workload); recompute it"
        )
    for name, expected in inputs.items():
        recorded = reference.get(name)
        if recorded is None or not torch.equal(recorded.cpu(), expected.cpu()):
            raise ValueError(
                f"The reference was computed for other {name} than this workload's; "
                "recompute it"
            )
    return reference["logits"]


def compare_logits(outputs: Any, reference: Any) -> dict[str, float]:
    """Top-1 agreement, KL divergence and logit error of ``[positions, vocab]`` logits."""
    actual = outputs.detach().double().cpu()
    expected = reference.double()
    if actual.shape != expected.shape:
        raise ValueError(
            f"Logits shape {tuple(actual.shape)} != {tuple(expected.shape)}"
        )
    kl = F.kl_div(
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


def _load(
    name: str, dtype: torch.dtype, device: torch.device, *, attn: str = "sdpa"
) -> torch.nn.Module:
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        name, dtype=dtype, attn_implementation=attn
    )
    return model.eval().to(device)


def _bytes_per_unit(
    model: torch.nn.Module, max_seq_len: int, ssm_dtype: torch.dtype
) -> float | None:
    try:
        weights = sum(p.numel() * p.element_size() for p in model.parameters())
        embedding = model.get_input_embeddings().weight
        return decode_bytes_per_token(
            model.config,
            weight_bytes=weights,
            embedding_bytes=embedding.numel() * embedding.element_size(),
            max_seq_len=max_seq_len,
            dtype_size=embedding.element_size(),
            ssm_dtype_size=torch.empty((), dtype=ssm_dtype).element_size(),
        )
    except Exception:  # noqa: BLE001 - a diagnostic must not fail the build
        return None


def _hf_decode_logits(
    model: torch.nn.Module, tokens: torch.Tensor, isl: int, osl: int
) -> torch.Tensor:
    """Teacher-forced decode logits ``[osl, vocab]`` from Hugging Face's own cache."""
    cache = prefill(model, tokens[:isl].view(1, -1), isl + osl)
    logits = []
    with torch.inference_mode():
        for j in range(osl):
            position = torch.full(
                (1, 1), isl + j, dtype=torch.int64, device=tokens.device
            )
            out = model(
                input_ids=tokens[isl + j].view(1, 1),
                position_ids=position,
                past_key_values=cache,
                use_cache=True,
            )
            logits.append(out.logits[0, -1].float())
    return torch.stack(logits)


def fix_flex_attention_masks() -> None:
    """Make transformers' FlexAttention masks correct and traceable in this process.

    Two changes to ``transformers.masking_utils`` (5.18):

    * Under a ``StaticCache``, a decode mask's query offset is the cache
      layer's ``cumulative_length`` tensor itself, which the mask function
      captures by reference. The first attention layer's cache update then
      advances it in place before attention runs, so every decode query also
      attends to the next, unwritten cache slot (SDPA and eager masks are
      materialized before the update and are unaffected). The offsets are
      copied when the mask function is built.
    * ``create_block_mask(_compile=True)``: this torch nightly deprecates the
      flag with a warning that Dynamo cannot trace; the decode step is
      compiled already.
    """
    import transformers.masking_utils as masking_utils
    from torch.nn.attention.flex_attention import create_block_mask

    if getattr(masking_utils, "_nemotron_masks_fixed", False):
        return
    add_offsets = masking_utils.add_offsets_to_mask_function

    def add_offsets_by_value(mask_function: Any, q_offset: Any, kv_offset: Any) -> Any:
        if isinstance(q_offset, torch.Tensor):
            q_offset = q_offset.clone()
        if isinstance(kv_offset, torch.Tensor):
            kv_offset = kv_offset.clone()
        return add_offsets(mask_function, q_offset, kv_offset)

    def block_mask(*args: Any, _compile: bool = False, **kwargs: Any) -> Any:
        return create_block_mask(*args, **kwargs)

    masking_utils.add_offsets_to_mask_function = add_offsets_by_value
    masking_utils.create_block_mask = block_mask
    masking_utils._nemotron_masks_fixed = True


class _DecodeCandidate(Candidate):
    """Shared teacher-forced protocol: ``osl`` steps from the prefilled state."""

    def __init__(self, *, tokens: torch.Tensor, isl: int, osl: int) -> None:
        self.isl, self.osl = isl, osl
        self.units, self.items = osl, 1
        device = tokens.device
        self.step_tokens = [tokens[isl + j].view(1, 1) for j in range(osl)]
        self.step_positions = [
            torch.full((1, 1), isl + j, dtype=torch.int64, device=device)
            for j in range(osl)
        ]
        self.details: dict[str, Any] = {}


class _TensorRTCandidate(_DecodeCandidate):
    """The compiled decode step, run ``osl`` times per call.

    With ``cudagraphs``, each engine replays its own CUDA graph (the runtime's
    per-engine mode), or, with ``whole_graph`` (Inductor regions between the
    engines), :meth:`setup` captures the whole step as one CUDA graph over
    static token and position buffers and the in-place cache and state
    tensors, and each step replays it. ``stack`` holds global settings the
    module runs under (the regions' Inductor options), restored on close.
    """

    def __init__(
        self,
        module: torch.nn.Module,
        *,
        tokens: torch.Tensor,
        isl: int,
        osl: int,
        start: list[torch.Tensor],
        n_kv: int,
        inplace: bool,
        cudagraphs: bool,
        whole_graph: bool = False,
        stack: contextlib.ExitStack | None = None,
    ) -> None:
        super().__init__(tokens=tokens, isl=isl, osl=osl)
        if whole_graph and not inplace:
            raise ValueError("Whole-step CUDA graphs need in-place state updates")
        self.module = module
        self.n_kv = n_kv
        self.inplace = inplace
        self.cudagraphs = cudagraphs
        self.whole_graph = whole_graph
        # The KV caches are written in place, only at decode positions and
        # identically on every call, so every call starts from these tensors.
        self.kv = start[:n_kv]
        # In-place states are overwritten every step: each call restores them
        # from a saved copy into the same buffers (a captured CUDA graph binds
        # their addresses). Functional states are only read.
        self.states = start[n_kv:]
        self.saved = [state.clone() for state in self.states] if inplace else []
        self._stack = stack or contextlib.ExitStack()
        self._graph: Any = None
        self._static: tuple[torch.Tensor, ...] = ()

    def setup(self) -> None:
        import torch_tensorrt

        if not self.cudagraphs:
            return
        if self.whole_graph:
            self._capture()
            self._stack.callback(self._release_graph)
        else:
            torch_tensorrt.runtime.set_cudagraphs_mode(True)
            self._stack.callback(torch_tensorrt.runtime.set_cudagraphs_mode, False)

    def warm_up(self, calls: int = 2) -> float:
        """Run the first decode step eagerly ``calls`` times; returns the seconds.

        The first call compiles (and autotunes) the Inductor regions.
        """
        start = time.perf_counter()
        with torch.inference_mode():
            for _ in range(calls):
                self._restore()
                self.module(
                    self.step_tokens[0], self.step_positions[0], *self.kv, *self.states
                )
        torch.cuda.synchronize()
        return time.perf_counter() - start

    def _restore(self) -> None:
        for state, saved in zip(self.states, self.saved):
            state.copy_(saved)

    def _capture(self) -> None:
        """Record one decode step, engines and regions, as one CUDA graph."""
        ids = self.step_tokens[0].clone()
        position = self.step_positions[0].clone()
        with torch.inference_mode():
            self._restore()
            # Warm up on a side stream, as CUDA graph capture expects.
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                self.module(ids, position, *self.kv, *self.states)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                out = self.module(ids, position, *self.kv, *self.states)
        torch.cuda.synchronize()
        self._graph, self._static = graph, (ids, position, out[0])

    def _release_graph(self) -> None:
        if self._graph is not None:
            self._graph.reset()
        self._graph, self._static = None, ()

    def __call__(self) -> torch.Tensor:
        with torch.inference_mode():
            if self._graph is not None:
                return self._replay()
            states = self.states
            if self.inplace:
                for state, saved in zip(states, self.saved):
                    state.copy_(saved)
            logits = []
            for j in range(self.osl):
                out = self.module(
                    self.step_tokens[j], self.step_positions[j], *self.kv, *states
                )
                logits.append(out[0][0].to(torch.float32, copy=True))
                if not self.inplace:
                    states = out[1 + self.n_kv :]
            return torch.stack(logits)

    def _replay(self) -> torch.Tensor:
        ids, position, out = self._static
        self._restore()
        logits = []
        for j in range(self.osl):
            ids.copy_(self.step_tokens[j])
            position.copy_(self.step_positions[j])
            self._graph.replay()
            logits.append(out[0].to(torch.float32, copy=True))
        return torch.stack(logits)

    def describe(self) -> dict[str, Any]:
        from torch_tensorrt.autotune import describe_module

        return {**describe_module(self.module), **self.details}

    def export(self, directory: Path) -> list[str]:
        return save_program(
            self.module, Path(directory) / "decode.ep", self.details.get("inductor")
        )

    def close(self) -> None:
        self._stack.close()


class _TorchCandidate(_DecodeCandidate):
    """Hugging Face's decode step and hybrid StaticCache, eager or under torch.compile."""

    def __init__(
        self,
        step: Any,
        cache: Any,
        *,
        kinds: Sequence[str],
        tokens: torch.Tensor,
        isl: int,
        osl: int,
        cudagraphs: bool,
    ) -> None:
        super().__init__(tokens=tokens, isl=isl, osl=osl)
        self.step, self.cache = step, cache
        self.cudagraphs = cudagraphs
        self.attention = [
            cache.layers[i] for i, kind in enumerate(kinds) if kind == "full_attention"
        ]
        self.mamba = [
            cache.layers[i]
            for i, kind in enumerate(kinds)
            if kind == "linear_attention"
        ]
        # The prefilled states; every call rewinds the cache to them in place
        # (the cache's tensors have static addresses for CUDA graphs). Cache
        # slots after the rewound position keep the previous call's keys and
        # values: a correct mask never lets a query see them.
        self.saved = [
            (layer.conv_states[0].clone(), layer.recurrent_states[0].clone())
            for layer in self.mamba
        ]

    def __call__(self) -> torch.Tensor:
        with torch.inference_mode():
            for layer in self.attention:
                layer.cumulative_length.fill_(self.isl)
            for layer, (conv, ssm) in zip(self.mamba, self.saved):
                layer.conv_states[0].copy_(conv)
                layer.recurrent_states[0].copy_(ssm)
            logits = []
            for j in range(self.osl):
                if self.cudagraphs:
                    torch.compiler.cudagraph_mark_step_begin()
                out = self.step(self.step_tokens[j], self.step_positions[j])
                logits.append(out[0, -1].to(torch.float32, copy=True))
            return torch.stack(logits)

    def describe(self) -> dict[str, Any]:
        return {
            "trt_engines": 0,
            "torch_partitions": 0,
            "plugin_regions": 0,
            **self.details,
        }

    def export(self, directory: Path) -> list[str]:
        return []
