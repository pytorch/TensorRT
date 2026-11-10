# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Hugging Face causal-LM decode workload for ``torch_tensorrt.autotune`` sessions.

The measured unit is one decoded token at batch 1. A candidate prefills a fixed
prompt once (untimed) and is then timed decoding ``osl`` tokens. Decoding is
teacher-forced on a fixed text, so one different token cannot derail the rest:
every step's logits are compared with an FP32 eager reference at the same
position (top-1 agreement, KL divergence and the largest logit error).

Token layouts (``prompt``):

* ``text``: the opening of "A Tale of Two Cities" (:data:`TEXT`), repeated if
  it is shorter than ``isl + osl`` tokens.
* ``recall``: for long contexts. Repeating one paragraph makes the check
  nearly blind: every decode token can be copied from the previous repeat a
  few hundred tokens back, so attention that silently drops the distant
  context still passes. Here the prompt starts with :data:`PASSAGE` (written
  for this workload, so no model has memorized it), continues with
  :data:`TEXT` repeated, and ends with the passage's first
  :data:`RECALL_CUE` tokens. Decode continues the passage for
  :data:`RECALL_DECODE` tokens, which needs attention across the whole
  context (at 8k tokens a 4096-token attention window, a dropped prefix or
  wrapped positions all fail the default limits), and then goes on with
  :data:`CODA`, text that appears nowhere in the prompt. Copied tokens are
  predicted almost surely, so errors elsewhere in the step barely move their
  logits; the unseen text keeps a natural uncertainty, so that, as at 128
  tokens, a skipped MLP, GEMV rows left out or a skipped norm fail the
  limits.
* ``auto`` (default): ``text`` where :data:`TEXT` covers every token without
  repeating (the 128-token workload), otherwise ``recall``.

The reference computes logits only at the decode positions, so long contexts
do not materialize the logits of the whole prompt.

Prefill for ``static_v3`` (``prefill``): ``engine`` runs the prompt one token
at a time through the decode engine itself (about 16 s for 8k tokens);
``eager`` runs one Hugging Face prefill in the build (same precision, SDPA
attention) and copies its keys and values into the engine's cache, the state
the engine's own prefill leaves up to rounding. ``auto`` (default) uses
``engine`` up to 1024 prompt tokens and ``eager`` above. Prefill is untimed.

Workload knobs:

* ``workload.cache``: the KV cache lowering. ``static_v2`` and ``static_v1``
  (see ``static_cache_v2.py``) share one engine between prefill and decode, but
  slice the cache to the current length, which costs a host synchronization
  per layer and step. ``static_v3`` (``static_cache_v3.py``) compiles a
  shape-static decode engine instead, which has no host synchronizations and
  can be captured in a CUDA graph (prefill: see above). Caches longer than
  192 slots attend through IAttention ``key_value_lengths`` instead of a
  position mask, which TensorRT 11.4's long-cache decode kernel ignores, and
  their unwritten slots are filled with noise before the first call, so an
  engine that reads past the written slots fails the first accuracy check.
* ``engine.precision`` and ``engine.use_fp32_acc``: model precision, and FP32
  accumulation for FP16 matrix multiplications.
* ``kernels.decode_blocks`` (Qwen3 models): whole decoder blocks as fused
  Triton kernels inside the TensorRT engine (``qwen3_regions.py``, through
  ``torch_tensorrt.region.fused_kernel``), with the static_v3 cache and CUDA
  graphs. Two editable files let an LLM planner edit the blocks' launch
  settings (``qwen3_regions.py``) and kernels (``qwen3_kernels.py``) inside
  their ``EDITABLE`` fences. ``generated`` fuses the ``layers`` blocks with
  kernels that PR 4914's ``KernelGenerator`` writes (``region_generation.py``)
  instead of the hand-written ones, and needs the same settings. It reuses
  the kernels cached in the ``kernel_cache`` directory
  (``region_generation.kernel_cache``) and generates a block's kernel during
  the build when the cache has none for it.
* ``runtime.step_graph``: capture the whole static_v3 decode step (every
  engine, plugin and PyTorch region) in one CUDA graph that binds the caches
  in place, instead of one graph per engine (``runtime.cudagraphs`` alone).
* ``kernels.inductor`` (Qwen3 models): hybrid variants that hand operations
  to Inductor's kernels (``qwen3_regions.py``): ``attention-flex`` hosts
  Inductor's generated flex-decoding kernels inside the engine; ``lm-head``,
  ``attention`` (flex-decoding) and ``attention-math`` (plain attention math)
  run as ``execute_in_torch(backend="inductor")`` regions, compiled in the
  build. The knob's requirements are per value: every variant needs
  static_v3, FP16 and CUDA graphs, and the region variants also the step
  graph. Region builds cannot be saved (``torch_tensorrt.save`` refuses
  Inductor regions), so their export raises ``NotImplementedError``.

plus the standard builder and runtime knobs. Every worker builds exactly one
candidate, so the process-global lowering passes of the cache variants never
mix.

Head-to-head benchmarks (``bench_compare.py``) reuse the workload through
``bench.*`` values, which are not knobs (a session never sets them):
``bench.backend`` ``"tensorrt"``
(default) builds the config as above; ``"eager"`` and ``"inductor"`` decode
with Hugging Face's ``StaticCache`` (the cache behind
``generate(cache_implementation="static")``), prefilled eagerly as
``generate`` does, with the decode step run eagerly or compiled by
``torch.compile(mode=bench.mode, fullgraph=True)`` (``bench.mode``: a
torch.compile mode, default ``"default"``). ``bench.attn`` is the Hugging Face
``attn_implementation`` (default ``"sdpa"``; ``"flex_attention"`` decodes with
FlexAttention's flex-decoding kernel) and ``bench.inductor`` a dict of
``torch._inductor.config`` overrides, such as
``{"coordinate_descent_tuning": True}`` (nested options by their dotted name,
such as ``"triton.cudagraph_trees"``). They use the same tokens, precision
(``engine.precision``), outputs and reference as the TensorRT candidates;
compilation and CUDA-graph recording happen in ``build``.
"""

from __future__ import annotations

import contextlib
import gc
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch_tensorrt.autotune import (
    STANDARD_KNOBS,
    Candidate,
    EditableFile,
    Knob,
    Timing,
    Workload,
    load_module,
)

# Opening of "A Tale of Two Cities" (Charles Dickens, 1859; public domain).
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
# A paragraph written for this workload (no model has memorized it); the
# "recall" layout's decode tokens continue it from the start of the prompt.
PASSAGE = (
    "The keeper of the lighthouse at Marrow Point wrote his log in green ink, "
    "because the supply boat had once brought nothing else. Every evening he "
    "noted the wind, the colour of the water near the northern rocks, the number "
    "of gulls on the rail, and whether the lamp had smoked. On the ninth of March "
    "he wrote that a crate of oranges had washed ashore unopened, that he had "
    "shared them with the cook of a passing trawler, and that the trawler's dog "
    "had refused every piece. Two pages later the ink turns blue, and the hand "
    "is not his. "
)
# Written for this workload too, and in no prompt: after RECALL_DECODE
# recalled passage tokens, "recall" decode continues with this unseen text.
# Worded so that no decode position of the 8k workload has two nearly tied
# best tokens (FP32 top-2 logit gaps of 0.15 and more), which correct FP16
# or BF16 rounding would flip.
CODA = (
    " and, on clear nights, the planets he could name without any chart. He "
    "kept the book in a tin box under his bed, and when the inspector came "
    "each autumn he read it aloud to him, slowly, as if it were a letter from "
    "someone far away. "
)
PROMPTS = ("auto", "text", "recall")
PREFILLS = ("auto", "engine", "eager")
# Passage tokens that end a "recall" prompt, so decode continues the passage.
RECALL_CUE = 16
# Passage tokens that "recall" decode copies from the start of the prompt
# (up to "the northern rocks,") before it continues with CODA.
RECALL_DECODE = 28
# Longest prompt that "auto" prefills one token at a time through the engine.
ENGINE_PREFILL_MAX = 1024
# Largest static_v3 cache that attends through a position mask: the 192 slots
# of the published 128-token results, where TensorRT 11.4 uses a kernel that
# honors it. Longer caches use IAttention key_value_lengths, because the
# long-cache decode kernel ignores masks (see static_cache_v3.py).
MASK_ATTENTION_MAX = 192

_DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
BACKENDS = ("tensorrt", "eager", "inductor")
# torch.compile modes that replay the decode step from CUDA graphs.
_CUDAGRAPH_MODES = ("reduce-overhead", "max-autotune")
# One-knob TensorRT build routes aimed at decode: GEMV generation and fusion,
# accumulation, code generation and the decode attention kernel. They need
# TensorRT with Global Performance Tuning (11.1+, not TensorRT-RTX); the knob
# is offered only where it is available.
BUILD_ROUTES = (
    "",
    "-kgen:lower_fc:all=on",
    "-cask_fusion:fp16_acc=off",
    "-kgen:codegen:cuda_tile=2",
    "-peep:match_gemv_mha=on",
    "-mha_fusion:gemv_mha_v2=off",
    "-tactic:splitk=off",
    "-late_h_fusion=off",
)
_STANDARD = (
    "runtime.cudagraphs",
    "runtime.pre_allocated_outputs",
    "engine.optimization_level",
    "engine.tiling_optimization_level",
)
# The kernels.decode_blocks value whose blocks run kernels that PR 4914's
# KernelGenerator writes instead of the hand-written ones, the preset of
# blocks it fuses, and the notes file of measured facts its generator reads
# (part of the kernel cache key).
GENERATED = "generated"
GENERATED_PRESET = "layers"
GENERATION_NOTES = "qwen3_decode_kernel_notes.md"


def create_case(
    model: str = "Qwen/Qwen3-0.6B",
    isl: int = 128,
    osl: int = 64,
    baseline: Mapping[str, Any] | None = None,
    prompt: str = "auto",
    prefill: str = "auto",
    kernel_cache: str | None = None,
) -> "DecodeWorkload":
    """Workload factory: ``WorkloadFactory("tools/llm/agentic_decode.py:create_case", {...})``."""
    return DecodeWorkload(
        model,
        isl=isl,
        osl=osl,
        baseline=baseline,
        prompt=prompt,
        prefill=prefill,
        kernel_cache=kernel_cache,
    )


class DecodeWorkload(Workload):
    """Batch-1 greedy decode of a Hugging Face causal LM with a static KV cache.

    The baseline is the documented ``tools/llm`` configuration: FP16 weights,
    ``static_v2`` cache, Python runtime, no CUDA graphs. ``prompt`` selects the
    token layout and ``prefill`` how ``static_v3`` candidates fill their cache
    (see the module docstring). ``kernel_cache`` is the KernelGenerator cache
    that ``kernels.decode_blocks="generated"`` takes its kernels from
    (``region_generation.kernel_cache``).
    """

    def __init__(
        self,
        model: str,
        *,
        isl: int,
        osl: int,
        baseline: Mapping[str, Any] | None = None,
        prompt: str = "auto",
        prefill: str = "auto",
        kernel_cache: str | None = None,
    ) -> None:
        if isl < 1 or osl < 2:
            raise ValueError("DecodeWorkload needs isl >= 1 and osl >= 2")
        if prompt not in PROMPTS:
            raise ValueError(f"prompt must be one of {PROMPTS}; got {prompt!r}")
        if prefill not in PREFILLS:
            raise ValueError(f"prefill must be one of {PREFILLS}; got {prefill!r}")
        self.model_name = model
        self.name = f"{model.rsplit('/', 1)[-1]}-decode-isl{isl}-osl{osl}"
        self.isl, self.osl = isl, osl
        self.max_seq = isl + osl
        self.prompt = prompt
        self.prefill = prefill
        self._baseline = dict(baseline or {})
        import region_generation

        self.kernel_cache = region_generation.kernel_cache(kernel_cache)
        self._tokens: torch.Tensor | None = None
        self._config: Any = None

    @property
    def eager_prefill(self) -> bool:
        """Whether static_v3 candidates copy the prompt cache from an eager prefill."""
        if self.prefill == "auto":
            return self.isl > ENGINE_PREFILL_MAX
        return self.prefill == "eager"

    @property
    def v3_attention(self) -> str:
        """How static_v3 engines limit attention to the written cache slots."""
        return "mask" if self.max_seq <= MASK_ATTENTION_MAX else "lengths"

    def _fused_shapes(self) -> dict[str, int] | None:
        """GEMV shapes of the fused decoder blocks, or None if they cannot run.

        The kernels are written for Qwen3 and keep a whole hidden row (and a
        head) in registers, so hidden_size and head_dim must be powers of two.
        """
        if self._config is None:
            from transformers import AutoConfig

            self._config = AutoConfig.from_pretrained(self.model_name)
        config = self._config
        if config.model_type != "qwen3":
            return None
        head_dim = getattr(config, "head_dim", None) or (
            config.hidden_size // config.num_attention_heads
        )
        if not all(_power_of_two(n) for n in (config.hidden_size, head_dim)):
            return None
        q, kv = (
            config.num_attention_heads * head_dim,
            config.num_key_value_heads * head_dim,
        )
        return {
            "hidden": config.hidden_size,
            "q": q,
            "kv": kv,
            "intermediate": config.intermediate_size,
            "vocab": config.vocab_size,
        }

    # ------------------------------------------------------------ contract

    def knobs(self) -> Sequence[Knob]:
        cache_description = (
            "KV cache lowering. static_v1/static_v2 share one dynamic-shape engine "
            "for prefill and decode and slice the cache to the current length, "
            "which forces a device-to-host synchronization per layer and step. "
            "static_v3 compiles a shape-static decode engine: in-place cache "
            "writes, masked full-length attention, no host synchronizations, "
            "CUDA-graph capturable."
        )
        if self.v3_attention == "lengths":
            cache_description += (
                f" At this length ({self.max_seq} slots) static_v3 limits attention "
                "with TensorRT IAttention key_value_lengths (position + 1) instead "
                "of a mask: TensorRT 11.4's long-cache decode kernel (gemv_mha_v2) "
                "ignores attention masks. static_v2 also copies the whole cache "
                "every step and runs the one-token attention with a GEMM kernel at "
                "this length."
            )
        return (
            Knob(
                "workload.cache",
                ("static_v2", "static_v3", "static_v1"),
                "static_v2",
                cache_description,
                ("host_syncs", "launch_overhead"),
            ),
            Knob(
                "engine.precision",
                ("fp16", "bf16", "fp32"),
                "fp16",
                "Model weight and activation precision (fp32 runs without TF32).",
                ("precision", "dominant_kernel"),
            ),
            Knob(
                "engine.use_fp32_acc",
                (True, False),
                True,
                "With fp16, accumulate matrix multiplications in FP32 by inserting "
                "casts; turning it off is faster and less accurate.",
                ("dominant_kernel",),
            ),
            *self._kernel_knobs(),
            *self._build_route_knobs(self.v3_attention == "lengths"),
            *(knob for knob in STANDARD_KNOBS if knob.name in _STANDARD),
            Knob(
                "runtime.step_graph",
                (False, True),
                False,
                "Capture the whole static_v3 decode step in one CUDA graph: every "
                "TensorRT engine, plugin and PyTorch region of the compiled module, "
                "with the KV caches bound in place (no copies) and the token and "
                "position copied into static buffers before each replay; an "
                "engine prefill runs through replays of the same graph. Without "
                "it, runtime.cudagraphs captures each TensorRT engine on its own and "
                "PyTorch regions between engines run eagerly. Requires the "
                "static_v3 cache and runtime.cudagraphs.",
                ("launch_overhead",),
                {"workload.cache": "static_v3", "runtime.cudagraphs": True},
            ),
        )

    def _kernel_knobs(self) -> tuple[Knob, ...]:
        import region_generation

        if self._fused_shapes() is None:
            return ()
        import qwen3_regions

        hybrid = {
            "workload.cache": "static_v3",
            "engine.precision": "fp16",
            "runtime.cudagraphs": True,
        }
        return (
            Knob(
                "kernels.decode_blocks",
                (*qwen3_regions.PRESETS, GENERATED),
                "",
                "Whole decoder blocks as fused Triton kernels inside the "
                "TensorRT engine (PR 4914 fused_kernel regions): 'mlp' fuses "
                "post-attention RMSNorm + gate/up GEMVs + SiLU*up, and the down "
                "GEMV + residual add; 'mlp+attn' adds input RMSNorm + q/k/v "
                "GEMVs and o_proj + residual add; 'layers' adds q/k RMSNorm + "
                "RoPE with the softmax scale folded into q; 'all' adds the "
                "final RMSNorm + lm_head. Each block replaces several small "
                "kernels and the gaps between them; the KV-cache write and "
                "attention stay in TensorRT. Every value but '' requires "
                "static_v3, fp16 and CUDA graphs (TensorRT 11.4 reloads plugin "
                "modules on every call outside CUDA graphs). Launch settings "
                "and kernels are the editable files decode_launch and "
                "decode_kernels. 'generated' fuses the 'layers' blocks with "
                "kernels that PR 4914's KernelGenerator writes and validates "
                "instead of the hand-written ones, so the editable files do not "
                "apply to it; it needs the same settings (static_v3, fp16, CUDA "
                "graphs). " + region_generation.CACHE_NOTE,
                ("roofline_gap", "launch_overhead"),
                {
                    "workload.cache": "static_v3",
                    "engine.precision": "fp16",
                    "runtime.cudagraphs": True,
                },
                prepare=(GENERATED,),
            ),
            Knob(
                "kernels.inductor",
                tuple(qwen3_regions.INDUCTOR_VARIANTS),
                "none",
                self._inductor_description(),
                ("dominant_kernel",),
                # Per value: the region variants also need the whole-step
                # graph, under which their regions are captured with the
                # engines (per-engine graphs would run them eagerly between
                # engines); attention-flex adds no region and runs under either.
                {
                    value: (
                        {**hybrid, "runtime.step_graph": True}
                        if value in qwen3_regions.REGION_VARIANTS
                        else hybrid
                    )
                    for value in qwen3_regions.INDUCTOR_VARIANTS
                    if value != "none"
                },
            ),
        )

    def _inductor_description(self) -> str:
        layers = int(self._config.num_hidden_layers)
        return (
            "Hybrid decode step: hand operations to Inductor's kernels and keep "
            "the rest in the config's TensorRT engine (kernels.decode_blocks "
            "blocks or plain static_v3). 'attention-flex': the kernels Inductor "
            "generates for FlexAttention decode (a split-KV template kernel and "
            "two combine reductions, compiled at build time for this model's "
            "shapes with mode='max-autotune-no-cudagraphs') run unchanged inside "
            "the engine as three fused_kernel plugins per layer, behind wrappers "
            "that restore the 16-byte pointer alignment Inductor's own launch "
            "assumes (the KV cache inputs must be 16-byte aligned, which setup "
            "checks); the engine computes the block mask once per step; no "
            "engine boundary; runs under per-engine CUDA graphs or the step "
            "graph. 'lm-head': [residual add +] final RMSNorm + lm_head in one "
            "execute_in_torch(backend='inductor') region after the engine "
            "(replaces the 'final' block): 1 engine-region boundary per token. "
            "'attention': every layer's decode attention in an "
            "execute_in_torch(backend='inductor') region that runs Inductor's "
            "flex-decoding: export captures flex_attention as a higher-order "
            "operator, which region capture rejects inside a scope, so the "
            "region calls an opaque custom operator whose implementation runs "
            "flex_attention compiled with mode='max-autotune-no-cudagraphs'. "
            "'attention-math': the same regions with plain attention math (two "
            "matmuls and a masked FP32 softmax over FP16 scores) that Inductor "
            "compiles itself. Both attention regions: "
            f"{layers} regions, {layers + 1} engines, {2 * layers} engine-region "
            "boundaries per token; the step takes its KV caches as inputs and "
            "TensorRT writes them in place (IKVCacheUpdate) before each region, "
            "which reads the written caches and masks the slots after the "
            "position. Regions compile in the build, on one step over scratch "
            "caches, without Inductor's CUDA graphs; lm-head and attention-math "
            "with Inductor's max-autotune options (max_autotune and "
            "coordinate_descent_tuning). Requires static_v3, fp16 and "
            "runtime.cudagraphs; 'lm-head', 'attention' and 'attention-math' "
            "also require runtime.step_graph (per-engine graphs would run their "
            "regions eagerly between engines). The region values cannot be "
            "saved: torch_tensorrt.save refuses execute_in_torch regions whose "
            "backend is 'inductor', so a session that keeps one writes its "
            "config but no artifact."
        )

    def editable_files(self) -> Sequence[EditableFile]:
        shapes = self._fused_shapes()
        if shapes is None:
            return ()
        here = Path(__file__).resolve().parent
        hidden, q, kv = shapes["hidden"], shapes["q"], shapes["kv"]
        inter, vocab = shapes["intermediate"], shapes["vocab"]
        return (
            EditableFile(
                "decode_launch",
                str(here / "qwen3_regions.py"),
                "DEFAULT_LAUNCH: launch settings of the fused decoder-block "
                "kernels that kernels.decode_blocks selects. BLOCK_N output "
                "rows per program, BLOCK_K the K chunk of the gemv_add kernels "
                "(o_proj, down), num_warps, EVICT (1: L2 evict_first weight "
                "loads) and FOLD_SCALE (softmax scale folded into q). BLOCK_N "
                "and BLOCK_K must be powers of two that divide N and K: "
                f"attn_in N={q} (q) and {kv} (k, v each) K={hidden}, attn_out "
                f"N={hidden} K={q}, mlp_in N={inter} K={hidden}, mlp_out "
                f"N={hidden} K={inter}, final N={vocab} K={hidden}. Matters only "
                "when kernels.decode_blocks selects hand-written blocks (not "
                "'generated').",
            ),
            EditableFile(
                "decode_kernels",
                str(here / "qwen3_kernels.py"),
                "The Triton kernels of the fused decoder blocks. Each computes "
                "one block's output vector for one token from FP16 row-major "
                "[N, K] weights with FP32 accumulation and one FP16 rounding. "
                "Keep every kernel's name and argument order (the plugin ABI: "
                "inputs, then outputs, then constexprs) and tl.multiple_of(..., "
                "16) hints on dereferenced pointers (without them loads are "
                "2-byte). Matters only when kernels.decode_blocks selects "
                "hand-written blocks (not 'generated').",
            ),
        )

    @staticmethod
    def _build_route_knobs(long_cache: bool = False) -> tuple[Knob, ...]:
        try:
            from torch_tensorrt.dynamo import is_global_perf_tuning_available
        except ImportError:
            return ()
        if not is_global_perf_tuning_available():
            return ()
        description = (
            "TensorRT Global Performance Tuning build route (Myelin knobs). "
            "'-kgen:lower_fc:all=on' generates every GEMV with kgen, which "
            "can fuse it with neighbouring norms and adds, instead of cuBLAS "
            "gemvx; '-cask_fusion:fp16_acc=off' forbids FP16-accumulation "
            "GEMM/GEMV kernels (FP32 accumulation without use_fp32_acc's "
            "casts); '-kgen:codegen:cuda_tile=2' uses CUDA Tile code "
            "generation wherever supported; '-peep:match_gemv_mha=on' and "
            "'-mha_fusion:gemv_mha_v2=off' change the decode attention "
            "kernel; '-tactic:splitk=off' and '-late_h_fusion=off' disable "
            "split-K tactics and late horizontal fusion."
        )
        if long_cache:
            description += (
                " At this cache length static_v3 attention is already "
                "gemv_mha_v2, so '-peep:match_gemv_mha=on' builds the same "
                "static_v3 engine as ''; '-mha_fusion:gemv_mha_v2=off' falls "
                "back to gemv_mha_v1."
            )
        return (
            Knob(
                "engine.build_route",
                BUILD_ROUTES,
                "",
                description,
                ("dominant_kernel",),
            ),
        )

    def baseline(self) -> dict[str, Any]:
        return dict(self._baseline)

    def timing(self) -> Timing:
        return Timing(warmup=2, rounds=10, calls=1, profile_calls=1)

    def accuracy_defaults(self) -> dict[str, tuple[str, float]]:
        return {"top1_agreement": (">=", 0.97), "mean_kl": ("<=", 0.005)}

    def tokens(self) -> torch.Tensor:
        """The fixed token sequence: ``isl`` prompt tokens, then ``osl`` decode inputs."""
        if self._tokens is None:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            def encode(text: str) -> torch.Tensor:
                return tokenizer(text, return_tensors="pt")["input_ids"][0]

            self._tokens = layout_tokens(
                encode(TEXT),
                encode(PASSAGE),
                encode(CODA),
                self.isl,
                self.osl,
                self.prompt,
            )
        return self._tokens

    def reference(self) -> Any:
        """FP32 eager logits at every decode position, given the same tokens.

        Only the decode positions (the last ``osl``) go through the LM head.
        """
        from transformers import AutoModelForCausalLM

        with torch.inference_mode():
            model = (
                AutoModelForCausalLM.from_pretrained(
                    self.model_name, attn_implementation="sdpa"
                )
                .eval()
                .cuda()
                .float()
            )
            tokens = self.tokens().cuda().unsqueeze(0)
            logits = model(tokens, logits_to_keep=self.osl).logits[0].float().cpu()
        del model
        torch.cuda.empty_cache()
        return logits

    def compare(self, outputs: Any, reference: Any) -> dict[str, float]:
        actual = outputs.detach().double().cpu()
        expected = reference.double()
        if actual.shape != expected.shape:
            raise ValueError(
                f"Logits shape {tuple(actual.shape)} != {tuple(expected.shape)}"
            )
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
            "max_abs_logit_error": (actual - expected).abs().max().item(),
        }

    # -------------------------------------------------------------- build

    def build(self, values: Mapping[str, Any]) -> Candidate:
        import torch_tensorrt
        from transformers import AutoModelForCausalLM

        backend = values.get("bench.backend") or "tensorrt"
        if backend not in BACKENDS:
            raise ValueError(
                f"bench.backend must be one of {BACKENDS}; got {backend!r}"
            )
        if backend != "tensorrt":
            return self._build_torch(values, backend)
        if values.get("runtime.step_graph") and (
            values["workload.cache"] != "static_v3" or not values["runtime.cudagraphs"]
        ):
            raise ValueError(
                "runtime.step_graph needs workload.cache=static_v3 and "
                "runtime.cudagraphs=True"
            )
        if (
            values.get("kernels.decode_blocks")
            or (values.get("kernels.inductor") or "none") != "none"
        ):
            return self._build_fused(values)
        cache = values["workload.cache"]
        precision = values["engine.precision"]
        device = torch.device("cuda:0")
        with torch.inference_mode():
            model = (
                AutoModelForCausalLM.from_pretrained(
                    self.model_name, use_cache=False, attn_implementation="sdpa"
                )
                .eval()
                .to(device)
                .to(_DTYPES[precision])
            )
        settings = {
            "use_fp32_acc": precision == "fp16" and values["engine.use_fp32_acc"],
            "disable_tf32": True,
            "device": device,
            "use_python_runtime": True,
            "offload_module_to_cpu": True,
            "min_block_size": 1,
            "optimization_level": values["engine.optimization_level"],
            "tiling_optimization_level": values["engine.tiling_optimization_level"],
            "build_route": values.get("engine.build_route") or None,
        }
        tokens = self.tokens().to(device)
        prompt_cache = None
        with torch.inference_mode():
            if cache == "static_v3":
                import static_cache_v3

                if self.eager_prefill:
                    prompt_cache = _eager_prefill(model, tokens[: self.isl])
                static_cache_v3.configure(self.max_seq, attention=self.v3_attention)
                ids = tokens[:1].view(1, 1)
                positions = torch.zeros((1, 1), dtype=torch.int64, device=device)
                program = torch.export.export(
                    static_cache_v3.DecodeStep(model), (ids, positions), strict=False
                )
                module = torch_tensorrt.dynamo.compile(
                    program, arg_inputs=[ids, positions], **settings
                )
            else:
                from torchtrt_ext import register_sdpa
                from utils import export_llm

                register_sdpa.enable_sdpa_converter(self.model_name, model.config)
                if cache == "static_v2":
                    import static_cache_v2  # noqa: F401  (registers the lowering pass)
                else:
                    import static_cache_v1  # noqa: F401  (registers the lowering pass)
                prompt = tokens[: self.isl].unsqueeze(0)
                program = export_llm(model, prompt, max_seq_len=self.max_seq)
                positions = torch.arange(self.isl, device=device).unsqueeze(0)
                module = torch_tensorrt.dynamo.compile(
                    program, arg_inputs=[prompt, positions], **settings
                )
        try:
            bytes_per_unit: float | None = _decode_bytes_per_token(
                model, cache, self.isl, self.osl, attention=self.v3_attention
            )
        except Exception:  # noqa: BLE001 - a diagnostic must not fail the build
            bytes_per_unit = None
        del model
        torch.cuda.empty_cache()
        candidate = _DecodeCandidate(
            module,
            cache=cache,
            tokens=tokens,
            isl=self.isl,
            osl=self.osl,
            cudagraphs=values["runtime.cudagraphs"],
            pre_allocated_outputs=values["runtime.pre_allocated_outputs"],
            prompt_cache=prompt_cache,
            fill_unwritten=cache == "static_v3" and self.v3_attention == "lengths",
            step_graph=bool(values.get("runtime.step_graph")),
        )
        candidate.bytes_per_unit = bytes_per_unit
        return candidate

    def _build_torch(self, values: Mapping[str, Any], backend: str) -> Candidate:
        """Hugging Face StaticCache decode, run eagerly or compiled by Inductor.

        Prefill runs eagerly, as in ``generate``; the decode step is compiled
        (and its CUDA graphs recorded) here, so the build time includes it.
        """
        import time

        from transformers import AutoModelForCausalLM, StaticCache

        mode = values.get("bench.mode") or "default"
        attn = values.get("bench.attn") or "sdpa"
        if backend == "inductor":
            import torch._inductor.config as inductor_config

            for key, value in dict(values.get("bench.inductor") or {}).items():
                if not hasattr(inductor_config, key):
                    raise ValueError(f"Unknown torch._inductor.config option {key!r}")
                setattr(inductor_config, key, value)
        if attn == "flex_attention":
            _drop_block_mask_compile_flag()
        device = torch.device("cuda:0")
        model = (
            AutoModelForCausalLM.from_pretrained(
                self.model_name, attn_implementation=attn
            )
            .eval()
            .to(device)
            .to(_DTYPES[values["engine.precision"]])
        )
        cache = StaticCache(config=model.config, max_cache_len=self.max_seq)
        tokens = self.tokens().to(device)

        def decode_step(
            input_ids: torch.Tensor, position_ids: torch.Tensor
        ) -> torch.Tensor:
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
        with torch.no_grad():
            model(
                input_ids=tokens[: self.isl].view(1, -1),
                position_ids=torch.arange(self.isl, device=device).view(1, -1),
                past_key_values=cache,
                use_cache=True,
                logits_to_keep=1,
            )
        candidate = _TorchDecodeCandidate(
            step,
            cache,
            tokens=tokens,
            isl=self.isl,
            osl=self.osl,
            cudagraphs=backend == "inductor" and mode in _CUDAGRAPH_MODES,
            # transformers 5.18's FlexAttention decode mask lets a token see
            # the cache slots after its position. A real generation leaves
            # them zero (unwritten), so every call clears them as one would.
            clear_future=attn == "flex_attention",
        )
        candidate.structure = {
            "backend": backend,
            "mode": mode if backend == "inductor" else None,
            "attn": attn,
            "inductor": dict(values.get("bench.inductor") or {}),
        }
        try:
            candidate.bytes_per_unit = _decode_bytes_per_token(
                model, "static_v3", self.isl, self.osl, attention=self.v3_attention
            )
        except Exception:  # noqa: BLE001 - a diagnostic must not fail the build
            candidate.bytes_per_unit = None
        start = time.perf_counter()
        for _ in range(3):
            candidate()
        torch.cuda.synchronize()
        candidate.structure["warmup_s"] = time.perf_counter() - start
        return candidate

    def _build_fused(self, values: Mapping[str, Any]) -> Candidate:
        """A static_v3 decode engine whose decoder blocks are fused Triton kernels.

        With ``kernels.inductor`` it is the hybrid step: Inductor regions, or
        Inductor's hosted kernels, in the config's engine (``preset`` may then
        be empty: plain static_v3 math, strictly exported). With
        ``kernels.decode_blocks="generated"`` the blocks run generated kernels
        (see :meth:`_blocks`).
        """
        import qwen3_regions
        from transformers import AutoModelForCausalLM

        preset = values.get("kernels.decode_blocks") or ""
        inductor = values.get("kernels.inductor") or "none"
        knob = "kernels.decode_blocks" if preset else "kernels.inductor"
        if values["workload.cache"] != "static_v3" or values["engine.precision"] != (
            "fp16"
        ):
            raise ValueError(
                f"{knob} needs workload.cache=static_v3 and engine.precision=fp16"
            )
        if not values["runtime.cudagraphs"]:
            raise ValueError(
                f"{knob} needs runtime.cudagraphs=True: TensorRT "
                "11.4 reloads plugin modules on every call outside CUDA graphs"
            )
        if inductor not in qwen3_regions.INDUCTOR_VARIANTS:
            raise ValueError(
                f"kernels.inductor must be one of {qwen3_regions.INDUCTOR_VARIANTS}; "
                f"got {inductor!r}"
            )
        regions_needed = inductor in qwen3_regions.REGION_VARIANTS
        if regions_needed and not values.get("runtime.step_graph"):
            raise ValueError(
                f"kernels.inductor={inductor} needs runtime.step_graph=True: "
                "per-engine CUDA graphs would run its Inductor regions eagerly "
                "between engines"
            )
        if inductor in ("lm-head", "attention-math"):
            # The regions compile on their first call with the global Inductor
            # config: use max-autotune's options (without Inductor's CUDA
            # graphs, which regions disable), as the head-to-head's Inductor
            # configurations do. ('attention' regions only call flex_decode,
            # which compiles flex_attention with that mode itself.)
            import torch._inductor.config as inductor_config

            inductor_config.max_autotune = True
            inductor_config.coordinate_descent_tuning = True
        regions: Any = qwen3_regions
        if values.get("edit.decode_launch"):
            regions = load_module(values["edit.decode_launch"], "qwen3_regions_edited")
        kernels = "qwen3_kernels"
        if values.get("edit.decode_kernels"):
            kernels = "qwen3_kernels_edited"
            load_module(values["edit.decode_kernels"], kernels)
        blocks, generated = self._blocks(regions, preset)
        device = torch.device("cuda:0")
        model = (
            AutoModelForCausalLM.from_pretrained(
                self.model_name, use_cache=False, attn_implementation="sdpa"
            )
            .eval()
            .to(device)
            .to(torch.float16)
        )
        tokens = self.tokens().to(device)
        prompt_cache = None
        if self.eager_prefill:
            # Before patching: the patched forward is a cache-less decode step.
            with torch.inference_mode():
                prompt_cache = _eager_prefill(model, tokens[: self.isl])
        # Only hybrid configs pass inductor=, and only generated blocks
        # generator=: an edited decode_launch file made from an earlier
        # qwen3_regions.py has no such parameters.
        hybrid = {} if inductor == "none" else {"inductor": inductor}
        regions.patch_qwen3(model, blocks, kernels=kernels, **hybrid, **generated)
        module = regions.compile_decode(
            model,
            max_seq_len=self.max_seq,
            device=device,
            attention=self.v3_attention,
            use_fp32_acc=values["engine.use_fp32_acc"],
            optimization_level=values["engine.optimization_level"],
            tiling_optimization_level=values["engine.tiling_optimization_level"],
            build_route=values.get("engine.build_route") or None,
        )
        region_compile_s = None
        if regions_needed:
            region_compile_s = _compile_regions(module, device)
        try:
            bytes_per_unit: float | None = _decode_bytes_per_token(
                model, "static_v3", self.isl, self.osl, attention=self.v3_attention
            )
        except Exception:  # noqa: BLE001 - a diagnostic must not fail the build
            bytes_per_unit = None
        flex = getattr(getattr(model, "_qwen3_blocks", None), "flex", None)
        # Regions build without offload_module_to_cpu, and strict export keeps
        # references to the parameters: move them off the GPU in place (the
        # engine holds its own copy), as offloading does on the other path.
        # The lm-head region reads the final norm and the (tied) LM head
        # weight, so only the decoder layers go.
        from torch_tensorrt.dynamo.utils import deallocate_module

        deallocate_module(model.model.layers if inductor == "lm-head" else model)
        del model
        gc.collect()
        torch.cuda.empty_cache()
        candidate = _DecodeCandidate(
            module,
            cache="static_v3",
            tokens=tokens,
            isl=self.isl,
            osl=self.osl,
            cudagraphs=True,
            pre_allocated_outputs=values["runtime.pre_allocated_outputs"],
            prompt_cache=prompt_cache,
            fill_unwritten=self.v3_attention == "lengths",
            step_graph=bool(values.get("runtime.step_graph")),
        )
        candidate.bytes_per_unit = bytes_per_unit
        if generated:
            candidate.kernel_cache = self.kernel_cache
        if inductor != "none":
            candidate.hybrid = {"inductor": inductor}
            if flex is not None:
                candidate.hybrid["inductor_flex_decoding"] = flex.describe()
            if region_compile_s is not None:
                candidate.hybrid["region_compile_s"] = round(region_compile_s, 1)
        return candidate

    def _blocks(
        self, regions: Any, preset: str
    ) -> tuple[tuple[str, ...], dict[str, Any]]:
        """The blocks a ``kernels.decode_blocks`` value fuses, and its patch_qwen3 options.

        A preset launches the hand-written kernels. ``generated`` fuses the
        :data:`GENERATED_PRESET` blocks with kernels that PR 4914's
        KernelGenerator writes, with the shared settings
        (``region_generation.make_generator``) and :data:`GENERATION_NOTES`, so
        every build asks for the kernels cached in ``kernel_cache``.
        """
        if preset != GENERATED:
            return regions.PRESETS[preset], {}
        import region_generation

        generator = region_generation.make_generator(
            self.kernel_cache, GENERATION_NOTES
        )
        return regions.PRESETS[GENERATED_PRESET], {"generator": generator}


def layout_tokens(
    text: torch.Tensor,
    passage: torch.Tensor,
    coda: torch.Tensor,
    isl: int,
    osl: int,
    prompt: str,
) -> torch.Tensor:
    """The ``isl + osl`` tokens of a ``prompt`` layout (see the module docstring).

    ``text``, ``passage`` and ``coda`` are the token ids of :data:`TEXT`,
    :data:`PASSAGE` and :data:`CODA`. ``auto`` is ``text`` while ``text``
    covers every token without repeating, otherwise ``recall`` if the prompt
    fits the passage.
    """
    total = isl + osl
    room = isl - RECALL_CUE - passage.numel()
    if prompt == "auto":
        prompt = "text" if total <= text.numel() or room < 0 else "recall"
    if prompt == "text":
        return text.repeat(-(-total // text.numel()))[:total].clone()
    if prompt != "recall":
        raise ValueError(f"prompt must be one of {PROMPTS}; got {prompt!r}")
    if room < 0:
        raise ValueError(
            f"The recall layout needs isl >= {passage.numel() + RECALL_CUE} "
            f"tokens; got {isl}"
        )
    filler = text.repeat(-(-room // text.numel()) if room else 0)[:room]
    cue = passage[:RECALL_CUE]
    # Decode: the passage after the cue (copied from the start of the prompt),
    # then the unseen coda, then the text if osl outlasts both.
    recalled = passage[RECALL_CUE : RECALL_CUE + RECALL_DECODE]
    decode = torch.cat([recalled, coda, text.repeat(-(-osl // text.numel()))])
    return torch.cat([passage, filler, cue, decode[:osl]]).clone()


def _eager_prefill(model: torch.nn.Module, prompt: torch.Tensor) -> list[torch.Tensor]:
    """Keys and values of ``prompt`` from one Hugging Face forward of ``model``.

    Returned per layer as ``key, value``: the order and the
    ``[batch, kv_heads, length, head_dim]`` layout (after RoPE and the q/k
    norms) of a static_v3 engine's cache inputs.
    """
    from transformers import DynamicCache

    cache = DynamicCache(config=model.config)
    positions = torch.arange(prompt.numel(), device=prompt.device).unsqueeze(0)
    model(
        prompt.view(1, -1),
        position_ids=positions,
        past_key_values=cache,
        use_cache=True,
        logits_to_keep=1,
    )
    tensors = []
    for layer in cache.layers:
        tensors += [layer.keys, layer.values]
    return tensors


def _fill_unwritten(caches: Sequence[torch.Tensor], start: int) -> None:
    """Fill every cache's slots from ``start`` on with noise at its own scale.

    A correct engine attends only to the slots written so far and writes each
    slot before it reads it, so its logits stay the same. One that reads past
    the written slots meets the noise on its first call, the call whose
    logits are checked first; later calls would find the previous call's keys
    and values there, which change the logits much less. Seeded, so every
    candidate starts from the same values.
    """
    generator = torch.Generator(device=caches[0].device)
    generator.manual_seed(0)
    for cache in caches:
        scale = cache[:, :, :start].float().pow(2).mean().sqrt()
        noise = torch.randn(
            cache[:, :, start:].shape, generator=generator, device=cache.device
        )
        cache[:, :, start:].copy_(noise * scale)


def _drop_block_mask_compile_flag() -> None:
    """Let Dynamo trace transformers' FlexAttention block masks.

    transformers 5.18 calls ``create_block_mask(_compile=True)``, which this
    torch nightly deprecates with a warning that Dynamo cannot trace inside a
    compiled decode step. The step is compiled already, so drop the flag, as
    the warning itself recommends.
    """
    import transformers.masking_utils as masking_utils
    from torch.nn.attention.flex_attention import create_block_mask

    def create_block_mask_without_compile_flag(
        *args: Any, _compile: bool = False, **kwargs: Any
    ) -> Any:
        return create_block_mask(*args, **kwargs)

    masking_utils.create_block_mask = create_block_mask_without_compile_flag


def _power_of_two(n: int) -> bool:
    return n > 0 and n & (n - 1) == 0


def _decode_bytes_per_token(
    model: torch.nn.Module, cache: str, isl: int, osl: int, attention: str = "mask"
) -> float:
    """Bytes one decode step must read: every weight once, plus the KV cache.

    A tied embedding is read in full by the LM head, so it counts once; an
    untied input embedding contributes only one row and is left out. A
    static_v3 cache with a position mask (``attention="mask"``) attends over
    all ``isl + osl`` slots; with ``key_value_lengths`` (``"lengths"``) it
    reads only the written slots, like the other caches, which slice to the
    current length: on average ``isl + osl / 2``.
    """
    weights = sum(p.numel() * p.element_size() for p in model.parameters())
    embedding = model.get_input_embeddings().weight
    head = model.get_output_embeddings()
    if head is None or head.weight.data_ptr() != embedding.data_ptr():
        weights -= embedding.numel() * embedding.element_size()
    config = model.config
    heads = getattr(config, "num_attention_heads", None) or config.n_head
    head_dim = (
        getattr(config, "head_dim", None)
        or (getattr(config, "hidden_size", None) or config.n_embd) // heads
    )
    kv_heads = getattr(config, "num_key_value_heads", None) or heads
    layers = getattr(config, "num_hidden_layers", None) or config.n_layer
    full = cache == "static_v3" and attention == "mask"
    length = isl + osl if full else isl + osl / 2
    kv = 2 * layers * kv_heads * head_dim * length * embedding.element_size()
    return float(weights + kv)


class _DecodeCandidate(Candidate):
    def __init__(
        self,
        module: torch.nn.Module,
        *,
        cache: str,
        tokens: torch.Tensor,
        isl: int,
        osl: int,
        cudagraphs: bool,
        pre_allocated_outputs: bool,
        prompt_cache: Sequence[torch.Tensor] | None = None,
        fill_unwritten: bool = False,
        step_graph: bool = False,
    ) -> None:
        self.module = module
        self.cache = cache
        self.isl, self.osl = isl, osl
        self.units = osl
        self.items = 1
        self.cudagraphs = cudagraphs
        self.pre_allocated_outputs = pre_allocated_outputs
        # One CUDA graph for the whole static_v3 step (see _StepGraph).
        if step_graph and (cache != "static_v3" or not cudagraphs):
            raise ValueError("A step graph needs the static_v3 cache and CUDA graphs")
        self.step_graph = step_graph
        self._graph: _StepGraph | None = None
        # kernels.inductor: the variant and what describe() reports about it.
        self.hybrid: dict[str, Any] = {}
        # kernels.decode_blocks=generated: the KernelGenerator cache it used.
        self.kernel_cache: str | None = None
        # static_v3 only: the prompt's keys and values from an eager prefill,
        # copied into the cache instead of prefilling through the engine.
        self.prompt_cache = list(prompt_cache) if prompt_cache is not None else None
        if self.prompt_cache is not None and cache != "static_v3":
            raise ValueError("An eager prompt cache needs the static_v3 cache")
        # static_v3 with key_value_lengths: noise in the decode slots until
        # the steps write them (see _fill_unwritten).
        self.fill_unwritten = fill_unwritten
        if fill_unwritten and cache != "static_v3":
            raise ValueError("Only the static_v3 cache has unwritten slots")
        device = tokens.device
        self.step_tokens = [tokens[isl + j].view(1, 1) for j in range(osl)]
        self.step_positions = [
            torch.full((1, 1), isl + j, dtype=torch.int64, device=device)
            for j in range(osl)
        ]
        self.prompt = tokens[:isl].view(1, isl)
        self._stack = contextlib.ExitStack()
        self._start: tuple[torch.Tensor, ...] = ()

    def _zero_caches(self) -> list[torch.Tensor]:
        placeholders = [n for n in self.module.graph.nodes if n.op == "placeholder"]
        caches = placeholders[2:] if self.cache == "static_v3" else placeholders[2:-2]
        return [
            torch.zeros(
                tuple(node.meta["val"].shape),
                dtype=node.meta["val"].dtype,
                device=self.prompt.device,
            )
            for node in caches
        ]

    def _step(
        self, token: torch.Tensor, position: torch.Tensor, index: int, caches: Any
    ) -> tuple[torch.Tensor, Any]:
        """Run one step; return its logits and the caches for the next step.

        A static v3 engine updates its cache inputs in place, so they stay the
        caches. In plain eager execution the runtime returns those same tensors;
        under tracing or a dispatch mode its op returns clones. Passing the same
        tensors every step keeps the addresses a captured CUDA graph binds.
        """
        if self.cache == "static_v3":
            if self._graph is not None and caches is self._graph.caches:
                return self._graph(token, position), caches
            out = self.module(token, position, *caches)
            return out[0], caches
        out = self.module(token, position, *caches, index, index + 1)
        return out[0], out[1:]

    def setup(self) -> None:
        import torch_tensorrt

        if self.pre_allocated_outputs:
            self._stack.enter_context(
                torch_tensorrt.runtime.enable_pre_allocated_outputs(self.module)
            )
        if self.cudagraphs and not self.step_graph:
            torch_tensorrt.runtime.set_cudagraphs_mode(True)
            self._stack.callback(torch_tensorrt.runtime.set_cudagraphs_mode, False)
        with torch.inference_mode():
            caches: Any = self._zero_caches()
            if self.hybrid.get("inductor") == "attention-flex":
                _check_cache_alignment(caches)
            if self.prompt_cache is not None:
                if len(self.prompt_cache) != len(caches):
                    raise ValueError(
                        f"The eager prefill has {len(self.prompt_cache)} cache "
                        f"tensors; the engine takes {len(caches)}"
                    )
                for cache, prompt in zip(caches, self.prompt_cache):
                    cache[:, :, : self.isl].copy_(prompt)
                self.prompt_cache = None
            elif self.cache == "static_v3":
                if self.step_graph:
                    # Capture first and prefill through replays: per-engine
                    # graphs are off, and outside CUDA graphs TensorRT 11.4
                    # reloads plugin modules on every enqueue. Warm-up writes
                    # slot 0 as the first prefill step does.
                    start = torch.zeros(
                        (1, 1), dtype=torch.int64, device=self.prompt.device
                    )
                    self._graph = _StepGraph(
                        self.module, caches, self.prompt[:, :1], start
                    )
                    caches = self._graph.caches
                for position in range(self.isl):
                    _, caches = self._step(
                        self.prompt[:, position : position + 1],
                        torch.full(
                            (1, 1),
                            position,
                            dtype=torch.int64,
                            device=self.prompt.device,
                        ),
                        position,
                        caches,
                    )
            else:
                positions = torch.arange(self.isl, device=self.prompt.device).unsqueeze(
                    0
                )
                out = self.module(self.prompt, positions, *caches, 0, self.isl)
                caches = out[1:]
            # Every timed call starts from this prefilled state. Static v3 writes
            # its caches in place, only at decode positions and identically on
            # every call, so it keeps these very tensors: a captured CUDA graph
            # binds their addresses. The other caches return new tensors.
            if self.cache == "static_v3":
                if self.step_graph and self._graph is None:
                    # After an eager prefill: warm-up steps write the first
                    # decode slot exactly as the first timed step does, before
                    # the unwritten slots get their noise.
                    self._graph = _StepGraph(
                        self.module,
                        caches,
                        self.step_tokens[0],
                        self.step_positions[0],
                    )
                    caches = self._graph.caches
                if self.fill_unwritten:
                    _fill_unwritten(caches, self.isl)
                self._start = tuple(caches)
                if self._graph is not None:
                    self._graph.caches = self._start
            else:
                self._start = tuple(tensor.clone() for tensor in caches)

    def __call__(self) -> torch.Tensor:
        with torch.inference_mode():
            caches: Any = self._start
            logits = []
            for j in range(self.osl):
                step_logits, caches = self._step(
                    self.step_tokens[j], self.step_positions[j], self.isl + j, caches
                )
                logits.append(step_logits[0, -1].to(torch.float32, copy=True))
            return torch.stack(logits)

    def describe(self) -> dict[str, Any]:
        from torch_tensorrt.autotune import describe_module

        structure = {**describe_module(self.module), "cache": self.cache}
        if self.kernel_cache is not None:
            structure["kernel_cache"] = self.kernel_cache
        if self.prompt_cache is not None:
            structure["prefill"] = "eager"
        if self.fill_unwritten:
            structure["unwritten_slots"] = "noise"
        if self.step_graph:
            structure["step_graph"] = True
        if self.hybrid:
            structure.update(self.hybrid)
            structure["torch_regions"] = _torch_regions(self.module)
        return structure

    def export(self, directory: Path) -> list[str]:
        """Save the compiled decode step as ``decode.ep``.

        ``kernels.inductor`` builds with Inductor regions (``lm-head``,
        ``attention``, ``attention-math``) cannot be saved: the exporter
        rejects ``execute_in_torch`` regions whose backend is not ``eager``,
        before it writes anything. That rejection is reported as
        ``NotImplementedError``, the contract's "cannot export", naming the
        value; other errors propagate unchanged.
        """
        import qwen3_regions
        import torch_tensorrt
        from torch_tensorrt import region

        path = Path(directory) / "decode.ep"
        try:
            torch_tensorrt.save(self.module, str(path), retrace=False)
        except region.RegionError as error:
            inductor = self.hybrid.get("inductor")
            if inductor not in qwen3_regions.REGION_VARIANTS:
                raise
            raise NotImplementedError(
                f"kernels.inductor={inductor!r} builds cannot be saved: their "
                "execute_in_torch(backend='inductor') regions run code that "
                "Inductor compiles at run time, and torch_tensorrt.save cannot "
                "preserve that backend. Rebuild the config instead "
                "('attention-flex' hosts Inductor's kernels in the engine and "
                "adds no region)."
            ) from error
        return [path.name]

    def close(self) -> None:
        self._graph = None
        self._stack.close()


def _compile_regions(module: torch.nn.Module, device: torch.device) -> float:
    """Run one static_v3 step on scratch caches, so that Inductor regions compile.

    Regions compile on their first call (an ``attention`` region's
    flex_decode compiles its flex_attention there too). Calling once here
    puts that compilation in the build, as the head-to-head's Inductor
    configurations compile in theirs, instead of in setup's warm-up, which
    then reuses it (same shapes, inference mode). Returns the seconds taken.
    """
    import time

    placeholders = [n for n in module.graph.nodes if n.op == "placeholder"]
    start = time.perf_counter()
    with torch.inference_mode():
        caches = [
            torch.zeros(
                tuple(node.meta["val"].shape),
                dtype=node.meta["val"].dtype,
                device=device,
            )
            for node in placeholders[2:]
        ]
        token = torch.zeros((1, 1), dtype=torch.int64, device=device)
        module(token, token.clone(), *caches)
        torch.cuda.synchronize(device)
    del caches
    torch.cuda.empty_cache()
    return time.perf_counter() - start


def _check_cache_alignment(caches: Sequence[torch.Tensor]) -> None:
    """Refuse cache tensors that are not 16-byte aligned (attention-flex).

    The hosted flex-decoding kernels assume 16-byte-aligned pointers, as
    Inductor's own launch does after checking; their wrappers round each
    pointer down, so a misaligned binding would read the wrong address.
    """
    for index, cache in enumerate(caches):
        if cache.data_ptr() % 16:
            raise ValueError(
                f"KV cache input {index} is not 16-byte aligned (address "
                f"{cache.data_ptr():#x}); kernels.inductor=attention-flex needs "
                "aligned caches"
            )


def _torch_regions(module: torch.nn.Module) -> dict[str, int]:
    """PyTorch regions of a compiled module, counted by backend."""
    try:
        from torch_tensorrt.dynamo.region_compilation import get_region_records
    except ImportError:
        return {}
    counts: dict[str, int] = {}
    for record in get_region_records(module):
        if record.route == "torch":
            counts[record.backend] = counts.get(record.backend, 0) + 1
    return counts


class _StepGraph:
    """One CUDA graph for a whole static_v3 decode step (``runtime.step_graph``).

    Captures ``module(token, position, *caches)``: every TensorRT engine (as
    plain enqueues, not per-engine graphs), plugin and PyTorch region,
    including Inductor-compiled ones, which compile on their first call (the
    build's, or else these warm-up calls) before the capture. The caches are
    bound in place, as the engines update
    them in place; the token and position are copied into static buffers
    before each replay; the logits come back in the graph's own buffer, valid
    until the next replay. Warm-up steps write the cache slot of ``position``.
    """

    def __init__(
        self,
        module: torch.nn.Module,
        caches: Sequence[torch.Tensor],
        token: torch.Tensor,
        position: torch.Tensor,
        warmup: int = 2,
    ) -> None:
        import torch_tensorrt

        if torch_tensorrt.runtime.get_cudagraphs_mode():
            raise RuntimeError(
                "A step graph captures the engines itself; turn per-engine CUDA "
                "graphs (set_cudagraphs_mode) off"
            )
        self.module = module
        self.caches = tuple(caches)
        self.token = token.clone()
        self.position = position.clone()
        stream = torch.cuda.Stream(device=token.device)
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(warmup):
                module(self.token, self.position, *self.caches)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            out = module(self.token, self.position, *self.caches)
        self.logits = out[0]

    def __call__(self, token: torch.Tensor, position: torch.Tensor) -> torch.Tensor:
        self.token.copy_(token)
        self.position.copy_(position)
        self.graph.replay()
        return self.logits


class _TorchDecodeCandidate(Candidate):
    """Decode with Hugging Face's StaticCache, eagerly or as a compiled step.

    Each call rewinds the cache to the prefilled prompt, so the decode steps
    write the same slots, identically, on every call.
    """

    def __init__(
        self,
        step: Any,
        cache: Any,
        *,
        tokens: torch.Tensor,
        isl: int,
        osl: int,
        cudagraphs: bool,
        clear_future: bool,
    ) -> None:
        self.step, self.cache = step, cache
        self.isl, self.osl = isl, osl
        self.units, self.items = osl, 1
        device = tokens.device
        self.step_tokens = [tokens[isl + j].view(1, 1) for j in range(osl)]
        self.step_positions = [
            torch.full((1, 1), isl + j, dtype=torch.int64, device=device)
            for j in range(osl)
        ]
        self.cudagraphs = cudagraphs
        self.clear_future = clear_future
        self.structure: dict[str, Any] = {}

    def __call__(self) -> torch.Tensor:
        with torch.no_grad():
            for layer in self.cache.layers:
                layer.cumulative_length.fill_(self.isl)
                if self.clear_future:
                    layer.keys[:, :, self.isl :].zero_()
                    layer.values[:, :, self.isl :].zero_()
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
            **self.structure,
        }

    def export(self, directory: Path) -> list[str]:
        return []
