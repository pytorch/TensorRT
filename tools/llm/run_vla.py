# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
.. _run_vla:

Running Alpamayo 1.5 (vision-language-action model) with Torch-TensorRT
==========================================================================

`Alpamayo 1.5 <https://huggingface.co/nvidia/Alpamayo-1.5-10B>`_ drives a Qwen3-VL backbone with a
flow-matching action expert. One rollout is

1. the vision tower on the camera images (fixed grid),
2. prefill + autoregressive decoding of the text decoder (reasoning tokens) with a KV cache,
3. ``--diffusion_steps`` denoising steps of the action expert, which cross-attends to that KV cache.

Each stage is one (or, for the decoder, several) Torch-TensorRT engine; the host keeps everything
data dependent (tokenization, embedding lookup, mRoPE index, sampling, expert mask, Euler loop).

The text decoder uses ``static_cache_v2`` (see ``static_cache_v2.py``) on the unmodified Hugging Face
decoder layers; the vision tower and the expert are exported with small wrappers (below).

Requires the ``alpamayo1_5`` package (and its dataset access) for the model and data loading::

    # build the engines, one process per component (keeps peak host memory down)
    python run_vla.py --mode build --component all --clip_id <clip uuid>
    # run on TensorRT / run the PyTorch reference and compare
    python run_vla.py --mode reference --clip_id <clip uuid> --save ref.pt
    python run_vla.py --mode run --clip_id <clip uuid> --ref ref.pt

The model is either the plain FP16 checkpoint or a ModelOpt FP8 checkpoint (a directory with a
``modelopt_state.pth``, see ``recipes/alpamayo1_5_quant``; needs ``nvidia-modelopt``). For FP8 the
weights are re-materialized as FP16 + fake-quant, so the exported graph carries plain Q/DQ nodes that
Torch-TensorRT folds into FP8 weights and activations.
"""

import argparse
import contextlib
import gc
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch_tensorrt

DEVICE = torch.device("cuda:0")
# additive attention-mask value that is safe in fp16
MASK_MIN = -30000.0


def get_args():
    ap = argparse.ArgumentParser(description="Alpamayo 1.5 with Torch-TensorRT")
    ap.add_argument("--mode", choices=["build", "run", "reference"], required=True)
    ap.add_argument(
        "--component",
        choices=["vision", "llm", "expert", "all"],
        default="all",
        help="build mode: which engine(s); 'all' builds each in its own process",
    )
    ap.add_argument("--model", type=str, default="nvidia/Alpamayo-1.5-10B")
    ap.add_argument("--engine_dir", type=str, default="vla_engines")
    ap.add_argument(
        "--clip_id", type=str, default=None, help="dataset clip (see --parquet)"
    )
    ap.add_argument(
        "--parquet",
        type=str,
        default=None,
        help="if --clip_id is not given: take the clip at --clip_index of this parquet's 'key' column",
    )
    ap.add_argument("--clip_index", type=int, default=0)
    ap.add_argument("--t0_us", type=int, default=5_100_100)
    ap.add_argument("--num_traj_samples", type=int, default=6)
    ap.add_argument("--max_generation_length", type=int, default=256)
    ap.add_argument(
        "--diffusion_steps",
        type=int,
        default=None,
        help="Euler steps of the action expert; defaults to the model's (10)",
    )
    ap.add_argument(
        "--max_seq_len",
        type=int,
        default=3456,
        help="KV-cache length: prompt (~3.1k tokens) + generated tokens + margin",
    )
    ap.add_argument(
        "--llm_segments",
        type=int,
        default=2,
        help="split the text decoder into this many engines; building one ~15 GB fp16 engine "
        "needs more host RAM than a 64 GB machine has",
    )
    ap.add_argument(
        "--chunk_len",
        type=int,
        default=64,
        help="max tokens per decoder call (profile max); the prompt is prefilled in chunks of this "
        "size. A larger profile makes TensorRT request tens of GB of scratch memory.",
    )
    ap.add_argument(
        "--fp8_attention_output_quant",
        action="store_true",
        help="FP8 checkpoints: also quantize the decoder's attention output (o_proj input) with "
        "TensorRT < 11.6. Off by default there: those versions build a wrong engine for some "
        "chunked-prefill windows (see maybe_disable_attention_output_quantizers)",
    )
    ap.add_argument("--top_p", type=float, default=0.98)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--resident",
        action="store_true",
        help="run mode: keep all engines loaded (the fp16 engines total ~28 GB); by default they "
        "are loaded one pipeline phase at a time",
    )
    ap.add_argument(
        "--save", type=str, default=None, help="reference mode: write results here"
    )
    ap.add_argument(
        "--ref", type=str, default=None, help="run mode: reference results to compare"
    )
    return ap.parse_args()


# %%
# Model and data
# ^^^^^^^^^^^^^^
def is_quantized(name: str) -> bool:
    """A ModelOpt-quantized checkpoint directory."""
    return (Path(name) / "modelopt_state.pth").exists()


def load_model(name: str):
    try:
        from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5
    except ImportError as e:
        raise SystemExit(f"run_vla.py needs the `alpamayo1_5` package: {e}")
    if is_quantized(name):
        import modelopt.torch.opt as mto

        mto.enable_huggingface_checkpointing()
    kwargs = {}
    try:
        import flash_attn  # noqa: F401
    except ImportError:
        # The checkpoint asks for FlashAttention-2; only the weights matter here.
        kwargs["attn_implementation"] = "sdpa"
    return Alpamayo1_5.from_pretrained(name, dtype=torch.float16, **kwargs).eval()


def load_inputs(model, args):
    from alpamayo1_5 import helper
    from alpamayo1_5.load_physical_aiavdataset import load_physical_aiavdataset

    clip_id = args.clip_id
    if clip_id is None:
        if args.parquet is None:
            raise SystemExit("pass --clip_id or --parquet")
        import pandas as pd

        df = pd.read_parquet(args.parquet)
        key = {c.lower(): c for c in df.columns}["key"]
        clip_id = str(df[key].iloc[args.clip_index])
    processor = helper.get_processor(model.tokenizer)
    data = load_physical_aiavdataset(clip_id, t0_us=args.t0_us)
    messages = helper.create_message(
        data["image_frames"].flatten(0, 1), camera_indices=data["camera_indices"]
    )
    tokenized = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        continue_final_message=True,
        return_dict=True,
        return_tensors="pt",
    )
    return data, tokenized


def cast_to_fp16(module: torch.nn.Module) -> None:
    """Cast leftover bf16/fp32 params and buffers (norms, Fourier frequencies) to fp16."""
    wide = (torch.bfloat16, torch.float32)
    for m in module.modules():
        for name, p in list(m._parameters.items()):
            if p is not None and p.dtype in wide:
                m._parameters[name] = torch.nn.Parameter(
                    p.data.to(torch.float16), False
                )
        for name, b in list(m._buffers.items()):
            if b is not None and b.dtype in wide:
                m._buffers[name] = b.to(torch.float16)


def dequantize_real_quant_weights(module: torch.nn.Module) -> None:
    """Turn ModelOpt real-quantized (compressed FP8) weights back into FP16 + fake-quant.

    The checkpoint stores weights as ``float8_e4m3fn`` and dequantizes inside the weight quantizer,
    which cannot be traced. Re-materializing the (already on-grid) FP16 weight and switching the
    quantizer to fake-quant gives a plain Q/DQ graph that TensorRT folds into FP8 weights. A no-op
    for the FP16 model.
    """
    from modelopt.torch.quantization.nn import TensorQuantizer

    for m in module.modules():
        wq = getattr(m, "weight_quantizer", None)
        if isinstance(wq, TensorQuantizer) and wq.is_enabled and wq._dequantize:
            with torch.no_grad():
                # ``m.weight`` is the raw compressed tensor; the quantizer (still in dequantize
                # mode) turns it back into values exactly as its forward does.
                w = wq(m._parameters["weight"]).detach().to(torch.float16)
            wq._dequantize = False
            wq._fake_quant = (
                True  # must precede the assignment, else the weight is re-compressed
            )
            m._parameters["weight"] = torch.nn.Parameter(w, requires_grad=False)
            m.allow_real_quant_gemm = False  # no real-quant GEMM kernels while tracing


def maybe_separate_sibling_input_scales(module: torch.nn.Module) -> None:
    """
    TensorRT < 11.5 needs the sibling input-scale workaround (11.5 builds these engines correctly).
    Fix for below versions by making the input scales of sibling FP8 linears (q/k/v, gate/up) differ
    in the last bit.

    The siblings read the same activation, so their calibrated input amax is identical, while their
    weight amax differs. With identical input Q/DQ and different weight scales TensorRT merges the
    GEMMs (e.g. ``[up]+[gate]`` followed by a ``MulAdd`` kernel) and builds an engine whose output is
    all NaN (fixed in TensorRT 11.5, so only applied before it). Reproduced on a single MLP of the
    Alpamayo decoder, with random weights too (H=4096, FF=12288, 64 tokens, ``mtq.FP8_DEFAULT_CFG``);
    equal weight scales or any difference in the input scales avoid it, and the Torch-TensorRT graph
    is the same in all cases. A 0.2% change of an input scale is far below the FP8 grid spacing (6%);
    applied to the eager FP8 model it changes minADE by 0.01 m or less.
    """
    import warnings
    import tensorrt
    from packaging.version import Version

    if Version(tensorrt.__version__) >= Version("11.5"):
        return

    warnings.warn(
        "Sibling input scales cause nan outputs for TensorRT versions below 11.5."
    )

    for m in module.modules():
        for group in (("q_proj", "k_proj", "v_proj"), ("gate_proj", "up_proj")):
            if not all(hasattr(m, name) for name in group):
                continue
            seen = set()
            for name in group:
                q = getattr(m, name).input_quantizer
                if q.amax is None:
                    continue
                while q.amax.item() in seen:
                    q.amax = (q.amax * 1.002).to(q.amax.dtype)
                seen.add(q.amax.item())


def maybe_disable_attention_output_quantizers(module: torch.nn.Module) -> None:
    """
    TensorRT < 11.6 needs the attention output (the ``o_proj`` input) of the decoder unquantized
    (fixed in TensorRT 11.6, so only applied before it).

    TensorRT fuses the FP8 Q/DQ of the attention output into the epilogue of the P.V matmul (Myelin
    kernel ``__myl_FcCastMulCast``, a ``*_by_fusion_tactic`` GEMM). With the decoder's dynamic key
    length (``static_cache_v2``) and a dynamic batch, that kernel is sometimes launched with the K
    extent and P/V strides of an earlier call: of key lengths 2240 and 2304 (batch 1, 64 tokens),
    whichever runs first is correct and the other is wrong. In layer 14 of the Alpamayo decoder,
    1 of 49 prefill chunks (2240..2303) has 38% error per token, and the errors accumulate over the
    layers (minADE +0.4 m). It is a TensorRT runtime bug (seen in 11.4 and 11.5, reproduced with the
    plain TensorRT API and with random weights, where peaked attention makes it visible). Without the
    quantizer the matmul is not fused and ``o_proj`` runs in FP16; the engine then matches the eager
    FP8 model (minADE within about 0.03 m).
    """
    import warnings
    import tensorrt
    from packaging.version import Version

    if Version(tensorrt.__version__) >= Version("11.6"):
        return

    warnings.warn(
        "Quantizing the attention output gives wrong outputs for TensorRT versions below 11.6; "
        "it is left unquantized."
    )

    for m in module.modules():
        o_proj = getattr(m, "o_proj", None)
        if o_proj is not None and hasattr(o_proj, "input_quantizer"):
            o_proj.input_quantizer.disable()


def prepare_weights(
    module: torch.nn.Module, quantized: bool, attention_output_quant: bool = True
) -> None:
    cast_to_fp16(module)
    if quantized:
        dequantize_real_quant_weights(module)
        maybe_separate_sibling_input_scales(module)
        if not attention_output_quant:
            maybe_disable_attention_output_quantizers(module)


def bmm_quantize(attn: torch.nn.Module, q, k, v):
    """ModelOpt's attention q/k/v fake-quantizers (FP8 checkpoints; identity for FP16).

    The Hugging Face decoder layers apply them themselves, the hand-written attention below does
    not. They matter: the layer-0 keys have attention-sink outliers that the k quantizer saturates.
    """
    qz = [getattr(attn, f"{n}_bmm_quantizer", None) for n in "qkv"]
    return tuple(f(t) if f is not None else t for f, t in zip(qz, (q, k, v)))


def min_ade(pred_xyz, data) -> float:
    gt = data["ego_future_xyz"].cpu()[0, 0, :, :2].numpy()
    pred = pred_xyz.detach().float().cpu().numpy()[0, 0, :, :, :2]
    return float(np.linalg.norm(pred - gt[None], axis=-1).mean(-1).min())


# %%
# Export wrappers
# ^^^^^^^^^^^^^^^
# Shapes the stock HF code derives from Python values (image grids, rope indices, masks) are
# computed on the host or precomputed, so that ``torch.export`` sees no data-dependent control flow.
class StaticVisionEncoder(torch.nn.Module):
    """Qwen3-VL vision tower for a fixed ``image_grid_thw``.

    The stock forward derives position embeddings, rotary tables and ``cu_seqlens`` from the
    *values* of ``grid_thw``, which cannot be traced. With a fixed camera rig the grid is constant,
    so everything is precomputed and the graph only takes ``pixel_values``. All images share one
    grid, so per-image attention is a plain batched SDPA instead of varlen attention.
    """

    def __init__(self, visual, grid_thw: torch.Tensor):
        super().__init__()
        assert (
            grid_thw == grid_thw[:1]
        ).all(), "all images must share the same grid_thw"
        t, h, w = grid_thw[0].tolist()
        self.num_images = grid_thw.shape[0]
        self.tokens_per_image = t * h * w
        self.visual = visual
        with torch.no_grad():
            pos_embeds = visual.fast_pos_embed_interpolate(grid_thw)
            rot = visual.rot_pos_emb(grid_thw)
            emb = torch.cat((rot, rot), dim=-1)
        self.register_buffer("pos_embeds", pos_embeds)
        self.register_buffer("cos", emb.cos())
        self.register_buffer("sin", emb.sin())

    def _attn(self, attn, x):
        from transformers.models.qwen3_vl.modeling_qwen3_vl import (
            apply_rotary_pos_emb_vision,
        )

        seq = x.shape[0]
        q, k, v = (
            attn.qkv(x)
            .reshape(seq, 3, attn.num_heads, -1)
            .permute(1, 0, 2, 3)
            .unbind(0)
        )
        q, k = apply_rotary_pos_emb_vision(q, k, self.cos, self.sin)
        q, k, v = (  # [S, H, D] -> [N, H, L, D]
            t.reshape(
                self.num_images, self.tokens_per_image, attn.num_heads, -1
            ).transpose(1, 2)
            for t in (q, k, v)
        )
        q, k, v = bmm_quantize(attn, q, k, v)
        out = F.scaled_dot_product_attention(q, k, v, scale=attn.scaling)
        return attn.proj(out.transpose(1, 2).reshape(seq, -1))

    def forward(self, pixel_values: torch.Tensor):
        v = self.visual
        x = v.patch_embed(pixel_values) + self.pos_embeds
        deepstack = []
        for i, blk in enumerate(v.blocks):
            x = x + self._attn(blk.attn, blk.norm1(x))
            x = x + blk.mlp(blk.norm2(x))
            if i in v.deepstack_visual_indexes:
                merger = v.deepstack_merger_list[v.deepstack_visual_indexes.index(i)]
                deepstack.append(merger(x))
        return v.merger(x), deepstack


class TextSegment(torch.nn.Module):
    """A contiguous range of Qwen3-VL text decoder layers (unmodified HF layers).

    The KV cache is added to the graph by ``static_cache_v2`` at compile time; the causal mask is
    that pass's causal SDPA, so the layers are called with ``attention_mask=None``. mRoPE is computed
    in the graph from ``position_ids`` [3, B, L]. The last segment applies the final norm and
    ``lm_head`` to all L positions (the host reads the last).

    The decoder is split into several engines because the host RAM needed to build and serialize one
    ~15 GB fp16 engine exceeds a 64 GB machine.
    """

    def __init__(
        self, text_model, lm_head, layer_range: Tuple[int, int], num_deepstack: int
    ):
        super().__init__()
        lo, hi = layer_range
        self.layers = text_model.layers[lo:hi]
        self.rotary_emb = text_model.rotary_emb
        self.first_layer, self.num_deepstack = lo, num_deepstack
        self.is_last = hi == len(text_model.layers)
        if self.is_last:
            self.norm, self.lm_head = text_model.norm, lm_head

    def _run(self, hidden, position_ids, deepstack):
        position_embeddings = self.rotary_emb(hidden, position_ids)
        for i, layer in enumerate(self.layers):
            hidden = layer(
                hidden, position_embeddings=position_embeddings, attention_mask=None
            )
            if deepstack is not None and self.first_layer + i < self.num_deepstack:
                hidden = (
                    hidden + deepstack[self.first_layer + i]
                )  # DeepStack: [1, L, H]
        return self.lm_head(self.norm(hidden)) if self.is_last else hidden

    def forward(self, hidden, position_ids):
        return self._run(hidden, position_ids, None)


class TextFirstSegment(TextSegment):
    """First segment: adds the dense DeepStack features [num_deepstack, 1, L, H] after layer i.

    (A separate class rather than an optional argument: ``torch_tensorrt.save`` cannot serialize a
    signature with a defaulted ``None`` argument that is absent from the graph.)
    """

    def forward(self, hidden, position_ids, deepstack):
        return self._run(hidden, position_ids, deepstack)


def _folded_gqa_sdpa(q, k, v, mask, scale):
    """SDPA with grouped-query attention folded into the query rows.

    q: [B, Hq, L, D]; k, v: [B, Hkv, S, D]; mask: additive, broadcastable to [B, 1, L, S]. Query
    head ``h`` attends kv head ``h // groups`` (as ``repeat_kv``), so stacking the query heads of a
    kv head along the row dim is exact and never expands K/V.
    """
    B, Hq, L, D = q.shape
    Hkv = k.shape[1]
    groups = Hq // Hkv
    q = q.reshape(B, Hkv, groups * L, D)
    mask = torch.cat([mask] * groups, dim=2) if groups > 1 else mask
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=scale).reshape(
        B, Hq, L, D
    )


class ExpertStep(torch.nn.Module):
    """``action_in_proj -> expert (cross-attending to the VLM KV cache) -> action_out_proj``.

    One flow-matching denoising step. The VLM KV cache is consumed read-only; the expert's own K/V
    for the diffusion tokens are appended to it. A host-built additive mask hides unwritten cache
    slots and the gap between the end of the reasoning and the diffusion tokens (this is a
    non-causal expert, which is why ``static_cache_v2`` does not apply).

    Inputs: x [B, T, C], t [B, 1, 1], cos/sin [B, T, D], mask [B, 1, T, S + T],
    caches 2*layers x [B, Hkv, S, D]. Output: predicted vector field [B, T, C].
    """

    def __init__(self, expert, action_in_proj, action_out_proj):
        super().__init__()
        self.layers, self.norm = expert.layers, expert.norm
        self.action_in_proj, self.action_out_proj = action_in_proj, action_out_proj

    def forward(self, x, t, cos, sin, mask, caches):
        from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb

        h = self.action_in_proj(x, t)
        for i, layer in enumerate(self.layers):
            attn = layer.self_attn
            B, L, _ = h.shape
            D = attn.head_dim
            n = layer.input_layernorm(h)
            q = attn.q_norm(attn.q_proj(n).view(B, L, -1, D)).transpose(1, 2)
            k = attn.k_norm(attn.k_proj(n).view(B, L, -1, D)).transpose(1, 2)
            v = attn.v_proj(n).view(B, L, -1, D).transpose(1, 2)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
            k = torch.cat([caches[2 * i], k], dim=2)
            v = torch.cat([caches[2 * i + 1], v], dim=2)
            # after the concat, as in the HF layer: the expert's quantizers see the whole K/V
            q, k, v = bmm_quantize(attn, q, k, v)
            out = _folded_gqa_sdpa(q, k, v, mask, attn.scaling)
            h = h + attn.o_proj(out.transpose(1, 2).reshape(B, L, -1))
            h = h + layer.mlp(layer.post_attention_layernorm(h))
        return self.action_out_proj(self.norm(h))


# %%
# Building the engines
# ^^^^^^^^^^^^^^^^^^^^
def trt_input(shape, dtype, min_shape=None, max_shape=None):
    """Static input unless a min/max range is given (``shape`` is then the optimization shape)."""
    if min_shape is None:
        return torch_tensorrt.Input(shape=shape, dtype=dtype)
    return torch_tensorrt.Input(
        min_shape=min_shape, opt_shape=shape, max_shape=max_shape, dtype=dtype
    )


def export_context(quantized: bool):
    """ModelOpt's export mode turns the quantizers into ``tensorrt::quantize_op`` Q/DQ nodes."""
    if not quantized:
        return contextlib.nullcontext()
    from modelopt.torch.quantization.utils import export_torch_mode

    return export_torch_mode()


def compile_and_save(build: Callable, out_path: Path, quantized: bool = False):
    """``build()`` -> (module, example_args, dynamic_shapes, trt_inputs).

    The builder owns the only references to the large module: it is dropped before ``save`` so host
    RAM does not hold the fp16 weights, the exported program and the serialized engine at once.
    """
    module, example_args, dynamic_shapes, trt_inputs = build()
    with torch.no_grad():
        with export_context(quantized):
            ep = torch.export.export(
                module, tuple(example_args), dynamic_shapes=dynamic_shapes, strict=False
            )
        trt_mod = torch_tensorrt.dynamo.compile(
            ep,
            inputs=trt_inputs,
            enabled_precisions={torch.float16},
            use_explicit_typing=False,
            min_block_size=1,
            offload_module_to_cpu=True,
        )
    del module, ep
    gc.collect()
    torch.cuda.empty_cache()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # pickle protocol 4: engines can exceed 4 GiB
    torch_tensorrt.save(trt_mod, str(out_path), retrace=False, pickle_protocol=4)
    print("saved", out_path)


def segment_ranges(num_layers: int, num_segments: int) -> List[Tuple[int, int]]:
    bounds = [round(i * num_layers / num_segments) for i in range(num_segments + 1)]
    return list(zip(bounds[:-1], bounds[1:]))


def build_vision(args):
    model = load_model(args.model).to(DEVICE)
    _, tokenized = load_inputs(model, args)
    pixel_values = tokenized["pixel_values"].to(DEVICE, torch.float16)
    grid_thw = tokenized["image_grid_thw"].to(DEVICE)
    quantized = is_quantized(args.model)
    visual = model.vlm.model.visual.eval()
    prepare_weights(visual, quantized)
    encoder = StaticVisionEncoder(visual, grid_thw).eval()
    del model, visual
    gc.collect()
    compile_and_save(
        lambda: (encoder, (pixel_values,), None, [pixel_values]),
        Path(args.engine_dir) / "vision.ep",
        quantized,
    )


def build_llm_segment(args, layer_range):
    model = load_model(args.model)
    text_model = model.vlm.model.language_model
    num_deepstack = len(model.vlm.config.vision_config.deepstack_visual_indexes)
    lo, hi = layer_range
    first = lo == 0
    assert (
        not first or hi >= num_deepstack
    ), "the first segment must hold all DeepStack layers"
    cls = TextFirstSegment if first else TextSegment
    seg = (
        cls(text_model, model.vlm.lm_head, layer_range, num_deepstack).to(DEVICE).eval()
    )
    prepare_weights(seg, is_quantized(args.model), args.fp8_attention_output_quant)
    H = text_model.config.hidden_size
    del model, text_model
    gc.collect()

    dt = torch.float16
    B_ex, L_ex = 2, 8  # example > 1 so export does not specialize the dynamic dims
    example = (
        torch.randn(B_ex, L_ex, H, device=DEVICE, dtype=dt),
        torch.arange(L_ex, device=DEVICE, dtype=torch.int32)
        .expand(3, B_ex, L_ex)
        .contiguous(),
    ) + (
        (torch.zeros(num_deepstack, 1, L_ex, H, device=DEVICE, dtype=dt),)
        if first
        else ()
    )
    batch = torch.export.Dim("batch", min=1, max=args.num_traj_samples)
    seq = torch.export.Dim("seq", min=1, max=args.chunk_len)
    dyn = ({0: batch, 1: seq}, {1: batch, 2: seq}) + (({2: seq},) if first else ())
    Bmax, C, Lopt = args.num_traj_samples, args.chunk_len, 1  # tuned for decoding
    inputs = [
        trt_input((Bmax, Lopt, H), dt, (1, 1, H), (Bmax, C, H)),
        trt_input((3, Bmax, Lopt), torch.int32, (3, 1, 1), (3, Bmax, C)),
    ]
    if first:
        inputs.append(
            trt_input(
                (num_deepstack, 1, Lopt, H),
                dt,
                (num_deepstack, 1, 1, H),
                (num_deepstack, 1, C, H),
            )
        )
    return seg, example, dyn, inputs


def build_llm(args):
    # The KV cache is added to every SDPA of the graph being compiled, so these are only enabled
    # for the decoder (the vision and expert engines are built in other processes).
    import static_cache_v2
    from torchtrt_ext import register_sdpa

    register_sdpa.enable_sdpa_converter("default", None)
    static_cache_v2.KV_CACHE_LEN = args.max_seq_len

    model = load_model(args.model)
    num_layers = len(model.vlm.model.language_model.layers)
    del model
    gc.collect()
    for idx, layer_range in enumerate(segment_ranges(num_layers, args.llm_segments)):
        print(f"[llm] segment {idx}: layers {layer_range}")
        compile_and_save(
            lambda r=layer_range: build_llm_segment(args, r),
            Path(args.engine_dir) / f"llm_{idx}.ep",
            is_quantized(args.model),
        )
        gc.collect()


def build_expert(args):
    def build():
        model = load_model(args.model)
        step = (
            ExpertStep(model.expert, model.action_in_proj, model.action_out_proj)
            .to(DEVICE)
            .eval()
        )
        prepare_weights(step, is_quantized(args.model))
        cfg = model.expert.config
        layers = len(model.expert.layers)
        T, C = model.action_space.get_action_space_dims()
        del model
        gc.collect()
        B, S, D, dt = (
            args.num_traj_samples,
            args.max_seq_len,
            cfg.head_dim,
            torch.float16,
        )
        kv = (B, cfg.num_key_value_heads, S, D)
        example = (
            torch.randn(B, T, C, device=DEVICE, dtype=dt),
            torch.zeros(B, 1, 1, device=DEVICE, dtype=dt),
            torch.randn(B, T, D, device=DEVICE, dtype=dt),
            torch.randn(B, T, D, device=DEVICE, dtype=dt),
            torch.zeros(B, 1, T, S + T, device=DEVICE, dtype=dt),
            [torch.zeros(kv, device=DEVICE, dtype=dt) for _ in range(2 * layers)],
        )
        inputs = [
            trt_input(tuple(e.shape), e.dtype)
            for e in torch.utils._pytree.tree_flatten(example)[0]
        ]
        return step, example, None, inputs

    compile_and_save(
        build, Path(args.engine_dir) / "expert.ep", is_quantized(args.model)
    )


def run_build(args):
    if args.component == "all":
        # One process per component: each ends with the host memory fully released.
        for component in ("vision", "llm", "expert"):
            subprocess.run(
                [sys.executable, os.path.abspath(__file__)]
                + sys.argv[1:]
                + ["--component", component],
                check=True,
            )
        return
    {"vision": build_vision, "llm": build_llm, "expert": build_expert}[args.component](
        args
    )


# %%
# PyTorch reference
# ^^^^^^^^^^^^^^^^^
@torch.inference_mode()
def run_reference(args):
    from alpamayo1_5 import helper

    model = load_model(args.model).to(DEVICE)
    data, tokenized = load_inputs(model, args)
    inputs = helper.to_device(
        {
            "tokenized_data": tokenized,
            "ego_history_xyz": data["ego_history_xyz"],
            "ego_history_rot": data["ego_history_rot"],
        },
        DEVICE,
    )
    # Deterministic check for the decoder: logits of the first generated token.
    ids = model.fuse_traj_tokens(
        inputs["tokenized_data"]["input_ids"],
        {
            "ego_history_xyz": inputs["ego_history_xyz"],
            "ego_history_rot": inputs["ego_history_rot"],
        },
    )
    td = {k: v for k, v in inputs["tokenized_data"].items() if k != "input_ids"}
    with torch.autocast(DEVICE.type, dtype=torch.float16):
        first_logits = model.vlm(input_ids=ids, **td).logits[0, -1].float().cpu()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    t = time.perf_counter()
    with torch.autocast(DEVICE.type, dtype=torch.float16):
        pred_xyz, _, extra = model.sample_trajectories_from_data_with_vlm_rollout(
            data=inputs,
            top_p=args.top_p,
            temperature=args.temperature,
            num_traj_samples=args.num_traj_samples,
            max_generation_length=args.max_generation_length,
            diffusion_kwargs={"inference_step": args.diffusion_steps},
            return_extra=True,
        )
    ms = (time.perf_counter() - t) * 1e3
    print(f"[reference] {ms:.0f} ms, minADE {min_ade(pred_xyz, data):.3f} m")
    for k, v in extra.items():
        print(f"[reference] {k}[0]: {str(v.flatten()[0])[:300]}")
    if args.save:
        torch.save(
            {"first_logits": first_logits, "pred_xyz": pred_xyz.cpu(), "ms": ms},
            args.save,
        )


# %%
# TensorRT pipeline
# ^^^^^^^^^^^^^^^^^
class TrtAlpamayo:
    def __init__(self, model, args):
        from alpamayo1_5.models.token_utils import to_special_token

        self.model, self.args = model, args
        self.S = args.max_seq_len
        self.engine_dir, self.resident, self._engines = (
            Path(args.engine_dir),
            args.resident,
            {},
        )
        self.llm_names = [
            p.stem
            for p in sorted(
                self.engine_dir.glob("llm_*.ep"),
                key=lambda p: int(p.stem.split("_")[1]),
            )
        ]
        # Host-side pieces: the embedding table and the expert's rotary tables (small).
        lm = model.vlm.model.language_model
        self.embed, self.expert_rotary = lm.embed_tokens, model.expert.rotary_emb
        for m in (self.embed, self.expert_rotary):
            cast_to_fp16(m)
            m.to(DEVICE)
        # Everything heavy lives in the engines; drop the CPU copies (~20 GB) of the weights.
        model.vlm.model.visual = torch.nn.Identity()
        lm.layers = torch.nn.ModuleList()
        model.vlm.lm_head = torch.nn.Identity()
        model.expert.layers = torch.nn.ModuleList()
        gc.collect()
        cfg = model.vlm.config.text_config
        self.layers, self.H = cfg.num_hidden_layers, cfg.hidden_size
        self.segment_ranges = segment_ranges(self.layers, len(self.llm_names))
        self.kv_shape = (cfg.num_key_value_heads, self.S, cfg.head_dim)
        self.image_token_id = model.vlm.config.image_token_id
        self.eos_id = model.tokenizer.convert_tokens_to_ids(
            to_special_token("traj_future_start")
        )
        self.pad_id = model.tokenizer.pad_token_id
        gen_eos = model.vlm.generation_config.eos_token_id
        self.gen_eos = torch.tensor(
            gen_eos if isinstance(gen_eos, list) else [gen_eos], device=DEVICE
        )

    # ----- helpers -------------------------------------------------------------------
    def engine(self, name):
        if name not in self._engines:
            t = time.perf_counter()
            self._engines[name] = torch.export.load(
                str(self.engine_dir / f"{name}.ep")
            ).module()
            print(f"[run] loaded {name} in {time.perf_counter() - t:.1f}s")
        return self._engines[name]

    def release(self, *names):
        if self.resident:
            return
        for n in names:
            self._engines.pop(n, None)
        gc.collect()
        torch.cuda.empty_cache()

    def run_llm(self, hidden, position_ids, caches, deepstack, start, end):
        """Chain the decoder segment engines; returns (logits [B, L, V], all updated caches)."""
        new_caches = []
        position_ids = position_ids.to(torch.int32)
        for i, (name, (lo, hi)) in enumerate(zip(self.llm_names, self.segment_ranges)):
            extra = (
                (deepstack,) if i == 0 else ()
            )  # only the first segment takes DeepStack
            out = self.engine(name)(
                hidden, position_ids, *extra, *caches[2 * lo : 2 * hi], start, end
            )
            hidden, new_caches = out[0], new_caches + list(out[1:])
        return hidden, new_caches

    def _sample(self, logits):
        """ExpertLogitsProcessor -> temperature -> top-p -> multinomial (HF's processor order)."""
        from transformers.generation.logits_process import (
            TemperatureLogitsWarper,
            TopPLogitsWarper,
        )

        c = self.model.config
        logits = logits.float()
        logits[
            :, c.traj_token_start_idx : c.traj_token_start_idx + c.traj_vocab_size
        ] = -float("inf")
        logits = TemperatureLogitsWarper(self.args.temperature)(None, logits)
        logits = TopPLogitsWarper(self.args.top_p)(None, logits)
        return torch.multinomial(torch.softmax(logits, -1), 1)[:, 0]

    # ----- stages --------------------------------------------------------------------
    def prefill(self, input_ids, tokenized):
        pv = tokenized["pixel_values"].to(DEVICE, torch.float16)
        grid = tokenized["image_grid_thw"].to(DEVICE)
        img, ds = self.engine("vision")(pv)
        self.release("vision")
        embeds = self.embed(input_ids)
        img_mask = input_ids == self.image_token_id
        embeds = embeds.masked_scatter(
            img_mask[..., None].expand_as(embeds), img.to(embeds.dtype)
        )
        deepstack = torch.zeros(
            len(ds), 1, input_ids.shape[1], self.H, device=DEVICE, dtype=torch.float16
        )
        for i, d in enumerate(ds):
            deepstack[i, 0, img_mask[0]] = d.to(torch.float16)

        position_ids, rope_deltas = self.model.vlm.model.get_rope_index(
            input_ids, grid, None, tokenized["attention_mask"].to(DEVICE)
        )
        L = input_ids.shape[1]
        caches = [
            torch.zeros(1, *self.kv_shape, device=DEVICE, dtype=torch.float16)
            for _ in range(2 * self.layers)
        ]
        # Chunked prefill: each chunk attends to the cache written by the previous ones.
        for s in range(0, L, self.args.chunk_len):
            e = min(s + self.args.chunk_len, L)
            logits, caches = self.run_llm(
                embeds[:, s:e],
                position_ids[:, :, s:e],
                caches,
                deepstack[:, :, s:e],
                s,
                e,
            )
        return logits[:, -1], caches, rope_deltas

    def generate(self, input_ids, logits, caches, rope_deltas):
        """Sampled decoding for N samples; mirrors HF generate + StopAfterEOS."""
        N, L = self.args.num_traj_samples, input_ids.shape[1]
        caches = [c.expand(N, -1, -1, -1).contiguous() for c in caches]
        seqs = input_ids.expand(N, -1)
        logits = logits.expand(N, -1)
        finished = torch.zeros(N, dtype=torch.bool, device=DEVICE)
        eos_found = torch.zeros(N, dtype=torch.bool, device=DEVICE)
        zero_ds = torch.zeros(3, 1, 1, self.H, device=DEVICE, dtype=torch.float16)
        for step in range(self.args.max_generation_length):
            tok = self._sample(logits)
            tok = torch.where(finished, torch.full_like(tok, self.pad_id), tok)
            seqs = torch.cat([seqs, tok[:, None]], dim=1)
            finished |= torch.isin(tok, self.gen_eos)
            # StopAfterEOS: stop once every sample has emitted EOS *and* one more token was fed,
            # because the token's KV entry is only written when it is fed to the model.
            all_eos_before = bool(eos_found.all())
            eos_found |= tok == self.eos_id
            if (
                all_eos_before
                or bool(finished.all())
                or step == self.args.max_generation_length - 1
            ):
                break
            pos = L + step  # cache slot of `tok`
            pid = (pos + rope_deltas).to(DEVICE).expand(3, N, 1)
            logits, caches = self.run_llm(
                self.embed(tok[:, None]), pid, caches, zero_ds, pos, pos + 1
            )
            logits = logits[:, 0]
        return seqs, caches

    def expert_rollout(self, seqs, caches, rope_deltas, data):
        model, N = self.model, self.args.num_traj_samples
        T, C = model.action_space.get_action_space_dims()
        offset = model._find_eos_offset(seqs, self.eos_id, DEVICE)  # [N]
        pos = torch.arange(T, device=DEVICE)[None, None].expand(3, N, T).clone()
        pos += (rope_deltas.to(DEVICE) + offset[:, None]).to(pos.dtype)
        cos, sin = self.expert_rotary(
            torch.zeros(1, device=DEVICE, dtype=torch.float16), pos
        )
        cos, sin = cos.to(torch.float16), sin.to(torch.float16)
        # additive mask [N, 1, T, S+T]: slots < offset visible, the gap up to the diffusion tokens
        # (and every unwritten slot) hidden, diffusion tokens fully visible (non-causal expert).
        slots = torch.arange(self.S + T, device=DEVICE)[None, :]
        visible = (slots < offset[:, None]) | (slots >= self.S)
        mask = torch.where(visible, 0.0, MASK_MIN).to(torch.float16)[:, None, None, :]
        mask = mask.expand(N, 1, T, self.S + T).contiguous()

        expert = self.engine("expert")

        def step_fn(x, t):
            return expert(
                x.to(torch.float16),
                t.to(torch.float16).contiguous(),
                cos,
                sin,
                mask,
                caches,
            ).float()

        action = model.diffusion.sample(
            batch_size=N,
            step_fn=step_fn,
            device=DEVICE,
            return_all_steps=False,
            inference_step=self.args.diffusion_steps,
        )
        hist_xyz = data["ego_history_xyz"][:, -1].repeat_interleave(N, 0).to(DEVICE)
        hist_rot = data["ego_history_rot"][:, -1].repeat_interleave(N, 0).to(DEVICE)
        pred_xyz, _ = model.action_space.action_to_traj(action, hist_xyz, hist_rot)
        self.release("expert")
        return pred_xyz.reshape(1, 1, N, *pred_xyz.shape[1:])

    @torch.inference_mode()
    def __call__(self, data, tokenized):
        from alpamayo1_5.models.token_utils import replace_padding_after_eos

        timings = {}

        def tick(name, t0):
            torch.cuda.synchronize()
            timings[name] = (time.perf_counter() - t0) * 1e3
            return time.perf_counter()

        t = time.perf_counter()
        traj = {
            "ego_history_xyz": data["ego_history_xyz"].to(DEVICE),
            "ego_history_rot": data["ego_history_rot"].to(DEVICE),
        }
        input_ids = self.model.fuse_traj_tokens(tokenized["input_ids"].to(DEVICE), traj)
        logits, caches, rope_deltas = self.prefill(input_ids, tokenized)
        self.first_logits = logits[0].float().cpu()
        t = tick("vision+prefill", t)
        seqs, caches = self.generate(input_ids, logits, caches, rope_deltas)
        self.release(*self.llm_names)
        t = tick("decode", t)
        seqs = replace_padding_after_eos(seqs, self.eos_id, self.pad_id)
        pred_xyz = self.expert_rollout(seqs, caches, rope_deltas, data)
        tick("expert", t)
        return pred_xyz, seqs, timings, seqs.shape[1] - input_ids.shape[1]


@torch.inference_mode()
def run_trt(args):
    from alpamayo1_5.models.token_utils import extract_text_tokens

    # CPU: only the tokenizer / host-side pieces are used
    model = load_model(args.model)
    data, tokenized = load_inputs(model, args)
    pipe = TrtAlpamayo(model, args)

    # 1st call includes lazy engine/context setup
    for i in range(2 if args.resident else 1):
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        t = time.perf_counter()
        pred_xyz, seqs, timings, n_gen = pipe(data, tokenized)
        torch.cuda.synchronize()
        total = (time.perf_counter() - t) * 1e3
        label = "warmup" if i == 0 else "run"
        print(
            f"[trt:{label}] {total:.0f} ms total; {n_gen} tokens; "
            + ", ".join(f"{k} {v:.0f} ms" for k, v in timings.items())
        )
    print(f"[trt] minADE {min_ade(pred_xyz, data):.3f} m")
    for k, v in extract_text_tokens(model.tokenizer, seqs).items():
        print(f"[trt] {k}[0]: {str(v[0])[:300]}")

    if args.ref:
        ref = torch.load(args.ref)
        a, b = pipe.first_logits, ref["first_logits"]
        cos = F.cosine_similarity(a, b, dim=0).item()
        top5 = lambda x: x.topk(5).indices.tolist()  # noqa: E731
        print(
            f"[compare] first-token logits cosine {cos:.5f}; top5 trt {top5(a)} ref {top5(b)}"
        )
        print(
            f"[compare] reference {ref['ms']:.0f} ms, minADE {min_ade(ref['pred_xyz'], data):.3f} m"
        )


if __name__ == "__main__":
    a = get_args()
    {"build": run_build, "reference": run_reference, "run": run_trt}[a.mode](a)
