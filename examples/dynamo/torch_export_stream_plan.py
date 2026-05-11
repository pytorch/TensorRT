"""
SmolVLA + apply_stream_plan + CUDA Green Contexts
==================================================
End-to-end real-world example of Torch-TensorRT's stream-plan system applied
to the lerobot/smolvla_base VLA pipeline.

Topology:

    cam0 (256x256) --> TRT encoder 0  (green ctx 0) --|
    cam1 (256x256) --> TRT encoder 1  (green ctx 1) --|--> VLM --> denoise --> actions
    cam2 (256x256) --> TRT encoder 2  (green ctx 2) --|

Each camera encoder (SigLIP + SmolVLM connector) compiles to its own TRT
engine.  The three engines are then bundled into one FX GraphModule with a
parallel-fan-out graph:

    forward(img0, img1, img2):
        out0 = _run_on_acc_0(img0)
        out1 = _run_on_acc_1(img1)
        out2 = _run_on_acc_2(img2)
        return out0, out1, out2

`apply_stream_plan(wrapper, streams=[g0, g1, g2])` rewrites this graph to:

  - `enter_compute_stream` at entry, `exit_compute_stream` at exit
  - `set_stream(g_i)` before each engine call  (single TLS write per call)
  - `sync_streams(g_i, caller)` events at fan-in for each branch
  - each engine call replaced by `call_trt_with_token(token, engine, img_i)`
    so the FX scheduler is data-flow forced to set the stream before calling

All dispatch happens from one Python thread.  Concurrency on the GPU comes
from green contexts splitting the device's SMs into three partitions, so
each engine occupies a disjoint set of SMs and they run truly in parallel
even when nominally on the same device.

Compared to the manual thread + `with torch.cuda.stream(...)` pattern this
file used to demonstrate, the stream-plan version:
  - eliminates Python threads and per-call CUDA stream context managers
  - produces a real FX GraphModule that downstream tools (torch.compile,
    AOTI lowering, FX visualizers) can consume
  - encodes the stream plan in the graph as effect-ordered ops, so the
    plan survives Inductor scheduling

First run downloads ~4 GB from HuggingFace and compiles three TRT engines.

Usage:
    uv run python examples/dynamo/torch_export_stream_plan.py
"""

from __future__ import annotations

import copy
import sys
import time
from typing import List

import torch
import torch.nn as nn
import torch_tensorrt  # must be imported before lerobot to avoid CUDA library conflicts
from torch_tensorrt.runtime import apply_stream_plan

# ── Green Context helpers ─────────────────────────────────────────────────────


def _green_ctx_available() -> bool:
    try:
        from cuda.bindings import driver as drv

        drv.cuGreenCtxCreate  # noqa: B018
        drv.cuDevSmResourceSplitByCount
        return True
    except (ImportError, AttributeError):
        return False


class GreenCtxStream:
    def __init__(self, green_ctx, raw_stream: int, device_id: int) -> None:
        self._green_ctx = green_ctx
        self.stream = torch.cuda.ExternalStream(
            raw_stream, device=torch.device("cuda", device_id)
        )

    def destroy(self) -> None:
        if self._green_ctx is not None:
            from cuda.bindings import driver as drv

            drv.cuGreenCtxDestroy(self._green_ctx)
            self._green_ctx = None

    def __del__(self) -> None:
        self.destroy()


def create_green_ctx_streams(device_id: int, n: int) -> List[GreenCtxStream]:
    from cuda.bindings import driver as drv

    torch.cuda.init()
    _ = torch.zeros(1, device=torch.device("cuda", device_id))

    err, total = drv.cuDeviceGetDevResource(
        device_id, drv.CUdevResourceType.CU_DEV_RESOURCE_TYPE_SM
    )
    if err != drv.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"cuDeviceGetDevResource failed ({err})")

    err, groups, nb, _ = drv.cuDevSmResourceSplitByCount(n, total, 0, 1)
    if err != drv.CUresult.CUDA_SUCCESS or nb < n:
        raise RuntimeError(
            f"Cannot split {total.sm.smCount} SMs into {n} partitions ({err}, got {nb})"
        )

    result = []
    for i in range(n):
        err, desc = drv.cuDevResourceGenerateDesc([groups[i]], 1)
        if err != drv.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"cuDevResourceGenerateDesc[{i}] failed ({err})")

        err, gctx = drv.cuGreenCtxCreate(
            desc, device_id, drv.CUgreenCtxCreate_flags.CU_GREEN_CTX_DEFAULT_STREAM
        )
        if err != drv.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"cuGreenCtxCreate[{i}] failed ({err})")

        err, ctx = drv.cuCtxFromGreenCtx(gctx)
        drv.cuCtxPushCurrent(ctx)
        err, raw = drv.cuGreenCtxStreamCreate(
            gctx, drv.CUstream_flags.CU_STREAM_NON_BLOCKING, 0
        )
        drv.cuCtxPopCurrent()

        if err != drv.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"cuGreenCtxStreamCreate[{i}] failed ({err})")

        result.append(GreenCtxStream(gctx, int(raw), device_id))
    return result


# ── SmolVLA loading helpers ───────────────────────────────────────────────────

SMOLVLA_MODEL_ID = "lerobot/smolvla_base"


def _stub_groot() -> None:
    """Stub out the GROOT policy which is incompatible with Python 3.13."""
    from unittest.mock import MagicMock

    for mod in [
        "lerobot.policies.groot",
        "lerobot.policies.groot.configuration_groot",
        "lerobot.policies.groot.modeling_groot",
        "lerobot.policies.groot.groot_n1",
    ]:
        if mod not in sys.modules:
            sys.modules[mod] = MagicMock()


def load_smolvla(device: torch.device):
    _stub_groot()
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    print(f"  Loading {SMOLVLA_MODEL_ID} (first run downloads ~4 GB) ...", flush=True)
    policy = SmolVLAPolicy.from_pretrained(SMOLVLA_MODEL_ID).eval()
    # SmolVLA loads with mixed dtypes (VLM: bfloat16, action projections: float32).
    # Move to device only — forcing a single dtype breaks the denoising loop.
    policy = policy.to(device=device)
    return policy


# ── TRT camera encoder ────────────────────────────────────────────────────────


class CameraEncoderModule(nn.Module):
    """SigLIP vision encoder + SmolVLM connector as a single compilable module.

    Input:  (B, 3, H, W) bfloat16 in [-1, 1]
    Output: (B, num_tokens, hidden_size) bfloat16
    """

    def __init__(self, vlm_with_expert) -> None:
        super().__init__()
        vlm = vlm_with_expert.get_vlm_model()
        self.vision_model = vlm.vision_model
        self.connector = vlm.connector

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        hidden = self.vision_model(
            pixel_values=pixel_values, patch_attention_mask=None
        ).last_hidden_state
        return self.connector(hidden)


def compile_camera_encoders(
    policy,
    n_cameras: int,
    img_size: tuple[int, int],
    device: torch.device,
) -> list:
    """Compile one TRT engine per camera from the same SigLIP + connector weights."""
    H, W = img_size
    inp = torch_tensorrt.Input(shape=(1, 3, H, W), dtype=torch.bfloat16)
    trt_kw = dict(
        ir="dynamo",
        min_block_size=1,
        device=device,
        cache_built_engines=False,
        reuse_cached_engines=False,
    )

    encoders = []
    for i in range(n_cameras):
        src = CameraEncoderModule(policy.model.vlm_with_expert).eval()
        if i > 0:
            # Independent engine with its own copy of weights
            src = copy.deepcopy(src)
        print(f"  Compiling camera {i} TRT engine ...", flush=True)
        enc = torch_tensorrt.compile(src, inputs=[inp], **trt_kw)
        encoders.append(enc)

    return encoders


# ── Stream-plan wrapper ───────────────────────────────────────────────────────


def _extract_single_trt_submodule(compiled: torch.fx.GraphModule) -> torch.nn.Module:
    """Pull the single TRT submodule out of a torch_tensorrt-compiled GM.

    For a fully-TRT-compatible model, ``torch_tensorrt.compile`` produces a
    GraphModule whose forward is a single passthrough call to one TRT
    submodule named ``_run_on_acc_0``.  This helper validates that shape and
    returns the inner submodule.  Raises if the encoder has any extra
    pre/post-processing nodes that would be lost by the extraction.
    """
    trt_subs = [
        (name, mod) for name, mod in compiled.named_children() if "_run_on_acc" in name
    ]
    if len(trt_subs) != 1:
        raise RuntimeError(
            f"expected exactly 1 TRT submodule per compiled encoder, got "
            f"{[n for n, _ in trt_subs]} — model is partially fallback to PyTorch?"
        )
    name, submod = trt_subs[0]

    call_modules = [n for n in compiled.graph.nodes if n.op == "call_module"]
    if len(call_modules) != 1 or call_modules[0].target != name:
        raise RuntimeError(
            f"compiled encoder forward is not a single TRT call — extracting "
            f"would drop pre/post-processing ops in the outer graph"
        )
    return submod


def build_stream_planned_encoders(
    trt_encoders: list,
    streams: list,
) -> torch.fx.GraphModule:
    """Bundle N independently-compiled TRT encoders into one stream-planned
    GraphModule.

    Builds an FX graph:
        forward(img0, ..., imgN-1) =
            (_run_on_acc_0(img0), ..., _run_on_acc_{N-1}(imgN-1))

    Then applies ``apply_stream_plan`` so the i-th branch runs on streams[i],
    with cross-stream syncs at fan-in.
    """
    if len(trt_encoders) != len(streams):
        raise ValueError(
            f"need one stream per encoder: got {len(trt_encoders)} encoders "
            f"and {len(streams)} streams"
        )

    parent = nn.Module()
    for i, enc in enumerate(trt_encoders):
        submod = _extract_single_trt_submodule(enc)
        parent.add_module(f"_run_on_acc_{i}", submod)

    g = torch.fx.Graph()
    placeholders = [g.placeholder(f"img{i}") for i in range(len(trt_encoders))]
    outputs = [
        g.call_module(f"_run_on_acc_{i}", args=(placeholders[i],))
        for i in range(len(trt_encoders))
    ]
    g.output(tuple(outputs))

    wrapper = torch.fx.GraphModule(parent, g)
    return apply_stream_plan(wrapper, streams=streams)


# ── Inference runner ──────────────────────────────────────────────────────────


class StreamPlanVLARunner:
    """Wraps SmolVLA to dispatch the camera encoders through a stream-planned
    GraphModule, then fan in to the VLM backbone and denoising loop unchanged.

    Single Python thread; GPU-level concurrency comes from the green contexts
    underlying the streams baked into the stream plan.
    """

    def __init__(self, policy, planned_encoders: torch.fx.GraphModule) -> None:
        self.policy = policy
        self.planned_encoders = planned_encoders
        self._cache: list | None = None
        self._cache_idx: int = 0

    def _run_cameras_parallel(self, images: list) -> list:
        bf16 = [img.to(dtype=torch.bfloat16) for img in images]
        with torch.no_grad():
            out = self.planned_encoders(*bf16)
        return list(out) if isinstance(out, (tuple, list)) else [out]

    def _patched_embed_image(self, _image: torch.Tensor) -> torch.Tensor:
        emb = self._cache[self._cache_idx]
        self._cache_idx += 1
        return emb

    @torch.no_grad()
    def predict_action_chunk(
        self, batch: dict, noise: torch.Tensor | None = None
    ) -> torch.Tensor:
        policy = self.policy
        vlm_with_expert = policy.model.vlm_with_expert

        images, _img_masks = policy.prepare_images(batch)

        # ── Parallel camera encoding via stream plan ──────────────────────────
        self._cache = self._run_cameras_parallel(images)
        self._cache_idx = 0

        original_embed_image = vlm_with_expert.embed_image
        vlm_with_expert.embed_image = self._patched_embed_image

        try:
            actions = policy.predict_action_chunk(batch, noise=noise)
        finally:
            vlm_with_expert.embed_image = original_embed_image
            self._cache = None

        return actions


# ── Batch preparation ─────────────────────────────────────────────────────────


def make_batch(policy, device: torch.device) -> dict:
    """Build a synthetic but correctly-shaped batch for one-shot inference."""
    from lerobot.utils.constants import (
        OBS_LANGUAGE_ATTENTION_MASK,
        OBS_LANGUAGE_TOKENS,
        OBS_STATE,
    )

    cfg = policy.config
    tok_len = cfg.tokenizer_max_length

    proc = policy.model.vlm_with_expert.processor
    enc = proc(text="pick up the red cube and place it in the bin", return_tensors="pt")
    raw_ids = enc["input_ids"][0][:tok_len]
    ids = torch.zeros(tok_len, dtype=torch.long, device=device)
    ids[: len(raw_ids)] = raw_ids.to(device)
    lang_mask = torch.zeros(tok_len, dtype=torch.bool, device=device)
    lang_mask[: len(raw_ids)] = True

    batch = {
        OBS_LANGUAGE_TOKENS: ids.unsqueeze(0),
        OBS_LANGUAGE_ATTENTION_MASK: lang_mask.unsqueeze(0),
        OBS_STATE: torch.zeros(
            1, cfg.max_state_dim, device=device, dtype=torch.float32
        ),
    }

    for key, feat in cfg.image_features.items():
        _, H, W = feat.shape
        batch[key] = torch.rand(1, 3, H, W, device=device, dtype=torch.float32)

    return batch


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    if not torch.cuda.is_available():
        print("CUDA not available — exiting.")
        sys.exit(1)

    device_id = 0
    device = torch.device("cuda", device_id)

    print("=" * 60)
    print("  SmolVLA  —  apply_stream_plan + CUDA Green Contexts + TRT")
    print("=" * 60)
    print()

    # ── Load model ────────────────────────────────────────────────────────────
    print("Loading SmolVLA ...", flush=True)
    policy = load_smolvla(device)
    cfg = policy.config
    img_keys = list(cfg.image_features.keys())
    img_size = cfg.resize_imgs_with_padding  # (512, 512)
    n_cameras = len(img_keys)
    action_dim = cfg.action_feature.shape[0]

    print(f"  cameras ({n_cameras}): {img_keys}")
    print(f"  resize to: {img_size[0]}x{img_size[1]}")
    print(
        f"  state_dim: {cfg.max_state_dim}  action_dim: {action_dim}  "
        f"chunk: {cfg.chunk_size}"
    )
    print()

    # ── Acquire green context streams (one per camera) ────────────────────────
    use_green_ctx = _green_ctx_available()
    if use_green_ctx:
        print(
            f"CUDA Green Contexts available — splitting SMs into {n_cameras} partitions"
        )
        gcs_list = create_green_ctx_streams(device_id, n_cameras)
        streams = [gcs.stream for gcs in gcs_list]
        stream_label = "green ctx"
    else:
        print("CUDA Green Contexts not available — using plain CUDA streams")
        streams = [torch.cuda.Stream(device=device) for _ in range(n_cameras)]
        gcs_list = []
        stream_label = "CUDA stream"

    print("  Stream handles:")
    for i, s in enumerate(streams):
        print(f"    {stream_label} {i}: 0x{s.cuda_stream:x}")
    print()

    # ── Compile TRT camera encoders ───────────────────────────────────────────
    print(f"Compiling {n_cameras} TRT camera encoders ...", flush=True)
    trt_encoders = compile_camera_encoders(policy, n_cameras, img_size, device)
    print()

    # ── Bundle into a stream-planned GraphModule ──────────────────────────────
    print("Building stream-planned encoder bundle ...", flush=True)
    planned_encoders = build_stream_planned_encoders(trt_encoders, streams)
    n_trt = sum(
        1
        for n in planned_encoders.graph.nodes
        if n.op == "call_function"
        and getattr(n.target, "__name__", "") == "call_trt_with_token"
    )
    print(
        f"  Stream plan applied across {n_trt} engine calls on {len(streams)} streams"
    )
    print()

    # ── Build runner + batch ──────────────────────────────────────────────────
    runner = StreamPlanVLARunner(policy, planned_encoders)
    batch = make_batch(policy, device)

    # Fixed noise so reference and TRT runs are deterministically comparable
    noise = torch.randn(
        1,
        policy.config.chunk_size,
        policy.config.max_action_dim,
        dtype=torch.float32,
        device=device,
    )

    # ── Reference: sequential inference via plain policy ─────────────────────
    print("Running reference (sequential, PyTorch vision) ...", flush=True)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        ref_actions = policy.predict_action_chunk(batch, noise=noise.clone())
    torch.cuda.synchronize()
    t_ref = time.perf_counter() - t0
    print(f"  actions shape: {ref_actions.shape}  (took {t_ref*1e3:.1f} ms)")
    print()

    # ── Stream-planned: parallel TRT camera encoders ──────────────────────────
    print("Running with apply_stream_plan + green ctx + TRT ...", flush=True)
    _ = runner.predict_action_chunk(batch)  # warmup
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    trt_actions = runner.predict_action_chunk(batch, noise=noise.clone())
    torch.cuda.synchronize()
    t_trt = time.perf_counter() - t0
    print(f"  actions shape: {trt_actions.shape}  (took {t_trt*1e3:.1f} ms)")
    print()

    # ── Verify ────────────────────────────────────────────────────────────────
    print("Verifying outputs ...", flush=True)
    cos = torch.nn.functional.cosine_similarity(
        ref_actions.flatten().unsqueeze(0),
        trt_actions.flatten().unsqueeze(0),
    ).item()
    max_diff = (ref_actions - trt_actions).abs().max().item()
    print(f"  cosine similarity: {cos:.6f}  (max abs diff: {max_diff:.4f})")
    if cos > 0.99:
        print("  Output matches reference")
    else:
        print("  WARNING: outputs diverge — check TRT precision settings")
    print()

    # ── Action summary ────────────────────────────────────────────────────────
    print("First 5 denoised action steps (6D):")
    for step in range(5):
        vals = trt_actions[0, step].tolist()
        print(f"  step {step}: {[f'{v:+.4f}' for v in vals]}")
    print()

    # ── Cleanup ───────────────────────────────────────────────────────────────
    for gcs in gcs_list:
        gcs.destroy()

    print("Done.")


if __name__ == "__main__":
    main()
