"""
.. _global_perf_tuning_attention_example:

Global Performance Tuning (Attention)
=====================================

This example shows how to use TensorRT's Global Performance Tuning (GPT) from
Torch-TensorRT: discover knobs, sweep a small route space on a simple
multi-head attention module, and apply the winning ``build_route``.

**Requirements:** TensorRT with GPT enabled. This feature is available since
TensorRT 11.1 and is currently not available in TensorRT-RTX or Windows. The
script exits early if GPT is not available.
"""

# %%
# Imports and model
# ^^^^^^^^^^^^^^^^^

import json
import os
import tempfile

import torch
import torch.nn as nn
import torch_tensorrt
from torch_tensorrt.dynamo import get_all_build_routes, is_global_perf_tuning_available


class SimpleAttention(nn.Module):
    """Minimal self-attention block suitable for a short GPT demo."""

    def __init__(self, embed_dim: int = 64, num_heads: int = 4) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True, bias=True
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, _ = self.attn(x, x, x, need_weights=False)
        return self.norm(x + y)


def main() -> None:
    torch.manual_seed(0)
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this example.")
    if not is_global_perf_tuning_available():
        raise SystemExit(
            "Global Performance Tuning is not available on this TensorRT build "
            "(IBuilderConfig.build_route / all_build_routes)."
        )

    batch, seq, embed = 1, 32, 64
    model = SimpleAttention(embed_dim=embed, num_heads=4).eval().cuda().half()
    example_inputs = [
        torch.randn((batch, seq, embed), device="cuda", dtype=torch.float16)
    ]

    # Discover knobs (``trtexec --helpBuildRoute``).
    knobs = get_all_build_routes()
    print(f"tuner_version={knobs.get('tuner_version')}")
    print(f"num_knobs={len(knobs.get('tuner_options', []))}")

    tune_expr = (
        "-match_ragged_mha=[on|off] "
        "-kgen:codegen:cuda_tile=[0|1|2|3] "
        "-cask_fusion:num_tactics=[10|20] "
        "-peep:max=[1|2]"
    )
    print(f"tune_build_routes={tune_expr}")

    # Sweep routes (``trtexec --tuneBuildRoutes``).
    cache_base = os.path.join(
        tempfile.gettempdir(), "torch_trt_attention_tune_example.jsonl"
    )
    cache_dir = os.path.dirname(cache_base) or "."
    cache_stem = os.path.splitext(os.path.basename(cache_base))[0]
    # Clear cache if you don't want to continue tuning
    # for name in os.listdir(cache_dir):
    #     if name.startswith(cache_stem) and name.endswith(".jsonl"):
    #         os.remove(os.path.join(cache_dir, name))
    print(f"tuning_cache_file base={cache_base}")

    optimized = torch_tensorrt.compile(
        model,
        ir="dynamo",
        arg_inputs=example_inputs,
        min_block_size=1,
        tune_build_routes=tune_expr,  # tuning_continue=True,
        tuning_search="mixed",
        accuracy_threshold=0.01,
        accuracy_algorithm="cos",
        tuning_cache_file=cache_base,
    )

    with torch.no_grad():
        ref = model(*example_inputs)
        out = optimized(*example_inputs)

    cos = torch.nn.functional.cosine_similarity(
        ref.flatten().float(), out.flatten().float(), dim=0
    ).item()
    print(f"cosine_similarity(torch_eager, tuned_trt)={cos:.6f}")

    partition_caches = sorted(
        os.path.join(cache_dir, name)
        for name in os.listdir(cache_dir)
        if name.startswith(cache_stem + ".") and name.endswith(".jsonl")
    )
    assert partition_caches, f"expected per-partition cache under {cache_base}"
    cache_path = partition_caches[0]
    print(f"partition cache_path={cache_path}")

    with open(cache_path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    header = json.loads(lines[0])
    print("cache header keys:", sorted(header.keys()))
    print(f"recorded iterations: {len(lines) - 1}")
    for line in lines[1:]:
        row = json.loads(line)
        print(
            f"  iter={row['iter']} crash={row['crash']} "
            f"gpu_time={row.get('gpu_time')} "
            f"route={row.get('build_route')}"
        )

    best_route = ""
    best_time = None
    for line in lines[1:]:
        row = json.loads(line)
        trial_time = row.get("gpu_time")
        if row.get("crash") or trial_time is None:
            continue
        if best_time is None or trial_time < best_time:
            best_time = trial_time
            best_route = row["build_route"]

    print(f"best cached route={best_route!r} gpu_time_ms={best_time}")
    if best_route:
        pinned = torch_tensorrt.compile(
            model,
            ir="dynamo",
            arg_inputs=example_inputs,
            require_full_compilation=True,
            min_block_size=1,
            build_route=best_route,
        )
        with torch.no_grad():
            pinned_out = pinned(*example_inputs)
        print(
            "pinned route max abs err vs torch eager:",
            (pinned_out.float() - ref.float()).abs().max().item(),
        )


if __name__ == "__main__":
    main()
