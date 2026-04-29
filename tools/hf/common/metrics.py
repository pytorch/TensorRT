"""
Metrics reporting helpers.  Each model family reports in different units:
  - Encoder / classifier : samples/s (throughput) + latency
  - LLM                  : tokens/s
  - Diffusion            : images/s (one full denoising pass)
  - Audio                : real-time factor (audio_duration / inference_time)
"""

from __future__ import annotations

import json


def print_table(rows: list[dict], title: str = "") -> None:
    if not rows:
        return
    if title:
        print(f"\n{'=' * 60}")
        print(f"  {title}")
        print(f"{'=' * 60}")
    keys = list(rows[0].keys())
    widths = {k: max(len(k), max(len(str(r.get(k, ""))) for r in rows)) for k in keys}
    header = "  ".join(k.ljust(widths[k]) for k in keys)
    print(header)
    print("-" * len(header))
    for row in rows:
        print("  ".join(str(row.get(k, "")).ljust(widths[k]) for k in keys))
    print()


def report_latency(stats: dict, backend: str, batch_size: int, precision: str) -> dict:
    row = {
        "backend": backend,
        "precision": precision,
        "batch_size": batch_size,
        "mean_lat_ms": f"{stats['mean_latency_ms']:.2f}",
        "p50_lat_ms": f"{stats['median_latency_ms']:.2f}",
        "p99_lat_ms": f"{stats['p99_latency_ms']:.2f}",
        "throughput": f"{stats['mean_throughput']:.2f}",
    }
    return row


def report_tokens_per_sec(
    total_tokens: int,
    elapsed_s: float,
    backend: str,
    precision: str,
) -> dict:
    tok_s = total_tokens / elapsed_s if elapsed_s > 0 else 0.0
    return {
        "backend": backend,
        "precision": precision,
        "tokens_per_sec": f"{tok_s:.1f}",
        "total_tokens": total_tokens,
        "elapsed_s": f"{elapsed_s:.3f}",
    }


def report_images_per_sec(
    n_images: int,
    elapsed_s: float,
    n_steps: int,
    backend: str,
    precision: str,
) -> dict:
    ips = n_images / elapsed_s if elapsed_s > 0 else 0.0
    step_ms = (elapsed_s / n_steps) * 1000 if n_steps > 0 else 0.0
    return {
        "backend": backend,
        "precision": precision,
        "images_per_sec": f"{ips:.2f}",
        "ms_per_step": f"{step_ms:.1f}",
        "n_steps": n_steps,
    }


def report_rtf(
    audio_duration_s: float,
    inference_s: float,
    backend: str,
    precision: str,
) -> dict:
    rtf = inference_s / audio_duration_s if audio_duration_s > 0 else 0.0
    return {
        "backend": backend,
        "precision": precision,
        "rtf": f"{rtf:.3f}",
        "inference_ms": f"{inference_s * 1000:.1f}",
        "audio_duration_s": f"{audio_duration_s:.2f}",
    }


def report_videos_per_sec(
    n_videos: int,
    elapsed_s: float,
    num_frames: int,
    n_steps: int,
    backend: str,
    precision: str,
) -> dict:
    vps = n_videos / elapsed_s if elapsed_s > 0 else 0.0
    fps = (n_videos * num_frames) / elapsed_s if elapsed_s > 0 else 0.0
    step_ms = (elapsed_s / n_steps) * 1000 if n_steps > 0 else 0.0
    return {
        "backend": backend,
        "precision": precision,
        "videos_per_sec": f"{vps:.3f}",
        "frames_per_sec": f"{fps:.2f}",
        "ms_per_step": f"{step_ms:.1f}",
        "num_frames": num_frames,
        "n_steps": n_steps,
    }


def dump_json(rows: list[dict], path: str) -> None:
    with open(path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"Results written to {path}")
