"""
3D Rotary Position Embedding (RoPE) + Attention compiled with Torch-TensorRT
=============================================================================

3D RoPE is the positional encoding used in video generation transformers such
as CogVideoX, Wan, and HunyuanVideo.  Unlike 1D RoPE (used in language models)
which encodes a single sequence index, 3D RoPE independently encodes three
axes — temporal (T), height (H), and width (W) — and assigns each axis a
dedicated slice of the per-head frequency vector:

    head-dim slots  0 ..  d//3-1  → temporal  frequencies
    head-dim slots  d//3.. 2d//3-1 → height    frequencies
    head-dim slots  2d//3.. d//2-1 → width     frequencies

The rotation is expressed with complex arithmetic:

    xq_rotated = view_as_real(view_as_complex(xq) * freqs_cis)

PyTorch complex ops (view_as_complex, complex mul) are not natively supported
by TensorRT.  Torch-TensorRT's ``complex_graph_detection`` lowering pass
intercepts them before partitioning and rewrites the subgraph to equivalent
real arithmetic — splitting the last dimension into (..., 2) real/imag pairs
and computing (ac-bd, ad+bc) manually — so the TRT engine only sees standard
float32 ops and the caller never needs to change anything.

This example:
  1. Defines a 3D-RoPE frequency precomputation helper (complex64 output).
  2. Defines a VideoAttentionBlock: linear QKV projection → 3D RoPE → SDPA.
  3. Runs a PyTorch baseline forward pass.
  4. Exports with torch.export.export() and dynamic T/H/W dimensions.
  5. Compiles to TensorRT via torch_tensorrt.dynamo.compile().
  6. Verifies numerical accuracy (cosine similarity on the output tensor).
  7. (Optional) benchmarks latency of both backends.

Usage
-----
# Quick correctness check (static shapes)
python examples/dynamo/torch_export_3d_rope.py

# Dynamic T/H/W shapes
python examples/dynamo/torch_export_3d_rope.py --dynamic

# Larger config + benchmark
python examples/dynamo/torch_export_3d_rope.py --heads 16 --head-dim 96 --t 8 --h 16 --w 16 --benchmark
"""

import argparse
import timeit

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_tensorrt
from torch.export import Dim

DEVICE = torch.device("cuda:0")


# ---------------------------------------------------------------------------
# Frequency precomputation
# ---------------------------------------------------------------------------


def precompute_freqs_3d(
    head_dim: int,
    t: int,
    h: int,
    w: int,
    theta: float = 10000.0,
) -> torch.Tensor:
    """Pre-compute 3D RoPE unit-complex frequency tensor.

    Returns a complex64 tensor of shape (t, h, w, head_dim // 2) where the
    last dimension is split evenly across the three spatial axes.

    Args:
        head_dim: Channels per attention head (must be even, head_dim//2
                  must be divisible by 3).
        t: Number of temporal frames.
        h: Spatial height in patches.
        w: Spatial width in patches.
        theta: Base for the geometric frequency progression.
    """
    half = head_dim // 2
    d_t = half // 3
    d_h = half // 3
    d_w = half - d_t - d_h  # absorbs any remainder from integer division

    def _axis_freqs(d: int, n: int) -> torch.Tensor:
        """1-D complex exponentials, shape (n, d)."""
        inv_freq = 1.0 / (theta ** (torch.arange(0, d * 2, 2).float() / (d * 2)))
        positions = torch.arange(n, dtype=torch.float32)
        angles = torch.outer(positions, inv_freq)
        return torch.polar(torch.ones_like(angles), angles)  # complex64

    freqs_t = _axis_freqs(d_t, t)[:, None, None, :].expand(t, h, w, d_t)
    freqs_h = _axis_freqs(d_h, h)[None, :, None, :].expand(t, h, w, d_h)
    freqs_w = _axis_freqs(d_w, w)[None, None, :, :].expand(t, h, w, d_w)

    # Concatenate along last dim → (t, h, w, half), complex64
    return torch.cat([freqs_t, freqs_h, freqs_w], dim=-1).contiguous()


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class VideoAttentionBlock(nn.Module):
    """Single attention block for video latents with 3D RoPE.

    Inputs
    ------
    x             : (B, T, H, W, C)  float32  video patch features
    freqs_cis_real: (T, H, W, C // n_heads)  float32
        The RoPE frequency tensor pre-flattened from complex64 via
        ``view_as_real(...).flatten(-2)``.  The module reconstructs the
        complex form internally with ``view_as_complex``.

        Passing frequencies as a plain real-valued input avoids exposing a
        complex tensor at the model boundary (TRT inputs must be real).

    Output
    ------
    (B, T, H, W, C)  float32
    """

    def __init__(self, channels: int = 512, n_heads: int = 8) -> None:
        super().__init__()
        assert channels % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = channels // n_heads
        self.norm = nn.LayerNorm(channels)
        self.qkv = nn.Linear(channels, 3 * channels, bias=False)
        self.proj = nn.Linear(channels, channels, bias=False)

    def _apply_rope(
        self, x: torch.Tensor, freqs_cis: torch.Tensor
    ) -> torch.Tensor:
        """Apply 3D RoPE to a single Q or K tensor.

        The complex multiply ``xc * freqs_cis`` is what Torch-TensorRT rewrites
        to real arithmetic via the complex_graph_detection lowering pass.

        Args:
            x        : (B, T, H, W, n_heads, head_dim)  float32
            freqs_cis: (T, H, W, head_dim // 2)  complex64
        Returns:
            Rotated tensor, same shape as ``x``, float32.
        """
        B, T, H, W, Nh, D = x.shape
        # Interpret consecutive pairs of head-dim channels as complex numbers.
        xc = torch.view_as_complex(x.reshape(B, T, H, W, Nh, D // 2, 2))
        # freqs_cis broadcast over batch (dim 0) and head (dim 4).
        freqs = freqs_cis[None, :, :, :, None, :]  # (1, T, H, W, 1, D//2)
        return torch.view_as_real(xc * freqs).flatten(-2)  # (B,T,H,W,Nh,D)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis_real: torch.Tensor,
    ) -> torch.Tensor:
        B, T, H, W, C = x.shape
        Nh, D = self.n_heads, self.head_dim

        h = self.norm(x)
        qkv = self.qkv(h).reshape(B, T, H, W, 3, Nh, D)
        q, k, v = qkv.unbind(dim=4)  # each (B, T, H, W, Nh, D)

        # Recover complex frequencies from the real-valued input.
        # freqs_cis_real: (T, H, W, D) → reshape to (T, H, W, D//2, 2) → complex
        freqs_cis = torch.view_as_complex(
            freqs_cis_real.reshape(T, H, W, D // 2, 2)
        )

        q = self._apply_rope(q, freqs_cis)
        k = self._apply_rope(k, freqs_cis)

        # Flatten spatial dims for attention: (B, Nh, T*H*W, D)
        N = T * H * W
        q = q.reshape(B, N, Nh, D).permute(0, 2, 1, 3)
        k = k.reshape(B, N, Nh, D).permute(0, 2, 1, 3)
        v = v.reshape(B, N, Nh, D).permute(0, 2, 1, 3)

        out = F.scaled_dot_product_attention(q, k, v)  # (B, Nh, N, D)
        out = out.permute(0, 2, 1, 3).reshape(B, T, H, W, C)
        return x + self.proj(out)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_inputs(
    B: int, T: int, H: int, W: int, C: int, n_heads: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (x, freqs_cis_real) on DEVICE."""
    x = torch.randn(B, T, H, W, C, dtype=torch.float32, device=DEVICE)
    freqs_cis = precompute_freqs_3d(C // n_heads, t=T, h=H, w=W).to(DEVICE)
    freqs_cis_real = torch.view_as_real(freqs_cis).flatten(-2)  # (T,H,W,D)
    return x, freqs_cis_real


def benchmark(fn, *args, iterations: int = 20, label: str = "") -> float:
    fn(*args)  # warmup
    torch.cuda.synchronize()
    total = 0.0
    for _ in range(iterations):
        t0 = timeit.default_timer()
        fn(*args)
        torch.cuda.synchronize()
        total += timeit.default_timer() - t0
    avg_ms = total / iterations * 1000
    print(f"[{label}] avg latency over {iterations} iters: {avg_ms:.2f} ms")
    return avg_ms


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(
        description="3D RoPE attention block compiled with Torch-TensorRT"
    )
    p.add_argument("--heads", type=int, default=8, help="Number of attention heads")
    p.add_argument(
        "--head-dim",
        dest="head_dim",
        type=int,
        default=48,
        help="Channels per head. head_dim//2 must be divisible by 3 (default: 48)",
    )
    p.add_argument("--t", type=int, default=4, help="Temporal frames (default: 4)")
    p.add_argument("--h", type=int, default=8, help="Spatial height patches (default: 8)")
    p.add_argument("--w", type=int, default=8, help="Spatial width patches (default: 8)")
    p.add_argument(
        "--dynamic",
        action="store_true",
        help="Export with dynamic T/H/W dims and compile with min/opt/max shapes",
    )
    p.add_argument("--benchmark", action="store_true", help="Benchmark PyTorch vs TRT latency")
    p.add_argument("--iterations", type=int, default=20)
    return p.parse_args()


def main():
    args = parse_args()

    if (args.head_dim // 2) % 3 != 0:
        raise ValueError(
            f"head_dim // 2 = {args.head_dim // 2} must be divisible by 3 "
            "for the T/H/W frequency split.  Try --head-dim 48, 60, 96, or 192."
        )

    B, T, H, W = 1, args.t, args.h, args.w
    C = args.heads * args.head_dim

    print(f"VideoAttentionBlock with 3D RoPE")
    print(f"  heads={args.heads}  head_dim={args.head_dim}  channels={C}")
    print(f"  input shape: ({B}, {T}, {H}, {W}, {C})")

    model = VideoAttentionBlock(channels=C, n_heads=args.heads).eval().to(DEVICE)

    # ------------------------------------------------------------------
    # 1. Build inputs
    # ------------------------------------------------------------------
    x, freqs_cis_real = make_inputs(B, T, H, W, C, args.heads)
    inputs = (x, freqs_cis_real)
    print(f"\n  x shape             : {x.shape}")
    print(f"  freqs_cis_real shape: {freqs_cis_real.shape}")

    # ------------------------------------------------------------------
    # 2. PyTorch baseline
    # ------------------------------------------------------------------
    with torch.inference_mode():
        pyt_out = model(*inputs)
    print(f"\n--- PyTorch baseline ---")
    print(f"  output shape: {pyt_out.shape}  dtype: {pyt_out.dtype}")

    # ------------------------------------------------------------------
    # 3. Export
    # ------------------------------------------------------------------
    print("\nExporting model ...")
    if args.dynamic:
        t_dim = Dim("T", min=1, max=32)
        h_dim = Dim("H", min=4, max=64)
        w_dim = Dim("W", min=4, max=64)
        dynamic_shapes = (
            # x: (B, T, H, W, C)
            {1: t_dim, 2: h_dim, 3: w_dim},
            # freqs_cis_real: (T, H, W, D)
            {0: t_dim, 1: h_dim, 2: w_dim},
        )
        ep = torch.export.export(model, inputs, dynamic_shapes=dynamic_shapes)
        print("  Exported with dynamic T / H / W dimensions.")
    else:
        ep = torch.export.export(model, inputs)
        print("  Exported with static shapes.")

    # ------------------------------------------------------------------
    # 4. Compile with Torch-TensorRT
    #
    # No special flags are required for the complex arithmetic rewrite.
    # The complex_graph_detection lowering pass automatically detects
    # view_as_complex / complex-mul / view_as_real subgraphs and rewrites
    # them to real-arithmetic ops before the TRT engine is built.
    # ------------------------------------------------------------------
    print("\nCompiling with Torch-TensorRT ...")
    D = C // args.heads  # freqs_cis_real last dim
    if args.dynamic:
        trt_inputs = [
            torch_tensorrt.Input(
                min_shape=(B, 1, 4, 4, C),
                opt_shape=(B, T, H, W, C),
                max_shape=(B, 32, 64, 64, C),
                dtype=torch.float32,
            ),
            torch_tensorrt.Input(
                min_shape=(1, 4, 4, D),
                opt_shape=(T, H, W, D),
                max_shape=(32, 64, 64, D),
                dtype=torch.float32,
            ),
        ]
    else:
        trt_inputs = list(inputs)

    trt_model = torch_tensorrt.dynamo.compile(
        ep,
        inputs=trt_inputs,
        enabled_precisions={torch.float32},
        min_block_size=1,
    )

    # ------------------------------------------------------------------
    # 5. TRT inference & accuracy check
    # ------------------------------------------------------------------
    with torch.inference_mode():
        trt_out = trt_model(*inputs)

    pyt_flat = pyt_out.float().flatten()
    trt_flat = trt_out.float().flatten()
    cos_sim = (pyt_flat @ trt_flat / (pyt_flat.norm() * trt_flat.norm())).item()
    max_diff = (pyt_out.float() - trt_out.float()).abs().max().item()

    print(f"\n--- TensorRT vs PyTorch ---")
    print(f"  output shape  : {trt_out.shape}")
    print(f"  cosine sim    : {cos_sim:.6f}")
    print(f"  max |Δ|       : {max_diff:.2e}")
    assert cos_sim > 0.99, f"Cosine similarity {cos_sim:.4f} below threshold 0.99!"
    print("  PASSED")

    # ------------------------------------------------------------------
    # 6. (Optional) benchmark
    # ------------------------------------------------------------------
    if args.benchmark:
        print("\n--- Benchmarking ---")
        with torch.inference_mode():
            benchmark(model, *inputs, iterations=args.iterations, label="PyTorch")
            benchmark(trt_model, *inputs, iterations=args.iterations, label="TensorRT")


if __name__ == "__main__":
    main()
