"""
Prototype: AOTAutograd backward pass with a custom torch.compile backend

Demonstrates that the backward graph from AOTAutograd is just another flat
ATen IR DAG — the same decomposition + partitioning pipeline used for the
forward can be applied to it directly.

Architecture:
    torch.compile(model, backend=trt_with_backward, dynamic=False)
        └── aot_autograd
              ├── fw_compiler  → apply TRT lowering passes → compile_module (TRT)
              └── bw_compiler  → apply TRT lowering passes → compile_module (TRT)
                                 (falls back to eager for ops without TRT converters)

Run:
    uv run python experiments/aot_backward_prototype.py
"""

import logging
import unittest.mock

import torch
import torch._dynamo
import torch.nn as nn
import torch_tensorrt
from torch._dynamo.backends.common import aot_autograd
from torch._dynamo.utils import detect_fake_mode
from torch_tensorrt.dynamo import CompilationSettings
from torch_tensorrt.dynamo._compiler import compile_module
from torch_tensorrt.dynamo.lowering import get_decompositions, post_lowering
from torch_tensorrt.dynamo.utils import prepare_inputs

logging.basicConfig(level=logging.WARNING)

# Force static shape specialization — every unique shape gets its own TRT
# engine rather than a single engine with symbolic (Sym) batch dimensions.
torch._dynamo.config.assume_static_by_default = True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_VERBOSE = True  # set to False in perf section to suppress graph dumps


def _compile_graph_with_trt(
    gm: torch.fx.GraphModule,
    sample_inputs: list,
    label: str,
    min_block_size: int = 1,
) -> callable:
    """Apply TRT lowering + compile_module to an already-traced ATen graph.

    This is the same pipeline as _pretraced_backend() in backends.py, but
    skips the aot_export_joint_simple step because AOTAutograd has already
    decomposed the graph before calling fw/bw compilers.
    """
    settings = CompilationSettings(
        min_block_size=min_block_size,
        pass_through_build_failures=True,
        use_explicit_typing=True,
        use_python_runtime=True,
        disable_tf32=True,
    )

    if _VERBOSE:
        print(f"\n{'=' * 60}")
        print(f"[{label}] ATen graph (post-AOT decomposition):")
        print(f"{'=' * 60}")
        gm.print_readable()
    else:
        print(f"[{label}] compiling...", end=" ", flush=True)

    # Apply the same lowering passes used for the forward path
    gm = post_lowering(gm, settings)

    # Filter to real tensors only (skip SymInt/SymFloat placeholders)
    torch_inputs = [
        i
        for i in sample_inputs
        if isinstance(i, torch.Tensor) and not isinstance(i, torch.SymInt)
    ]
    if not torch_inputs:
        print(f"[{label}] No tensor inputs found, returning eager graph")
        return gm

    # If AOTAutograd is running inside a FakeTensorMode context, allow real
    # tensors so compile_module can inspect shapes for TRT engine building.
    fake_mode = detect_fake_mode(sample_inputs)
    ctx = (
        unittest.mock.patch.object(fake_mode, "allow_non_fake_inputs", True)
        if fake_mode is not None
        else unittest.mock.MagicMock()  # no-op context manager
    )

    try:
        with ctx:
            torchtrt_inputs = prepare_inputs(
                torch_inputs, disable_memory_format_check=True
            )
            compiled = compile_module(gm, torchtrt_inputs, settings=settings)
        print(f"[{label}] TRT compilation succeeded")
        return compiled
    except Exception as e:
        if _VERBOSE:
            print(f"[{label}] TRT compilation failed ({type(e).__name__}: {e})")
        print(f"[{label}] falling back to eager")
        return gm


# ---------------------------------------------------------------------------
# Custom fw / bw compilers
# ---------------------------------------------------------------------------


def trt_fw_compiler(gm: torch.fx.GraphModule, sample_inputs: list) -> callable:
    return _compile_graph_with_trt(gm, sample_inputs, label="FW")


def trt_bw_compiler(gm: torch.fx.GraphModule, sample_inputs: list) -> callable:
    # The backward graph is just another ATen DAG — same pipeline applies.
    # Ops like aten.mm (used in linear backward) already have TRT converters.
    # Ops without converters fall back to PyTorch subgraphs via the partitioner.
    return _compile_graph_with_trt(gm, sample_inputs, label="BW")


# native_layer_norm returns (out, mean, rstd) — the mean/rstd outputs can be
# None after partitioning, which breaks TRT converters. Decompose it into
# primitive aten ops (sub, var, mul, add) that TRT handles without issue.
# The backward (native_layer_norm_backward) is already in get_decompositions().
_extra_decomps = torch._decomp.get_decompositions(
    [
        torch.ops.aten.native_layer_norm.default,
        # native_layer_norm decomposes to var_mean, which returns a tuple TRT
        # can't handle. Decompose var_mean into separate var + mean calls.
        torch.ops.aten.var_mean.correction,
    ]
)

_decompositions = {
    **get_decompositions(
        enable_experimental_decompositions=True,
        decompose_attention=False,
    ),
    **_extra_decomps,
}

# Register the combined backend
trt_with_backward = aot_autograd(
    fw_compiler=trt_fw_compiler,
    bw_compiler=trt_bw_compiler,
    decompositions=_decompositions,
)


# ---------------------------------------------------------------------------
# Demo model
# ---------------------------------------------------------------------------


class TransformerBlock(nn.Module):
    """Single transformer block: multi-head self-attention + feed-forward."""

    def __init__(self, d_model: int, n_heads: int, ffn_dim: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.ff1 = nn.Linear(d_model, ffn_dim)
        self.ff2 = nn.Linear(ffn_dim, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq, d_model)
        B, S, D = x.shape
        H, Dh = self.n_heads, self.head_dim

        # --- self-attention ---
        qkv = self.qkv(x)  # (B, S, 3D)
        qkv = qkv.reshape(B, S, 3, H, Dh)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, S, Dh)
        q, k, v = qkv[0], qkv[1], qkv[2]

        scale = self.head_dim**-0.5
        attn = (q @ k.transpose(-2, -1)) * scale  # (B, H, S, S)
        attn = attn.softmax(dim=-1)
        out = attn @ v  # (B, H, S, Dh)
        out = out.permute(0, 2, 1, 3).reshape(B, S, D)
        x = self.norm1(x + self.out_proj(out))

        # --- feed-forward ---
        x = self.norm2(x + self.ff2(torch.relu(self.ff1(x))))
        return x


class SmallTransformer(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        ffn_dim: int = 512,
        n_layers: int = 20,
        out_features: int = 16,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, ffn_dim) for _ in range(n_layers)]
        )
        self.head = nn.Linear(d_model, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq, d_model)
        for block in self.blocks:
            x = block(x)
        return self.head(x[:, 0])  # classify from first token


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def run_training(
    model: nn.Module, compiled_model: nn.Module, device: str, steps: int = 5
):
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    loss_fn = nn.CrossEntropyLoss()

    print("\n" + "=" * 60)
    print("Training: TRT forward + TRT backward (w/ eager fallback)")
    print("=" * 60)

    for step in range(steps):
        x = torch.randn(8, 64, device=device)
        y = torch.randint(0, 16, (8,), device=device)

        optimizer.zero_grad()
        out = compiled_model(x)
        loss = loss_fn(out, y)
        loss.backward()
        optimizer.step()

        print(f"  step {step + 1}/{steps}  loss={loss.item():.4f}")


# ---------------------------------------------------------------------------
# Gradient correctness check
# ---------------------------------------------------------------------------


def check_gradients(
    model: nn.Module,
    compiled_model: nn.Module,
    device: str,
    batch_size: int = 8,
    in_features: int = 64,
):
    print("\n" + "=" * 60)
    print("Gradient correctness check (compiled vs eager)")
    print("=" * 60)

    x = torch.randn(batch_size, in_features, device=device)

    # Pre-warm with requires_grad=True so the TRT fw+bw engines for this
    # input variant are already compiled before we time/check anything.
    x_warm = x.detach().clone().requires_grad_(True)
    compiled_model(x_warm).sum().backward()
    compiled_model.zero_grad()
    torch.cuda.synchronize()

    # TensorRT enables TF32 matmuls by default on Ampere+, giving ~1e-3
    # precision. Disable TF32 in the PyTorch reference so both sides use
    # the same strict FP32 arithmetic, making the comparison meaningful.
    prev_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False

    ref_model = MLP(in_features=in_features).to(device)
    ref_model.load_state_dict(model.state_dict())
    x_ref = x.detach().clone().requires_grad_(True)
    ref_model(x_ref).sum().backward()

    torch.backends.cuda.matmul.allow_tf32 = prev_tf32

    # Compiled (uses already-built TRT engines — no recompile)
    x_compiled = x.detach().clone().requires_grad_(True)
    compiled_model(x_compiled).sum().backward()

    max_abs_diff = (x_compiled.grad - x_ref.grad).abs().max().item()
    rel_diff = max_abs_diff / (x_ref.grad.abs().max().item() + 1e-8)

    print(f"  Max absolute diff in input gradients : {max_abs_diff:.3e}")
    print(f"  Max relative diff in input gradients : {rel_diff:.3e}")

    atol, rtol = 1e-3, 1e-2
    passed = torch.allclose(x_compiled.grad, x_ref.grad, atol=atol, rtol=rtol)
    print(f"  Result: {'PASS' if passed else 'FAIL'} (atol={atol}, rtol={rtol})")
    return passed


# ---------------------------------------------------------------------------
# Perf benchmark
# ---------------------------------------------------------------------------


def benchmark(
    model: nn.Module,
    compiled_model: nn.Module,
    device: str,
    in_features: int,
    out_features: int,
    batch_size: int = 64,
    warmup: int = 10,
    iters: int = 100,
):
    """Time forward+backward for eager vs TRT-compiled.

    Both branches run with gradients active to avoid triggering a recompile
    (torch.compile creates separate guards for no_grad / grad contexts).
    """
    loss_fn = nn.CrossEntropyLoss()

    def make_batch():
        x = torch.randn(batch_size, in_features, device=device)
        y = torch.randint(0, out_features, (batch_size,), device=device)
        return x, y

    def time_fn(fn):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / iters  # ms per iter

    def eager_step():
        x, y = make_batch()
        loss_fn(model(x), y).backward()
        model.zero_grad()

    def compiled_step():
        x, y = make_batch()
        loss_fn(compiled_model(x), y).backward()
        compiled_model.zero_grad()

    print("\n" + "=" * 60)
    print(f"Perf benchmark  (batch={batch_size}, warmup={warmup}, iters={iters})")
    print("=" * 60)

    t_eager = time_fn(eager_step)
    t_compiled = time_fn(compiled_step)

    speedup = t_eager / t_compiled
    print(f"  eager    : {t_eager:.3f} ms/step")
    print(f"  trt      : {t_compiled:.3f} ms/step")
    print(f"  speedup  : {speedup:.2f}x")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    global _VERBOSE
    device = "cuda"
    loss_fn = nn.CrossEntropyLoss()

    # d_model=256, 4 heads, ffn=512, 4 layers, seq_len=16
    D, H, F, L, S, OUT = 256, 4, 512, 4, 16, 16

    model = SmallTransformer(
        d_model=D, n_heads=H, ffn_dim=F, n_layers=L, out_features=OUT
    ).to(device)
    compiled = torch.compile(model, backend=trt_with_backward, dynamic=False)

    def make_batch(bs):
        x = torch.randn(bs, S, D, device=device)
        y = torch.randint(0, OUT, (bs,), device=device)
        return x, y

    # --- Correctness (verbose: graphs printed on first compile) ---
    print("\n" + "=" * 60)
    print("Training: TRT forward + TRT backward (w/ eager fallback)")
    print("=" * 60)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    for step in range(3):
        x, y = make_batch(8)
        optimizer.zero_grad()
        loss_fn(compiled(x), y).backward()
        optimizer.step()
        print(f"  step {step + 1}/3  loss={loss_fn(model(x), y).item():.4f}")

    _VERBOSE = False

    # Gradient check
    print("\n" + "=" * 60)
    print("Gradient correctness check (compiled vs eager)")
    print("=" * 60)
    x_chk, _ = make_batch(8)

    # Pre-warm requires_grad variant so no recompile during check
    x_w = x_chk.detach().clone().requires_grad_(True)
    compiled(x_w).sum().backward()
    compiled.zero_grad()
    torch.cuda.synchronize()

    # TRT now uses disable_tf32=True, so both sides run strict FP32.
    ref = SmallTransformer(
        d_model=D, n_heads=H, ffn_dim=F, n_layers=L, out_features=OUT
    ).to(device)
    ref.load_state_dict(model.state_dict())
    x_ref = x_chk.detach().clone().requires_grad_(True)
    ref(x_ref).sum().backward()

    x_trt = x_chk.detach().clone().requires_grad_(True)
    compiled(x_trt).sum().backward()

    max_abs = (x_trt.grad - x_ref.grad).abs().max().item()
    rel = max_abs / (x_ref.grad.abs().max().item() + 1e-8)
    print(f"  Max absolute diff : {max_abs:.3e}")
    print(f"  Max relative diff : {rel:.3e}")
    passed = torch.allclose(x_trt.grad, x_ref.grad, atol=1e-3, rtol=1e-2)
    print(f"  Result: {'PASS' if passed else 'FAIL'} (atol=1e-3, rtol=1e-2)")

    # --- Perf benchmark ---
    bs_bench = 32
    print(f"\n--- perf: SmallTransformer bs={bs_bench} seq={S} d={D} {L}L ---")
    x_w2, y_w2 = make_batch(bs_bench)
    print("  [warmup]", end=" ", flush=True)
    loss_fn(compiled(x_w2), y_w2).backward()
    compiled.zero_grad()
    torch.cuda.synchronize()
    print("done")

    def eager_step():
        x, y = make_batch(bs_bench)
        loss_fn(model(x), y).backward()
        model.zero_grad()

    def trt_step():
        x, y = make_batch(bs_bench)
        loss_fn(compiled(x), y).backward()
        compiled.zero_grad()

    warmup, iters = 10, 100
    for fn in (eager_step, trt_step):
        for _ in range(warmup):
            fn()
    torch.cuda.synchronize()

    def time_fn(fn):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / iters

    t_eager = time_fn(eager_step)
    t_trt = time_fn(trt_step)
    print(f"  eager   : {t_eager:.3f} ms/step")
    print(f"  trt     : {t_trt:.3f} ms/step")
    print(f"  speedup : {t_eager / t_trt:.2f}x")

    print(
        "\nDone." + (" Gradients match." if passed else " WARNING: gradient mismatch.")
    )


if __name__ == "__main__":
    main()
