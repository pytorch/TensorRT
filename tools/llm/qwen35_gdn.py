# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Export- and TensorRT-friendly Gated DeltaNet prefill for Qwen3.5 models.

Qwen3.5 (``Qwen/Qwen3.5-0.8B``, ``qwen3_5_text``) interleaves Gated DeltaNet
linear-attention layers with gated full attention. Without the ``fla`` Triton
kernels, Hugging Face runs ``torch_chunk_gated_delta_rule``: a Python loop
over 64-token chunks that writes its outputs in place, with
``torch.linalg.solve_triangular`` for the per-chunk UT transform in eager
mode and a 63-step forward substitution with in-place row writes under
export. TensorRT has no layer for either.

:func:`chunk_gated_delta_rule` computes the same thing as a functional graph
of batched matrix products, all in FP32 like the Hugging Face fallback:

* the decays, the masked ``k_beta k^T`` system and the intra-chunk
  ``q k^T`` scores of every chunk at once;
* ``(I + L)^-1`` of every chunk's strictly lower triangular ``L`` without a
  sequential solve: block forward substitution whose block size doubles
  each step, ``D <- D - D E D`` (:func:`unit_lower_inverse`; five steps of
  two products for 64-token chunks). The Neumann product ``(I + N)(I +
  N^2)(I + N^4)...`` costs the same but overflows FP32 on real layers;
* the inter-chunk state recurrence ``S <- S * gamma + k_dec^T (u - w S)``,
  the only sequential part. ``scan="loop"`` unrolls it into two small batched
  matrix products per chunk; ``scan="triton"`` runs it as one Triton kernel
  per layer (``gdn_scan_kernel``), compiled into the TensorRT engine
  through ``torch_tensorrt.region.fused_kernel``. :func:`check_scan_kernel`
  compiles that kernel ahead of the build and refuses one the TensorRT 11.4
  plugin launch cannot run (over 48 KB of shared memory, or Triton scratch);
  ``scan="generated"`` runs the same region (``gdn_scan``, the same inputs and
  body) with a kernel that PR 4914's ``torch_tensorrt.region.KernelGenerator``
  writes during the build (:func:`scan_states_generated`);
* the outputs of all chunks at once from the per-chunk start states.

:func:`install` swaps the ``forward`` of every ``Qwen3_5GatedDeltaNet`` layer
of a loaded model for :func:`functional_forward` (prefill without a cache),
which also offers the causal depthwise convolution as Hugging Face's
channel-major ``F.conv1d`` or as four token-major shifted multiply-adds
(``conv="shifts"``). With ``prologue="inductor"`` it binds
:func:`prologue_region_forward` instead: the same layer, with the prologue
(:func:`gdn_prologue`: ``in_proj_qkv``, the convolution and SiLU, the q/k/v
split and the FP32 L2 norms) inside a
``torch_tensorrt.region.execute_in_torch(backend="inductor")`` scope, which a
strict ``torch_tensorrt.compile`` hands to Inductor between two TensorRT
engines.

The numerics, the scan kernel and the layer forward (conv, q/k/v split,
gating) sit inside ``EDITABLE`` fences: ``qwen35_prefill.py`` declares this
file as an editable file, so an Auto-Tuner session may edit them (each edit
is compiled, checked against the FP32 reference and measured in the model
like any other candidate). The resource check of the scan kernel stays
outside them.
"""

from __future__ import annotations

import functools
import math
import types
from dataclasses import dataclass, field
from typing import Any, Optional

import torch
import torch.nn.functional as F

CHUNK_SIZES = (64, 32, 128)
SCANS = ("loop", "triton", "generated")
# The generator's notes for the gdn_scan region (scan="generated"): measured
# facts about its kernels. Part of the kernel cache key; editing the file
# starts a new cache.
GENERATION_NOTES = "qwen35_scan_kernel_notes.md"
CONVS = ("conv1d", "shifts")
# Where the layer prologue (gdn_prologue) runs: in the TensorRT engine with the
# rest of the layer, or in an Inductor region (prologue_region_forward), which
# with "inductor-decay" also takes the gates and the chunk rule's elementwise
# head (chunk_rule_head).
PROLOGUES = ("tensorrt", "inductor", "inductor-decay")


@dataclass(frozen=True)
class GDNSettings:
    """How :func:`functional_forward` computes a Gated DeltaNet layer.

    ``chunk_size``: tokens per chunk (a power of two); ``scan``: the
    inter-chunk state recurrence (``"loop"``, ``"triton"`` or
    ``"generated"``) and ``scan_tf32`` the Triton scan's dot precision
    (launch settings in :data:`SCAN_LAUNCH`); ``generator``: the
    ``torch_tensorrt.region.KernelGenerator`` that writes the kernel of the
    ``"generated"`` scan (created before tracing); ``conv``: the causal
    depthwise convolution (``"conv1d"`` or ``"shifts"``); ``prologue``:
    ``"inductor"`` or ``"inductor-decay"`` makes :func:`install` bind
    :func:`prologue_region_forward`.
    """

    chunk_size: int = 64
    scan: str = "loop"
    conv: str = "conv1d"
    scan_tf32: bool = False
    prologue: str = "tensorrt"
    generator: Any = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        if not _power_of_two(self.chunk_size) or self.chunk_size < 2:
            raise ValueError(f"chunk_size={self.chunk_size} is not a power of two")
        if self.scan not in SCANS:
            raise ValueError(f"scan={self.scan!r}; choose one of {SCANS}")
        if (self.scan == "generated") != (self.generator is not None):
            raise ValueError(
                "scan='generated' needs a KernelGenerator (generator), and only "
                "it takes one"
            )
        if self.conv not in CONVS:
            raise ValueError(f"conv={self.conv!r}; choose one of {CONVS}")
        if self.prologue not in PROLOGUES:
            raise ValueError(f"prologue={self.prologue!r}; choose one of {PROLOGUES}")


def _power_of_two(n: int) -> bool:
    return n > 0 and n & (n - 1) == 0


def l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Hugging Face's (and fla's) query/key L2 normalization."""
    return x * torch.rsqrt((x * x).sum(dim=-1, keepdim=True) + eps)


# Launch settings of the Triton state scan by dot precision: value columns
# per program (``block_v``, a power of two of at least 16 dividing the value
# head dim), chunk rows per step (``block_rows``, a power of two of at least
# 16; chunks longer than this are processed in row blocks so the kernel's
# shared memory does not grow with the chunk), warps and software-pipelining
# stages. From an isolated sweep on an RTX 5090 at 2k tokens (16 heads, 32
# chunks of 64): IEEE FP32 200 us per layer, TF32 137 us. The kernel must
# stay within SHARED_MEMORY_LIMIT (check_scan_kernel enforces it at build
# time): with 2 stages these configs need 77-90 KB and would run in 196 and
# 105 us. Wider blocks run slower.
# EDITABLE-START
SCAN_LAUNCH: dict[str, dict[str, int]] = {
    "ieee": {"block_v": 16, "block_rows": 64, "num_warps": 8, "num_stages": 1},
    "tf32": {"block_v": 16, "block_rows": 64, "num_warps": 4, "num_stages": 1},
}
# EDITABLE-END

# The TensorRT 11.4 QDP plugin launch passes Triton's shared-memory size as
# dynamic shared memory without opting the kernel in to more than the 48 KB
# default, so a larger kernel builds and then fails at its first enqueue
# (cudaErrorStreamCaptureInvalidated under CUDA graphs).
SHARED_MEMORY_LIMIT = 48 * 1024


# ----------------------------------------------------------------- numerics
# EDITABLE-START
def unit_lower_inverse(lower: torch.Tensor) -> torch.Tensor:
    """``(I + lower)^-1`` for strictly lower triangular ``[..., C, C]`` matrices.

    Block forward substitution with the block size doubling each step: if
    ``D`` inverts the block-diagonal part of ``I + lower`` at block size
    ``s``, and ``E`` holds the entries of ``lower`` between the two
    ``s``-blocks of each ``2s``-block, then ``D - D E D`` inverts it at block
    size ``2s`` (``E D E = 0``). ``ceil(log2 C)`` steps of two batched matrix
    products, no sequential row solve. Every intermediate is a set of
    diagonal blocks of the exact inverse, so nothing grows beyond the
    inverse itself (unlike the Neumann product ``(I + N)(I + N^2)...``,
    whose powers of ``N`` overflow FP32 on real Gated DeltaNet systems).
    """
    size = lower.shape[-1]
    index = torch.arange(size, device=lower.device)
    eye = (index[:, None] == index[None, :]).to(lower.dtype)
    inverse = eye - lower * _between_blocks(index, 1).to(lower.dtype)
    block = 2
    while block < size:
        between = lower * _between_blocks(index, block).to(lower.dtype)
        inverse = inverse - (inverse @ between) @ inverse
        block *= 2
    return inverse


def _between_blocks(index: torch.Tensor, block: int) -> torch.Tensor:
    """Entries in the same ``2 * block`` block but in different ``block`` blocks."""
    rows, cols = index[:, None], index[None, :]
    same_pair = (rows // (2 * block)) == (cols // (2 * block))
    return same_pair & ((rows // block) != (cols // block))


def chunk_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    chunk_size: int = 64,
    use_qk_l2norm: bool = True,
    scan: str = "loop",
    scan_tf32: bool = False,
    scan_generator: Any = None,
) -> torch.Tensor:
    """The gated delta rule over a whole sequence, from a zero state.

    Same inputs and output as Hugging Face's ``torch_chunk_gated_delta_rule``
    (``[batch, seq, heads, dim]`` query, key and value; ``[batch, seq,
    heads]`` log decay ``g`` and ``beta``) without an initial or final state.
    Computes in FP32 and returns the output in the query's dtype.
    ``scan_generator`` is the KernelGenerator of ``scan="generated"``.
    """
    out_dtype = query.dtype
    batch, length, heads, dk = key.shape
    dv = value.shape[-1]
    q, k, v = (x.transpose(1, 2).to(torch.float32) for x in (query, key, value))
    b, decay = (x.transpose(1, 2).to(torch.float32) for x in (beta, g))
    if use_qk_l2norm:
        q, k = l2norm(q), l2norm(k)
    q = q * dk**-0.5
    pad = (-length) % chunk_size
    if pad:
        q, k, v = (F.pad(x, (0, 0, 0, pad)) for x in (q, k, v))
        b, decay = (F.pad(x, (0, pad)) for x in (b, decay))
    chunks = (length + pad) // chunk_size
    shape = (batch, heads, chunks, chunk_size)
    q, k = q.reshape(*shape, dk), k.reshape(*shape, dk)
    v = v.reshape(*shape, dv)
    b, decay = b.reshape(shape), decay.reshape(shape)

    # Log decay accumulated from the chunk start, and pairwise within a chunk:
    # pairwise[i, j] = exp(G_i - G_j) for j <= i, else 0 (masked before exp,
    # where G_i - G_j > 0 could overflow).
    cumulative = decay.cumsum(-1)
    rows = torch.arange(chunk_size, device=q.device)
    causal = rows[:, None] >= rows[None, :]
    strict = (rows[:, None] > rows[None, :]).to(torch.float32)
    difference = cumulative.unsqueeze(-1) - cumulative.unsqueeze(-2)
    pairwise = torch.where(causal, difference, float("-inf")).exp()

    k_beta = k * b.unsqueeze(-1)
    v_beta = v * b.unsqueeze(-1)
    # UT transform: T = (I + strictly_lower(k_beta k^T * pairwise))^-1.
    system = (k_beta @ k.transpose(-1, -2)) * pairwise * strict
    transform = unit_lower_inverse(system)
    u = transform @ v_beta
    w = transform @ (k_beta * cumulative.exp().unsqueeze(-1))
    scores = (q @ k.transpose(-1, -2)) * pairwise
    q_decayed = q * cumulative.exp().unsqueeze(-1)
    last = cumulative[..., -1:]
    k_decayed = k * (last - cumulative).exp().unsqueeze(-1)
    gamma = last.exp()  # [batch, heads, chunks, 1]

    if scan == "triton":
        states, v_new = scan_states_triton(w, u, k_decayed, gamma, tf32=scan_tf32)
    elif scan == "generated":
        states, v_new = scan_states_generated(
            w, u, k_decayed, gamma, generator=scan_generator
        )
    elif scan == "loop":
        states, v_new = scan_states(w, u, k_decayed, gamma)
    else:
        raise ValueError(f"scan={scan!r}; choose one of {SCANS}")
    out = q_decayed @ states + scores @ v_new
    out = out.reshape(batch, heads, chunks * chunk_size, dv)[:, :, :length]
    return out.transpose(1, 2).contiguous().to(out_dtype)


def scan_states(
    w: torch.Tensor, u: torch.Tensor, k_decayed: torch.Tensor, gamma: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """The inter-chunk recurrence as an unrolled loop of batched matrix products.

    With ``S_0 = 0``: ``v_n = u_n - w_n S_n`` and ``S_{n+1} = S_n gamma_n +
    k_decayed_n^T v_n``. Returns the start state of every chunk ``[batch,
    heads, chunks, dk, dv]`` and every chunk's ``v_n`` ``[batch, heads,
    chunks, chunk, dv]``.
    """
    batch, heads, chunks, _, dk = w.shape
    dv = u.shape[-1]
    state = w.new_zeros(batch, heads, dk, dv)
    states, values = [], []
    for n in range(chunks):
        states.append(state)
        v_n = u[:, :, n] - w[:, :, n] @ state
        values.append(v_n)
        state = state * gamma[:, :, n].unsqueeze(-1) + (
            k_decayed[:, :, n].transpose(-1, -2) @ v_n
        )
    return torch.stack(states, 2), torch.stack(values, 2)


# EDITABLE-END


# ------------------------------------------------------------ Triton scan
def gdn_scan_kernel_factory() -> Any:
    """The Triton kernel of the state scan (resolved outside tracing)."""
    return _triton_kernels().gdn_scan_kernel


def _scan_grid(dv: int, block_v: int, rows: int, inputs: Any, outputs: Any) -> tuple:
    return (dv // block_v, rows)


@functools.lru_cache(maxsize=None)
def _triton_kernels() -> types.SimpleNamespace:
    import triton
    import triton.language as tl

    # Runtime parameters follow the plugin ABI: input pointers (w, u, k,
    # gamma), then output pointers (h, v); every size is a tl.constexpr. The
    # plugin path compiles without Triton's pointer-alignment specialization,
    # so dereferenced pointers carry 16-element tl.multiple_of hints (every
    # row is 16-aligned: dims are multiples of 16).
    # EDITABLE-START
    @triton.jit
    def gdn_scan_kernel(
        w_ptr,
        u_ptr,
        k_ptr,
        gamma_ptr,
        h_ptr,
        v_ptr,
        CHUNKS: tl.constexpr,
        C: tl.constexpr,
        DK: tl.constexpr,
        DV: tl.constexpr,
        BV: tl.constexpr,
        ROWS: tl.constexpr,
        TF32: tl.constexpr,
    ):
        """One (value block, batch-head) program: carry S[DK, BV] over every chunk.

        Inputs are contiguous FP32 ``w, k [rows, CHUNKS, C, DK]``, ``u [rows,
        CHUNKS, C, DV]`` and ``gamma [rows, CHUNKS]``; outputs ``h [rows,
        CHUNKS, DK, DV]`` (start states) and ``v [rows, CHUNKS, C, DV]``.
        Each chunk is processed in ``C / ROWS`` blocks of ``ROWS`` rows: every
        block's ``v_new`` uses the chunk's start state, and the state update
        sums the blocks. Products run in IEEE FP32, like the Hugging Face
        fallback, or on TF32 tensor cores with ``TF32=1``.
        """
        iv = tl.program_id(0)
        row = tl.program_id(1)
        t = tl.arange(0, ROWS)
        kk = tl.arange(0, DK)
        vv = iv * BV + tl.arange(0, BV)
        state = tl.zeros((DK, BV), dtype=tl.float32)
        for n in range(CHUNKS):
            chunk = row * CHUNKS + n
            h_ptrs = h_ptr + chunk * DK * DV + kk[:, None] * DV + vv[None, :]
            tl.store(tl.multiple_of(h_ptrs, [16, 16]), state)
            update = tl.zeros((DK, BV), dtype=tl.float32)
            for block in tl.static_range(C // ROWS):
                tr = block * ROWS + t
                w = tl.load(
                    tl.multiple_of(
                        w_ptr + chunk * C * DK + tr[:, None] * DK + kk[None, :],
                        [16, 16],
                    )
                )
                u = tl.load(
                    tl.multiple_of(
                        u_ptr + chunk * C * DV + tr[:, None] * DV + vv[None, :],
                        [16, 16],
                    )
                )
                if TF32:
                    v_new = u - tl.dot(w, state, input_precision="tf32")
                else:
                    v_new = u - tl.dot(w, state, input_precision="ieee")
                tl.store(
                    tl.multiple_of(
                        v_ptr + chunk * C * DV + tr[:, None] * DV + vv[None, :],
                        [16, 16],
                    ),
                    v_new,
                )
                k = tl.load(
                    tl.multiple_of(
                        k_ptr + chunk * C * DK + tr[:, None] * DK + kk[None, :],
                        [16, 16],
                    )
                )
                if TF32:
                    update = tl.dot(tl.trans(k), v_new, update, input_precision="tf32")
                else:
                    update = tl.dot(tl.trans(k), v_new, update, input_precision="ieee")
            gamma = tl.load(gamma_ptr + chunk)
            state = state * gamma + update

    # EDITABLE-END
    return types.SimpleNamespace(gdn_scan_kernel=gdn_scan_kernel)


def scan_launch(chunk: int, dv: int, tf32: bool) -> dict[str, int]:
    """The launch of ``gdn_scan_kernel`` for one shape: :data:`SCAN_LAUNCH` checked.

    Returns ``block_v``, ``rows`` (``min(chunk, block_rows)``), ``num_warps``
    and ``num_stages``; raises ``ValueError`` for settings the kernel cannot
    run (``tl.dot`` needs blocks of at least 16 rows and columns).
    """
    launch = SCAN_LAUNCH["tf32" if tf32 else "ieee"]
    block_v = int(launch["block_v"])
    rows = min(chunk, int(launch.get("block_rows", chunk)))
    if dv % block_v or not _power_of_two(block_v) or block_v < 16:
        raise ValueError(
            f"block_v={block_v} must be a power of two of at least 16 dividing the "
            f"value head dim {dv}"
        )
    if chunk % rows or not _power_of_two(rows) or rows < 16:
        raise ValueError(
            f"The Triton scan needs chunks of at least 16 tokens split into power-"
            f"of-two row blocks of at least 16; got chunk {chunk}, block_rows {rows}"
        )
    return {
        "block_v": block_v,
        "rows": rows,
        "num_warps": int(launch["num_warps"]),
        "num_stages": int(launch["num_stages"]),
    }


def _scan_constexprs(
    chunks: int, chunk: int, dk: int, dv: int, launch: dict[str, int], tf32: bool
) -> dict[str, int]:
    return {
        "CHUNKS": chunks,
        "C": chunk,
        "DK": dk,
        "DV": dv,
        "BV": launch["block_v"],
        "ROWS": launch["rows"],
        "TF32": int(tf32),
    }


def scan_kernel_resources(
    chunks: int, chunk: int, dk: int, dv: int, tf32: bool, target: Any = None
) -> dict[str, int]:
    """Compile ``gdn_scan_kernel`` as the plugin does and report its resources.

    ``target`` is a ``triton.backends.compiler.GPUTarget`` (default: the
    current GPU). Returns Triton's ``shared`` memory (bytes) and its
    ``global_scratch_size`` and ``profile_scratch_size``.
    """
    import triton

    if target is None:
        target = triton.runtime.driver.active.get_current_target()
    launch = scan_launch(chunk, dv, tf32)
    pointers = ("w_ptr", "u_ptr", "k_ptr", "gamma_ptr", "h_ptr", "v_ptr")
    source = triton.compiler.ASTSource(
        fn=gdn_scan_kernel_factory(),
        signature=dict.fromkeys(pointers, "*fp32"),
        constexprs=_scan_constexprs(chunks, chunk, dk, dv, launch, tf32),
    )
    compiled = triton.compile(
        source,
        target=target,
        options={
            "arch": f"sm{target.arch}",
            "num_warps": launch["num_warps"],
            "num_stages": launch["num_stages"],
        },
    )
    metadata = compiled.metadata
    return {
        "shared": int(metadata.shared),
        "global_scratch_size": int(getattr(metadata, "global_scratch_size", 0) or 0),
        "profile_scratch_size": int(getattr(metadata, "profile_scratch_size", 0) or 0),
    }


def check_scan_kernel(
    chunks: int, chunk: int, dk: int, dv: int, tf32: bool, target: Any = None
) -> dict[str, int]:
    """:func:`scan_kernel_resources`, refusing what the TensorRT plugin cannot launch.

    Raises ``ValueError`` when the kernel needs more than
    :data:`SHARED_MEMORY_LIMIT` bytes of shared memory, or any Triton
    scratch buffer (the plugin passes null for them), so a config or edit
    that would fail at its first enqueue fails at build time instead.
    """
    resources = scan_kernel_resources(chunks, chunk, dk, dv, tf32, target)
    if resources["shared"] > SHARED_MEMORY_LIMIT:
        raise ValueError(
            f"gdn_scan_kernel needs {resources['shared']} bytes of shared memory at "
            f"chunk {chunk} ({'tf32' if tf32 else 'ieee'}); the TensorRT 11.4 plugin "
            f"launch allows {SHARED_MEMORY_LIMIT}. Use fewer stages, smaller "
            "block_v or block_rows in SCAN_LAUNCH."
        )
    if resources["global_scratch_size"] or resources["profile_scratch_size"]:
        raise ValueError(
            f"gdn_scan_kernel needs Triton scratch memory {resources}; the TensorRT "
            "plugin launch passes null scratch pointers"
        )
    return resources


def scan_states_triton(
    w: torch.Tensor,
    u: torch.Tensor,
    k_decayed: torch.Tensor,
    gamma: torch.Tensor,
    *,
    tf32: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """:func:`scan_states` as one Triton kernel inside a TensorRT engine.

    Under ``torch_tensorrt.compile(..., strict=True)`` the body becomes a
    ``fused_kernel`` plugin launching ``gdn_scan_kernel`` with
    :func:`scan_launch` settings on a ``(dv / block_v, batch * heads)`` grid;
    elsewhere (eager, ``torch.export``) it runs the PyTorch loop. TensorRT
    11.4 reloads plugin modules on every enqueue outside CUDA graphs, so run
    such engines under CUDA graphs, and check the kernel's resources with
    :func:`check_scan_kernel` before building.
    """
    from torch_tensorrt import region

    batch, heads, chunks, chunk, dk = w.shape
    dv = u.shape[-1]
    launch = scan_launch(chunk, dv, tf32)
    w, u, k_decayed, gamma = _scan_region_inputs(w, u, k_decayed, gamma)
    with region.fused_kernel(
        kernel=gdn_scan_kernel_factory,
        inputs=[w, u, k_decayed, gamma],
        name="gdn_scan",
        constexprs=_scan_constexprs(chunks, chunk, dk, dv, launch, tf32),
        grid=functools.partial(_scan_grid, dv, launch["block_v"], batch * heads),
        num_warps=launch["num_warps"],
        num_stages=launch["num_stages"],
    ):
        states, values = scan_states(w, u, k_decayed, gamma.unsqueeze(-1))
    return states, values


def scan_states_generated(
    w: torch.Tensor,
    u: torch.Tensor,
    k_decayed: torch.Tensor,
    gamma: torch.Tensor,
    *,
    generator: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """:func:`scan_states` as the ``gdn_scan`` region, with a generated kernel.

    The region of :func:`scan_states_triton` (the same name, inputs and body)
    under ``fused_kernel(generator=...)``: strict ``torch_tensorrt.compile``
    replaces it with the kernel that ``generator`` (a
    ``torch_tensorrt.region.KernelGenerator``) writes, checks against the body
    and caches, or with the one it finds cached; the layers' regions are
    identical, so they share one kernel. Elsewhere (eager,
    ``torch.export``) it runs the PyTorch loop. TensorRT 11.4 reloads plugin
    modules on every enqueue outside CUDA graphs, so run such engines under
    CUDA graphs.
    """
    from torch_tensorrt import region

    w, u, k_decayed, gamma = _scan_region_inputs(w, u, k_decayed, gamma)
    with region.fused_kernel(
        generator=generator, inputs=[w, u, k_decayed, gamma], name="gdn_scan"
    ):
        states, values = scan_states(w, u, k_decayed, gamma.unsqueeze(-1))
    return states, values


def _scan_region_inputs(
    w: torch.Tensor, u: torch.Tensor, k_decayed: torch.Tensor, gamma: torch.Tensor
) -> tuple[torch.Tensor, ...]:
    """The ``gdn_scan`` region's inputs: contiguous ``w``, ``u``, ``k_decayed``
    and ``gamma`` as ``[batch, heads, chunks]`` (static, contiguous FP32)."""
    batch, heads, chunks = w.shape[:3]
    w, u, k_decayed = (x.contiguous() for x in (w, u, k_decayed))
    return w, u, k_decayed, gamma.reshape(batch, heads, chunks).contiguous()


# ------------------------------------------------------------- the layer
# The conv, the q/k/v split and the layer's elementwise work around the chunk
# rule: TensorRT fuses them into large elementwise kernels whose layout (and,
# with kernels.gdn_scan, the plugin region boundary) sets their cost.
# EDITABLE-START
def causal_conv1d_silu(
    x: torch.Tensor, weight: torch.Tensor, conv: str
) -> torch.Tensor:
    """SiLU of a causal depthwise convolution over tokens.

    ``x`` is token-major ``[B, T, D]``, ``weight`` ``[D, K]``; the output is
    ``[B, T, D]`` in ``x``'s dtype. ``"conv1d"`` is Hugging Face's
    ``causal_conv1d_fn``: transpose to channel-major ``[B, D, T]``, a grouped
    ``F.conv1d`` padded on both sides and cut, and transpose back (TensorRT
    then reads the following split and L2 norms across a strided layout).
    ``"shifts"`` stays token-major: ``K`` shifted copies of the left-padded
    input times their weights, summed in FP32 and rounded once (pure
    elementwise work that fuses with the SiLU).
    """
    width = weight.shape[-1]
    length = x.shape[1]
    if conv == "conv1d":
        out = F.conv1d(
            x.transpose(1, 2).to(weight.dtype),
            weight.unsqueeze(1),
            padding=width - 1,
            groups=x.shape[-1],
        )[..., :length]
        return F.silu(out).to(x.dtype).transpose(1, 2)
    if conv == "shifts":
        padded = F.pad(x, (0, 0, width - 1, 0)).to(torch.float32)
        taps = weight.to(torch.float32).t()  # [K, D]
        out = padded[:, :length] * taps[0]
        for j in range(1, width):
            out = out + padded[:, j : j + length] * taps[j]
        return F.silu(out).to(x.dtype)
    raise ValueError(f"conv={conv!r}; choose one of {CONVS}")


def functional_forward(
    self: torch.nn.Module,
    hidden_states: torch.Tensor,
    cache_params: Any = None,
    attention_mask: Optional[torch.Tensor] = None,
    **kwargs: Any,
) -> torch.Tensor:
    """Prefill forward of a ``Qwen3_5GatedDeltaNet`` layer without a cache.

    Bound to a layer by :func:`install`; computes what Hugging Face's forward
    computes for a full prompt from an empty state.
    """
    if cache_params is not None:
        raise ValueError("The functional Gated DeltaNet forward is cache-free prefill")
    settings: GDNSettings = self._gdn_settings
    if attention_mask is not None:
        hidden_states = (hidden_states * attention_mask[:, :, None]).to(
            hidden_states.dtype
        )
    batch, length, _ = hidden_states.shape
    mixed = self.in_proj_qkv(hidden_states)
    z = self.in_proj_z(hidden_states)
    b = self.in_proj_b(hidden_states)
    a = self.in_proj_a(hidden_states)
    mixed = causal_conv1d_silu(mixed, self.conv1d.weight.squeeze(1), settings.conv)
    query, key, value = torch.split(
        mixed, [self.key_dim, self.key_dim, self.value_dim], dim=-1
    )
    query = query.reshape(batch, length, -1, self.head_k_dim)
    key = key.reshape(batch, length, -1, self.head_k_dim)
    value = value.reshape(batch, length, -1, self.head_v_dim)
    beta = b.sigmoid()
    g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
    repeat = self.num_v_heads // self.num_k_heads
    if repeat > 1:
        query = query.repeat_interleave(repeat, dim=2)
        key = key.repeat_interleave(repeat, dim=2)
    core = chunk_gated_delta_rule(
        query,
        key,
        value,
        g,
        beta,
        chunk_size=settings.chunk_size,
        scan=settings.scan,
        scan_tf32=settings.scan_tf32,
        scan_generator=settings.generator,
    )
    core = self.norm(
        core.reshape(-1, self.head_v_dim), z.reshape(-1, self.head_v_dim)
    ).reshape(batch, length, -1)
    return self.out_proj(core)


# EDITABLE-END


# ------------------------------------------------ the prologue as a region
def gdn_prologue(
    layer: torch.nn.Module, hidden_states: torch.Tensor, conv: str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The prologue of a Gated DeltaNet layer: L2-normalized q and k, and v.

    ``in_proj_qkv``, :func:`causal_conv1d_silu`, the q/k/v split, the key
    head repeat for grouped value heads and the FP32 L2 norms of q and k
    that :func:`chunk_gated_delta_rule` applies with ``use_qk_l2norm`` (the
    same operations in the same order as :func:`functional_forward` and the
    head of the chunk rule). Returns contiguous head-major ``[batch, heads,
    seq, dim]`` tensors, the layout the chunk math reads: q and k in FP32, v
    in the activation dtype.
    """
    batch, length, _ = hidden_states.shape
    mixed = layer.in_proj_qkv(hidden_states)
    mixed = causal_conv1d_silu(mixed, layer.conv1d.weight.squeeze(1), conv)
    query, key, value = torch.split(
        mixed, [layer.key_dim, layer.key_dim, layer.value_dim], dim=-1
    )
    query = query.reshape(batch, length, -1, layer.head_k_dim)
    key = key.reshape(batch, length, -1, layer.head_k_dim)
    value = value.reshape(batch, length, -1, layer.head_v_dim)
    repeat = layer.num_v_heads // layer.num_k_heads
    if repeat > 1:
        query = query.repeat_interleave(repeat, dim=2)
        key = key.repeat_interleave(repeat, dim=2)
    q, k = (
        l2norm(x.transpose(1, 2).to(torch.float32)).contiguous() for x in (query, key)
    )
    return q, k, value.transpose(1, 2).contiguous()


def gdn_gates(
    layer: torch.nn.Module, b: torch.Tensor, a: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """``beta`` and the log decay ``g`` of a layer, as :func:`functional_forward`."""
    beta = b.sigmoid()
    g = -layer.A_log.float().exp() * F.softplus(a.float() + layer.dt_bias)
    return beta, g


def chunk_rule_head(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int,
) -> tuple[torch.Tensor, ...]:
    """The elementwise head of :func:`chunk_gated_delta_rule`.

    From :func:`gdn_prologue`'s head-major outputs (q and k L2-normalized)
    and the token-major gates ``g`` and ``beta``: the query scale, padding and
    chunking, the cumulative log decays and the pairwise decay mask,
    ``k_beta``, ``v_beta``, the ``w`` input ``k_beta * exp(G)``, the decayed
    queries and keys and the chunk decays ``gamma``, computed as the rule
    computes them (a copy of that part of the fenced rule, which an
    ``edit.gdn`` edit does not change). Returns the inputs of :func:`chunk_rule_body`:
    everything the chunk matrix products, the inverse and the scan read.
    """
    batch, heads, length, dk = k.shape
    dv = v.shape[-1]
    v = v.to(torch.float32)
    b, decay = (x.transpose(1, 2).to(torch.float32) for x in (beta, g))
    q = q * dk**-0.5
    pad = (-length) % chunk_size
    if pad:
        q, k, v = (F.pad(x, (0, 0, 0, pad)) for x in (q, k, v))
        b, decay = (F.pad(x, (0, pad)) for x in (b, decay))
    chunks = (length + pad) // chunk_size
    shape = (batch, heads, chunks, chunk_size)
    q, k = q.reshape(*shape, dk), k.reshape(*shape, dk)
    v = v.reshape(*shape, dv)
    b, decay = b.reshape(shape), decay.reshape(shape)
    cumulative = decay.cumsum(-1)
    rows = torch.arange(chunk_size, device=q.device)
    causal = rows[:, None] >= rows[None, :]
    difference = cumulative.unsqueeze(-1) - cumulative.unsqueeze(-2)
    pairwise = torch.where(causal, difference, float("-inf")).exp()
    k_beta = k * b.unsqueeze(-1)
    v_beta = v * b.unsqueeze(-1)
    w_input = k_beta * cumulative.exp().unsqueeze(-1)
    q_decayed = q * cumulative.exp().unsqueeze(-1)
    last = cumulative[..., -1:]
    k_decayed = k * (last - cumulative).exp().unsqueeze(-1)
    gamma = last.exp()
    return q, k, k_beta, v_beta, w_input, q_decayed, k_decayed, gamma, pairwise


def chunk_rule_body(
    q: torch.Tensor,
    k: torch.Tensor,
    k_beta: torch.Tensor,
    v_beta: torch.Tensor,
    w_input: torch.Tensor,
    q_decayed: torch.Tensor,
    k_decayed: torch.Tensor,
    gamma: torch.Tensor,
    pairwise: torch.Tensor,
    *,
    length: int,
    scan: str = "loop",
    scan_tf32: bool = False,
    scan_generator: Any = None,
) -> torch.Tensor:
    """The rest of :func:`chunk_gated_delta_rule` after :func:`chunk_rule_head`.

    The masked UT system and :func:`unit_lower_inverse`, ``u`` and ``w``,
    the intra-chunk scores, the state scan (:func:`scan_states`,
    :func:`scan_states_triton` or :func:`scan_states_generated`) and the
    outputs; returns the token-major FP32 output ``[batch, length, heads,
    dv]`` (the rule rounds it to the input dtype).
    """
    chunk_size = pairwise.shape[-1]
    rows = torch.arange(chunk_size, device=q.device)
    strict = (rows[:, None] > rows[None, :]).to(torch.float32)
    system = (k_beta @ k.transpose(-1, -2)) * pairwise * strict
    transform = unit_lower_inverse(system)
    u = transform @ v_beta
    w = transform @ w_input
    scores = (q @ k.transpose(-1, -2)) * pairwise
    if scan == "triton":
        states, v_new = scan_states_triton(w, u, k_decayed, gamma, tf32=scan_tf32)
    elif scan == "generated":
        states, v_new = scan_states_generated(
            w, u, k_decayed, gamma, generator=scan_generator
        )
    elif scan == "loop":
        states, v_new = scan_states(w, u, k_decayed, gamma)
    else:
        raise ValueError(f"scan={scan!r}; choose one of {SCANS}")
    out = q_decayed @ states + scores @ v_new
    batch, heads, chunks, _, dv = out.shape
    out = out.reshape(batch, heads, chunks * chunk_size, dv)[:, :, :length]
    return out.transpose(1, 2).contiguous()


def prologue_region_forward(
    self: torch.nn.Module,
    hidden_states: torch.Tensor,
    cache_params: Any = None,
    attention_mask: Optional[torch.Tensor] = None,
    **kwargs: Any,
) -> torch.Tensor:
    """:func:`functional_forward` with the prologue in an Inductor region.

    :func:`gdn_prologue` runs inside ``torch_tensorrt.region.execute_in_torch(
    name="gdn_prologue", backend="inductor")``. Under ``torch_tensorrt.compile(
    ..., strict=True)`` the scope leaves TensorRT and Inductor compiles it
    (its projection on cuBLAS, the rest as generated Triton kernels): the
    engine ends at the region's inputs (the normalized hidden states) and the
    next one starts at its outputs, two engine boundaries per layer. The
    other input projections (``in_proj_z``, ``in_proj_b``, ``in_proj_a``) stay
    in TensorRT, which merges them into one GEMM instead of merging all four.

    ``settings.prologue == "inductor"``: the gates,
    :func:`chunk_gated_delta_rule` (its L2 norms already applied), the gated
    norm and ``out_proj`` follow in TensorRT as in :func:`functional_forward`.
    ``"inductor-decay"``: the region also computes the gates and the rule's
    elementwise head (:func:`chunk_rule_head`), so TensorRT keeps the chunk
    matrix products, the inverse, the scan and the outputs
    (:func:`chunk_rule_body`). Both compute the same operations as
    :func:`functional_forward` (the second through its own copy of the rule,
    so an ``edit.gdn`` edit of :func:`chunk_gated_delta_rule` does not reach
    it; ``qwen35_prefill`` refuses such an edit with these values). Run eagerly
    (or exported), the region is ordinary code with the same roundings.
    Compiled by Inductor, its fused kernels may skip intermediate roundings
    to the activation dtype (Inductor's ``emulate_precision_casts`` is off by
    default): the SiLU output that feeds the q/k L2 norms, for instance, stays
    FP32 where the eager code rounds it to FP16.
    """
    if cache_params is not None:
        raise ValueError("The functional Gated DeltaNet forward is cache-free prefill")
    from torch_tensorrt import region

    settings: GDNSettings = self._gdn_settings
    if attention_mask is not None:
        hidden_states = (hidden_states * attention_mask[:, :, None]).to(
            hidden_states.dtype
        )
    batch, length, _ = hidden_states.shape
    z = self.in_proj_z(hidden_states)
    b = self.in_proj_b(hidden_states)
    a = self.in_proj_a(hidden_states)
    if settings.prologue == "inductor-decay":
        with region.execute_in_torch(name="gdn_prologue", backend="inductor"):
            query, key, value = gdn_prologue(self, hidden_states, settings.conv)
            beta, g = gdn_gates(self, b, a)
            head = chunk_rule_head(query, key, value, g, beta, settings.chunk_size)
        core = chunk_rule_body(
            *head,
            length=length,
            scan=settings.scan,
            scan_tf32=settings.scan_tf32,
            scan_generator=settings.generator,
        )
    else:
        with region.execute_in_torch(name="gdn_prologue", backend="inductor"):
            query, key, value = gdn_prologue(self, hidden_states, settings.conv)
        beta, g = gdn_gates(self, b, a)
        # The chunk rule takes token-major views; transposed back, they are
        # the region's contiguous head-major outputs. It returns FP32 (the
        # query's dtype).
        core = chunk_gated_delta_rule(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            g,
            beta,
            chunk_size=settings.chunk_size,
            use_qk_l2norm=False,
            scan=settings.scan,
            scan_tf32=settings.scan_tf32,
            scan_generator=settings.generator,
        )
    # Rounded once, as functional_forward's chunk rule does.
    core = core.to(hidden_states.dtype)
    core = self.norm(
        core.reshape(-1, self.head_v_dim), z.reshape(-1, self.head_v_dim)
    ).reshape(batch, length, -1)
    return self.out_proj(core)


def gdn_layers(model: torch.nn.Module) -> list[torch.nn.Module]:
    """The Gated DeltaNet layers of a Qwen3.5 model (any wrapper)."""
    return [m for m in model.modules() if type(m).__name__.endswith("GatedDeltaNet")]


def install(model: torch.nn.Module, settings: GDNSettings) -> int:
    """Bind :func:`functional_forward` with ``settings`` to every GDN layer.

    With ``settings.prologue`` other than ``"tensorrt"`` it binds
    :func:`prologue_region_forward` instead. Returns the number of layers
    patched. The model is changed in place.
    """
    forward = (
        functional_forward
        if getattr(settings, "prologue", "tensorrt") == "tensorrt"
        else prologue_region_forward
    )
    layers = gdn_layers(model)
    for layer in layers:
        layer._gdn_settings = settings
        layer.forward = types.MethodType(forward, layer)
    return len(layers)


def flops_per_layer(
    length: int, heads: int, dk: int, dv: int, chunk_size: int
) -> float:
    """Matrix-product FLOPs of one :func:`chunk_gated_delta_rule` call.

    Counts the chunk-local systems and scores, the inverse, ``u``/``w``, the
    scan and the outputs; elementwise work is left out.
    """
    chunks = -(-length // chunk_size)
    c = chunk_size
    steps = max(math.ceil(math.log2(c)) - 1, 0)
    per_chunk = (
        2 * c * c * dk * 2  # k_beta k^T and q k^T
        + 2 * steps * 2 * c**3  # two products per step of the inverse
        + 2 * c * c * (dv + dk)  # u and w
        + 2 * c * dk * dv * 2  # w S and k_decayed^T v_new
        + 2 * c * dk * dv  # q_decayed S
        + 2 * c * c * dv  # scores v_new
    )
    return float(per_chunk * chunks * heads)
