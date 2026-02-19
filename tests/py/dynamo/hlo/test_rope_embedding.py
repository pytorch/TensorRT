"""
Tests for Rotary Position Embedding (RoPE) compilation with torch-tensorrt.

RoPE is a critical subgraph used in modern LLMs (LLaMA, Qwen, Mistral, etc.).
Two common forms are tested:
  1. HuggingFace-style: rotate_half + apply_rotary_pos_emb using cos/sin tensors
  2. Complex-number form: view_as_complex + complex multiply + view_as_real

Both static and dynamic shapes (varying seq_len, batch) are covered, as well as
RoPE embedded inside a larger attention block (a common failure mode).
"""

import os

import pytest
import torch
import torch.nn as nn
import torch_tensorrt as torchtrt
from torch.export import Dim
from torch_tensorrt.dynamo.utils import COSINE_THRESHOLD, cosine_similarity


# ---------------------------------------------------------------------------
# Shared helper modules
# ---------------------------------------------------------------------------


class HFRotaryEmbedding(nn.Module):
    """HuggingFace-style RoPE as used in LLaMA / Qwen / Mistral.

    Identical to ``apply_rotary_pos_emb`` in transformers:
        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
    """

    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ):
        # cos/sin shape: (batch, seq_len, head_dim) – unsqueeze head dim
        cos = cos.unsqueeze(1)  # (batch, 1, seq_len, head_dim)
        sin = sin.unsqueeze(1)
        q_embed = (q * cos) + (self.rotate_half(q) * sin)
        k_embed = (k * cos) + (self.rotate_half(k) * sin)
        return q_embed, k_embed


class ComplexRotaryEmbedding(nn.Module):
    """Complex-number RoPE as used in original LLaMA / Meta models.

    Applies pre-computed complex frequency tensor via:
        x_complex = view_as_complex(x.reshape(..., -1, 2))
        out = view_as_real(x_complex * freqs_cis).flatten(-2)
    """

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, n_heads, head_dim)
        # freqs_cis: (seq_len, head_dim // 2) complex
        x_ = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        freqs_cis = freqs_cis[None, :, None, :]  # broadcast over batch and heads
        x_out = torch.view_as_real(x_ * freqs_cis).flatten(3)
        return x_out.type_as(x)


def _make_freqs_cis(seq_len: int, head_dim: int, theta: float = 10000.0) -> torch.Tensor:
    """Pre-compute complex frequency tensor on CUDA."""
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(seq_len, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs).cuda()


# ---------------------------------------------------------------------------
# Test 1: HuggingFace-style RoPE – static shapes
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_rope_hf_style_static():
    """HF rotate_half RoPE compiles and produces correct outputs (static shapes)."""
    model = HFRotaryEmbedding().eval().cuda()

    q = torch.randn(1, 12, 5, 128, dtype=torch.float32).cuda()
    k = torch.randn(1, 12, 5, 128, dtype=torch.float32).cuda()
    # cos/sin: (batch, seq_len, head_dim)
    cos = torch.randn(1, 5, 128, dtype=torch.float32).cuda()
    sin = torch.randn(1, 5, 128, dtype=torch.float32).cuda()
    inputs = (q, k, cos, sin)

    compile_spec = {
        "inputs": inputs,
        "ir": "dynamo",
        "min_block_size": 1,
        "pass_through_build_failures": True,
    }
    trt_model = torchtrt.compile(model, **compile_spec)

    py_q, py_k = model(*inputs)
    trt_q, trt_k = trt_model(*inputs)

    cos_sim_q = cosine_similarity(py_q, trt_q)
    cos_sim_k = cosine_similarity(py_k, trt_k)
    assert cos_sim_q > COSINE_THRESHOLD, (
        f"test_rope_hf_style_static: q outputs differ. "
        f"Cosine sim: {cos_sim_q:.4f} < threshold {COSINE_THRESHOLD}"
    )
    assert cos_sim_k > COSINE_THRESHOLD, (
        f"test_rope_hf_style_static: k outputs differ. "
        f"Cosine sim: {cos_sim_k:.4f} < threshold {COSINE_THRESHOLD}"
    )
    torch._dynamo.reset()


# ---------------------------------------------------------------------------
# Test 2: HuggingFace-style RoPE – dynamic seq_len
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_rope_hf_style_dynamic():
    """HF rotate_half RoPE compiles and produces correct outputs (dynamic seq_len)."""
    model = HFRotaryEmbedding().eval().cuda()

    q = torch.randn(1, 12, 5, 128, dtype=torch.float32).cuda()
    k = torch.randn(1, 12, 5, 128, dtype=torch.float32).cuda()
    cos = torch.randn(1, 5, 128, dtype=torch.float32).cuda()
    sin = torch.randn(1, 5, 128, dtype=torch.float32).cuda()
    inputs = (q, k, cos, sin)

    seq_len = Dim("seq_len", min=2, max=2048)
    # q/k: (batch, n_heads, seq_len, head_dim) – seq_len is dim 2
    # cos/sin: (batch, seq_len, head_dim) – seq_len is dim 1
    dynamic_shapes = (
        {2: seq_len},
        {2: seq_len},
        {1: seq_len},
        {1: seq_len},
    )
    exp_program = torch.export.export(model, inputs, dynamic_shapes=dynamic_shapes)

    compile_spec = {
        "inputs": inputs,
        "min_block_size": 1,
        "pass_through_build_failures": True,
    }
    trt_model = torchtrt.dynamo.compile(exp_program, **compile_spec)

    py_q, py_k = model(*inputs)
    trt_q, trt_k = trt_model(*inputs)

    cos_sim_q = cosine_similarity(py_q, trt_q)
    cos_sim_k = cosine_similarity(py_k, trt_k)
    assert cos_sim_q > COSINE_THRESHOLD, (
        f"test_rope_hf_style_dynamic: q outputs differ. "
        f"Cosine sim: {cos_sim_q:.4f} < threshold {COSINE_THRESHOLD}"
    )
    assert cos_sim_k > COSINE_THRESHOLD, (
        f"test_rope_hf_style_dynamic: k outputs differ. "
        f"Cosine sim: {cos_sim_k:.4f} < threshold {COSINE_THRESHOLD}"
    )
    torch._dynamo.reset()


# ---------------------------------------------------------------------------
# Test 3: Complex-number RoPE – static shapes
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_rope_complex_form_static():
    """Complex (view_as_complex/view_as_real) RoPE compiles correctly (static shapes)."""
    BATCH, SEQ_LEN, N_HEADS, HEAD_DIM = 2, 8, 4, 64
    model = ComplexRotaryEmbedding().eval().cuda()

    x = torch.randn(BATCH, SEQ_LEN, N_HEADS, HEAD_DIM, dtype=torch.float32).cuda()
    freqs_cis = _make_freqs_cis(SEQ_LEN, HEAD_DIM)
    inputs = (x, freqs_cis)

    compile_spec = {
        "inputs": inputs,
        "ir": "dynamo",
        "min_block_size": 1,
        "pass_through_build_failures": True,
    }
    trt_model = torchtrt.compile(model, **compile_spec)

    py_out = model(*inputs)
    trt_out = trt_model(*inputs)

    cos_sim = cosine_similarity(py_out, trt_out)
    assert cos_sim > COSINE_THRESHOLD, (
        f"test_rope_complex_form_static: outputs differ. "
        f"Cosine sim: {cos_sim:.4f} < threshold {COSINE_THRESHOLD}"
    )
    torch._dynamo.reset()


# ---------------------------------------------------------------------------
# Test 4: Complex-number RoPE – dynamic batch and seq_len
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_rope_complex_form_dynamic():
    """Complex RoPE compiles correctly with dynamic batch and seq_len."""
    BATCH, SEQ_LEN, N_HEADS, HEAD_DIM = 2, 8, 4, 64
    model = ComplexRotaryEmbedding().eval().cuda()

    x = torch.randn(BATCH, SEQ_LEN, N_HEADS, HEAD_DIM, dtype=torch.float32).cuda()
    freqs_cis = _make_freqs_cis(SEQ_LEN, HEAD_DIM)
    inputs = (x, freqs_cis)

    batch = Dim("batch", min=1, max=4)
    seq_len = Dim("seq_len", min=2, max=512)
    # x: (batch, seq_len, n_heads, head_dim)
    # freqs_cis: (seq_len, head_dim//2) complex – dim 0 is seq_len
    dynamic_shapes = (
        {0: batch, 1: seq_len},
        {0: seq_len},
    )
    exp_program = torch.export.export(model, inputs, dynamic_shapes=dynamic_shapes)

    compile_spec = {
        "inputs": inputs,
        "min_block_size": 1,
        "pass_through_build_failures": True,
    }
    trt_model = torchtrt.dynamo.compile(exp_program, **compile_spec)

    py_out = model(*inputs)
    trt_out = trt_model(*inputs)

    cos_sim = cosine_similarity(py_out, trt_out)
    assert cos_sim > COSINE_THRESHOLD, (
        f"test_rope_complex_form_dynamic: outputs differ. "
        f"Cosine sim: {cos_sim:.4f} < threshold {COSINE_THRESHOLD}"
    )
    torch._dynamo.reset()


# ---------------------------------------------------------------------------
# Test 5: RoPE embedded inside an attention block – static shapes
# ---------------------------------------------------------------------------


class AttentionWithRoPE(nn.Module):
    """Minimal self-attention block with HF-style RoPE, as found in LLaMA/Qwen.

    This exercises RoPE inside a larger graph—a common failure mode where
    the shape inference for cos/sin unsqueeze interacts with the projection
    output shapes.
    """

    def __init__(self, embed_dim: int = 64, n_heads: int = 4):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.o_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def apply_rotary_pos_emb(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ):
        cos = cos.unsqueeze(1)  # add head dim
        sin = sin.unsqueeze(1)
        return (q * cos) + (self.rotate_half(q) * sin), (k * cos) + (
            self.rotate_half(k) * sin
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        batch, seq_len, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        q, k = self.apply_rotary_pos_emb(q, k, cos, sin)

        attn_out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn_out = attn_out.transpose(1, 2).reshape(batch, seq_len, -1)
        return self.o_proj(attn_out)


@pytest.mark.unit
def test_rope_in_attention_block_static():
    """RoPE inside a full attention block compiles correctly (static shapes)."""
    EMBED_DIM, N_HEADS, BATCH, SEQ_LEN = 64, 4, 2, 16
    HEAD_DIM = EMBED_DIM // N_HEADS

    model = AttentionWithRoPE(EMBED_DIM, N_HEADS).eval().cuda()

    hidden = torch.randn(BATCH, SEQ_LEN, EMBED_DIM, dtype=torch.float32).cuda()
    cos = torch.randn(BATCH, SEQ_LEN, HEAD_DIM, dtype=torch.float32).cuda()
    sin = torch.randn(BATCH, SEQ_LEN, HEAD_DIM, dtype=torch.float32).cuda()
    inputs = (hidden, cos, sin)

    compile_spec = {
        "inputs": inputs,
        "ir": "dynamo",
        "min_block_size": 1,
        "pass_through_build_failures": True,
    }
    trt_model = torchtrt.compile(model, **compile_spec)

    py_out = model(*inputs)
    trt_out = trt_model(*inputs)

    cos_sim = cosine_similarity(py_out, trt_out)
    assert cos_sim > COSINE_THRESHOLD, (
        f"test_rope_in_attention_block_static: outputs differ. "
        f"Cosine sim: {cos_sim:.4f} < threshold {COSINE_THRESHOLD}"
    )
    torch._dynamo.reset()


# ---------------------------------------------------------------------------
# Test 6: RoPE embedded inside an attention block – dynamic seq_len
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_rope_in_attention_block_dynamic():
    """RoPE inside a full attention block compiles correctly (dynamic seq_len)."""
    EMBED_DIM, N_HEADS, BATCH, SEQ_LEN = 64, 4, 2, 16
    HEAD_DIM = EMBED_DIM // N_HEADS

    model = AttentionWithRoPE(EMBED_DIM, N_HEADS).eval().cuda()

    hidden = torch.randn(BATCH, SEQ_LEN, EMBED_DIM, dtype=torch.float32).cuda()
    cos = torch.randn(BATCH, SEQ_LEN, HEAD_DIM, dtype=torch.float32).cuda()
    sin = torch.randn(BATCH, SEQ_LEN, HEAD_DIM, dtype=torch.float32).cuda()
    inputs = (hidden, cos, sin)

    seq_len = Dim("seq_len", min=2, max=2048)
    # hidden: (batch, seq_len, embed_dim) – seq_len is dim 1
    # cos/sin: (batch, seq_len, head_dim) – seq_len is dim 1
    dynamic_shapes = (
        {1: seq_len},
        {1: seq_len},
        {1: seq_len},
    )
    exp_program = torch.export.export(model, inputs, dynamic_shapes=dynamic_shapes)

    compile_spec = {
        "inputs": inputs,
        "min_block_size": 1,
        "pass_through_build_failures": True,
    }
    trt_model = torchtrt.dynamo.compile(exp_program, **compile_spec)

    py_out = model(*inputs)
    trt_out = trt_model(*inputs)

    cos_sim = cosine_similarity(py_out, trt_out)
    assert cos_sim > COSINE_THRESHOLD, (
        f"test_rope_in_attention_block_dynamic: outputs differ. "
        f"Cosine sim: {cos_sim:.4f} < threshold {COSINE_THRESHOLD}"
    )
    torch._dynamo.reset()


# ---------------------------------------------------------------------------
# Test 7: Complex RoPE – serialization with retrace=True then inference
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_rope_complex_form_serialization_retrace(tmp_path):
    """Complex RoPE survives save(retrace=True) + load + inference round-trip.

    When retrace=True, torch_tensorrt.save re-exports the compiled GraphModule
    via torch.export.export (strict=False), inlining the view_as_real unpacking
    ops that live in the Python runtime forward(). The reloaded ExportedProgram
    must accept the original complex inputs and produce correct results.
    """
    BATCH, SEQ_LEN, N_HEADS, HEAD_DIM = 2, 8, 4, 64
    model = ComplexRotaryEmbedding().eval().cuda()

    x = torch.randn(BATCH, SEQ_LEN, N_HEADS, HEAD_DIM, dtype=torch.float32).cuda()
    freqs_cis = _make_freqs_cis(SEQ_LEN, HEAD_DIM)
    inputs = (x, freqs_cis)

    # Step 1: compile
    compile_spec = {
        "inputs": inputs,
        "ir": "dynamo",
        "min_block_size": 1,
        "pass_through_build_failures": True,
    }
    trt_model = torchtrt.compile(model, **compile_spec)

    py_out = model(*inputs)
    trt_out_before = trt_model(*inputs)

    cos_sim_before = cosine_similarity(py_out, trt_out_before)
    assert cos_sim_before > COSINE_THRESHOLD, (
        f"test_rope_complex_form_serialization_retrace: pre-save TRT output wrong. "
        f"Cosine sim: {cos_sim_before:.4f} < threshold {COSINE_THRESHOLD}"
    )

    # Step 2: save with retrace=True — re-exports the compiled GraphModule so
    # the view_as_real input-unpacking is inlined into the exported graph.
    ep_path = str(tmp_path / "rope_complex_trt.ep")
    torchtrt.save(
        trt_model,
        ep_path,
        output_format="exported_program",
        arg_inputs=list(inputs),
        retrace=True,
    )
    assert os.path.exists(ep_path), "Serialized .ep file was not created"

    # Step 3: reload
    loaded_ep = torchtrt.load(ep_path)
    # torch_tensorrt.load returns ExportedProgram; call .module() to get the
    # callable GraphModule.
    loaded_module = loaded_ep.module()

    # Step 4: inference on reloaded model
    trt_out_after = loaded_module(*inputs)

    cos_sim_after = cosine_similarity(py_out, trt_out_after)
    assert cos_sim_after > COSINE_THRESHOLD, (
        f"test_rope_complex_form_serialization_retrace: post-load TRT output wrong. "
        f"Cosine sim: {cos_sim_after:.4f} < threshold {COSINE_THRESHOLD}"
    )
    torch._dynamo.reset()


# ---------------------------------------------------------------------------
# Test 8: Complex output – model whose output is a complex tensor
# ---------------------------------------------------------------------------


class ComplexOutputModel(nn.Module):
    """A model that outputs a complex tensor.

    This exercises the post-partition complex output restoration pass:
    complex_graph_detection rewrites the internal complex ops to real
    arithmetic before partitioning, and the compiler must re-insert
    view_as_complex at the output boundary when the tail block is TRT.
    """

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, n_heads, head_dim) – real
        # freqs_cis: (seq_len, head_dim // 2) – complex
        # Returns: complex tensor of shape (batch, seq_len, n_heads, head_dim // 2)
        x_ = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        freqs_cis = freqs_cis[None, :, None, :]
        return x_ * freqs_cis  # complex output – no view_as_real


@pytest.mark.unit
def test_complex_output_static():
    """Model with a complex tensor output compiles and produces correct results."""
    model = ComplexOutputModel().eval().cuda()

    x = torch.randn(1, 4, 8, 64, dtype=torch.float32).cuda()
    freqs_cis = _make_freqs_cis(4, 64)  # shape (4, 32), complex64
    inputs = (x, freqs_cis)

    compile_spec = {
        "inputs": inputs,
        "ir": "dynamo",
        "min_block_size": 1,
        "pass_through_build_failures": True,
    }
    trt_model = torchtrt.compile(model, **compile_spec)

    py_out = model(*inputs)
    trt_out = trt_model(*inputs)

    assert trt_out.is_complex(), (
        f"test_complex_output_static: TRT output should be complex, got dtype {trt_out.dtype}"
    )
    assert trt_out.shape == py_out.shape, (
        f"test_complex_output_static: shape mismatch {trt_out.shape} vs {py_out.shape}"
    )
    # Compare real and imaginary parts via cosine similarity
    cos_sim_real = cosine_similarity(py_out.real, trt_out.real)
    cos_sim_imag = cosine_similarity(py_out.imag, trt_out.imag)
    assert cos_sim_real > COSINE_THRESHOLD, (
        f"test_complex_output_static: real part cosine sim {cos_sim_real:.4f} < {COSINE_THRESHOLD}"
    )
    assert cos_sim_imag > COSINE_THRESHOLD, (
        f"test_complex_output_static: imag part cosine sim {cos_sim_imag:.4f} < {COSINE_THRESHOLD}"
    )
    torch._dynamo.reset()
