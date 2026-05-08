"""
Example: ViT Attention TensorRT Plugin Integration
================================================

This example shows how to use custom TensorRT plugin that implements
ViT self-attention using a fused QKV input.

The Python code demonstrates:
- loading the TensorRT plugin shared library
- registering a placeholder custom op for TorchDynamo conversion
- converting that custom op to a TensorRT plugin layer
- comparing a PyTorch reference self-attention implementation with the plugin model

attention_plugin_example.py is model-agnostic around this invariant:
LLM decode attention = RoPE + KV cache + causal/GQA attention

vit_attention_plugin_example.py is model-agnostic around this invariant:
ViT/VLA visual attention = full/window bidirectional attention over image tokens
"""

import os
import ctypes
from dataclasses import dataclass
from typing import Callable, Dict

import numpy as np
import tensorrt as trt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_tensorrt
from torch_tensorrt.dynamo.conversion import ConversionContext, dynamo_tensorrt_converter
from torch_tensorrt.dynamo.conversion.converter_utils import get_trt_tensor

# Enable plugin debug logging
# ----------------------------
#os.environ["TRT_EDGELLM_DEBUG_PLUGIN"] = "1"

# Initialize CUDA and Load Plugin
# --------------------------------
# CUDA must be initialized before loading the TensorRT plugin library
print("Initializing CUDA context...")
DEVICE = torch.device("cuda:0")
_ = torch.zeros(1, device=DEVICE)  # Initialize CUDA
print(f"CUDA initialized on {DEVICE}\n")

PLUGIN_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
    "TensorRT-Edge-LLM",
    "build",
    "libNvInfer_edgellm_plugin.so",
)
if not os.path.exists(PLUGIN_PATH):
    PLUGIN_PATH = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "TensorRT-Edge-LLM",
        "build",
        "libNvInfer_edgellm_plugin.so",
    )
ctypes.CDLL(PLUGIN_PATH)
print(f"Loaded plugin: {PLUGIN_PATH}\n")

# -----------------------------------------------------------------------------
# Model configuration
# -----------------------------------------------------------------------------

BATCH_SIZE = 1
NUM_HEADS = 4
HEAD_DIM = 16
EMBED_DIM = NUM_HEADS * HEAD_DIM
SEQ_LEN = 16
DTYPE = torch.float16
DEVICE = torch.device("cuda:0")
NUM_WARMUP = 10
NUM_BENCHMARK_RUNS = 100

def apply_qwen_vl_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply Qwen-VL rotate-half RoPE to [B, H, S, D] tensors."""
    cos = cos.view(1, 1, cos.shape[0], cos.shape[1]).to(dtype=x.dtype)
    sin = sin.view(1, 1, sin.shape[0], sin.shape[1]).to(dtype=x.dtype)
    half_dim = x.shape[-1] // 2
    x1 = x[..., :half_dim]
    x2 = x[..., half_dim:]
    rotated = torch.cat((-x2, x1), dim=-1)
    return x * cos + rotated * sin


def create_identity_rope(seq_len: int, dtype: torch.dtype, device: torch.device):
    cos = torch.ones(seq_len, HEAD_DIM, dtype=dtype, device=device)
    sin = torch.zeros_like(cos)
    return cos, sin


def create_qwen_vl_rope(seq_len: int, dtype: torch.dtype, device: torch.device):
    inv_freq = 1.0 / (
        10000.0 ** (torch.arange(0, HEAD_DIM, 2, dtype=torch.float32, device=device) / HEAD_DIM)
    )
    positions = torch.arange(seq_len, dtype=torch.float32, device=device)
    freqs = torch.outer(positions, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(dtype=dtype), emb.sin().to(dtype=dtype)


def create_zero_mask(batch_size: int, seq_len: int, dtype: torch.dtype, device: torch.device):
    return torch.zeros(batch_size, seq_len, seq_len, dtype=dtype, device=device)


def create_window_mask(batch_size: int, seq_len: int, dtype: torch.dtype, device: torch.device, window_size: int = 4):
    mask = torch.full((batch_size, seq_len, seq_len), torch.finfo(dtype).min, dtype=dtype, device=device)
    for start in range(0, seq_len, window_size):
        end = min(start + window_size, seq_len)
        mask[:, start:end, start:end] = 0
    return mask


# -----------------------------------------------------------------------------
# Model-agnostic attention module
# -----------------------------------------------------------------------------


class ModelAgnosticViTAttention(nn.Module):
    """
    Model-agnostic attention wrapper for reference and plugin paths.

    Projection layout is inferred from the wrapped module:
    - ``qkv`` + ``proj/out/out_proj`` => fused QKV attention
    - ``q_proj/k_proj/v_proj`` + ``o_proj/out_proj/out`` => separate QKV attention

    RoPE and masks are inferred from forward inputs:
    - provided ``cos`` and ``sin`` means apply Qwen-VL-style rotate-half RoPE
    - no ``cos``/``sin`` means identity RoPE
    - provided ``attention_mask`` is used as-is
    - no ``attention_mask`` means full bidirectional attention
    """

    def __init__(self, original_attn: nn.Module, use_plugin: bool):
        super().__init__()
        self.original_attn = original_attn
        self.use_plugin = use_plugin
        self.projection_layout = self._detect_projection_layout(original_attn)
        self.output_proj = self._detect_output_projection(original_attn)

    @staticmethod
    def _detect_projection_layout(attn: nn.Module) -> str:
        if hasattr(attn, "qkv"):
            return "fused_qkv"
        if all(hasattr(attn, attr) for attr in ("q_proj", "k_proj", "v_proj")):
            return "separate_qkv"
        if all(hasattr(attn, attr) for attr in ("query", "key", "value")):
            return "separate_hf_vit"
        raise ValueError(
            f"Cannot detect QKV projection layout for {attn.__class__.__name__}"
        )

    @staticmethod
    def _detect_output_projection(attn: nn.Module) -> nn.Module:
        for attr in ("proj", "o_proj", "out_proj", "out"):
            if hasattr(attn, attr):
                return getattr(attn, attr)
        if hasattr(attn, "output"):
            output = getattr(attn, "output")
            return output.dense if hasattr(output, "dense") else output
        raise ValueError(f"Cannot detect output projection for {attn.__class__.__name__}")

    def _project_qkv(self, x: torch.Tensor) -> torch.Tensor:
        if self.projection_layout == "fused_qkv":
            return self.original_attn.qkv(x)

        if self.projection_layout == "separate_qkv":
            q = self.original_attn.q_proj(x)
            k = self.original_attn.k_proj(x)
            v = self.original_attn.v_proj(x)
        elif self.projection_layout == "separate_hf_vit":
            q = self.original_attn.query(x)
            k = self.original_attn.key(x)
            v = self.original_attn.value(x)
        else:
            raise ValueError(f"Unknown projection layout: {self.projection_layout}")
        return torch.cat([q, k, v], dim=-1)

    def _reference_attention(
        self,
        qkv: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: torch.Tensor,
        has_position_embeddings: bool,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = qkv.shape
        qkv = qkv.view(batch_size, seq_len, 3, NUM_HEADS, HEAD_DIM).permute(2, 0, 3, 1, 4)
        query, key, value = qkv[0], qkv[1], qkv[2]

        if has_position_embeddings:
            query = apply_qwen_vl_rope(query, cos, sin)
            key = apply_qwen_vl_rope(key, cos, sin)

        attn = F.scaled_dot_product_attention(
            query.contiguous(),
            key.contiguous(),
            value.contiguous(),
            attn_mask=attention_mask.unsqueeze(1),
            dropout_p=0.0,
            is_causal=False,
        )
        return attn.transpose(1, 2).contiguous().view(batch_size, seq_len, EMBED_DIM)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor = None,
        cos: torch.Tensor = None,
        sin: torch.Tensor = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        qkv = self._project_qkv(x)
        has_position_embeddings = cos is not None and sin is not None
        if not has_position_embeddings:
            cos, sin = create_identity_rope(seq_len, x.dtype, x.device)
        else:
            cos = cos.to(dtype=x.dtype)
            sin = sin.to(dtype=x.dtype)

        if attention_mask is None:
            attention_mask = create_zero_mask(batch_size, seq_len, x.dtype, x.device)
        else:
            attention_mask = attention_mask.to(dtype=x.dtype)

        if self.use_plugin:
            attn_out = torch.ops.tensorrt_vit_attention.attn.default(
                qkv,
                cos,
                sin,
                attention_mask,
                NUM_HEADS,
                HEAD_DIM,
                1,
            )
        else:
            attn_out = self._reference_attention(
                qkv,
                cos,
                sin,
                attention_mask,
                has_position_embeddings,
            )

        return self.output_proj(attn_out)

class FusedQKVFakeAttention(nn.Module):
    """Fake module with Qwen/ViT-like fused QKV names."""

    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(EMBED_DIM, 3 * EMBED_DIM)
        self.proj = nn.Linear(EMBED_DIM, EMBED_DIM)

class SeparateQKVFakeAttention(nn.Module):
    """Fake module with Llama Vision/SigLip/GR00T-like separate QKV names."""

    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(EMBED_DIM, EMBED_DIM)
        self.k_proj = nn.Linear(EMBED_DIM, EMBED_DIM)
        self.v_proj = nn.Linear(EMBED_DIM, EMBED_DIM)
        self.o_proj = nn.Linear(EMBED_DIM, EMBED_DIM)


@dataclass(frozen=True)
class AttentionCase:
    name: str
    attention_factory: Callable[[], nn.Module]
    kwargs_factory: Callable[[torch.Tensor], Dict[str, torch.Tensor]]


def no_extra_inputs(x: torch.Tensor) -> Dict[str, torch.Tensor]:
    return {}


def qwen_vl_inputs(x: torch.Tensor) -> Dict[str, torch.Tensor]:
    batch_size, seq_len, _ = x.shape
    cos, sin = create_qwen_vl_rope(seq_len, x.dtype, x.device)
    return {
        "attention_mask": create_window_mask(batch_size, seq_len, x.dtype, x.device),
        "cos": cos,
        "sin": sin,
    }


ATTENTION_CASES = [
    AttentionCase(
        name="Plain ViT Attention",
        attention_factory=FusedQKVFakeAttention,
        kwargs_factory=no_extra_inputs,
    ),
    AttentionCase(
        name="QwenVL-Style Attention",
        attention_factory=FusedQKVFakeAttention,
        kwargs_factory=qwen_vl_inputs,
    ),
    AttentionCase(
        name="LlamaVision-Style Attention",
        attention_factory=SeparateQKVFakeAttention,
        kwargs_factory=no_extra_inputs,
    ),
    AttentionCase(
        name="GR00T/SigLip2-Style Attention",
        attention_factory=SeparateQKVFakeAttention,
        kwargs_factory=no_extra_inputs,
    ),
]

# -----------------------------------------------------------------------------
# Plugin operation registration
# -----------------------------------------------------------------------------

def register_vit_attention_op():
    @torch.library.custom_op("tensorrt_vit_attention::attn", mutates_args=())
    def vit_attention(
        qkv: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: torch.Tensor,
        num_heads: int,
        head_dim: int,
        qkv_fused: int = 1,
    ) -> torch.Tensor:
        batch_size = qkv.shape[0]
        seq_len = qkv.shape[1]
        return torch.zeros(batch_size, seq_len, EMBED_DIM, dtype=qkv.dtype, device=qkv.device)

    @torch.library.register_fake("tensorrt_vit_attention::attn")
    def _(qkv, cos, sin, attention_mask, num_heads, head_dim, qkv_fused=1):
        batch_size = qkv.shape[0]
        seq_len = qkv.shape[1]
        return torch.empty(batch_size, seq_len, EMBED_DIM, dtype=qkv.dtype, device=qkv.device)


register_vit_attention_op()


@dynamo_tensorrt_converter(torch.ops.tensorrt_vit_attention.attn.default, supports_dynamic_shapes=True)
def convert_vit_attention(ctx: ConversionContext, target, args, kwargs, name):
    qkv, cos, sin, attention_mask, num_heads, head_dim = args[:6]
    qkv_fused = args[6] if len(args) > 6 else kwargs.get("qkv_fused", 1)

    creator = trt.get_plugin_registry().get_plugin_creator("ViTAttentionPlugin", "1", "")
    if creator is None:
        raise RuntimeError("ViTAttentionPlugin not found! Make sure the plugin library is loaded.")

    field_list = [
        trt.PluginField("num_heads", np.array([num_heads], dtype=np.int32), trt.PluginFieldType.INT32),
        trt.PluginField("head_size", np.array([head_dim], dtype=np.int32), trt.PluginFieldType.INT32),
        trt.PluginField("qkv_fused", np.array([qkv_fused], dtype=np.int32), trt.PluginFieldType.INT32),
    ]
    fields = trt.PluginFieldCollection(field_list)
    plugin = creator.create_plugin(name, fields)
    if plugin is None:
        raise RuntimeError("Failed to create ViTAttentionPlugin")

    input_tensors = [
        get_trt_tensor(ctx, qkv, "qkv"),
        get_trt_tensor(ctx, cos, "cos"),
        get_trt_tensor(ctx, sin, "sin"),
        get_trt_tensor(ctx, attention_mask, "attention_mask"),
    ]
    layer = ctx.net.add_plugin_v2(input_tensors, plugin)
    return layer.get_output(0)

# -----------------------------------------------------------------------------
# Example execution
# -----------------------------------------------------------------------------

def benchmark_model(fn, num_warmup: int = NUM_WARMUP, num_runs: int = NUM_BENCHMARK_RUNS):
    for _ in range(num_warmup):
        fn()
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    timings = []

    for _ in range(num_runs):
        start_event.record()
        fn()
        end_event.record()
        torch.cuda.synchronize()
        timings.append(start_event.elapsed_time(end_event))

    timings_tensor = torch.tensor(timings, dtype=torch.float32)
    return (
        timings_tensor.mean().item(),
        timings_tensor.median().item(),
        timings_tensor.std(unbiased=False).item(),
    )


def run_attention_case(
    case_name: str,
    reference_model: nn.Module,
    plugin_model: nn.Module,
    x: torch.Tensor,
    kwargs,
):
    plugin_model.load_state_dict(reference_model.state_dict())

    with torch.no_grad():
        ref_out = reference_model(x, **kwargs)

    print(f"\n=== {case_name} ===")
    print("Compiling TensorRT ViT attention plugin model...")
    dynamic_shapes = {"x": {}}
    dynamic_shapes.update({key: {} for key in kwargs})
    ep = torch.export.export(
        plugin_model,
        args=(x,),
        kwargs=kwargs,
        dynamic_shapes=dynamic_shapes,
        strict=False,
    )
    trt_model = torch_tensorrt.dynamo.compile(
        ep,
        inputs=[x, *kwargs.values()],
        use_explicit_typing=True,
        use_fp32_acc=True,
        device=DEVICE,
        disable_tf32=True,
        min_block_size=1,
    )

    with torch.no_grad():
        plugin_out = trt_model(x, **kwargs)

    print("Reference output shape:", ref_out.shape)
    print("Plugin output shape:", plugin_out.shape)
    max_abs_diff = (ref_out - plugin_out).abs().max().item()
    cosine = F.cosine_similarity(ref_out.flatten().float(), plugin_out.flatten().float(), dim=0).item()
    passed = cosine >= 0.99
    print(f"Max absolute difference: {max_abs_diff:.6f}")
    print(f"Cosine similarity: {cosine:.6f}")
    print(f"Result: {'PASS' if passed else 'FAIL'}")

    ref_mean, ref_median, ref_std = benchmark_model(lambda: reference_model(x, **kwargs))
    trt_mean, trt_median, trt_std = benchmark_model(lambda: trt_model(x, **kwargs))
    print("Latency:")
    print(f"  PyTorch SDPA | Mean: {ref_mean:.4f} ms | Median: {ref_median:.4f} ms | Std: {ref_std:.4f} ms")
    print(f"  TensorRT     | Mean: {trt_mean:.4f} ms | Median: {trt_median:.4f} ms | Std: {trt_std:.4f} ms")

    return passed, cosine, max_abs_diff, ref_mean, trt_mean


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(BATCH_SIZE, SEQ_LEN, EMBED_DIM, dtype=DTYPE, device=DEVICE)

    print("\nViT Attention Plugin - Correctness and Latency Validation")
    results = []
    for attention_case in ATTENTION_CASES:
        reference_attn = attention_case.attention_factory()
        plugin_attn = attention_case.attention_factory()
        reference_model = ModelAgnosticViTAttention(
            reference_attn,
            use_plugin=False,
        ).to(device=DEVICE, dtype=DTYPE).eval()
        plugin_model = ModelAgnosticViTAttention(
            plugin_attn,
            use_plugin=True,
        ).to(device=DEVICE, dtype=DTYPE).eval()
        kwargs = attention_case.kwargs_factory(x)
        results.append(
            (
                attention_case.name,
                run_attention_case(
                    attention_case.name,
                    reference_model,
                    plugin_model,
                    x,
                    kwargs,
                ),
            )
        )

    print("\nSUMMARY")
    for name, (passed, cosine, max_abs_diff, ref_mean, trt_mean) in results:
        status = "PASS" if passed else "FAIL"
        speedup = ref_mean / trt_mean if trt_mean > 0 else float("inf")
        print(f"{name}: {status}")
        print(f"  Cosine: {cosine:.4f}, Max abs diff: {max_abs_diff:.6f}")
        print(f"  PyTorch: {ref_mean:.4f} ms, TensorRT: {trt_mean:.4f} ms, Speedup: {speedup:.2f}x")

    all_passed = all(result[0] for _, result in results)
    if all_passed:
        print("SUCCESS: All ViT attention plugin tests passed!")
    else:
        print("FAILURE: Some ViT attention plugin tests failed")
