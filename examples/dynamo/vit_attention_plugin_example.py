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
HEAD_DIM = 64
EMBED_DIM = NUM_HEADS * HEAD_DIM
SEQ_LEN = 256
WINDOW_SIZE = 64
MASK_TYPE_DENSE = 0
MASK_TYPE_CU_SEQLENS = 1
DTYPE = torch.float16
DEVICE = torch.device("cuda:0")

def apply_qwen_vl_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    Apply Qwen-VL rotate-half RoPE to query/key tensors in [B, H, S, D].

    Qwen-style RoPE stores the first half and second half of each head as the
    two rotation components. This must match the C++ plugin's RoPE prepass for
    the dense and cu_seqlens paths to compare against the same reference.
    """
    cos = cos.view(1, 1, cos.shape[0], cos.shape[1]).to(dtype=x.dtype)
    sin = sin.view(1, 1, sin.shape[0], sin.shape[1]).to(dtype=x.dtype)
    half_dim = x.shape[-1] // 2
    x1 = x[..., :half_dim]
    x2 = x[..., half_dim:]
    rotated = torch.cat((-x2, x1), dim=-1)
    return x * cos + rotated * sin


def create_identity_rope(seq_len: int, dtype: torch.dtype, device: torch.device):
    """
    Create a no-op RoPE table.

    The plugin always receives cos/sin tensors, so plain ViT attention uses
    cos=1 and sin=0 instead of a special "no RoPE" code path.
    """
    cos = torch.ones(seq_len, HEAD_DIM, dtype=dtype, device=device)
    sin = torch.zeros_like(cos)
    return cos, sin


def create_qwen_vl_rope(seq_len: int, dtype: torch.dtype, device: torch.device):
    """
    Build Qwen-VL-style RoPE tables with shape [S, D].

    The embedding duplicates the frequency matrix across the head dimension so
    it aligns with rotate-half RoPE: [freqs, freqs].
    """
    inv_freq = 1.0 / (
        10000.0 ** (torch.arange(0, HEAD_DIM, 2, dtype=torch.float32, device=device) / HEAD_DIM)
    )
    positions = torch.arange(seq_len, dtype=torch.float32, device=device)
    freqs = torch.outer(positions, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(dtype=dtype), emb.sin().to(dtype=dtype)


def create_zero_mask(batch_size: int, seq_len: int, dtype: torch.dtype, device: torch.device):
    """
    Create a dense additive mask for full bidirectional attention.

    A value of zero means every token can attend to every other token.
    """
    return torch.zeros(batch_size, seq_len, seq_len, dtype=dtype, device=device)


def create_window_mask(batch_size: int, seq_len: int, dtype: torch.dtype, device: torch.device, window_size: int = WINDOW_SIZE):
    """
    Create a dense additive mask for independent local attention windows.

    Values outside each window are set to the minimum representable value for
    the dtype, which behaves like -inf after the attention score add.
    """
    mask = torch.full((batch_size, seq_len, seq_len), torch.finfo(dtype).min, dtype=dtype, device=device)
    for start in range(0, seq_len, window_size):
        end = min(start + window_size, seq_len)
        mask[:, start:end, start:end] = 0
    return mask


def create_full_cu_seqlens(seq_len: int, device: torch.device):
    """
    Create cu_seqlens for one packed full-attention segment.

    cu_seqlens is a prefix-sum array. [0, S] means one independent sequence
    containing S tokens.
    """
    return torch.tensor([0, seq_len], dtype=torch.int32, device=device)


def create_window_cu_seqlens(seq_len: int, device: torch.device, window_size: int = WINDOW_SIZE):
    """
    Create cu_seqlens for independent contiguous attention windows.

    For S=256 and window=64 this returns [0, 64, 128, 192, 256], meaning the
    plugin should launch four independent 64-token attention problems over one
    packed QKV tensor.
    """
    boundaries = list(range(0, seq_len, window_size))
    if boundaries[-1] != seq_len:
        boundaries.append(seq_len)
    return torch.tensor(boundaries, dtype=torch.int32, device=device)


# -----------------------------------------------------------------------------
# PyTorch SDPA and TensorRT plugin models
# -----------------------------------------------------------------------------

class ViTSDPAModel(nn.Module):
    """
    PyTorch SDPA reference model for ViT/VLA attention.

    Projection layout is selected explicitly:
    - ``fused_qkv`` => one Linear produces Q, K, and V together
    - ``separate_qkv`` => separate q_proj, k_proj, and v_proj layers

    This model is the correctness baseline. It uses torch.nn.functional
    scaled_dot_product_attention with the same dense additive mask semantics
    that the plugin should reproduce.
    """

    def __init__(self, projection_layout: str):
        """
        Create projection layers for the requested model-style layout.

        The layer names intentionally match ViTPluginModel so load_state_dict()
        can copy the exact same weights into the plugin model.
        """
        super().__init__()
        self.projection_layout = projection_layout

        if self.projection_layout == "fused_qkv":
            self.qkv = nn.Linear(EMBED_DIM, 3 * EMBED_DIM)
            self.output_proj = nn.Linear(EMBED_DIM, EMBED_DIM)
        elif self.projection_layout == "separate_qkv":
            self.q_proj = nn.Linear(EMBED_DIM, EMBED_DIM)
            self.k_proj = nn.Linear(EMBED_DIM, EMBED_DIM)
            self.v_proj = nn.Linear(EMBED_DIM, EMBED_DIM)
            self.output_proj = nn.Linear(EMBED_DIM, EMBED_DIM)
        else:
            raise ValueError(f"Unsupported projection layout: {projection_layout}")

    def project_qkv(self, x: torch.Tensor) -> torch.Tensor:
        """
        Produce a fused [B, S, 3 * H * D] QKV tensor.

        The plugin consumes fused QKV, so the SDPA reference uses the same
        representation before splitting into [Q, K, V].
        """
        if self.projection_layout == "fused_qkv":
            return self.qkv(x)

        if self.projection_layout == "separate_qkv":
            q = self.q_proj(x)
            k = self.k_proj(x)
            v = self.v_proj(x)
        else:
            raise ValueError(f"Unknown projection layout: {self.projection_layout}")
        return torch.cat([q, k, v], dim=-1)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor = None,
        cu_seqlens: torch.Tensor = None,
        cos: torch.Tensor = None,
        sin: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Run PyTorch SDPA using the dense equivalent of the requested attention.

        cu_seqlens is accepted so the reference and plugin models can share the
        same call signature, but SDPA consumes the dense additive mask.
        """
        batch_size, seq_len, _ = x.shape
        qkv = self.project_qkv(x)
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
        attn_out = attn.transpose(1, 2).contiguous().view(batch_size, seq_len, EMBED_DIM)
        return self.output_proj(attn_out)


class ViTPluginModel(nn.Module):
    """
    TensorRT plugin model for ViT/VLA attention.

    The projection layers mirror ViTSDPAModel. The attention math is replaced by
    the custom Torch op that convert_vit_attention lowers to ViTAttentionPlugin.
    """

    def __init__(
        self,
        projection_layout: str,
        plugin_mask_type: int = MASK_TYPE_DENSE,
        plugin_max_seq_len: int = 0,
    ):
        """
        Create projection layers and remember plugin execution settings.

        plugin_max_seq_len is only used for cu_seqlens FMHA. It is the maximum
        segment length represented by the prefix-sum array, which can be smaller
        than the total packed token count for windowed visual attention.
        """
        super().__init__()
        self.projection_layout = projection_layout
        self.plugin_mask_type = plugin_mask_type
        self.plugin_max_seq_len = plugin_max_seq_len

        if self.projection_layout == "fused_qkv":
            self.qkv = nn.Linear(EMBED_DIM, 3 * EMBED_DIM)
            self.output_proj = nn.Linear(EMBED_DIM, EMBED_DIM)
        elif self.projection_layout == "separate_qkv":
            self.q_proj = nn.Linear(EMBED_DIM, EMBED_DIM)
            self.k_proj = nn.Linear(EMBED_DIM, EMBED_DIM)
            self.v_proj = nn.Linear(EMBED_DIM, EMBED_DIM)
            self.output_proj = nn.Linear(EMBED_DIM, EMBED_DIM)
        else:
            raise ValueError(f"Unsupported projection layout: {projection_layout}")

    def project_qkv(self, x: torch.Tensor) -> torch.Tensor:
        """
        Produce the fused [B, S, 3 * H * D] tensor consumed by the plugin.

        Separate projection layouts are concatenated here so the plugin sees the
        same input layout regardless of upstream model style.
        """
        if self.projection_layout == "fused_qkv":
            return self.qkv(x)

        if self.projection_layout == "separate_qkv":
            q = self.q_proj(x)
            k = self.k_proj(x)
            v = self.v_proj(x)
        else:
            raise ValueError(f"Unknown projection layout: {self.projection_layout}")
        return torch.cat([q, k, v], dim=-1)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor = None,
        cu_seqlens: torch.Tensor = None,
        cos: torch.Tensor = None,
        sin: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Run the TensorRT ViTAttentionPlugin through a Torch custom op.

        Dense mode passes an additive mask. cu_seqlens mode passes prefix-sum
        boundaries that describe independent packed attention regions.
        """
        batch_size, seq_len, _ = x.shape
        qkv = self.project_qkv(x)
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

        mask_or_cu_seqlens = attention_mask
        if self.plugin_mask_type == MASK_TYPE_CU_SEQLENS:
            if cu_seqlens is None:
                cu_seqlens = create_full_cu_seqlens(seq_len, x.device)
            mask_or_cu_seqlens = cu_seqlens

        attn_out = torch.ops.tensorrt_vit_attention.attn.default(
            qkv,
            cos,
            sin,
            mask_or_cu_seqlens,
            NUM_HEADS,
            HEAD_DIM,
            1,
            self.plugin_mask_type,
            self.plugin_max_seq_len,
        )

        return self.output_proj(attn_out)


@dataclass(frozen=True)
class AttentionCase:
    """Bundle one model-style attention layout with its runtime inputs."""

    name: str
    projection_layout: str
    kwargs_factory: Callable[[torch.Tensor], Dict[str, torch.Tensor]]

def no_extra_inputs(x: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Create inputs for full bidirectional attention with identity RoPE.

    The dense path will synthesize a zero mask in forward(). The cu_seqlens path
    uses [0, S] to represent the same full attention region.
    """
    _, seq_len, _ = x.shape
    return {
        "cu_seqlens": create_full_cu_seqlens(seq_len, x.device),
    }

def qwen_vl_inputs(x: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Create inputs for QwenVL-style windowed visual attention.

    This is the key validation case for packed cu_seqlens: the dense mask and
    cu_seqlens tensor represent the same four independent windows.
    """
    batch_size, seq_len, _ = x.shape
    cos, sin = create_qwen_vl_rope(seq_len, x.dtype, x.device)
    return {
        "attention_mask": create_window_mask(batch_size, seq_len, x.dtype, x.device),
        "cu_seqlens": create_window_cu_seqlens(seq_len, x.device),
        "cos": cos,
        "sin": sin,
    }


ATTENTION_CASES = [
    AttentionCase(
        name="Plain ViT Attention",
        projection_layout="fused_qkv",
        kwargs_factory=no_extra_inputs,
    ),
    AttentionCase(
        name="QwenVL-Style Attention",
        projection_layout="fused_qkv",
        kwargs_factory=qwen_vl_inputs,
    ),
    AttentionCase(
        name="LlamaVision-Style Attention",
        projection_layout="separate_qkv",
        kwargs_factory=no_extra_inputs,
    ),
    AttentionCase(
        name="GR00T/SigLip2-Style Attention",
        projection_layout="separate_qkv",
        kwargs_factory=no_extra_inputs,
    ),
]

# -----------------------------------------------------------------------------
# Plugin operation registration
# -----------------------------------------------------------------------------

def register_vit_attention_op():
    """
    Register a Torch custom op that TorchDynamo can trace.

    The Python implementation is only a placeholder shape function. During
    Torch-TensorRT conversion, convert_vit_attention replaces this op with the
    real TensorRT plugin layer.
    """
    @torch.library.custom_op("tensorrt_vit_attention::attn", mutates_args=())
    def vit_attention(
        qkv: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask_or_cu_seqlens: torch.Tensor,
        num_heads: int,
        head_dim: int,
        qkv_fused: int = 1,
        mask_type: int = MASK_TYPE_DENSE,
        max_seq_len: int = 0,
    ) -> torch.Tensor:
        """Return an empty-shaped output for eager/tracing fallback."""
        batch_size = qkv.shape[0]
        seq_len = qkv.shape[1]
        output_dim = num_heads * head_dim
        return torch.zeros(batch_size, seq_len, output_dim, dtype=qkv.dtype, device=qkv.device)

    @torch.library.register_fake("tensorrt_vit_attention::attn")
    def _(qkv, cos, sin, mask_or_cu_seqlens, num_heads, head_dim, qkv_fused=1, mask_type=MASK_TYPE_DENSE, max_seq_len=0):
        """Provide fake tensor propagation for torch.export."""
        batch_size = qkv.shape[0]
        seq_len = qkv.shape[1]
        output_dim = num_heads * head_dim
        return torch.empty(batch_size, seq_len, output_dim, dtype=qkv.dtype, device=qkv.device)

register_vit_attention_op()

@dynamo_tensorrt_converter(torch.ops.tensorrt_vit_attention.attn.default, supports_dynamic_shapes=True)
def convert_vit_attention(ctx: ConversionContext, target, args, kwargs, name):
    """
    Convert the traced custom op into a ViTAttentionPlugin layer.

    Scalar arguments become TensorRT plugin fields. Tensor arguments become
    plugin inputs. max_seq_len is a field because FMHA needs it when cu_seqlens
    packs several smaller attention regions into one QKV tensor.
    """
    qkv, cos, sin, mask_or_cu_seqlens, num_heads, head_dim = args[:6]
    qkv_fused = args[6] if len(args) > 6 else kwargs.get("qkv_fused", 1)
    mask_type = args[7] if len(args) > 7 else kwargs.get("mask_type", MASK_TYPE_DENSE)
    max_seq_len = args[8] if len(args) > 8 else kwargs.get("max_seq_len", 0)

    creator = trt.get_plugin_registry().get_plugin_creator("ViTAttentionPlugin", "1", "")
    if creator is None:
        raise RuntimeError("ViTAttentionPlugin not found! Make sure the plugin library is loaded.")

    field_list = [
        trt.PluginField("num_heads", np.array([num_heads], dtype=np.int32), trt.PluginFieldType.INT32),
        trt.PluginField("head_size", np.array([head_dim], dtype=np.int32), trt.PluginFieldType.INT32),
        trt.PluginField("qkv_fused", np.array([qkv_fused], dtype=np.int32), trt.PluginFieldType.INT32),
        trt.PluginField("mask_type", np.array([mask_type], dtype=np.int32), trt.PluginFieldType.INT32),
        trt.PluginField("max_seq_len", np.array([max_seq_len], dtype=np.int32), trt.PluginFieldType.INT32),
    ]
    fields = trt.PluginFieldCollection(field_list)
    plugin = creator.create_plugin(name, fields)
    if plugin is None:
        raise RuntimeError("Failed to create ViTAttentionPlugin")

    input_tensors = [
        get_trt_tensor(ctx, qkv, "qkv"),
        get_trt_tensor(ctx, cos, "cos"),
        get_trt_tensor(ctx, sin, "sin"),
        get_trt_tensor(ctx, mask_or_cu_seqlens, "mask_or_cu_seqlens"),
    ]
    layer = ctx.net.add_plugin_v2(input_tensors, plugin)
    return layer.get_output(0)

# -----------------------------------------------------------------------------
# Correctness validation
# -----------------------------------------------------------------------------

def run_attention_case(
    case_name: str,
    plugin_label: str,
    plugin_mask_type: int,
    reference_model: nn.Module,
    plugin_model: nn.Module,
    x: torch.Tensor,
    kwargs,
):
    """
    Compile and validate one attention case for one plugin mask mode.

    The reference model and plugin model share weights. Inputs are normalized to
    positional tensors so this example can use the same torch_tensorrt.compile()
    style as attention_plugin_example.py.
    """
    plugin_model.load_state_dict(reference_model.state_dict())
    batch_size, seq_len, _ = x.shape
    attention_mask = kwargs.get(
        "attention_mask",
        create_zero_mask(batch_size, seq_len, x.dtype, x.device),
    )
    cu_seqlens = kwargs.get("cu_seqlens", create_full_cu_seqlens(seq_len, x.device))
    cos, sin = kwargs.get("cos"), kwargs.get("sin")
    if cos is None or sin is None:
        cos, sin = create_identity_rope(seq_len, x.dtype, x.device)

    runtime_inputs = (x, attention_mask, cu_seqlens, cos, sin)

    with torch.no_grad():
        ref_out = reference_model(*runtime_inputs)

    print(f"\n=== {case_name} | {plugin_label} ===")
    print("Compiling TensorRT ViT attention plugin model...")
    inputs_spec = [
        torch_tensorrt.Input(shape=tuple(x.shape), dtype=x.dtype),
        torch_tensorrt.Input(shape=tuple(attention_mask.shape), dtype=attention_mask.dtype),
        torch_tensorrt.Input(shape=tuple(cu_seqlens.shape), dtype=cu_seqlens.dtype),
        torch_tensorrt.Input(shape=tuple(cos.shape), dtype=cos.dtype),
        torch_tensorrt.Input(shape=tuple(sin.shape), dtype=sin.dtype),
    ]
    with torch_tensorrt.logging.errors():
        trt_model = torch_tensorrt.compile(
            plugin_model,
            inputs=inputs_spec,
            use_explicit_typing=True,
            use_fp32_acc=True,
            device=DEVICE,
            disable_tf32=True,
            min_block_size=1,
        )

    with torch.no_grad():
        plugin_out = trt_model(*runtime_inputs)

    print("Reference output shape:", ref_out.shape)
    print("Plugin output shape:", plugin_out.shape)
    max_abs_diff = (ref_out - plugin_out).abs().max().item()
    cosine = F.cosine_similarity(ref_out.flatten().float(), plugin_out.flatten().float(), dim=0).item()
    passed = cosine >= 0.99
    print(f"Max absolute difference: {max_abs_diff:.6f}")
    print(f"Cosine similarity: {cosine:.6f}")
    print(f"Result: {'PASS' if passed else 'FAIL'}")

    return passed, cosine, max_abs_diff


def get_plugin_max_seq_len(plugin_mask_type: int, kwargs: Dict[str, torch.Tensor]) -> int:
    """
    Return the maximum segment length required by the FMHA cu_seqlens path.

    For full attention this is S. For windowed attention this is the window
    length. Dense mode returns 0 so the plugin falls back to runtime S.
    """
    if plugin_mask_type != MASK_TYPE_CU_SEQLENS or "cu_seqlens" not in kwargs:
        return 0
    cu_seqlens = kwargs["cu_seqlens"]
    return int((cu_seqlens[1:] - cu_seqlens[:-1]).max().item())


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(BATCH_SIZE, SEQ_LEN, EMBED_DIM, dtype=DTYPE, device=DEVICE)

    print("\nViT Attention Plugin - Dense Mask vs cu_seqlens Correctness Validation")
    print(f"Config: B={BATCH_SIZE}, S={SEQ_LEN}, H={NUM_HEADS}, D={HEAD_DIM}, window={WINDOW_SIZE}")
    results = []
    for attention_case in ATTENTION_CASES:
        reference_model = ViTSDPAModel(
            attention_case.projection_layout,
        ).to(device=DEVICE, dtype=DTYPE).eval()
        kwargs = attention_case.kwargs_factory(x)

        for plugin_label, plugin_mask_type in (
            ("Dense additive mask", MASK_TYPE_DENSE),
            ("cu_seqlens FMHA", MASK_TYPE_CU_SEQLENS),
        ):
            plugin_model = ViTPluginModel(
                attention_case.projection_layout,
                plugin_mask_type=plugin_mask_type,
                plugin_max_seq_len=get_plugin_max_seq_len(plugin_mask_type, kwargs),
            ).to(device=DEVICE, dtype=DTYPE).eval()
            results.append(
                (
                    attention_case.name,
                    plugin_label,
                    run_attention_case(
                        attention_case.name,
                        plugin_label,
                        plugin_mask_type,
                        reference_model,
                        plugin_model,
                        x,
                        kwargs,
                    ),
                )
            )

    print("\nSUMMARY")
    for name, plugin_label, (passed, cosine, max_abs_diff) in results:
        status = "PASS" if passed else "FAIL"
        print(f"{name} | {plugin_label}: {status}")
        print(f"  Cosine: {cosine:.4f}, Max abs diff: {max_abs_diff:.6f}")

    all_passed = all(result[0] for _, _, result in results)
    if all_passed:
        print("SUCCESS: All ViT attention plugin tests passed!")
    else:
        print("FAILURE: Some ViT attention plugin tests failed")
