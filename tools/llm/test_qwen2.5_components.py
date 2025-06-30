import torch

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

import argparse
import os
import sys
from contextlib import nullcontext

import torch.nn as nn
import torch_tensorrt
from torch.testing._internal.common_utils import TestCase, run_tests
from transformers import AutoModelForCausalLM
from transformers.models.llama.configuration_llama import LlamaConfig

# Register SDPA as a standalone operator. Converter and lowering pass are defined in register_sdpa.py
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from register_sdpa import *

ATOL = 1e-5
RTOL = 1e-5


qwen2_5_model_name = "Qwen/Qwen2.5-1.5B-Instruct"
qwen2_5_model = (
    AutoModelForCausalLM.from_pretrained(
        qwen2_5_model_name,
        use_cache=False,
        attn_implementation="sdpa",
        num_hidden_layers=1,
    )
    .eval()
    .cuda()
)
QWEN_CONFIG = qwen2_5_model.config


def print_diff(tensor1, tensor2, prefix=""):
    """
    Print the diff between two tensors
    """
    print(
        f"[{prefix}] Diff between tensor1 and tensor2: {torch.mean(torch.abs(tensor1 - tensor2))}"
    )


def test_qwen_apply_rotary_pos_emb(args):
    class QwenApplyRotaryPosEmb(nn.Module):
        def __init__(self):
            super().__init__()

        def rotate_half(self, x):
            x1 = x[..., : x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2 :]
            return torch.cat((-x2, x1), dim=-1)

        def apply_rotary_pos_emb(self, q, k, cos, sin, unsqueeze_dim=1):
            cos = cos.unsqueeze(unsqueeze_dim)
            sin = sin.unsqueeze(unsqueeze_dim)
            q_embed = (q * cos) + (self.rotate_half(q) * sin)
            k_embed = (k * cos) + (self.rotate_half(k) * sin)
            return q_embed, k_embed

        def forward(self, q, k, cos, sin, unsqueeze_dim=1):
            return self.apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim)

    DTYPE = torch.float32
    if args.precision == "FP16":
        DTYPE = torch.float16
    elif args.precision == "BF16":
        DTYPE = torch.bfloat16

    # Set precision specific flags
    use_fp32_acc = False
    use_explicit_typing = False
    if args.precision == "FP16":
        enabled_precisions = {torch.float32}
        use_fp32_acc = True
        use_explicit_typing = True
    elif args.precision == "BF16":
        enabled_precisions = {torch.bfloat16}
        use_fp32_acc = False
    else:
        enabled_precisions = {torch.float32}

    model = QwenApplyRotaryPosEmb().eval().cuda().to(DTYPE)
    # Shapes for Qwen 2.5
    q = torch.randn((1, 12, 5, 128), dtype=DTYPE).cuda()
    k = torch.randn((1, 12, 5, 128), dtype=DTYPE).cuda()
    cos = torch.randn((1, 5, 128), dtype=DTYPE).cuda()
    sin = torch.randn((1, 5, 128), dtype=DTYPE).cuda()

    pyt_output = model(q, k, cos, sin)

    seq_len = torch.export.Dim("seq_len", min=2, max=2176)
    dynamic_shapes = ({2: seq_len}, {2: seq_len}, {1: seq_len}, {1: seq_len})
    ep = torch.export.export(model, (q, k, cos, sin), dynamic_shapes=dynamic_shapes)
    with torch_tensorrt.logging.debug() if args.debug else nullcontext():
        trt_model = torch_tensorrt.dynamo.compile(
            ep,
            inputs=[q, k, cos, sin],
            enabled_precisions=enabled_precisions,
            disable_tf32=True,
            use_fp32_acc=use_fp32_acc,
            use_explicit_typing=use_explicit_typing,
            debug=args.debug,
        )
    trt_output = trt_model(q, k, cos, sin)

    if isinstance(pyt_output, tuple):
        print_diff(pyt_output[0], trt_output[0], "Diff b/w pyt and trt")
        # print_diff(pyt_output[1], trt_output[1], "Diff b/w pyt and trt")
        assert torch.allclose(pyt_output[0], trt_output[0], atol=ATOL, rtol=RTOL)
    else:
        print_diff(pyt_output, trt_output, "Diff b/w pyt and trt")
        assert torch.allclose(pyt_output, trt_output, atol=ATOL, rtol=RTOL)


def test_qwen_attention(args):

    DTYPE = torch.float32
    if args.precision == "FP16":
        DTYPE = torch.float16
    elif args.precision == "BF16":
        DTYPE = torch.bfloat16

    # Set precision specific flags
    use_fp32_acc = False
    use_explicit_typing = False
    if args.precision == "FP16":
        enabled_precisions = {torch.float32}
        use_fp32_acc = True
        use_explicit_typing = True
    elif args.precision == "BF16":
        enabled_precisions = {torch.bfloat16}
        use_fp32_acc = False
    else:
        enabled_precisions = {torch.float32}

    model = qwen2_5_model.model.layers[0].self_attn.to(DTYPE)
    # qwen2.5
    hidden_states = torch.randn((1, 5, 1536), dtype=DTYPE).cuda()
    position_embeddings = (
        torch.randn((1, 5, 128), dtype=DTYPE).cuda(),
        torch.randn((1, 5, 128), dtype=DTYPE).cuda(),
    )

    pyt_output = model(hidden_states, position_embeddings, None)

    seq_len = torch.export.Dim("seq_len", min=2, max=2176)
    dynamic_shapes = ({1: seq_len}, ({1: seq_len}, {1: seq_len}), None)
    ep = torch.export.export(
        model, (hidden_states, position_embeddings, None), dynamic_shapes=dynamic_shapes
    )

    with torch_tensorrt.logging.debug() if args.debug else nullcontext():
        trt_model = torch_tensorrt.dynamo.compile(
            ep,
            inputs=[hidden_states, position_embeddings, None],
            enabled_precisions=enabled_precisions,
            disable_tf32=True,
            use_fp32_acc=use_fp32_acc,
            use_explicit_typing=use_explicit_typing,
            debug=args.debug,
        )
    trt_output = trt_model(hidden_states, position_embeddings, None)

    if isinstance(pyt_output, tuple):
        print_diff(pyt_output[0], trt_output[0], "Diff b/w pyt and trt")
        assert torch.allclose(pyt_output[0], trt_output[0], atol=ATOL, rtol=RTOL)
    else:
        print_diff(pyt_output, trt_output, "Diff b/w pyt and trt")
        assert torch.allclose(pyt_output, trt_output, atol=ATOL, rtol=RTOL)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(
        description="Run test cases for llama attention and decoder"
    )
    arg_parser.add_argument(
        "--debug", action="store_true", help="Enable debug (default: False)"
    )
    arg_parser.add_argument(
        "--precision",
        type=str,
        default="FP16",
        help="Precision to use in the model. Options: FP16, BF16, FP32",
    )
    args = arg_parser.parse_args()
    with torch.inference_mode():
        # test_qwen_apply_rotary_pos_emb(args)
        test_qwen_attention(args)
