import torch

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

import torch.nn as nn
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.common_utils import TestCase
from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention, Qwen3DecoderLayer
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers import AutoModelForCausalLM
import torch_tensorrt
from contextlib import nullcontext
import argparse
import sys
import os

# Register SDPA as a standalone operator. Converter and lowering pass are defined in register_sdpa.py
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from register_sdpa import *

ATOL = 1e-5
RTOL = 1e-5


qwen3_model_name = "Qwen/Qwen3-0.6B"
qwen3_model = AutoModelForCausalLM.from_pretrained(
                qwen3_model_name,
                use_cache=False,
                attn_implementation="sdpa",
                num_hidden_layers=1,
            ).eval().cuda()
QWEN_CONFIG = qwen3_model.config

def print_diff(tensor1, tensor2, prefix=""):
    """
    Print the diff between two tensors
    """
    print(f"[{prefix}] Diff between tensor1 and tensor2: {torch.mean(torch.abs(tensor1 - tensor2))}")

    
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

    model = qwen3_model.model.layers[0].self_attn.to(DTYPE)
    # qwen2.5     
    hidden_states = torch.randn((1, 5, 1024), dtype=DTYPE).cuda()
    position_embeddings = (torch.randn((1, 5, 128), dtype=DTYPE).cuda(), torch.randn((1, 5, 128), dtype=DTYPE).cuda())

    pyt_output = model(hidden_states, position_embeddings, None)
    
    seq_len = torch.export.Dim("seq_len", min=2, max=2176)
    dynamic_shapes = ({1: seq_len}, ({1: seq_len}, {1: seq_len}), None)
    ep = torch.export.export(model, (hidden_states, position_embeddings, None), dynamic_shapes=dynamic_shapes)

    with (torch_tensorrt.logging.debug() if args.debug else nullcontext()):
        trt_model = torch_tensorrt.dynamo.compile(ep, 
                                                inputs=[hidden_states, position_embeddings, None], 
                                                enabled_precisions=enabled_precisions,
                                                disable_tf32=True,
                                                use_fp32_acc=use_fp32_acc,
                                                use_explicit_typing=use_explicit_typing,
                                                debug=args.debug)
    trt_output = trt_model(hidden_states, position_embeddings, None)
    
    if isinstance(pyt_output, tuple):
        print_diff(pyt_output[0], trt_output[0], "Diff b/w pyt and trt")
        assert torch.allclose(pyt_output[0], trt_output[0], atol=ATOL, rtol=RTOL)
    else:
        print_diff(pyt_output, trt_output, "Diff b/w pyt and trt")
        assert torch.allclose(pyt_output, trt_output, atol=ATOL, rtol=RTOL)

def test_qwen3_decoder(args):
    
    class QwenDecoderLayerBlock(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.config = QWEN_CONFIG
            self.model = model
        def forward(self, hidden_states, position_ids, position_embeddings):
            return self.model(hidden_states, position_ids=position_ids, position_embeddings=position_embeddings)

    DTYPE = torch.float32
    if args.precision == "FP16":
        DTYPE = torch.float16
    elif args.precision == "BF16":
        DTYPE = torch.bfloat16

    model = QwenDecoderLayerBlock(qwen3_model.model.layers[0].to(DTYPE))
    # qwen3 
    hidden_states = torch.randn((1, 5, 1024), dtype=DTYPE).cuda()
    position_ids = torch.randint(0, 5, (1, 5), dtype=torch.int64).cuda()
    position_embeddings = (torch.randn((1, 5, 128), dtype=DTYPE).cuda(), torch.randn((1, 5, 128), dtype=DTYPE).cuda())

    pyt_output = model(hidden_states, position_ids, position_embeddings)
    seq_len = torch.export.Dim("seq_len", min=2, max=2176)
    dynamic_shapes = ({1: seq_len}, {1: seq_len}, ({1: seq_len}, {1: seq_len}))
    ep = torch.export.export(model, (hidden_states, position_ids, position_embeddings), dynamic_shapes=dynamic_shapes)
    
    with (torch_tensorrt.logging.debug() if args.debug else nullcontext()):
        trt_model = torch_tensorrt.dynamo.compile(ep, 
                                                inputs=[hidden_states, position_ids, position_embeddings], 
                                                enabled_precisions={torch.float32},
                                                debug=args.debug)
    trt_output = trt_model(hidden_states, position_ids, position_embeddings)

    print(f"Diff b/w pyt and trt: {torch.mean(torch.abs(pyt_output[0] - trt_output[0]))}")
    assert torch.allclose(pyt_output[0], trt_output[0], atol=ATOL, rtol=RTOL)

def test_qwen3_model(args):

    DTYPE = torch.float32
    if args.precision == "FP16":
        DTYPE = torch.float16
    elif args.precision == "BF16":
        DTYPE = torch.bfloat16

    model = qwen3_model.model.to(DTYPE)
    # qwen3 
    input_ids = torch.randint(0, 5, (1, 5), dtype=torch.int64).cuda()
    position_ids = torch.arange(input_ids.shape[1], dtype=torch.int64).cuda().unsqueeze(0)

    pyt_output = model(input_ids, position_ids)
    seq_len = torch.export.Dim("seq_len", min=2, max=2176)
    dynamic_shapes = ({1: seq_len}, {1: seq_len})
    ep = torch.export.export(model, (input_ids, position_ids), dynamic_shapes=dynamic_shapes)
    
    with (torch_tensorrt.logging.debug() if args.debug else nullcontext()):
        trt_model = torch_tensorrt.dynamo.compile(ep,
                                                inputs=[input_ids, position_ids], 
                                                enabled_precisions={torch.float32},
                                                use_python_runtime=True,
                                                disable_tf32=True,
                                                debug=args.debug)
    # breakpoint()
    trt_output = trt_model(input_ids, position_ids)

    print(f"Diff b/w pyt and trt: {torch.mean(torch.abs(pyt_output[0] - trt_output[0]))}")
    print(f"Diff b/w pyt and trt: {torch.mean(torch.abs(pyt_output[1] - trt_output[1]))}")
    print(f"Diff b/w pyt and trt: {torch.mean(torch.abs(pyt_output[2] - trt_output[2]))}")
    assert torch.allclose(pyt_output[0], trt_output[0], atol=ATOL, rtol=RTOL)

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(
        description="Run test cases for llama attention and decoder"
    )
    arg_parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug (default: False)"
    )
    arg_parser.add_argument("--precision", type=str, default="FP32", help="Precision to use in the model. Options: FP16, BF16, FP32")
    args = arg_parser.parse_args()
    with torch.inference_mode():
        # test_qwen_attention(args)
        # test_qwen3_decoder(args)
        test_qwen3_model(args)
