import torch

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

import torch.nn as nn
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.common_utils import TestCase
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaDecoderLayer
from transformers.models.llama.configuration_llama import LlamaConfig
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


# llama2_model_name = "meta-llama/Llama-2-7b-hf"
llama3_model_name = "meta-llama/Llama-3.2-1B-Instruct"
llama_model = AutoModelForCausalLM.from_pretrained(
                llama3_model_name,
                use_cache=False,
                attn_implementation="sdpa",
                num_hidden_layers=1,
            ).eval().cuda()
LLAMA_CONFIG = llama_model.config

def test_llama_attention(args):
    class LlamaAttentionBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = LLAMA_CONFIG
            self.attn = LlamaAttention(
                config=self.config,
                layer_idx=0
            )
        def forward(self, hidden_states, position_embeddings):
            attn_output, attn_weights = self.attn(hidden_states, position_embeddings, None)
            return attn_output
    
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

    # model = LlamaAttentionBlock().eval().cuda().to(DTYPE)
    model = llama_model.model.layers[0].self_attn.to(DTYPE)
    # llama3 
    hidden_states = torch.randn((1, 6, 2048), dtype=DTYPE).cuda()
    position_embeddings = (torch.randn((1, 6, 64), dtype=DTYPE).cuda(), torch.randn((1, 6, 64), dtype=DTYPE).cuda())

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
        print(f"Diff b/w pyt and trt: {torch.mean(torch.abs(pyt_output[0] - trt_output[0]))}")
        breakpoint()
        assert torch.allclose(pyt_output[0], trt_output[0], atol=ATOL, rtol=RTOL)
    else:
        print(f"Diff b/w pyt and trt: {torch.mean(torch.abs(pyt_output - trt_output))}")
        assert torch.allclose(pyt_output, trt_output, atol=ATOL, rtol=RTOL)

def print_diff(tensor1, tensor2, prefix=""):
    """
    Print the diff between two tensors
    """
    print(f"[{prefix}] Diff between tensor1 and tensor2: {torch.mean(torch.abs(tensor1 - tensor2))}")

def test_llama_attention_with_static_cache(args):
    class LlamaAttentionBlock(nn.Module):
        def __init__(self):
            super().__init__()
                self.config = LLAMA_CONFIG
            self.attn = LlamaAttention(
                config=self.config,
                layer_idx=0
            )
        def forward(self, hidden_states, position_embeddings):
            attn_output, attn_weights = self.attn(hidden_states, position_embeddings, None)
            return attn_output
    
    DTYPE = torch.float32
    model = llama_model.model.layers[0].self_attn.to(DTYPE)

    # Inputs 
    ISL = 2048
    NUM_TOKENS = 128
    OSL = ISL + NUM_TOKENS
    hidden_states = torch.randn((1, ISL, 2048), dtype=DTYPE).cuda()
    position_embeddings = (torch.randn((1, ISL, 64), dtype=DTYPE).cuda(), torch.randn((1, ISL, 64), dtype=DTYPE).cuda())
    key_cache = torch.zeros(1, 32, OSL, 64).cuda().to(DTYPE)
    value_cache = torch.zeros(1, 32, OSL, 64).cuda().to(DTYPE)
    start_idx = 0
    end_idx = ISL
    is_causal = True

    pyt_output = model(hidden_states, position_embeddings, None)
    seq_len = torch.export.Dim("seq_len", min=2, max=2176)
    dynamic_shapes = ({1: seq_len}, ({1: seq_len}, {1: seq_len}), None)
    ep = torch.export.export(model, (hidden_states, position_embeddings, None), dynamic_shapes=dynamic_shapes)
    import register_sdpa
    import static_cache2
    with (torch_tensorrt.logging.debug() if args.debug else nullcontext()):
        trt_model = torch_tensorrt.dynamo.compile(ep, 
                                                inputs=[hidden_states, position_embeddings, None, key_cache, value_cache, start_idx, end_idx, is_causal], 
                                                enabled_precisions={torch.float32},
                                                disable_tf32=True,
                                                debug=args.debug, 
                                                # offload_module_to_cpu=True, 
                                                use_python_runtime=True)
    
    # Test Prefill
    trt_output, _, key_cache, value_cache = trt_model(hidden_states, position_embeddings, None, key_cache, value_cache, start_idx, end_idx, is_causal)
    print_diff(pyt_output[0], trt_output[0], "pyt_output[0] vs trt_output[0] [Prefill]")

    # Test Generate
    for start_idx in range(2048, 2176):
        end_idx = start_idx + 1
        hidden_states_curr = torch.randn((1, 1, 2048), dtype=DTYPE).cuda()
        position_embeddings_curr = (torch.randn((1, 1, 64), dtype=DTYPE).cuda(), torch.randn((1, 1, 64), dtype=DTYPE).cuda())
        # Concatenate the current  hidden_states with the previous ones
        hidden_states_full = torch.cat((hidden_states, hidden_states_curr), dim=1)
        position_embeddings_full = (torch.cat((position_embeddings[0], position_embeddings_curr[0]), dim=1), torch.cat((position_embeddings[1], position_embeddings_curr[1]), dim=1))
        
        is_causal = False
        out_no_cache, _ = model(hidden_states_full, position_embeddings_full, None)
        out_trt, _, key_cache, value_cache = trt_model(hidden_states_curr, position_embeddings_curr, None, key_cache, value_cache, start_idx, end_idx, is_causal)
        out_pyt = out_no_cache[:, -1:, :]
        print_diff(out_pyt, out_trt, f"pyt_curr_output vs out_trt for idx {start_idx}")

        hidden_states = hidden_states_full
        position_embeddings = position_embeddings_full


def test_llama_decoder(args):
    
    DTYPE = torch.float32
    model = llama_model.model.layers[0].to(DTYPE)
    # llama3 
    hidden_states = torch.randn((1, 6, 2048), dtype=DTYPE).cuda()
    position_embeddings = (torch.randn((1, 6, 64), dtype=DTYPE).cuda(), torch.randn((1, 6, 64), dtype=DTYPE).cuda())

    pyt_output = model(hidden_states, position_embeddings)
    seq_len = torch.export.Dim("seq_len", min=2, max=2176)
    dynamic_shapes = ({1: seq_len}, ({1: seq_len}, {1: seq_len}))
    ep = torch.export.export(model, (hidden_states, position_embeddings), dynamic_shapes=dynamic_shapes)
    
    with (torch_tensorrt.logging.debug() if args.debug else nullcontext()):
        trt_model = torch_tensorrt.dynamo.compile(ep, 
                                                inputs=[hidden_states, position_embeddings], 
                                                enabled_precisions={torch.float32},
                                                debug=args.debug)
    trt_output = trt_model(hidden_states, position_embeddings)

    print(f"Diff b/w pyt and trt: {torch.mean(torch.abs(pyt_output - trt_output))}")
    assert torch.allclose(pyt_output, trt_output, atol=ATOL, rtol=RTOL)

def test_llama_decoder_with_static_cache(args):

    class LlamaDecoderLayerBlock(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.config = LLAMA_CONFIG
            self.decoder = LlamaDecoderLayer(
                config=self.config,
                layer_idx=0)
            self.model = model
        def forward(self, hidden_states, position_embeddings):
            return self.model(hidden_states, position_embeddings=position_embeddings)

    DTYPE = torch.float32
    model = LlamaDecoderLayerBlock(llama_model.model.layers[0].to(DTYPE))
    
    # Inputs 
    ISL = 2048
    NUM_TOKENS = 128
    OSL = ISL + NUM_TOKENS
    hidden_states = torch.randn((1, ISL, 2048), dtype=DTYPE).cuda()
    position_embeddings = (torch.randn((1, ISL, 64), dtype=DTYPE).cuda(), torch.randn((1, ISL, 64), dtype=DTYPE).cuda())
    key_cache = torch.zeros(1, 32, OSL, 64).cuda().to(DTYPE)
    value_cache = torch.zeros(1, 32, OSL, 64).cuda().to(DTYPE)
    start_idx = 0
    end_idx = ISL
    is_causal = True

    pyt_output = model(hidden_states, position_embeddings)
    seq_len = torch.export.Dim("seq_len", min=2, max=2176)
    dynamic_shapes = ({1: seq_len}, ({1: seq_len}, {1: seq_len}))
    ep = torch.export.export(model, args=(hidden_states, position_embeddings), dynamic_shapes=dynamic_shapes)
    import register_sdpa
    import static_cache2
    with (torch_tensorrt.logging.debug() if args.debug else nullcontext()):
        trt_model = torch_tensorrt.dynamo.compile(ep, 
                                                arg_inputs=[hidden_states, position_embeddings, key_cache, value_cache, start_idx, end_idx, is_causal], 
                                                enabled_precisions={torch.float32},
                                                disable_tf32=True,
                                                debug=args.debug, 
                                                # offload_module_to_cpu=True, 
                                                use_python_runtime=True)
    
    # Test Prefill
    trt_output, key_cache, value_cache = trt_model(hidden_states, position_embeddings, key_cache, value_cache, start_idx, end_idx, is_causal)
    print_diff(pyt_output[0], trt_output, "pyt_output vs trt_output [Prefill]")

    # Test Generate
    for start_idx in range(2048, 2176):
        end_idx = start_idx + 1
        hidden_states_curr = torch.randn((1, 1, 2048), dtype=DTYPE).cuda()
        position_embeddings_curr = (torch.randn((1, 1, 64), dtype=DTYPE).cuda(), torch.randn((1, 1, 64), dtype=DTYPE).cuda())
        # Concatenate the current  hidden_states with the previous ones
        hidden_states_full = torch.cat((hidden_states, hidden_states_curr), dim=1)
        position_embeddings_full = (torch.cat((position_embeddings[0], position_embeddings_curr[0]), dim=1), torch.cat((position_embeddings[1], position_embeddings_curr[1]), dim=1))
        
        is_causal = False
        out_no_cache = model(hidden_states_full, position_embeddings_full)

        out_trt, key_cache, value_cache = trt_model(hidden_states_curr, position_embeddings_curr, key_cache, value_cache, start_idx, end_idx, is_causal)
        out_pyt = out_no_cache[0][:, -1:, :]
        print_diff(out_pyt, out_trt, f"pyt_curr_output vs out_trt for idx {start_idx}")
        hidden_states = hidden_states_full
        position_embeddings = position_embeddings_full

def test_llama_model_with_static_cache(args):

    DTYPE = torch.float32
    model = llama_model.model.to(DTYPE)

    # Inputs 
    ISL = 2048
    NUM_TOKENS = 128
    OSL = ISL + NUM_TOKENS
    input_ids = torch.randint(1, 20, (1, ISL), dtype=torch.int64).cuda()
    position_ids = torch.arange(input_ids.shape[1]).unsqueeze(0).cuda()
    key_cache = torch.zeros(1, 32, OSL, 64).cuda().to(DTYPE)
    value_cache = torch.zeros(1, 32, OSL, 64).cuda().to(DTYPE)
    start_idx = 0
    end_idx = ISL
    is_causal = True

    pyt_output = model(input_ids)
    seq_len = torch.export.Dim("seq_len", min=2, max=2176)
    dynamic_shapes = ({1: seq_len}, {1: seq_len})
    kwarg_inputs = {"input_ids":input_ids, "position_ids":position_ids}
    ep = torch.export.export(model, args=(), kwargs=kwarg_inputs, dynamic_shapes=dynamic_shapes)

    import register_sdpa
    import static_cache2
    with (torch_tensorrt.logging.debug() if args.debug else nullcontext()):
        trt_model = torch_tensorrt.dynamo.compile(ep, 
                                                arg_inputs=[], 
                                                kwarg_inputs=kwarg_inputs,
                                                enabled_precisions={torch.float32},
                                                disable_tf32=True,
                                                debug=args.debug, 
                                                # offload_module_to_cpu=True, 
                                                use_python_runtime=True)
    
    # Test Prefill
    trt_output, key_cache, value_cache = trt_model(input_ids, position_ids, key_cache, value_cache, start_idx, end_idx, is_causal)
    pyt_output = pyt_output.last_hidden_state
    print_diff(pyt_output, trt_output, "pyt_output vs trt_output [Prefill]")

    # Test Generate
    for start_idx in range(2048, 2176):
        end_idx = start_idx + 1
        input_ids_curr = torch.randint(1, 20, (1, 1), dtype=torch.int64).cuda()
        position_ids_curr = torch.tensor([[start_idx]], dtype=torch.int64).cuda()
        
        # Concatenate the current  hidden_states with the previous ones
        input_ids_full = torch.cat((input_ids, input_ids_curr), dim=1)
        position_ids_full = torch.cat((position_ids, position_ids_curr), dim=1)
        is_causal = False
        kwarg_inputs = {"input_ids":input_ids_full, "position_ids":position_ids_full}
        out_no_cache = model(**kwarg_inputs)

        out_trt, key_cache, value_cache = trt_model(input_ids_curr, position_ids_curr, key_cache, value_cache, start_idx, end_idx, is_causal)
        out_pyt = out_no_cache.last_hidden_state[:, -1:, :]
        print_diff(out_pyt, out_trt, f"pyt_curr_output vs out_trt for idx {start_idx}")
        input_ids = input_ids_full
        position_ids = position_ids_full

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(
        description="Run test cases for llama attention and decoder"
    )
    arg_parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug (default: False)"
    )
    arg_parser.add_argument(
        "--precision",
        type=str,
        default="FP16",
        help="Precision (default: FP16)"
    )
    args = arg_parser.parse_args()
    with torch.inference_mode():
        test_llama_attention(args)
        # test_llama_decoder(args)
        # test_llama_attention_with_static_cache(args)
        # test_llama_decoder_with_static_cache(args)
        # test_llama_model_with_static_cache(args)