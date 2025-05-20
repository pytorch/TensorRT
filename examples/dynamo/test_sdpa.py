import torch
import torch.nn as nn
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.common_utils import TestCase
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaDecoderLayer
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers import AutoModelForCausalLM
import torch_tensorrt
from contextlib import nullcontext
import argparse

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
    # model = LlamaAttentionBlock().eval().cuda().to(DTYPE)
    model = llama_model.model.layers[0].self_attn.to(DTYPE)
    # llama3 
    # hidden_states = torch.randn((1, 6, 2048), dtype=DTYPE).cuda()
    # position_embeddings = (torch.randn((1, 6, 64), dtype=DTYPE).cuda(), torch.randn((1, 6, 64), dtype=DTYPE).cuda())
    hidden_states = torch.load("hidden_states.pt")
    position_embeddings = torch.load("position_embeddings.pt")
    # breakpoint()
    pyt_output = model(hidden_states, position_embeddings, None)
    
    seq_len = torch.export.Dim("seq_len", min=2, max=2176)
    dynamic_shapes = ({1: seq_len}, ({1: seq_len}, {1: seq_len}), None)
    ep = torch.export.export(model, (hidden_states, position_embeddings, None), dynamic_shapes=dynamic_shapes)
    
    with torch_tensorrt.logging.debug():
        trt_model = torch_tensorrt.dynamo.compile(ep, 
                                                inputs=[hidden_states, position_embeddings, None], 
                                                enabled_precisions={torch.float32},
                                                disable_tf32=True,
                                                debug=True)
    trt_output = trt_model(hidden_states, position_embeddings, None)
    breakpoint()
    if isinstance(pyt_output, tuple):
        print(f"Diff b/w pyt and trt: {torch.mean(torch.abs(pyt_output[0] - trt_output[0]))}")
    else:
        print(f"Diff b/w pyt and trt: {torch.mean(torch.abs(pyt_output - trt_output))}")
    

def test_llama_decoder(args):
    class LlamaDecoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = LLAMA_CONFIG
            self.decoder_layer = LlamaDecoderLayer(
                config=self.config,
                layer_idx=0
            )
        def forward(self, hidden_states, position_embeddings):
            decoder_output = self.decoder_layer(hidden_states, position_embeddings=position_embeddings)
            return decoder_output[0]
    
    DTYPE = torch.float32
    model = LlamaDecoder().eval().cuda().to(DTYPE)
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


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(
        description="Run test cases for llama attention and decoder"
    )
    arg_parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug (default: False)"
    )
    args = arg_parser.parse_args()
    with torch.inference_mode():
        test_llama_attention(args)
        # test_llama_decoder(args)