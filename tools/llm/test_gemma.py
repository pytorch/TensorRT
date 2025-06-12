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
from transformers.models.gemma3.configuration_gemma3 import Gemma3Config
from transformers.models.gemma3.modeling_gemma3 import (
    Gemma3Attention,
    Gemma3DecoderLayer,
)

# Register SDPA as a standalone operator. Converter and lowering pass are defined in register_sdpa.py
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from register_sdpa import *

ATOL = 1e-5
RTOL = 1e-5


gemma3_model_name = "google/gemma-3-1b-it"
gemma3_model = (
    AutoModelForCausalLM.from_pretrained(
        gemma3_model_name,
        use_cache=False,
        attn_implementation="sdpa",
        num_hidden_layers=1,
    )
    .eval()
    .cuda()
)
GEMMA3_CONFIG = gemma3_model.config


def print_diff(tensor1, tensor2, prefix=""):
    """
    Print the diff between two tensors
    """
    print(
        f"[{prefix}] Diff between tensor1 and tensor2: {torch.mean(torch.abs(tensor1 - tensor2))}"
    )


def test_gemma3_attention(args):

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

    model = gemma3_model.model.layers[0].self_attn.to(DTYPE)

    # gemma3
    hidden_states = torch.randn((1, 5, 1152), dtype=DTYPE).cuda()
    position_embeddings = (
        torch.randn((1, 5, 256), dtype=DTYPE).cuda(),
        torch.randn((1, 5, 256), dtype=DTYPE).cuda(),
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


def test_gemma3_attention_with_static_cache(args):

    import static_cache_v2

    DTYPE = torch.float32
    model = gemma3_model.model.layers[0].self_attn.to(DTYPE)

    # Inputs
    ISL = 2048
    NUM_TOKENS = 128
    OSL = ISL + NUM_TOKENS
    hidden_states = torch.randn((1, ISL, 1152), dtype=DTYPE).cuda()
    position_embeddings = (
        torch.randn((1, ISL, 256), dtype=DTYPE).cuda(),
        torch.randn((1, ISL, 256), dtype=DTYPE).cuda(),
    )
    key_cache = torch.zeros(1, 4, OSL, 64).cuda().to(DTYPE)
    value_cache = torch.zeros(1, 4, OSL, 64).cuda().to(DTYPE)
    start_idx = 0
    end_idx = ISL
    is_causal = True

    pyt_output = model(hidden_states, position_embeddings, None)
    seq_len = torch.export.Dim("seq_len", min=2, max=2176)
    dynamic_shapes = ({1: seq_len}, ({1: seq_len}, {1: seq_len}), None)
    ep = torch.export.export(
        model, (hidden_states, position_embeddings, None), dynamic_shapes=dynamic_shapes
    )
    with torch_tensorrt.logging.debug() if args.debug else nullcontext():
        trt_model = torch_tensorrt.dynamo.compile(
            ep,
            inputs=[
                hidden_states,
                position_embeddings,
                None,
                key_cache,
                value_cache,
                start_idx,
                end_idx,
                is_causal,
            ],
            enabled_precisions={torch.float32},
            disable_tf32=True,
            debug=args.debug,
            # offload_module_to_cpu=True,
            use_python_runtime=True,
        )

    # Test Prefill
    trt_output, _, key_cache, value_cache = trt_model(
        hidden_states,
        position_embeddings,
        None,
        key_cache,
        value_cache,
        start_idx,
        end_idx,
        is_causal,
    )
    print_diff(pyt_output[0], trt_output[0], "pyt_output[0] vs trt_output[0] [Prefill]")

    # Test Generate
    for start_idx in range(2048, 2176):
        end_idx = start_idx + 1
        hidden_states_curr = torch.randn((1, 1, 1152), dtype=DTYPE).cuda()
        position_embeddings_curr = (
            torch.randn((1, 1, 256), dtype=DTYPE).cuda(),
            torch.randn((1, 1, 256), dtype=DTYPE).cuda(),
        )
        # Concatenate the current  hidden_states with the previous ones
        hidden_states_full = torch.cat((hidden_states, hidden_states_curr), dim=1)
        position_embeddings_full = (
            torch.cat((position_embeddings[0], position_embeddings_curr[0]), dim=1),
            torch.cat((position_embeddings[1], position_embeddings_curr[1]), dim=1),
        )

        is_causal = False
        out_no_cache, _ = model(hidden_states_full, position_embeddings_full, None)
        out_trt, _, key_cache, value_cache = trt_model(
            hidden_states_curr,
            position_embeddings_curr,
            None,
            key_cache,
            value_cache,
            start_idx,
            end_idx,
            is_causal,
        )
        out_pyt = out_no_cache[:, -1:, :]
        print_diff(out_pyt, out_trt, f"pyt_curr_output vs out_trt for idx {start_idx}")

        hidden_states = hidden_states_full
        position_embeddings = position_embeddings_full


def test_gemma3_decoder(args):

    DTYPE = torch.float32
    if args.precision == "FP16":
        DTYPE = torch.float16
    elif args.precision == "BF16":
        DTYPE = torch.bfloat16
    model = gemma3_model.model.layers[0].to(DTYPE)
    # model.self_attn.is_sliding = False

    # gemma3
    hidden_states = torch.randn((1, 6, 1152), dtype=DTYPE).cuda()
    position_embeddings_global = (
        torch.randn((1, 6, 256), dtype=DTYPE).cuda(),
        torch.randn((1, 6, 256), dtype=DTYPE).cuda(),
    )
    position_embeddings_local = (
        torch.randn((1, 6, 256), dtype=DTYPE).cuda(),
        torch.randn((1, 6, 256), dtype=DTYPE).cuda(),
    )

    pyt_output = model(
        hidden_states, position_embeddings_global, position_embeddings_local
    )
    seq_len = torch.export.Dim("seq_len", min=2, max=2176)
    dynamic_shapes = (
        {1: seq_len},
        ({1: seq_len}, {1: seq_len}),
        ({1: seq_len}, {1: seq_len}),
    )
    ep = torch.export.export(
        model,
        (hidden_states, position_embeddings_global, position_embeddings_local),
        dynamic_shapes=dynamic_shapes,
    )

    with torch_tensorrt.logging.debug() if args.debug else nullcontext():
        trt_model = torch_tensorrt.dynamo.compile(
            ep,
            inputs=[
                hidden_states,
                position_embeddings_global,
                position_embeddings_local,
            ],
            enabled_precisions={torch.float32},
            debug=args.debug,
        )
    trt_output = trt_model(
        hidden_states, position_embeddings_global, position_embeddings_local
    )

    print(
        f"Diff b/w pyt and trt: {torch.mean(torch.abs(pyt_output[0] - trt_output[0]))}"
    )
    # breakpoint()
    assert torch.allclose(pyt_output[0], trt_output[0], atol=ATOL, rtol=RTOL)


def test_gemma3_decoder_with_static_cache(args):

    class Gemma3DecoderLayerBlock(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.config = GEMMA3_CONFIG
            self.decoder = Gemma3DecoderLayer(config=self.config, layer_idx=0)
            self.model = model

        def forward(self, hidden_states, position_embeddings):
            return self.model(hidden_states, position_embeddings=position_embeddings)

    DTYPE = torch.float32
    model = Gemma3DecoderLayerBlock(gemma3_model.model.layers[0].to(DTYPE))

    import static_cache_v2

    # Inputs
    ISL = 2048
    NUM_TOKENS = 128
    OSL = ISL + NUM_TOKENS
    hidden_states = torch.randn((1, ISL, 1152), dtype=DTYPE).cuda()
    position_embeddings_global = (
        torch.randn((1, ISL, 256), dtype=DTYPE).cuda(),
        torch.randn((1, ISL, 256), dtype=DTYPE).cuda(),
    )
    position_embeddings_local = (
        torch.randn((1, NUM_TOKENS, 256), dtype=DTYPE).cuda(),
        torch.randn((1, NUM_TOKENS, 256), dtype=DTYPE).cuda(),
    )
    key_cache = torch.zeros(1, 4, OSL, 64).cuda().to(DTYPE)
    value_cache = torch.zeros(1, 4, OSL, 64).cuda().to(DTYPE)
    start_idx = 0
    end_idx = ISL
    is_causal = True

    pyt_output = model(
        hidden_states, position_embeddings_global, position_embeddings_local
    )
    seq_len = torch.export.Dim("seq_len", min=2, max=2176)
    dynamic_shapes = ({1: seq_len}, ({1: seq_len}, {1: seq_len}))
    ep = torch.export.export(
        model, args=(hidden_states, position_embeddings), dynamic_shapes=dynamic_shapes
    )

    with torch_tensorrt.logging.debug() if args.debug else nullcontext():
        trt_model = torch_tensorrt.dynamo.compile(
            ep,
            arg_inputs=[
                hidden_states,
                position_embeddings,
                key_cache,
                value_cache,
                start_idx,
                end_idx,
                is_causal,
            ],
            enabled_precisions={torch.float32},
            disable_tf32=True,
            debug=args.debug,
            # offload_module_to_cpu=True,
            use_python_runtime=True,
        )

    # Test Prefill
    trt_output, key_cache, value_cache = trt_model(
        hidden_states,
        position_embeddings,
        key_cache,
        value_cache,
        start_idx,
        end_idx,
        is_causal,
    )
    print_diff(pyt_output[0], trt_output, "pyt_output vs trt_output [Prefill]")

    # Test Generate
    for start_idx in range(2048, 2176):
        end_idx = start_idx + 1
        hidden_states_curr = torch.randn((1, 1, 1152), dtype=DTYPE).cuda()
        position_embeddings_curr = (
            torch.randn((1, 1, 256), dtype=DTYPE).cuda(),
            torch.randn((1, 1, 256), dtype=DTYPE).cuda(),
        )
        # Concatenate the current  hidden_states with the previous ones
        hidden_states_full = torch.cat((hidden_states, hidden_states_curr), dim=1)
        position_embeddings_full = (
            torch.cat((position_embeddings[0], position_embeddings_curr[0]), dim=1),
            torch.cat((position_embeddings[1], position_embeddings_curr[1]), dim=1),
        )

        is_causal = False
        out_no_cache = model(hidden_states_full, position_embeddings_full)

        out_trt, key_cache, value_cache = trt_model(
            hidden_states_curr,
            position_embeddings_curr,
            key_cache,
            value_cache,
            start_idx,
            end_idx,
            is_causal,
        )
        out_pyt = out_no_cache[0][:, -1:, :]
        print_diff(out_pyt, out_trt, f"pyt_curr_output vs out_trt for idx {start_idx}")
        hidden_states = hidden_states_full
        position_embeddings = position_embeddings_full


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
        # test_gemma3_attention(args)
        # test_gemma3_attention_with_static_cache(args)
        test_gemma3_decoder(args)
        # test_gemma3_decoder_with_static_cache(args)
