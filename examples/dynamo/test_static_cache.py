import torch
import torch.nn as nn
from torch.export import export
import torch_tensorrt
from contextlib import nullcontext
import argparse
from lower_sdpa import *
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaDecoderLayer
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers import AutoModelForCausalLM
from torch_tensorrt.dynamo.lowering import (
    get_decompositions,
    post_lowering,
    pre_export_lowering,
)

ATOL = 1e-5
RTOL = 1e-5
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

class DynamicCacheModel(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, q, k, v, k1, v1, flag):
        def true_fn(q, k, v, k1, v1):   
            k_new = torch.cat((k, k1), dim=2)
            v_new = torch.cat((v, v1), dim=2)
            return torch._C._nn.scaled_dot_product_attention(q, k_new, v_new)

        def false_fn(q, k, v, k1, v1):
            return torch._C._nn.scaled_dot_product_attention(q, k, v)

        out = torch.cond(flag, true_fn, false_fn, (q, k, v, k1, v1))

        return 2 * out
    
class ModelNoCache(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, q, k, v):
        return torch._C._nn.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=True)

class StaticCacheModel(nn.Module):
    def __init__(self):
        super().__init__()
        
    # def forward(self, q, k, v, key_cache, value_cache, start_idx, end_idx, is_causal=True): 
    #     new_key_cache = torch.cat((key_cache[:, :, :start_idx, :], k, key_cache[:, :, end_idx:, :]), dim=2)
    #     new_value_cache = torch.cat((value_cache[:, :, :start_idx, :], v, value_cache[:, :, end_idx:, :]), dim=2)
    #     out = torch._C._nn.scaled_dot_product_attention(q, new_key_cache[:, :, :end_idx, :], new_value_cache[:, :, :end_idx, :], dropout_p=0.0, is_causal=is_causal)
        
    #     return out, new_key_cache, new_value_cache
    
    def forward(self, q, k, v, key_cache, value_cache, start_idx, end_idx, is_causal=True): 
        concat_keys = torch.cat((key_cache[:, :, :start_idx, :], k), dim=2)  # key_cache[:, :, :6, :] + curr_keys + key_cache[:, : 7: ,: ]
        concat_values = torch.cat((value_cache[:, :, :start_idx, :], v), dim=2)
        new_key_cache = torch.cat((concat_keys, key_cache[:, :, end_idx:, :]), dim=2)
        new_value_cache = torch.cat((concat_values, value_cache[:, :, end_idx:, :]), dim=2)
        out = torch._C._nn.scaled_dot_product_attention(q, concat_keys, concat_values, dropout_p=0.0, is_causal=is_causal)
        
        return out, new_key_cache, new_value_cache


def eager_sdpa(query, key, value, attn_mask=None, dropout_p=0.0,
        is_causal=False, scale=None, enable_gqa=False) -> torch.Tensor:
    """
    Eager implementation of SDPA
    """
    import math
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
    breakpoint()
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0).cuda()
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias = attn_mask + attn_bias

    if enable_gqa:
        key = key.repeat_interleave(query.size(-3)//key.size(-3), -3)
        value = value.repeat_interleave(query.size(-3)//value.size(-3), -3)

    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value

def print_diff(tensor1, tensor2, prefix=""):
    """
    Print the diff between two tensors
    """
    print(f"[{prefix}] Diff between tensor1 and tensor2: {torch.mean(torch.abs(tensor1 - tensor2))}")

def test_static_cache_model(args):
    """
    Test the static cache model
    """
    with torch.inference_mode():
        model_no_cache = ModelNoCache().eval().cuda()
        model_static_cache = StaticCacheModel().eval().cuda()
        q = torch.randn(1, 32, 2048, 64).cuda()
        k = torch.randn(1, 32, 2048, 64).cuda()
        v = torch.randn(1, 32, 2048, 64).cuda()
        key_cache = torch.zeros(1, 32, 2176, 64).cuda()
        value_cache = torch.zeros(1, 32, 2176, 64).cuda()

        # Test Prefill
        start_idx = 0
        end_idx = 2048
        out_no_cache = model_no_cache(q, k, v)
        out_static_cache, new_key_cache, new_value_cache = model_static_cache(q, k, v, key_cache, value_cache, start_idx, end_idx, is_causal=True)
        assert torch.allclose(out_no_cache, out_static_cache, atol=ATOL, rtol=RTOL)
        
        # Test Generate
        for start_idx in range(2048, 2176):
            end_idx = start_idx + 1
            q_curr = torch.randn(1, 32, 1, 64).cuda()
            k_curr = torch.randn(1, 32, 1, 64).cuda()
            v_curr = torch.randn(1, 32, 1, 64).cuda()

            # Concatenate the current query, key, and value with the previous ones
            q_full = torch.cat((q, q_curr), dim=2)
            k_full = torch.cat((k, k_curr), dim=2)
            v_full = torch.cat((v, v_curr), dim=2)

            out_no_cache = model_no_cache(q_full, k_full, v_full)
            out_static_cache, new_key_cache, new_value_cache = model_static_cache(q_curr, k_curr, v_curr, new_key_cache, new_value_cache, start_idx, end_idx, is_causal=False)

            assert torch.allclose(out_no_cache[:, :, -1:, :], out_static_cache, atol=ATOL, rtol=RTOL)
            q = q_full 
            k = k_full
            v = v_full
        print("============== test_static_cache passed ==============")

def transform_gm_with_kv_cache(exported_program: torch.export.ExportedProgram, args):
    """
    Transform the graph module by adding key and value cache to the graph
    """
    gm = exported_program.module()
    # Post lower the model
    settings = torch_tensorrt.dynamo.conversion.CompilationSettings(
        enabled_precisions={torch.float32},
        disable_tf32=True,
        use_python_runtime=True,
        debug=args.debug,
        min_block_size=1,
    )
    exported_program = pre_export_lowering(exported_program, settings)
    exported_program = exported_program.run_decompositions(
        get_decompositions(False)
    )

    gm = exported_program.module()
    gm = post_lowering(gm, settings)

    return gm

def test_static_cache_lowering(args):
    """
    Test static cache lowering pass applied to the model with no cache and run the graph module 
    and compare the output with the model with no cache
    """
    import static_cache2

    model_no_cache = ModelNoCache().eval().cuda()
    q = torch.randn(1, 32, 2, 64).cuda()
    k = torch.randn(1, 32, 2048, 64).cuda()
    v = torch.randn(1, 32, 2048, 64).cuda()
    key_cache = torch.zeros(1, 32, 2176, 64).cuda()
    value_cache = torch.zeros(1, 32, 2176, 64).cuda()
    
    # Export the model
    q_seq_len = torch.export.Dim("q_seq_len", min=2, max=2176)
    kv_seq_len = torch.export.Dim("kv_seq_len", min=2, max=2176)
    exported_program = export(
        model_no_cache,
        args=(q, k, v),
        dynamic_shapes=({2 : q_seq_len}, {2 : kv_seq_len}, {2 : kv_seq_len}),
        strict=False
    )

    gm = transform_gm_with_kv_cache(exported_program, args)

    # Test Prefill
    start_idx = 0
    end_idx = 2048
    is_causal = True
    q = torch.randn(1, 32, 2048, 64).cuda()
    out_no_cache = model_no_cache(q, k, v)
    out_pyt_cache, key_cache, value_cache = gm(q, k, v, is_causal, key_cache, value_cache, start_idx, end_idx)
    assert torch.allclose(out_no_cache, out_pyt_cache, atol=ATOL, rtol=RTOL)

    # Test Generate
    for start_idx in range(2048, 2176):
        end_idx = start_idx + 1
        is_causal = False
        q_curr = torch.randn(1, 32, 1, 64).cuda()
        k_curr = torch.randn(1, 32, 1, 64).cuda()
        v_curr = torch.randn(1, 32, 1, 64).cuda()
        # Concatenate the current query, key, and value with the previous ones
        q_full = torch.cat((q, q_curr), dim=2)
        k_full = torch.cat((k, k_curr), dim=2)
        v_full = torch.cat((v, v_curr), dim=2)   
        
        out_no_cache = model_no_cache(q_full, k_full, v_full)
        out_pyt_static_cache, key_cache, value_cache = gm(q_curr, k_curr, v_curr, is_causal, key_cache, value_cache, start_idx, end_idx)
        assert torch.allclose(out_no_cache[:, :, -1:, :], out_pyt_static_cache, atol=ATOL, rtol=RTOL)
        q = q_full 
        k = k_full
        v = v_full
    
    print("============== test_static_cache_lowering passed ==============")

def test_static_cache_export(args):
    """
    Test the static cache model export
    """
    model_static_cache = StaticCacheModel().eval().cuda()
    q = torch.randn(1, 32, 2048, 64).cuda()
    k = torch.randn(1, 32, 2048, 64).cuda()
    v = torch.randn(1, 32, 2048, 64).cuda()
    key_cache = torch.zeros(1, 32, 2176, 64).cuda()
    value_cache = torch.zeros(1, 32, 2176, 64).cuda()
    # Test Prefill
    start_idx = 0
    end_idx = 2048
    is_causal = True
    # Export the model
    seq_len = torch.export.Dim("seq_len", min=2, max=2048)
    seq_len_dyn_dim = {2 : seq_len}
    exported_program = export(
        model_static_cache,
        args=(q, k, v, key_cache, value_cache, start_idx, end_idx, is_causal),
        dynamic_shapes=(seq_len_dyn_dim, seq_len_dyn_dim, seq_len_dyn_dim, None, None, torch.export.Dim.DYNAMIC, torch.export.Dim.DYNAMIC, None),
        strict=False
    )
    
        
def test_static_cache_with_torch_tensorrt(args):
    """
    Test the static cache model with torch_tensorrt
    """
    import static_cache2

    model_no_cache = ModelNoCache().eval().cuda()
    q = torch.randn(1, 32, 2, 64).cuda()
    k = torch.randn(1, 32, 2048, 64).cuda()
    v = torch.randn(1, 32, 2048, 64).cuda()
    key_cache = torch.zeros(1, 32, 2176, 64).cuda()
    value_cache = torch.zeros(1, 32, 2176, 64).cuda()
    
    # Export the model
    q_seq_len = torch.export.Dim("q_seq_len", min=2, max=2176)
    kv_seq_len = torch.export.Dim("kv_seq_len", min=2, max=2176)
    exported_program = export(
        model_no_cache,
        args=(q, k, v),
        dynamic_shapes=({2 : q_seq_len}, {2 : kv_seq_len}, {2 : kv_seq_len}),
        strict=False
    )
    with (torch_tensorrt.logging.debug() if args.debug else nullcontext()):
        trt_model = torch_tensorrt.dynamo.compile(
            exported_program,
            inputs=[q, k, v],
            enabled_precisions={torch.float32},
            disable_tf32=True,
            use_python_runtime=True,
            debug=args.debug,
            min_block_size=1,
        )
    
    start_idx = 0
    end_idx = 2048
    is_causal = True
    q = torch.randn(1, 32, 2048, 64).cuda()
    # out_eager = eager_sdpa(q, k, v, is_causal=is_causal)
    out_no_cache = model_no_cache(q, k, v)
    out_trt, trt_key_cache, trt_value_cache = trt_model(q, k, v, is_causal, key_cache, value_cache, start_idx, end_idx)
    # breakpoint()
    assert torch.allclose(out_no_cache, out_trt, atol=ATOL, rtol=RTOL), "Prefill TRT logits don't match"
    assert torch.allclose(trt_key_cache[:, :, :end_idx, :], k, atol=ATOL, rtol=RTOL), "Prefill TRT key cache don't match"
    assert torch.allclose(trt_value_cache[:, :, :end_idx, :], v, atol=ATOL, rtol=RTOL), "Prefill TRT value cache don't match"
    
    # Test Generate
    for start_idx in range(2048, 2176):
        end_idx = start_idx + 1
        q_curr = torch.randn(1, 32, 1, 64).cuda()
        k_curr = torch.randn(1, 32, 1, 64).cuda()
        v_curr = torch.randn(1, 32, 1, 64).cuda()   
        # Concatenate the current query, key, and value with the previous ones
        q_full = torch.cat((q, q_curr), dim=2)
        k_full = torch.cat((k, k_curr), dim=2)
        v_full = torch.cat((v, v_curr), dim=2)   
        is_causal = False
        out_no_cache = model_no_cache(q_full, k_full, v_full)
        out_trt, trt_key_cache, trt_value_cache = trt_model(q_curr, k_curr, v_curr, is_causal, trt_key_cache, trt_value_cache, start_idx, end_idx)
        # breakpoint()
        # print_diff(out_no_cache[:, :, -1:, :], out_trt, f"out_no_cache[:, :, -1:, :] vs out_trt for idx {start_idx}")
        # print_diff(trt_key_cache[:, :, :end_idx, :], k_full, f"trt_key_cache[:, :, :end_idx, :] vs k_full for idx {start_idx}")
        # print_diff(trt_value_cache[:, :, :end_idx, :], v_full, f"trt_value_cache[:, :, :end_idx, :] vs v_full for idx {start_idx}")
        assert torch.allclose(out_no_cache[:, :, -1:, :], out_trt, atol=ATOL, rtol=RTOL), f"Generate TRT logits don't match for idx {start_idx}"
        assert torch.allclose(trt_key_cache[:, :, :end_idx, :], k_full, atol=ATOL, rtol=RTOL), f"Generate TRT key cache don't match for idx {start_idx}"
        assert torch.allclose(trt_value_cache[:, :, :end_idx, :], v_full, atol=ATOL, rtol=RTOL), f"Generate TRT value cache don't match for idx {start_idx}"
        q = q_full 
        k = k_full
        v = v_full

    print("============== test_static_cache_with_torch_tensorrt passed ==============")
    

def main():
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
        # test_static_cache_model(args)
        # test_static_cache_lowering(args)
        test_static_cache_with_torch_tensorrt(args)
    

if __name__ == "__main__":
    main()