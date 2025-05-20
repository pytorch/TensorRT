import torch
import torch.nn as nn
from torch.export import export

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
        
    def forward(self, q, k, v, key_cache, value_cache, start_idx, end_idx, is_causal=True): 
        new_key_cache = torch.cat((key_cache[:, :, :start_idx, :], k, key_cache[:, :, end_idx:, :]), dim=2)
        new_value_cache = torch.cat((value_cache[:, :, :start_idx, :], v, value_cache[:, :, end_idx:, :]), dim=2)
        out = torch._C._nn.scaled_dot_product_attention(q, new_key_cache[:, :, :end_idx, :], new_value_cache[:, :, :end_idx, :], dropout_p=0.0, is_causal=is_causal)
        
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

def print_diff(tensor1, tensor2):
    """
    Print the diff between two tensors
    """
    print(f"[Diff] Diff between tensor1 and tensor2: {torch.mean(torch.abs(tensor1 - tensor2))}")

def test_static_cache():
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
        print(f"[Prefill] Diff between no cache and static cache: {torch.mean(torch.abs(out_no_cache - out_static_cache))}")
        
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
            print_diff(out_no_cache[:, :, -1:, :], out_static_cache)
            q = q_full 
            k = k_full
            v = v_full
        

def main():
    # Create model
    # model = ConditionalModel2()
    # model.eval()  # Set to evaluation mode
    with torch.inference_mode():
        test_static_cache()
    # # Create example inputs
    # q = torch.randn(1, 32, 2048, 64).cuda()
    # k = torch.randn(1, 32, 2048, 64).cuda()
    # v = torch.randn(1, 32, 2048, 64).cuda()
    # k1 = torch.zeros(1, 32, 2176, 64).cuda()
    # v1 = torch.zeros(1, 32, 2176, 64).cuda()
    # # example_flag = torch.tensor(True)
    # start_idx = 0
    # end_idx = 2048
    # out_pyt = model(q, k, v, k1, v1, start_idx, end_idx)
    # out_pyt2 = model(q, k, v, k1, v1, 17, 18)

    # exported_program = export(
    #     model,
    #     args=(q, k, v, k1, v1, start_idx, end_idx),
    #     dynamic_shapes=(None, None, None, None, None, torch.export.Dim.DYNAMIC, torch.export.Dim.DYNAMIC),
    #     strict=False
    # )
    # import torch_tensorrt
    # with torch_tensorrt.logging.debug():
    #     trt_model = torch_tensorrt.dynamo.compile(
    #         exported_program,
    #         inputs=[q, k, v, k1, v1, start_idx, end_idx],
    #         enabled_precisions={torch.float32},
    #         # truncate_double=True,
    #         disable_tf32=True,
    #         use_python_runtime=True,
    #         debug=True,
    #         min_block_size=1,
    #     )
    
    # gm = exported_program.module() 
    # breakpoint()
    # out_ep = gm(q, k, v, k1, v1, start_idx, end_idx) 
    # out_ep2 = gm(q, k, v, k1, v1, 2048, 2049)
    # out_trt = trt_model(q, k, v, k1, v1, start_idx, end_idx)
    # out_trt2 = trt_model(q, k, v, k1, v1, 2048, 2049)
    # # breakpoint()
    # # Print the graph
    # print("\nExported Graph:")
    # print(exported_program.graph_module.graph)
    # # breakpoint()
    # # print("done")
    

if __name__ == "__main__":
    main()