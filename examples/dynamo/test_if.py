import torch
import torch.nn as nn
from torch.export import export

class ConditionalModel(nn.Module):
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

class ConditionalModel2(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, q, k, v, k1, v1, start_idx, end_idx): 
        
        new_k1 = torch.cat((k1[:, :, :start_idx, :], k, k1[:, :, end_idx:, :]), dim=2)
        new_v1 = torch.cat((v1[:, :, :start_idx, :], v, v1[:, :, end_idx:, :]), dim=2)
        out = torch._C._nn.scaled_dot_product_attention(q, new_k1[:,:,:end_idx,:], new_v1[:,:,:end_idx,:])
        
        return out, new_k1, new_v1


def main():
    # Create model
    model = ConditionalModel2()
    model.eval()  # Set to evaluation mode
    
    # Create example inputs
    q = torch.randn(1, 32, 2048, 64).cuda()
    k = torch.randn(1, 32, 2048, 64).cuda()
    v = torch.randn(1, 32, 2048, 64).cuda()
    k1 = torch.zeros(1, 32, 2176, 64).cuda()
    v1 = torch.zeros(1, 32, 2176, 64).cuda()
    # example_flag = torch.tensor(True)
    start_idx = 0
    end_idx = 2048
    out_pyt = model(q, k, v, k1, v1, start_idx, end_idx)
    out_pyt2 = model(q, k, v, k1, v1, 17, 18)

    exported_program = export(
        model,
        args=(q, k, v, k1, v1, start_idx, end_idx),
        dynamic_shapes=(None, None, None, None, None, torch.export.Dim.DYNAMIC, torch.export.Dim.DYNAMIC),
        strict=False
    )
    import torch_tensorrt
    with torch_tensorrt.logging.debug():
        trt_model = torch_tensorrt.dynamo.compile(
            exported_program,
            inputs=[q, k, v, k1, v1, start_idx, end_idx],
            enabled_precisions={torch.float32},
            # truncate_double=True,
            disable_tf32=True,
            use_python_runtime=True,
            debug=True,
            min_block_size=1,
        )
    
    gm = exported_program.module() 
    breakpoint()
    out_ep = gm(q, k, v, k1, v1, start_idx, end_idx) 
    out_ep2 = gm(q, k, v, k1, v1, 2048, 2049)
    out_trt = trt_model(q, k, v, k1, v1, start_idx, end_idx)
    out_trt2 = trt_model(q, k, v, k1, v1, 2048, 2049)
    # breakpoint()
    # Print the graph
    print("\nExported Graph:")
    print(exported_program.graph_module.graph)
    # breakpoint()
    # print("done")
    

if __name__ == "__main__":
    main()