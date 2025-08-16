import torch
import torch_tensorrt
from torch.export import Dim
from torchtrt_ext import register_sdpa


class SimpleNetwork(torch.nn.Module):
    def __init__(self):
        super(SimpleNetwork, self).__init__()

    def forward(self, query, key, value, attn_mask):
        with torch.backends.cuda.sdp_kernel(
            enable_flash=False,
            enable_math=False,
            enable_mem_efficient=True,
        ):
            return torch.nn.functional.scaled_dot_product_attention(
                query, key, value, attn_mask, 0.0, False, scale=0.0625
            )


dtype = torch.float32

dyn_dim = Dim("dyn_dim", min=3, max=32)

query = torch.randn((1, 4, 13, 256), dtype=dtype).cuda()
key = torch.randn((1, 4, 13, 256), dtype=dtype).cuda()
value = torch.randn((1, 4, 13, 256), dtype=dtype).cuda()
attn_mask = torch.ones((13, 13), dtype=torch.bool).tril(diagonal=0).cuda()
inputs = (query, key, value, attn_mask)

model = SimpleNetwork().eval().cuda()
output_pyt = model(*inputs)
exp_program = torch.export.export(
    model,
    inputs,
    strict=False,
    dynamic_shapes={
        "query": {2: dyn_dim},
        "key": {2: dyn_dim},
        "value": {2: dyn_dim},
        "attn_mask": {0: dyn_dim, 1: dyn_dim},
    },
)
DEBUG_LOGGING_DIR = "./debug_logs"
with torch_tensorrt.dynamo.Debugger(
    "graphs",
    logging_dir=DEBUG_LOGGING_DIR,
    capture_fx_graph_after=["complex_graph_detection"],
    save_engine_profile=True,
    profile_format="trex",
    engine_builder_monitor=True,
):
    trt_model = torch_tensorrt.dynamo.compile(
        exp_program,
        inputs=inputs,
        enabled_precisions={dtype},
        min_block_size=1,
        cache_built_engines=False,
        reuse_cached_engines=False,
        truncate_double=True,
        use_python_runtime=False,
    )
    outputs_trt = trt_model(*inputs)
    breakpoint()
    assert torch.allclose(output_pyt, outputs_trt, rtol=1e-2, atol=1e-2)

print("Done")
