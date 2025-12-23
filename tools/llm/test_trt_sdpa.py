import torch
import torch_tensorrt
from torchtrt_ext import register_sdpa


class ModelNoCache(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v):
        return torch._C._nn.scaled_dot_product_attention(
            q, k, v, dropout_p=0.0, is_causal=True, scale=1.0
        )


model = ModelNoCache().cuda().eval().to(torch.float16)
q = torch.randn(1, 32, 6, 64).cuda().to(torch.float16)
k = torch.randn(1, 32, 6, 64).cuda().to(torch.float16)
v = torch.randn(1, 32, 6, 64).cuda().to(torch.float16)
pyt_outputs = model(q, k, v)

register_sdpa.enable_sdpa_converter("default", None)
seq_len_query = torch.export.Dim("seq_len_query", min=2, max=128)
seq_len_key = torch.export.Dim("seq_len_key", min=2, max=128)
dynamic_shapes = {"q": {2: seq_len_key}, "k": {2: seq_len_key}, "v": {2: seq_len_key}}
ep = torch.export.export(model, (q, k, v), dynamic_shapes=dynamic_shapes, strict=False)

with torch_tensorrt.dynamo.Debugger():
    trt_gm = torch_tensorrt.dynamo.compile(
        ep,
        inputs=(q, k, v),
        enabled_precisions={torch.float32},
        min_block_size=1,
        disable_tf32=True,
        use_explicit_typing=True,
    )

trt_outputs = trt_gm(q, k, v)
print("Diff between pyt and trt: ", torch.mean(torch.abs(pyt_outputs - trt_outputs)))
# breakpoint()
# q = torch.randn(1, 32, 1, 64).cuda().to(torch.float16)
# k = torch.randn(1, 32, 10, 64).cuda().to(torch.float16)
# v = torch.randn(1, 32, 10, 64).cuda().to(torch.float16)
# pyt_outputs = model(q, k, v)
# trt_outputs = trt_gm(q, k, v)
# print("Diff between pyt and trt: ", torch.mean(torch.abs(pyt_outputs - trt_outputs)))
