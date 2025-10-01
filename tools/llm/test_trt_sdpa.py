import torch
import torch_tensorrt
from torchtrt_ext import register_sdpa


class ModelNoCache(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v):
        return torch._C._nn.scaled_dot_product_attention(
            q, k, v, dropout_p=0.0, is_causal=True
        )


model = ModelNoCache().cuda().eval().to(torch.float16)
q = torch.randn(1, 32, 6, 64).cuda().to(torch.float16)
k = torch.randn(1, 32, 6, 64).cuda().to(torch.float16)
v = torch.randn(1, 32, 6, 64).cuda().to(torch.float16)
pyt_outputs = model(q, k, v)
register_sdpa.enable_sdpa_converter("default", None)
ep = torch.export.export(model, (q, k, v), strict=False)

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
