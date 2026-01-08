import logging

import torch
import torch.nn as nn
import torch_tensorrt

logging.basicConfig(level=logging.DEBUG)

torch.manual_seed(0)


class ExpandReshapeModel(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.embed_dim = embed_dim
        self.qkv_proj = nn.Linear(self.embed_dim, self.embed_dim * 3)

    def forward(self, x: torch.Tensor):
        batch_size = x.shape[0]
        cls_token = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        x = self.qkv_proj(x)
        reshaped_qkv = x.reshape(batch_size, x.size(1), 3, 12, -1)
        return reshaped_qkv


model = ExpandReshapeModel(embed_dim=768).cuda().eval()
x = torch.randn(4, 196, 768).cuda()

# 1. JIT
x1 = x.clone()
torch._dynamo.mark_dynamic(x1, index=0, min=2, max=32)
trt_module = torch.compile(model, backend="tensorrt")
out1 = trt_module(x1)

# 2. torch_tensorrt.compile API
x2 = x.clone()
example_input = torch_tensorrt.Input(
    min_shape=[1, 196, 768],
    opt_shape=[4, 196, 768],
    max_shape=[32, 196, 768],
    dtype=torch.float32,
)
trt_module = torch_tensorrt.compile(model, ir="dynamo", inputs=example_input)
out2 = trt_module(x2)

# 3. torch.export + Dynamo compile API
x3 = x.clone()
bs = torch.export.Dim("bs", min=1, max=32)
dynamic_shapes = {"x": {0: bs}}
exp_program = torch.export.export(model, (x3,), dynamic_shapes=dynamic_shapes)
trt_module = torch_tensorrt.dynamo.compile(exp_program, (x3,))
out3 = trt_module(x3)

assert torch.allclose(out1, out2)
assert torch.allclose(out1, out3)
assert torch.allclose(out2, out3)
