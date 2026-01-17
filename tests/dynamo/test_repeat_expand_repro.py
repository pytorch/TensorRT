import pytest
import torch
import torch.nn as nn

try:
    import torch_tensorrt
except Exception:
    torch_tensorrt = None

REQUIRES_TRT = torch.cuda.is_available() and (torch_tensorrt is not None)

pytestmark = pytest.mark.skipif(not REQUIRES_TRT, reason="requires CUDA + Torch-TensorRT runtime")

class CosmosLearnablePositionalEmbed(nn.Module):
    def __init__(self, hidden_size, max_size, patch_size):
        super().__init__()
        self.patch_size = patch_size
        self.pos_emb_t = nn.Parameter(torch.zeros(max_size[0] // patch_size[0], hidden_size))
        self.pos_emb_h = nn.Parameter(torch.zeros(max_size[1] // patch_size[1], hidden_size))
        self.pos_emb_w = nn.Parameter(torch.zeros(max_size[2] // patch_size[2], hidden_size))

    def forward(self, hidden_states):
        batch_size, _, num_frames, height, width = hidden_states.shape
        pe_size = [num_frames // self.patch_size[0], height // self.patch_size[1], width // self.patch_size[2]]
        emb_t = self.pos_emb_t[:pe_size[0]][None, :, None, None, :].repeat(batch_size, 1, pe_size[1], pe_size[2], 1)
        emb_h = self.pos_emb_h[:pe_size[1]][None, None, :, None, :].repeat(batch_size, pe_size[0], 1, pe_size[2], 1)
        emb_w = self.pos_emb_w[:pe_size[2]][None, None, None, :, :].repeat(batch_size, pe_size[0], pe_size[1], 1, 1)
        emb = emb_t + emb_h + emb_w
        emb = emb.flatten(1, 3)
        return emb

def test_repeat_expand_lowering_repro():
    device = torch.device("cuda")
    hidden_size = 4096
    model = CosmosLearnablePositionalEmbed(hidden_size=hidden_size, max_size=(128,240,240), patch_size=(1,2,2)).to(device).eval()
    hidden_states = torch.randn(1, 17, 16, 88, 160, dtype=torch.bfloat16, device=device)

    with torch.no_grad():
        pyt_out = model(hidden_states)

    ep = torch.export.export(model, args=(hidden_states,), strict=False)
    trt_mod = torch_tensorrt.dynamo.compile(ep, inputs=[hidden_states], enabled_precisions={torch.bfloat16}, use_python_runtime=True)
    trt_out = trt_mod(hidden_states)

    assert pyt_out.shape == trt_out.shape
    maxdiff = (pyt_out.float() - trt_out.float()).abs().max().item()
    assert maxdiff < 1e-2
