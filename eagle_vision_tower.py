import torch
import torch_tensorrt
from transformers import AutoModel, AutoProcessor
import time
from torch.fx import GraphModule
# eagle_vision_tower.py 맨 위쪽 ― 모델 로드 직후에 삽입
from transformers.models.siglip import modeling_siglip as ms


def patched_attention_forward(self, hidden_states, attention_mask=None, output_attentions=False):
    """Scaled-dot-product attention (export-friendly).

    • Matches the computation path of `flash_attention_forward` but avoids any custom
      CUDA kernels so that torch.export can symbolically trace the graph.
    • Softmax is executed in fp32 for numerical stability, then cast back to the
      original dtype – identical to HuggingFaceʼs implementation.
    """

    B, S, _ = hidden_states.shape

    # 1. Linear projections
    q = self.q_proj(hidden_states)
    k = self.k_proj(hidden_states)
    v = self.v_proj(hidden_states)

    # 2. Shape to (B, nH, S, dH)
    q = q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
    k = k.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
    v = v.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

    # 3. Scaled dot-product attention
    attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, nH, S, S)
    if attention_mask is not None:
        attn_scores = attn_scores + attention_mask  # additive mask

    # 4. Softmax in fp32 then cast back
    attn_probs = torch.softmax(attn_scores, dim=-1, dtype=torch.float32).to(q.dtype)

    # 5. Dropout (train-time only, mirrors HF behaviour)
    if self.training and self.dropout > 0:
        attn_probs = torch.nn.functional.dropout(attn_probs, p=self.dropout, training=True)

    # 6. Weighted sum over values
    context = torch.matmul(attn_probs, v)  # (B, nH, S, dH)

    # 7. Restore original shape and project out
    context = context.transpose(1, 2).contiguous().view(B, S, self.embed_dim)
    context = self.out_proj(context)

    if not output_attentions:
        attn_probs = None

    return context, attn_probs

# 원래 forward 교체
ms.SiglipAttention.forward = patched_attention_forward

############################################################
# 1) 모델 준비
############################################################
DEVICE = torch.device("cuda:0")
model_id = "nvidia/Eagle2-2B"
model = (
    AutoModel.from_pretrained(model_id, trust_remote_code=True, torch_dtype=torch.float16)
    .to(DEVICE)
    .eval()
)

vision_tower = model.vision_model  # vision part
print(f"Vision model type: {type(vision_tower).__name__}")
print(f"Vision config - Image size: {model.config.vision_config.image_size}, Patch size: {model.config.vision_config.patch_size}")

############################################################
# 2) 더미 입력 및 dynamic_shapes 정의
############################################################
# 실제 모델 입력 크기인 (7, 3, 448, 448) 사용
batch_size = 7
channels = 3
height = 448
width = 448

dummy_pixel_values = torch.randn(
    batch_size,
    channels,
    height,
    width,
    dtype=torch.float16, 
    device=DEVICE
)

# # 모델 래핑
# class VisionTowerWrapper(torch.nn.Module):
#     def __init__(self, vision_model):
#         super().__init__()
#         self.vision_model = vision_model            # SiglipVisionModel

#     @torch.no_grad()
#     def forward(self, pixel_values):
#         return self.vision_model(pixel_values).last_hidden_state   # <-- Tensor 1개만 반환

vision_wrapper = vision_tower  #VisionTowerWrapper(vision_tower)  # (수정) wrapper 없이 적용

# torch.export.Dim을 이용해 pixel_values의 차원을 동적으로 정의
BATCH = torch.export.Dim("batch_dim", min=1, max=8)
dynamic_shapes = {
    "pixel_values": {
        0: BATCH,   # batch dimension
    }
}

############################################################
# 3) Vision Tower export하기
############################################################
with torch.inference_mode():
    exported_program = torch.export.export(
        vision_wrapper,
        args=(dummy_pixel_values,),
        dynamic_shapes=dynamic_shapes,
        strict=False
    )

gm: GraphModule = exported_program.graph_module

print("\n=== FX nodes & meta keys ===")
for n in gm.graph.nodes:
    print(f"{n.format_node():60}  meta={list(n.meta.keys())}")

# # 노드 메타데이터 확인 함수
# def print_node_metadata():
#     print("Graph nodes metadata:")
#     for node in exported_program.graph_module.graph.nodes:
#         print(f"Node: {node.name}, Target: {node.target}, Has meta: {'val' in node.meta}")
#         if node.target == torch.ops.aten._to_copy.default:
#             print(f"  _to_copy node found: {node.name}")
#             print(f"  Args: {node.args}")
#             print(f"  Kwargs: {node.kwargs}")
#             print(f"  Meta: {node.meta}")

# # Export 후 호출
# print_node_metadata()

    
############################################################
# 4) PyTorch vs Torch-TensorRT 결과 비교
############################################################
# (4-1) PyTorch로 기본 forward (시간 측정)
def measure_inference_time(func, num_warmup=10, num_runs=50):
    # 워밍업
    for _ in range(num_warmup):
        func()
    
    torch.cuda.synchronize()
    
    # 시간 측정
    start_time = time.time()
    for _ in range(num_runs):
        func()
    torch.cuda.synchronize()
    end_time = time.time()
    
    avg_time = (end_time - start_time) * 1000 / num_runs  # ms로 변환
    return avg_time

# 기본 PyTorch 모델 추론 함수
def run_pytorch():
    with torch.inference_mode():
        return vision_wrapper(dummy_pixel_values)

# 기본 PyTorch 모델 실행
baseline_outputs = run_pytorch()
if hasattr(baseline_outputs, 'last_hidden_state'):
    baseline_hidden = baseline_outputs.last_hidden_state
else:
    baseline_hidden = baseline_outputs

print(f"Vision Tower output shape: {baseline_hidden.shape}")

# PyTorch 실행 시간 측정
pytorch_time = measure_inference_time(run_pytorch)
print(f"PyTorch forward time: {pytorch_time:.2f} ms")

# (4-2) Torch-TensorRT로 compile
with torch_tensorrt.logging.debug():
    trt_vision_tower = torch_tensorrt.dynamo.compile(
        exported_program,
        inputs=[dummy_pixel_values],
        enabled_precisions={torch.float16},
        device=DEVICE,
        # disable_tf32=True,
        # use_explicit_typing=True,
        # use_fp32_acc=True,
        truncate_double=True,
    )

# TensorRT 모델 추론 함수
def run_tensorrt():
    with torch.inference_mode():
        return trt_vision_tower(dummy_pixel_values)

# TensorRT 모델 실행
trt_outputs = run_tensorrt()
if hasattr(trt_outputs, 'last_hidden_state'):
    trt_hidden = trt_outputs.last_hidden_state
else:
    trt_hidden = trt_outputs

# TensorRT 실행 시간 측정
tensorrt_time = measure_inference_time(run_tensorrt)
print(f"TensorRT forward time: {tensorrt_time:.2f} ms")
print(f"Speedup: {pytorch_time / tensorrt_time:.2f}x")

############################################################
# 5) 두 결과의 차이 계산
############################################################
# 최대 절대 오차
max_abs_diff = (baseline_hidden - trt_hidden).abs().max().item()
print(f"[Vision Tower] max absolute difference: {max_abs_diff:.6f}")

# 평균 절대 오차
mean_abs_diff = (baseline_hidden - trt_hidden).abs().mean().item()
print(f"[Vision Tower] mean absolute difference: {mean_abs_diff:.6f}")

# 코사인 유사도 계산
def compute_cosine_similarity(tensor1, tensor2):
    tensor1_flat = tensor1.reshape(-1)
    tensor2_flat = tensor2.reshape(-1)
    return torch.nn.functional.cosine_similarity(tensor1_flat.unsqueeze(0), tensor2_flat.unsqueeze(0), dim=1).item()

cosine_sim = compute_cosine_similarity(baseline_hidden.float(), trt_hidden.float())
print(f"[Vision Tower] cosine similarity: {cosine_sim:.6f}")
