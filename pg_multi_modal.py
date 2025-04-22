import torch
import torch_tensorrt

from transformers import PaliGemmaForConditionalGeneration

############################################################
# 1) 모델 준비
############################################################
DEVICE = torch.device("cuda:0")
model_id = "google/paligemma2-3b-pt-224"

model = (
    PaliGemmaForConditionalGeneration
    .from_pretrained(model_id, torch_dtype=torch.float16)
    .to(DEVICE)
    .eval()
)

multi_modal_projector = model.multi_modal_projector
vision_hidden_size = model.config.vision_config.hidden_size

############################################################
# 2) 더미 입력 & dynamic_shapes 설정
#    multi_modal_projector는 forward(image_features)를 받습니다.
#    보통 vision_tower의 마지막 hidden_state가 shape:
#    [batch_size, seq_len, vision_config.hidden_size]
############################################################
batch_size = 2
seq_len = 196

dummy_image_features = torch.randn(
    batch_size,
    seq_len,
    vision_hidden_size,
    dtype=torch.float16,
    device=DEVICE
)

# 동적 범위를 PyTorch Export의 Dim 객체로 설정
B = torch.export.Dim("batch_dim", min=1, max=4)
S = torch.export.Dim("seq_dim", min=1, max=512)
dynamic_shapes = {
    "image_features": {
        0: B,   # batch dimension
        1: S,   # sequence length dimension
    }
}

# torch.export.export에 넘길 kwargs
dummy_kwargs = {
    "image_features": dummy_image_features
}

############################################################
# 3) Export: multi_modal_projector만 export
############################################################
with torch.inference_mode():
    exported_program = torch.export.export(
        multi_modal_projector,
        args=(),
        kwargs=dummy_kwargs,
        dynamic_shapes=dynamic_shapes,
        strict=False
    )

############################################################
# 4) PyTorch vs Torch-TensorRT 결과 비교
############################################################
# (4-1) PyTorch로 기본 forward
with torch.inference_mode():
    baseline_outputs = multi_modal_projector(dummy_image_features)

# (4-2) Torch-TensorRT로 compile
with torch_tensorrt.logging.debug():
    trt_projector = torch_tensorrt.dynamo.compile(
        exported_program,
        inputs=[dummy_image_features],  # forward()에 실제 줄 입력
        enabled_precisions={torch.float32},
        device=DEVICE,
        disable_tf32=True,
        use_explicit_typing=True,
        use_fp32_acc=True,
        truncate_double=True,
    )

with torch.inference_mode():
    trt_outputs = trt_projector(dummy_image_features)

############################################################
# 5) 두 결과의 차이 계산 (최댓값)
############################################################
diff = (baseline_outputs - trt_outputs).abs().max()
print("[Multi-Modal Projector] max difference:", diff.item())