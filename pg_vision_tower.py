import torch
import torch_tensorrt

from transformers import PaliGemmaForConditionalGeneration

############################################################
# 1) 모델 준비
############################################################
DEVICE = torch.device("cuda:0")
model_id = "google/paligemma2-3b-pt-224"
model = (
    PaliGemmaForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float16)
    .to(DEVICE)
    .eval()
)

vision_tower = model.vision_tower  # vision part

############################################################
# 2) 더미 입력 및 dynamic_shapes 정의
############################################################
# 예: batch=2, (3,224,224) 이미지를 float16으로
batch_size = 2
channels = 3
height = 224
width = 224

dummy_pixel_values = torch.randn(
    batch_size,
    channels,
    height,
    width,
    dtype=torch.float16,
    device=DEVICE
)

# torch.export.Dim을 이용해 pixel_values의 차원을 동적으로 정의
BATCH = torch.export.Dim("batch_dim", min=1, max=4)

dynamic_shapes = {
    "pixel_values": {
        0: BATCH,   # batch dimension
    }
}

# torch.export.export에 넣을 kwargs
dummy_kwargs = {
    "pixel_values": dummy_pixel_values
}

############################################################
# 3) Vision Tower만 export하기
############################################################
with torch.inference_mode():
    exported_program = torch.export.export(
        vision_tower,
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
    baseline_outputs = vision_tower(dummy_pixel_values)
# Vision Tower 결과는 BaseModelOutput (마지막 히든스테이트 등)
baseline_hidden = baseline_outputs.last_hidden_state

# (4-2) Torch-TensorRT로 compile
with torch_tensorrt.logging.debug():
    trt_vision_tower = torch_tensorrt.dynamo.compile(
        exported_program,
        inputs=[dummy_pixel_values],       # forward에 들어갈 실제 입력 텐서
        enabled_precisions={torch.float32},
        device=DEVICE,
        disable_tf32=True,
        use_explicit_typing=True,
        use_fp32_acc=True,
        truncate_double=True,
    )

with torch.inference_mode():
    trt_outputs = trt_vision_tower(dummy_pixel_values)
trt_hidden = trt_outputs.last_hidden_state

############################################################
# 5) 두 결과의 차이 계산 (최댓값 기준)
############################################################
diff = (baseline_hidden - trt_hidden).abs().max()
print("[Vision Tower] max difference:", diff.item())