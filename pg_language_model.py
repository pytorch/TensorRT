import torch
import torch.nn as nn
import torch_tensorrt
import torch.nn.functional as F

from transformers import PaliGemmaForConditionalGeneration

############################################################
# 1) 래퍼 모듈 정의
############################################################
class LMNoCache(nn.Module):
    def __init__(self, lm):
        super().__init__()
        self.lm = lm  # 실제 language_model (GPT-like decoder)

    def forward(self, inputs_embeds=None, attention_mask=None, input_ids=None):
        # input_ids 또는 inputs_embeds 중 하나는 반드시 제공되어야 함
        if inputs_embeds is None and input_ids is not None:
            inputs_embeds = self.lm.get_input_embeddings()(input_ids)
        elif inputs_embeds is None and input_ids is None:
            raise ValueError("inputs_embeds와 input_ids 중 하나는 반드시 제공되어야 합니다.")
            
        # attention_mask 형태 확인 및 변환
        if attention_mask is not None and attention_mask.dim() == 2:
            # [batch, seq_len] -> [batch, 1, seq_len, seq_len]
            batch_size, seq_length = attention_mask.size()
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = attention_mask.expand(batch_size, 1, seq_length, seq_length)
            
        # use_cache=False를 주어 캐시를 사용하지 않게 함
        return self.lm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            use_cache=False
        )

############################################################
# 2) 모델 준비
############################################################
DEVICE = torch.device("cuda:0")
model_id = "google/paligemma2-3b-pt-224"

model = (
    PaliGemmaForConditionalGeneration
    .from_pretrained(model_id, torch_dtype=torch.float16)
    .eval()
    .to(DEVICE)
)

# Language model 부분만 추출
language_model = model.language_model
wrapped_lm = LMNoCache(language_model).to(DEVICE).eval()

############################################################
# 3) 더미 입력 준비
############################################################
batch_size = 2
seq_len = 16

dummy_input_ids = torch.randint(
    0,
    model.config.text_config.vocab_size,  # ex) 32000
    (batch_size, seq_len),
    device=DEVICE,
    dtype=torch.long
)

# 4D attention mask로 변경 (batch_size, 1, seq_len, seq_len)
# 값은 0.0 (비마스킹)으로 채움 (causal 마스크는 PaliGemma 내부에서 처리됨)
dummy_attention_mask = torch.zeros(
    batch_size,
    1,
    seq_len,
    seq_len,
    dtype=torch.float16,  # 모델과 동일한 dtype 사용
    device=DEVICE
)

############################################################
# 4) Dynamic shapes 정의
############################################################
B = torch.export.Dim("batch_dim", min=1, max=4)
_seq_dim = torch.export.Dim("_seq_dim", min=1, max=16)  # underlying dim
seq_dim = 8 * _seq_dim  # total sequence dimension = 8, 16, ..., 128
ext_dim = torch.export.Dim("ext_dim", min=8, max=128)  # 확장된 시퀀스 길이 (attention_mask 4번째 차원)

# 임베딩 계산 (실제 export에서는 input_ids 대신 inputs_embeds 사용)
inputs_embeds = model.language_model.get_input_embeddings()(dummy_input_ids)

dynamic_shapes = {
    "inputs_embeds": {
        0: B,         # batch dimension
        1: seq_dim,   # sequence dimension
    },
    "attention_mask": {
        0: B,         # batch dimension
        1: 1,         # head dimension
        2: seq_dim,   # query sequence dimension
        3: ext_dim,   # key sequence dimension
    },
}


############################################################
# 5) export
############################################################
with torch.inference_mode():
    exported_program = torch.export.export(
        wrapped_lm,
        args=(inputs_embeds, dummy_attention_mask),
        kwargs={},
        dynamic_shapes=dynamic_shapes,
        strict=False
    )

############################################################
# 6) Torch-TensorRT로 compile & 결과 비교
############################################################
with torch.inference_mode():
    # PyTorch 원본 결과
    baseline_outputs = wrapped_lm(inputs_embeds=inputs_embeds, attention_mask=dummy_attention_mask)

with torch_tensorrt.logging.debug():
    trt_lm = torch_tensorrt.dynamo.compile(
        exported_program,
        inputs=[inputs_embeds, dummy_attention_mask],
        enabled_precisions={torch.float32}, 
        device=DEVICE,
        disable_tf32=False,  # TF32 활성화 (더 나은 성능과 정확도 균형)
        use_explicit_typing=True,
        use_fp32_acc=True,
        truncate_double=True
    )

with torch.inference_mode():
    trt_outputs = trt_lm(inputs_embeds, dummy_attention_mask)

# language_model(...) 출력은 CausalLMOutputWithPast 형태이므로,
# logits를 꺼내 비교
baseline_logits = baseline_outputs.logits
trt_logits = trt_outputs.logits

diff = (baseline_logits - trt_logits).abs().max()
print("[Language Model] max difference:", diff.item())
print("Baseline logits shape:", baseline_logits.shape)
print("TRT logits shape:", trt_logits.shape)

# 2. fp32로 변환 (정확도 비교를 위한 권장 처리)
baseline_fp32 = baseline_logits.float()
trt_fp32 = trt_logits.float()

# 3. Flatten
baseline_flat = baseline_fp32.flatten(1)  # (B, -1)
trt_flat = trt_fp32.flatten(1)

# 4. Cosine similarity 계산 (batch-wise 평균)
cos_sim = F.cosine_similarity(baseline_flat, trt_flat, dim=1)
avg_cos_sim = cos_sim.mean()

print("Cosine similarity (per example):", cos_sim)
print(f"Average cosine similarity: {avg_cos_sim:.6f}")

# # 코사인 유사도 계산
# def compute_cosine_similarity(tensor1, tensor2):
#     tensor1_flat = tensor1.reshape(-1)
#     tensor2_flat = tensor2.reshape(-1)
#     return F.cosine_similarity(tensor1_flat.unsqueeze(0), tensor2_flat.unsqueeze(0), dim=1).item()

# cosine_sim = compute_cosine_similarity(baseline_logits, trt_logits)
# print(f"Cosine similarity between baseline and TRT model logits: {cosine_sim:.6f}")

# # 추가 통계값
# mean_diff = (baseline_logits - trt_logits).abs().mean().item()
# print(f"Mean absolute difference: {mean_diff:.6f}")

# # 두 logits에서 가장 큰 값(argmax)의 일치율 계산 (기존 방식)
# baseline_argmax = baseline_logits.argmax(dim=-1)
# trt_argmax = trt_logits.argmax(dim=-1)
# argmax_match_rate = (baseline_argmax == trt_argmax).float().mean().item()
# print(f"Simple argmax token match rate: {argmax_match_rate:.6f} ({argmax_match_rate*100:.2f}%)")

# ############################################################
# # 7) 생성 기반 평가: 실제 모델 생성과 유사한 방식
# ############################################################
# from transformers import PreTrainedTokenizerFast

# # 토크나이저 로드
# tokenizer = PreTrainedTokenizerFast.from_pretrained(model_id)

# # 간단한 프롬프트 생성 - 실제 테스트 용도
# prompt = "Translate this to English: Bonjour, comment ça va?"
# prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
# prompt_attn_mask = torch.ones_like(prompt_ids, dtype=torch.float16, device=DEVICE)
# prompt_embeds = model.language_model.get_input_embeddings()(prompt_ids)

# # 4D attention mask 생성
# batch_size, seq_length = prompt_ids.shape
# prompt_attn_mask_4d = prompt_attn_mask.unsqueeze(1).unsqueeze(2).expand(batch_size, 1, seq_length, seq_length)

# print("\n생성 방식 유사도 비교:")
# print(f"프롬프트: \"{prompt}\"")

# # 기본 모델 추론
# with torch.inference_mode():
#     baseline_output = wrapped_lm(inputs_embeds=prompt_embeds, attention_mask=prompt_attn_mask_4d)
#     baseline_next_token_logits = baseline_output.logits[:, -1, :]
#     baseline_next_token = baseline_next_token_logits.argmax(dim=-1)
    
# # TensorRT 모델 추론
# with torch.inference_mode():
#     trt_output = trt_lm(prompt_embeds, prompt_attn_mask_4d)
#     trt_next_token_logits = trt_output.logits[:, -1, :]
#     trt_next_token = trt_next_token_logits.argmax(dim=-1)

# # 다음 토큰 예측 비교
# next_token_match = (baseline_next_token == trt_next_token).all().item()
# print(f"다음 토큰 일치 여부: {next_token_match}")
# print(f"기본 모델 다음 토큰: {tokenizer.decode(baseline_next_token)}")
# print(f"TRT 모델 다음 토큰: {tokenizer.decode(trt_next_token)}")

# # Top-5 토큰 비교 (실제 생성에서는 토큰 분포가 중요)
# baseline_top5 = baseline_next_token_logits.topk(5, dim=-1)
# trt_top5 = trt_next_token_logits.topk(5, dim=-1)

# print("\nTop-5 토큰 비교:")
# print("기본 모델 Top-5:", [tokenizer.decode(t) for t in baseline_top5.indices[0].cpu().numpy()])
# print("TRT 모델 Top-5:", [tokenizer.decode(t) for t in trt_top5.indices[0].cpu().numpy()])

# # Top-5 중 일치하는 토큰 수 계산
# top5_matches = sum(t1 == t2 for t1, t2 in zip(baseline_top5.indices[0], trt_top5.indices[0]))
# print(f"Top-5 토큰 일치율: {top5_matches/5:.2f} ({top5_matches}/5)")

# # 토큰 확률 분포 유사도 (KL 다이버전스)
# import torch.nn.functional as F
# import numpy as np

# def calculate_distribution_similarity(logits1, logits2):
#     # 소프트맥스 확률 분포로 변환
#     probs1 = F.softmax(logits1, dim=-1)
#     probs2 = F.softmax(logits2, dim=-1)
    
#     # 유효한 확률값만 사용 (0이 아닌 값)
#     mask = (probs1 > 1e-6) & (probs2 > 1e-6)
    
#     if not mask.any():
#         return 0.0  # 유효한 확률이 없는 경우
    
#     probs1_valid = probs1[mask]
#     probs2_valid = probs2[mask]
    
#     # Jensen-Shannon 다이버전스 (대칭적 측정치)
#     m = 0.5 * (probs1_valid + probs2_valid)
#     js_div = 0.5 * (
#         F.kl_div(probs1_valid.log(), m, reduction='batchmean') +
#         F.kl_div(probs2_valid.log(), m, reduction='batchmean')
#     )
    
#     # JS 다이버전스를 0~1 유사도로 변환 (1이 완전 일치)
#     similarity = np.exp(-js_div.item())
#     return similarity

# dist_similarity = calculate_distribution_similarity(
#     baseline_next_token_logits[0], trt_next_token_logits[0]
# )
# print(f"토큰 확률 분포 유사도: {dist_similarity:.6f}")