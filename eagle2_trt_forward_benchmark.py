#!/usr/bin/env python
# benchmark_forward_once.py
# ------------------------
import torch, requests, copy
from PIL import Image
from collections import defaultdict
from transformers import GenerationMixin                    # type hint용

from eagle2_trt_replace import build_trt_model, PROF_TIMINGS, STEP_TIMINGS


# ────────────────────────────────────────────────────────────
# 0)  질문에서 제공해 주신 모든 유틸리티( build_trt_model 등 )를
#     동일한 파일에 복사하거나, 별도 모듈로 import 해 주세요.
#     여기서는 이미 import 가능한 상태라고 가정합니다.

# ------------------------------------------------------------------
# ✦ 1) 단 1 회 forward()만 실행하는 벤치마크 헬퍼
# ------------------------------------------------------------------
def run_forward_once(model: GenerationMixin, *, model_inputs: dict, label: str):
    """
    model.forward()를 단 한 번 호출하면서
       전체 실행시간 · vis · mlp · lm 세 구간의 시간을 기록/출력합니다.
    """
    global PROF_TIMINGS, STEP_TIMINGS          # 동일 객체 재사용
    PROF_TIMINGS.clear()
    STEP_TIMINGS.clear()

    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    # GPU 이벤트로 전체 wall-time 측정
    s_all = torch.cuda.Event(enable_timing=True)
    e_all = torch.cuda.Event(enable_timing=True)
    s_all.record()

    with torch.inference_mode():
        # pixel_values 는 첫 스텝에만 필요
        outputs = model(
            pixel_values = model_inputs.get("pixel_values", None),
            input_ids    = model_inputs["input_ids"],
            attention_mask = model_inputs.get("attention_mask", None),
            use_cache=False,            # 두 모델 모두 동일 조건
        )

    e_all.record()
    torch.cuda.synchronize()

    total_s = s_all.elapsed_time(e_all) / 1000.0   # ms → s
    vis_s  = PROF_TIMINGS.get("vis", 0.0)
    mlp_s  = PROF_TIMINGS.get("mlp", 0.0)
    lm_s   = PROF_TIMINGS.get("lm",  0.0)
    oh_s   = max(total_s - (vis_s + mlp_s + lm_s), 0.0)

    print(f"[{label}] forward(): {total_s:.4f}s  |  "
          f"vis={vis_s:.4f}s  mlp={mlp_s:.4f}s  lm={lm_s:.4f}s  overhead={oh_s:.4f}s")

    return total_s, vis_s, mlp_s, lm_s


# ------------------------------------------------------------------
# ✦ 2) 실행 예시 (Eagle 2-2B · 1280×960 샘플 이미지)
# ------------------------------------------------------------------
if __name__ == "__main__":
    dev = "cuda:1"
    torch.cuda.set_device(dev)

    # ── (1) Torch 원본 모델 & TensorRT 통합 모델 생성 ───────────────
    torch_model, trt_model, proc = build_trt_model(device=dev)

    # ── (2) 단순 테스트용 프롬프트/이미지 준비 ─────────────────────
    url   = "https://cdn.pixabay.com/photo/2019/08/08/23/33/car-4393990_1280.jpg"
    image = Image.open(requests.get(url, stream=True).raw)

    prompt = "Describe the image."
    msgs = [{"role":"user","content":
            [{"type":"image","image":image},
             {"type":"text","text":prompt}]}]

    txt = [proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)]
    ims, vids = proc.process_vision_info(msgs)
    inp = proc(text=txt, images=ims, videos=vids,
               return_tensors="pt", padding=True).to(dev)

        # Ensure dtypes are what PyTorch-SDPA expects
    if "attention_mask" in inp and inp["attention_mask"].dtype != torch.bool:
        inp["attention_mask"] = inp["attention_mask"].bool()
    # fp16 로 통일
    if "pixel_values" in inp and inp["pixel_values"].dtype != torch.float16:
        inp["pixel_values"] = inp["pixel_values"].to(torch.float16)
    for k in ("image_sizes","image_flags"):
        inp.pop(k, None)

    # ── (3) Torch → 1-step forward ────────────────────────────────
    print("\n▶ Baseline (Pure Torch)")
    t_tot, t_vis, t_mlp, t_lm = run_forward_once(torch_model, model_inputs=inp, label="Torch")

    # ── (4) TensorRT → 1-step forward ─────────────────────────────
    print("\n▶ TensorRT-optimised")
    r_tot, r_vis, r_mlp, r_lm = run_forward_once(trt_model,  model_inputs=inp, label="TensorRT")

    # ── (5) Speed-up 산출·요약 ────────────────────────────────────
    def _spd(a,b): return a/b if b>0 else float("inf")

    print("\n━━━━━━━━ Speed-up ━━━━━━━━")
    print(f"total      : ×{_spd(t_tot, r_tot):5.2f}")
    print(f"vision(ViT): ×{_spd(t_vis, r_vis):5.2f}")
    print(f"MLP        : ×{_spd(t_mlp, r_mlp):5.2f}")
    print(f"LM         : ×{_spd(t_lm,  r_lm ):5.2f}")
