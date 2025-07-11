import argparse
from contextlib import nullcontext
import torch
import torch.nn as nn
import torch_tensorrt
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

DEVICE = "cuda:0"

# ----------------------------------------------------------------------
# SDPA register
# ----------------------------------------------------------------------
from torchtrt_ext import register_sdpa  # noqa: F401

# ----------------------------------------------------------------------
# TextModel Wrapper
# ----------------------------------------------------------------------
class TextModelWrapper(nn.Module):
    """
    Qwen2_5_VLTextModel 의 forward 중 input_ids / attention_mask 두 인자만
    받아 last_hidden_state 만 반환하도록 얇게 감싼 래퍼.
    """
    def __init__(self, text_model):
        super().__init__()
        self.text_model = text_model

    def forward(self, input_ids, attention_mask=None):
        out = self.text_model(
            input_ids=input_ids, attention_mask=attention_mask, use_cache=False
        )
        return out.last_hidden_state  # shape: (bs, seq, hidden)


# ----------------------------------------------------------------------
# 테스트 드라이버
# ----------------------------------------------------------------------
def main(args):
    # 1) Load model -------------------------------------------------------

    model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
    # default: Load the model on the available device(s)
    full_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map=DEVICE
    )

    # We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
    # model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    #     "Qwen/Qwen2.5-VL-3B-Instruct",
    #     torch_dtype=torch.bfloat16,
    #     attn_implementation="flash_attention_2",
    #     device_map="auto",
    # )

    # default processer
    processor = AutoProcessor.from_pretrained(model_name)
    text_model = full_model.model # .language_model  # From transformers==4.54.0dev, refactorized by language model Qwen2_5_VLTextModel
    hidden_size = text_model.config.hidden_size
    vocab_size = text_model.config.vocab_size

    # 2) dummu input 
    B, S = args.batch_size, args.seq_len
    input_ids = torch.randint(0, vocab_size, (B, S), device=DEVICE)
    attention_mask = torch.ones_like(input_ids).to(DEVICE)

    # 3) PyTorch output
    with torch.inference_mode():
        pyt_logits = text_model(
            input_ids=input_ids, attention_mask=attention_mask, use_cache=False
        ).last_hidden_state

    # 4) Torch-TensorRT 컴파일 ------------------------------------------
    wrapper = TextModelWrapper(text_model).eval().to(DEVICE)

    # Set precision specific flags
    use_fp32_acc = False
    use_explicit_typing = False
    if args.precision == "FP16":
        enabled_precisions = {torch.float32}
        use_fp32_acc = True
        use_explicit_typing = True
    elif args.precision == "BF16":
        enabled_precisions = {torch.bfloat16}
        use_fp32_acc = False
    else:
        enabled_precisions = {torch.float32}

    # dynamic shape
    # S_dyn = torch.export.Dim("seq", min=1, max=args.seq_len_max)
    _seq = torch.export.Dim('_seq', min=1, max=272)
    seq = 8*_seq
    S_dyn = seq
    dyn_shapes = {"input_ids": {1: S_dyn}, "attention_mask": {1: S_dyn}}

    with torch.inference_mode():
        export_mod = torch.export.export(
            wrapper, (input_ids, attention_mask), dynamic_shapes=dyn_shapes, strict=False
        )

    with torch_tensorrt.logging.debug() if args.debug else nullcontext():
        trt_mod = torch_tensorrt.dynamo.compile(
            export_mod,
            inputs=[input_ids, attention_mask],
            enabled_precisions=enabled_precisions,
            use_fp32_acc=use_fp32_acc,
            use_explicit_typing=use_explicit_typing,
            disable_tf32=True,
            debug=args.debug,
        )

    # 5) TensorRT 출력 ---------------------------------------------------
    with torch.inference_mode():
        trt_logits = trt_mod(input_ids, attention_mask)

    # 6) 비교 ------------------------------------------------------------
    diff = (pyt_logits - trt_logits).abs().mean()
    print(f"\nMean abs diff PyTorch vs TensorRT: {diff.item():.4e}")
    tol = 1e-1 if args.precision == "FP16" else 1e-3
    if diff < tol:
        print("✅  Outputs match within tolerance.")
    else:
        print("❌  Outputs differ beyond tolerance!")

# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Qwen2_5_VLTextModel Torch-TensorRT test")
    parser.add_argument("--precision", choices=["FP16", "BF16", "FP32"], default="FP16")
    parser.add_argument("--seq-len", type=int, default=128, dest="seq_len")
    parser.add_argument("--seq-len-max", type=int, default=2176, dest="seq_len_max")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args) 