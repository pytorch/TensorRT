import os

import huggingface_hub
import modelopt.torch.quantization as mtq
import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen2.5-0.5B"
DEVICE = torch.device("cuda:0")
# two supported quant_cfg: mtq.NVFP4_DEFAULT_CFG and mtq.FP8_DEFAULT_CFG
quant_cfg = mtq.FP8_DEFAULT_CFG

quantized_model_save_path = os.path.expanduser(
    "~/.cache/huggingface/hub/models--lan--Qwen2.5-0.5B-modelopt-fp8"
)
os.makedirs(quantized_model_save_path, exist_ok=True)


def quantize_model(model, tokenizer, quant_cfg):
    from modelopt.torch.utils.dataset_utils import (
        create_forward_loop,
        get_dataset_dataloader,
    )

    calib_dataloader = get_dataset_dataloader(
        tokenizer=tokenizer,
        batch_size=32,
        num_samples=512,
        device="cuda:0",
    )
    if not quant_cfg in (mtq.FP8_DEFAULT_CFG, mtq.NVFP4_DEFAULT_CFG):
        raise ValueError(
            f"Unsupported quantization configuration: {quant_cfg}, only fp8 and nvfp4 are supported"
        )

    calibrate_loop = create_forward_loop(dataloader=calib_dataloader)

    model = mtq.quantize(model, quant_cfg, forward_loop=calibrate_loop)

    mtq.print_quant_summary(model)

    return model


with torch.inference_mode():
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = (
        AutoModelForCausalLM.from_pretrained(
            model_name,
        )
        .eval()
        .cuda()
    )
    breakpoint()
    print(f"successfully get the unquantized model: {model_name}")

    unquantized_model_path = snapshot_download(
        model_name,
        local_files_only=huggingface_hub.constants.HF_HUB_OFFLINE,
        ignore_patterns=["original/**/*"],
        revision=None,
    )
    print(
        f"successfully downloaded unquantized model: {model_name} to {unquantized_model_path}"
    )
    breakpoint()
    model = quantize_model(model, tokenizer, quant_cfg)
    print(f"successfully quantized {model_name} with {quant_cfg}")

    model.save_pretrained(quantized_model_save_path)
    tokenizer.save_pretrained(quantized_model_save_path)
    print(
        f"successfully saved quantized model: {model_name} to {quantized_model_save_path}"
    )
