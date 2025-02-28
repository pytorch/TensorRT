import torch
import torch_tensorrt
from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration
from transformers.image_utils import load_image

DEVICE = "cuda:0"

model_id = "google/paligemma2-3b-pt-224"
url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg"
image = load_image(url)


model = PaliGemmaForConditionalGeneration.from_pretrained(
    model_id, torch_dtype=torch.float16).eval()
model.to(DEVICE).to(torch.float16)
# model.forward = model.forward.to(torch.float16).eval()

processor = PaliGemmaProcessor.from_pretrained(model_id)
prompt = ""
model_inputs = processor(text=prompt, images=image, return_tensors="pt").to(torch.float16).to(DEVICE) # to(DEVICE) # .to(torch.float16).to(DEVICE)
input_len = model_inputs["input_ids"].shape[-1]

# model.config.token_healing = False

with torch.inference_mode():
    pyt_generation = model.generate(**model_inputs, max_new_tokens=100, do_sample=False)
    pyt_generation_out = pyt_generation[0][input_len:]
    pyt_decoded = processor.decode(pyt_generation_out, skip_special_tokens=True)
    print("=============================")
    print("pyt_generation whole text:")
    print(pyt_generation)
    print("=============================")
    print("=============================")
    print("PyTorch generated text:")
    print(pyt_decoded)
    print("=============================")

with torch_tensorrt.logging.debug():
    torch._dynamo.mark_dynamic(model_inputs["input_ids"], 1, min=2, max=256)
    model.forward = torch.compile(
        model.forward,
        backend="tensorrt",
        dynamic=None,
        options={
            "enabled_precisions": {torch.float16},
            "disable_tf32": True,
            "min_block_size": 1,
            # "use_explicit_typing": True,
            # "use_fp32_acc": True,
            "debug": True,
            # "use_aot_joint_export":False,
        },
    )
    
    with torch.inference_mode():
        trt_generation = model.generate(**model_inputs, max_new_tokens=100, do_sample=False) 
        trt_generation_out = trt_generation[0][input_len:]
        trt_decoded = processor.decode(trt_generation_out, skip_special_tokens=True)
        print(trt_generation)
        print("TensorRT generated text:")
        print(trt_decoded)