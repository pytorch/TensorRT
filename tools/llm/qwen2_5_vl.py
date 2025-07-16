from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch

# default: Load the model on the available device(s)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-3B-Instruct", torch_dtype=torch.bfloat16, device_map="cuda:0"
)

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
# model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen2.5-VL-3B-Instruct",
#     torch_dtype=torch.bfloat16,
#     attn_implementation="flash_attention_2",
#     device_map="auto",
# )

# default processer
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")

# The default range for the number of visual tokens per image in the model is 4-16384.
# You can set min_pixels and max_pixels according to your needs, such as a token range of 256-1280, to balance performance and cost.
# min_pixels = 256*28*28
# max_pixels = 1280*28*28
# processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
            },
            {"type": "text", "text": "Describe this image."},
        ],
    }
]

# Preparation for inference
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to("cuda")

# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=128, do_sample=False)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)


# from PIL import Image
# import requests
# from transformers import AutoProcessor, AutoModel
# import torch
# model = AutoModel.from_pretrained("nvidia/Eagle2-1B",trust_remote_code=True, torch_dtype=torch.bfloat16)
# processor = AutoProcessor.from_pretrained("nvidia/Eagle2-1B", trust_remote_code=True, use_fast=True)
# processor.tokenizer.padding_side = "left"

# messages = [
#     {
#         "role": "user",
#         "content": [
#             {
#                 "type": "image",
#                 "image": "https://www.ilankelman.org/stopsigns/australia.jpg",
#             },
#             {"type": "text", "text": "Describe this image."},
#         ],
#     }
# ]

# text_list = [processor.apply_chat_template(
#     messages, tokenize=False, add_generation_prompt=True
# )]
# image_inputs, video_inputs = processor.process_vision_info(messages)
# inputs = processor(text = text_list, images=image_inputs, videos=video_inputs, return_tensors="pt", padding=True)
# inputs = inputs.to("cuda")
# model = model.to("cuda")
# generated_ids = model.generate(**inputs, max_new_tokens=1024)
# output_text = processor.batch_decode(
#     generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
# )
# print(output_text)