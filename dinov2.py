import os
import torch
import torch_tensorrt
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import requests

def main():
    # 1) Load image from URL
    url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    image = Image.open(requests.get(url, stream=True).raw)

    # 2) Load DINOv2 processor & model
    processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
    model = AutoModel.from_pretrained('facebook/dinov2-base')

    # Set return_dict=False if tuple outputs are needed
    model.config.return_dict = False

    # 3) Switch model to FP16, eval mode, and GPU
    model = model.half().eval().cuda()

    # 4) Preprocess image -> pixel_values (FP16)
    inputs = processor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"].half().cuda()

    # 5) PyTorch (FP16) inference
    with torch.no_grad():
        pt_outputs = model(pixel_values)
        pt_last_hidden = pt_outputs[0]

    # 6) Torch-TRT compile with engine caching
    compiled_model = torch.compile(
        model,
        backend="tensorrt",
        options={
            "use_python_runtime": False,
            "enabled_precisions": {torch.float16},
            "debug": True,
            "min_block_size": 1,
            "cache_built_engines": True,
            "reuse_cached_engines": True,
        },
    )

    # 7) Torch-TRT inference
    with torch.no_grad():
        trt_outputs = compiled_model(pixel_values)
        trt_last_hidden = trt_outputs[0]

    # 8) Cosine similarity calculation
    pt_flat = pt_last_hidden.flatten().float()
    trt_flat = trt_last_hidden.flatten().float()
    cos_sim = F.cosine_similarity(pt_flat.unsqueeze(0), trt_flat.unsqueeze(0), dim=1)
    cos_val = cos_sim.item()

    print(f"[PyTorch vs Torch-TRT] Cosine Similarity = {cos_val:.6f}")

if __name__ == "__main__":
    main()