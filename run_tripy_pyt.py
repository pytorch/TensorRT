import timeit

import numpy as np
import torch
from PIL import Image
from sam2.sam2_image_predictor import SAM2ImagePredictor

checkpoint = "./checkpoints/sam2_hiera_large.pt"
model_cfg = "sam2_hiera_l.yaml"

predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-large")
image = Image.open("./truck.jpg")
image = np.array(image.convert("RGB"))
input_point = np.array([[500, 375]])
input_label = np.array([1])

timings = []
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    for i in range(100):
        start_time = timeit.default_timer()
        # Image encoder
        predictor.set_image(image)
        torch.cuda.synchronize()
        timings.append(timeit.default_timer() - start_time)
        # Mask prediction
        masks, scores, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )
        torch.cuda.synchronize()


times = np.array(timings)
time_med = np.median(times) * 1000
print("Median time: ", time_med)
