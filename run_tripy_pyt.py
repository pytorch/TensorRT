import timeit

import numpy as np
import torch
from PIL import Image
from sam2.sam2_image_predictor import SAM2ImagePredictor

device = torch.device("cuda:0") 

checkpoint = "./checkpoints/sam2_hiera_large.pt"
model_cfg = "sam2_hiera_l.yaml"

predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-large")
image = Image.open("./truck.jpg")
image = np.array(image.convert("RGB"))
input_point = np.array([[500, 375]])
input_label = np.array([1])

timings_overall = []
timings_set_image = []
timings_predict = []

with torch.cuda.device(device):
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        # Warm Up
        # torch.cuda.synchronize()
        for i in range(50):            
            predictor.set_image(image)
            torch.cuda.synchronize()
            masks, scores, logits = predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=True,
            )
            torch.cuda.synchronize()

        # overall model    
        for i in range(100):
            torch.cuda.synchronize()
            start_time = timeit.default_timer()
            predictor.set_image(image)
            torch.cuda.synchronize()
            masks, scores, logits = predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=True,
            )
            torch.cuda.synchronize()
            timings_overall.append(timeit.default_timer() - start_time)
        

        for i in range(100):
            start_time = timeit.default_timer()
            predictor.set_image(image)
            torch.cuda.synchronize()
            timings_set_image.append(timeit.default_timer() - start_time)


        predictor.set_image(image)
        torch.cuda.synchronize()
        for i in range(100):
            start_time = timeit.default_timer()
            masks, scores, logits = predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=True,
            )
            torch.cuda.synchronize()
            timings_predict.append(timeit.default_timer() - start_time)



times_overall = np.array(timings_overall)
time_med_overall = np.median(times_overall) * 1000
print("Median time: ", time_med_overall)

timings_ms_overall = np.array(timings_overall) * 1000
formatted_timings_overall = [f"{value:.1f} ms" for value in timings_ms_overall]

print("All timings_overall in ms:", formatted_timings_overall)


times_set_image = np.array(timings_set_image)
time_med_set_image = np.median(times_set_image) * 1000
print("Median time: ", time_med_set_image)

timings_ms_set_image = np.array(timings_set_image) * 1000
formatted_timings_set_image = [f"{value:.1f} ms" for value in timings_ms_set_image]

print("All timings_set_image in ms:", formatted_timings_set_image)



times_predict = np.array(timings_predict)
time_med_predict = np.median(times_predict) * 1000
print("Median time: ", time_med_predict)

timings_ms_predict = np.array(timings_predict) * 1000
formatted_timings_predict = [f"{value:.1f} ms" for value in timings_ms_predict]

print("All timings_predict in ms:", formatted_timings_predict)


