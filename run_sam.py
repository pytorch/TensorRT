import timeit
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch_tensorrt
from PIL import Image
from sam2.sam2_image_predictor import SAM2ImagePredictor

WARMUP_ITER = 10


def recordStats(backend, timings, precision, batch_size=1, compile_time_s=None):
    """
    Records different timing stats and adds it to the result
    """
    results = []
    times = np.array(timings)
    speeds = batch_size / times
    time_mean = np.mean(times)
    time_med = np.median(times)
    time_99th = np.percentile(times, 99)
    time_std = np.std(times, ddof=0)
    speed_mean = np.mean(speeds)
    speed_med = np.median(speeds)

    stats = {
        "Backend": backend,
        "Precision": precision,
        # "Batch size": batch_size,
        "Median(FPS)": speed_med,
        # "Mean(FPS)": speed_mean,
        "Median-Latency(ms)": time_med * 1000,
        # "Mean-Latency(ms)": time_mean * 1000,
        # "Latency-StdDev(ms)": time_std * 1000,
    }
    results.append(stats)

    return results


def record_perf(
    model,
    backend,
    input_tensors,
    precision,
    iterations,
    batch_size,
    compile_time_s=None,
):
    """
    Run the model for certain number of iterations and record the perf.
    Model is warmed up initially
    """
    # Warm up
    with torch.no_grad():
        for _ in range(WARMUP_ITER):
            model(*input_tensors)

    torch.cuda.synchronize()

    timings = []
    with torch.no_grad():
        for i in range(iterations):
            start_time = timeit.default_timer()
            _ = model(*input_tensors)
            torch.cuda.synchronize()
            end_time = timeit.default_timer()
            timings.append(end_time - start_time)

    results = recordStats(
        "Torch-TensorRT " + backend, timings, precision, batch_size, compile_time_s
    )

    return results


def infer(
    predictor, image, input_point, input_label, mode="enc", multimask_output=True
):

    if mode == "enc":
        predictor.set_image(image)
    elif mode == "head":
        masks, scores, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=multimask_output,
        )


# Raw input
image = Image.open("./truck.jpg")
image = np.array(image.convert("RGB"))
input_point = np.array([[500, 375]])
input_label = np.array([1])

# Predictor
predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-small")

# measure time for image enc or prediction head
mode = "head"
if mode == "head":
    # Pre-process input image
    predictor.set_image(image)
timings = []
for _ in range(10):
    start_time = timeit.default_timer()
    infer(predictor, image, input_point, input_label, mode)
    end_time = timeit.default_timer()
    timings.append(end_time - start_time)

results = recordStats("Torch-TensorRT SAM " + mode, timings, "fp32", 1)
print(results)


# https://github.com/pytorch/pytorch/issues/115534


class MyModule(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, torch_img: torch.Tensor):
        return self.module.forward_image(torch_img)


# pre process
input_image = predictor._transforms(image)
input_image = input_image[None, ...].to("cuda:0").half()
precision = "fp16"

pyt_model = MyModule(predictor.model).eval().cuda().half()
pyt_results = record_perf(pyt_model, "Torch", [input_image], precision, 3, 1)

ep = torch.export.export(pyt_model, (input_image,))
with torch_tensorrt.logging.debug():
    trt_gm = torch_tensorrt.dynamo.compile(
        ep,
        inputs=[input_image],
        debug=True,
        min_block_size=1,
        enabled_precisions={torch.float16},
    )

trt_results = record_perf(trt_gm, "TensorRT", [input_image], precision, 3, 1)

print("==================================")
print(pd.DataFrame(pyt_results))
print("==================================")
print(pd.DataFrame(trt_results))
