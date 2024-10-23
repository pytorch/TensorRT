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
        "Batch size": batch_size,
        "Median(FPS)": speed_med,
        "Mean(FPS)": speed_mean,
        "Median-Latency(ms)": time_med * 1000,
        "Mean-Latency(ms)": time_mean * 1000,
        "Latency-StdDev(ms)": time_std * 1000,
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


image = Image.open("./truck.jpg")
image = np.array(image.convert("RGB"))

predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-small")
# with torch_tensorrt.logging.debug():
#     predictor.set_image = torch.compile(predictor.set_image, backend="tensorrt", options={"debug": True, "min_block_size": 1})
#     predictor.set_image(image)

# https://github.com/pytorch/pytorch/issues/115534

# Just model
model = predictor.model  # .image_encoder


class MyModule(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, torch_img: torch.Tensor):
        return self.module.forward_image(torch_img)


# pre process
input_image = predictor._transforms(image)
input_image = input_image[None, ...].to("cuda:0")

pyt_model = MyModule(model)
pyt_results = record_perf(pyt_model, "Torch", [input_image], "fp32", 3, 1)

ep = torch.export.export(pyt_model, (input_image,))
with torch_tensorrt.logging.debug():
    trt_gm = torch_tensorrt.dynamo.compile(
        ep, inputs=[input_image], debug=True, min_block_size=1
    )

trt_results = record_perf(trt_gm, "Dynamo", [input_image], "fp32", 3, 1)

print("==================================")
print(pd.DataFrame(pyt_results))
print("==================================")
print(pd.DataFrame(trt_results))
