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
image = Image.open("./images/truck.jpg")
image = np.array(image.convert("RGB"))
input_point = np.array([[500, 375]])
input_label = np.array([1])

# Predictor
predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-small")

# measure time for image enc or prediction head
mode = "enc"
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
# Prepare inputs in half precision
predictor.set_image(image)

point_coords = torch.tensor([[500, 375]], dtype=torch.float).unsqueeze(0).to("cuda:0").half()
point_labels = torch.tensor([1], dtype=torch.int).unsqueeze(0).to("cuda:0")
# point_labels = torch.tensor([1], dtype=torch.int).unsqueeze(0).to("cuda:0").half()
mask_input = torch.zeros(1, 1, 256, 256).to("cuda:0").half()

image_embedding = predictor.get_image_embedding().to("cuda:0").half()
image_pe = predictor.model.sam_prompt_encoder.get_dense_pe().to("cuda:0").half()

high_res_features = [feat.to("cuda:0").half() for feat in predictor._features["high_res_feats"]]

class MyHeadModule(torch.nn.Module):
    def __init__(self, predictor):
        super().__init__()
        self.model = predictor.model  

    def forward(self, image_embedding, image_pe, point_coords, point_labels, mask_input, high_res_features):
        sparse_embeddings, dense_embeddings = self.model.sam_prompt_encoder(
            points=(point_coords, point_labels),
            boxes=None, 
            masks=mask_input, 
        )

        batched_mode = point_coords.shape[0] > 1

        low_res_masks, iou_predictions, _, _ = self.model.sam_mask_decoder(
            image_embeddings=image_embedding, 
            image_pe=image_pe, 
            sparse_prompt_embeddings=sparse_embeddings.half(),
            dense_prompt_embeddings=dense_embeddings.half(),
            multimask_output=True,
            repeat_image=batched_mode,
            high_res_features=high_res_features,
        )
        # # Upscale the masks to the original image resolution
        # masks = self._transforms.postprocess_masks(
        #     low_res_masks, self._orig_hw[img_idx]
        # )
        # low_res_masks = torch.clamp(low_res_masks, -32.0, 32.0)
        # if not return_logits:
        #     masks = masks > self.mask_threshold
        return low_res_masks, iou_predictions # sparse_embeddings, dense_embeddings # 

pyt_model = MyHeadModule(predictor).eval().cuda().half()

# Measure performance without TensorRT
print("Measuring Head with Torch (before Torch-TensorRT)")
pyt_results = record_perf(
    pyt_model,
    "Torch",
    [image_embedding, image_pe, point_coords, point_labels, mask_input, high_res_features],
    "fp16",
    iterations=10,
    batch_size=1,
)

print("Compiling Head with Torch-TensorRT")
ep_head = torch.export.export(
    pyt_model,
    (image_embedding, image_pe, point_coords, point_labels, mask_input, high_res_features)
)

# with torch_tensorrt.logging.debug():
trt_gm = torch_tensorrt.dynamo.compile(
    ep_head,
    ir="dynamo",
    inputs=[image_embedding, image_pe, point_coords, point_labels, mask_input, high_res_features],
    # debug=True,
    min_block_size=1,
    enabled_precisions={torch.float16},
)

# TensorRT 엔진 저장
engine_path = "sam_head_model_fp16.engine"
with torch.no_grad():
    torch_tensorrt.save(trt_gm, "trt.ep", inputs=[image_embedding, image_pe, point_coords, point_labels, mask_input, high_res_features])
    torch_tensorrt.save(trt_gm, "trt.ts", output_format="torchscript", inputs=[image_embedding, image_pe, point_coords, point_labels, mask_input, high_res_features])


# Measure performance after TensorRT
print("Measuring Head with Torch-TensorRT")
trt_results = record_perf(
    trt_gm,
    "TensorRT",
    [image_embedding, image_pe, point_coords, point_labels, mask_input, high_res_features],
    "fp16",
    iterations=10,
    batch_size=1,
)

print("==================================")
print("Head Performance Results (before TensorRT):")
print(pd.DataFrame(pyt_results))

print("==================================")
print("Head Performance Results (after TensorRT):")
print(pd.DataFrame(trt_results))
