import argparse
import timeit
from typing import Tuple

import modelopt.torch.quantization as mtq
import numpy as np
import pandas as pd
import torch
import torch_tensorrt
from modelopt.torch.quantization.utils import export_torch_mode
from PIL import Image
from sam2.sam2_image_predictor import SAM2ImagePredictor

# Github issue related to SAM
# https://github.com/pytorch/pytorch/issues/115534

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


class MyModule(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module
        self._bb_feat_sizes = [
            (256, 256),
            (128, 128),
            (64, 64),
        ]

    def forward(self, torch_img: torch.Tensor):
        backbone_out = self.module.forward_image(torch_img)
        _, vision_feats, _, _ = self.module._prepare_backbone_features(backbone_out)
        # Add no_mem_embed, which is added to the lowest rest feat. map during training on videos
        if self.module.directly_add_no_mem_embed:
            vision_feats[-1] = vision_feats[-1] + self.module.no_mem_embed

        feats = [
            feat.permute(1, 2, 0).view(1, -1, *feat_size)
            for feat, feat_size in zip(vision_feats[::-1], self._bb_feat_sizes[::-1])
        ][::-1]
        image_features = {"image_embed": feats[-1], "high_res_feats": feats[:-1]}
        return image_features


def build_model(args):

    # Predictor
    predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-small")
    pyt_model = predictor.model
    if args.precision == "fp16":
        pyt_model = pyt_model.half()
        full_model = MyModule(pyt_model).eval().cuda()
    elif args.precision == "fp8":
        full_model = MyModule(pyt_model).eval().cuda()
        input_tensor = torch.randn((1, 3, 1024, 1024), dtype=torch.float32).cuda()

        def calibrate_loop(model):
            """Simple calibration function for testing."""
            model(input_tensor)

        quant_cfg = mtq.FP8_DEFAULT_CFG
        mtq.quantize(full_model, quant_cfg, forward_loop=calibrate_loop)
        breakpoint()
        print("done")
    else:
        full_model = MyModule(pyt_model).eval().cuda()

    return full_model, predictor


def build_input(args, file_path, predictor):

    # Raw input
    image = Image.open(file_path)
    image = np.array(image.convert("RGB"))
    input_point = np.array([[500, 375]])
    input_label = np.array([1])
    # pre process
    input_image = predictor._transforms(image)
    input_image = input_image[None, ...].to("cuda:0")
    if args.precision == "fp16":
        input_image = input_image.half()

    return input_image


def compile_with_torchtrt(model, inputs, args):

    precision = torch.float32
    if args.precision == "fp16":
        precision = torch.float16

    # ep = torch.export.export(model, inputs)
    ep = torch.export._trace._export(
        model,
        inputs,
        strict=False,
        allow_complex_guards_as_runtime_asserts=True,
    )
    with torch_tensorrt.logging.debug():
        trt_gm = torch_tensorrt.dynamo.compile(
            ep,
            inputs=[input_image],
            debug=True,
            min_block_size=1,
            enabled_precisions={precision},
        )

    return trt_gm


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="Run inference on SAM")
    # The following options are manual user provided settings
    arg_parser.add_argument(
        "--precision",
        type=str,
        default="fp32",
        help="Precision of the model to compile for TensorRT",
    )

    args = arg_parser.parse_args()

    pyt_model, predictor = build_model(args)

    input_image = build_input(args, "./truck.jpg", predictor)

    trt_model = compile_with_torchtrt(pyt_model, (input_image,), args)

    pyt_results = record_perf(pyt_model, "Torch", [input_image], args.precision, 3, 1)
    trt_results = record_perf(
        trt_model, "TensorRT", [input_image], args.precision, 3, 1
    )

    print("==================================")
    print(pd.DataFrame(pyt_results))
    print("==================================")
    print(pd.DataFrame(trt_results))
