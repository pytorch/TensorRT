import argparse
import timeit
import numpy as np
import pandas as pd
import torch
import torch_tensorrt
from PIL import Image
from sam2.sam2_image_predictor import SAM2ImagePredictor

from sam_components import ImageEncoder, HeadModule, SAM2FullModel

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
        "Median(FPS)": speed_med,
        "Median-Latency(ms)": time_med * 1000,
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
            out = model(*input_tensors)
            torch.cuda.synchronize()
            end_time = timeit.default_timer()
            timings.append(end_time - start_time)

    results = recordStats(
        backend, timings, precision, batch_size, compile_time_s
    )

    return results, out


def evaluate_optimization_accuracy(original_output, optimized_output, precision, cos_sim_tol=0.99):
    results = []

    def compute_cosine_similarity(orig, opt, convert_dtype=None):
        """Compute cosine similarity, with optional dtype conversion."""
        if convert_dtype:
            orig, opt = orig.to(convert_dtype), opt.to(convert_dtype)

        # Calculate cosine similarity and prevent NaN by adding a small epsilon to denominators
        cosine_similarity = torch.nn.functional.cosine_similarity(
            orig.flatten() + 1e-8, opt.flatten() + 1e-8, dim=0
        ).item()
        
        return cosine_similarity

    for key, orig in original_output.items():
        opt = optimized_output[key]

        if isinstance(orig, list) and isinstance(opt, list):  # Lists (e.g., high_res_feats)
            for i, (orig_feat, opt_feat) in enumerate(zip(orig, opt)):
                label = f"{key}[{i}]"
                
                # Compute original cosine similarity
                original_cosine = compute_cosine_similarity(orig_feat, opt_feat)
                result = {"Element": label, "Cosine Similarity (Original)": original_cosine}
                
                # Additional cosine similarity for dtype conversion based on precision
                if precision == "fp16" and orig_feat.dtype == torch.float16:
                    converted_cosine = compute_cosine_similarity(orig_feat, opt_feat, convert_dtype=torch.float32)
                    result["Cosine Similarity (Converted to torch.float32)"] = converted_cosine
                elif precision == "fp32" and orig_feat.dtype == torch.float32:
                    converted_cosine = compute_cosine_similarity(orig_feat, opt_feat, convert_dtype=torch.float16)
                    result["Cosine Similarity (Converted to torch.float16)"] = converted_cosine

                results.append(result)

        else:  # Single tensors (e.g., image_embed)
            label = key
            
            # Compute original cosine similarity
            original_cosine = compute_cosine_similarity(orig, opt)
            result = {"Element": label, "Cosine Similarity (Original)": original_cosine}

            # Additional cosine similarity for dtype conversion based on precision
            if precision == "fp16" and orig.dtype == torch.float16:
                converted_cosine = compute_cosine_similarity(orig, opt, convert_dtype=torch.float32)
                result["Cosine Similarity (Converted to torch.float32)"] = converted_cosine
            elif precision == "fp32" and orig.dtype == torch.float32:
                converted_cosine = compute_cosine_similarity(orig, opt, convert_dtype=torch.float16)
                result["Cosine Similarity (Converted to torch.float16)"] = converted_cosine

            results.append(result)
    
    return results




def build_model(args):
    predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-small")
    model = predictor.model.eval().cuda()
    
    image = Image.open("./truck.jpg").convert("RGB")
    predictor.set_image(image)

    if args.precision == "fp16":
        model = model.half()

    if args.mode == "encoder":
        return ImageEncoder(model).eval().cuda(), predictor
    elif args.mode == "head":
        return HeadModule(model).eval().cuda(), predictor
    elif args.mode == "all":
        return SAM2FullModel(model).eval().cuda(), predictor

def build_input(args, file_path, predictor):
    # Raw input
    image = Image.open(file_path).convert("RGB")
    input_image = predictor._transforms(np.array(image))[None, ...].to("cuda:0")
    point_coords = torch.tensor([[500, 375]], dtype=torch.float).unsqueeze(0).to("cuda:0")
    point_labels = torch.tensor([1], dtype=torch.int).unsqueeze(0).to("cuda:0")
    mask_input = torch.zeros(1, 1, 256, 256).to("cuda:0")

    if args.precision == "fp16":
        input_image = input_image.half()
        point_coords = point_coords.half()
        mask_input = mask_input.half()

    if args.mode == "encoder":
        return (input_image,)
    elif args.mode == "head":
        image_embedding = predictor.get_image_embedding().to("cuda:0").half()
        high_res_features = [feat.to("cuda:0").half() for feat in predictor._features["high_res_feats"]]

        return (image_embedding, point_coords, point_labels, mask_input, high_res_features)
    elif args.mode == "all":
        return (input_image, point_coords, point_labels, mask_input)

def compile_with_torchtrt(model, inputs, precision):
    # from torch.export._trace import _export
    # ep = _export(
    #     model,
    #     inputs,
    #     strict=False,
    #     allow_complex_guards_as_runtime_asserts=True,
    # )
    ep = torch.export.export(model, inputs, strict=False)

    with torch_tensorrt.logging.debug():
        trt_gm = torch_tensorrt.dynamo.compile(
            ep,
            inputs=inputs, 
            debug=True,
            min_block_size=1,
            enabled_precisions={torch.float16 if precision == "fp16" else torch.float32},
        )

    return trt_gm

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="Run inference on SAM")
    arg_parser.add_argument(
        "--precision",
        type=str,
        default="fp16",
        help="Precision of the model to compile for TensorRT",
    )
    arg_parser.add_argument(
        "--mode",
        type=str,
        choices=["encoder", "head", "all"],
        default="head",
        help="Supported options include encoder | prediction_head",
    )
    args = arg_parser.parse_args()

    pyt_model, predictor = build_model(args)
    inputs = build_input(args, "./truck.jpg", predictor)

    # PyTorch 
    pyt_results, pyt_out = record_perf(pyt_model, "Torch", inputs, args.precision, 10, 1)

    # Torch-TensorRT 
    trt_model = compile_with_torchtrt(pyt_model, inputs, args.precision)
    trt_results, trt_out = record_perf(trt_model, "TensorRT", inputs, args.precision, 10, 1)

    print("==================================")
    print("PyTorch Results:", pd.DataFrame(pyt_results))
    print("==================================")
    print("TensorRT Results:", pd.DataFrame(trt_results))

    # numerical accuracy
    accuracy_results = evaluate_optimization_accuracy(pyt_out, trt_out, args.precision)
    print("Cosine Similarity Results:", pd.DataFrame(accuracy_results))