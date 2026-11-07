# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Generate one SDXL image with FP16 PyTorch or a Torch-TensorRT UNet.

Run from the repository root:
    python examples/apps/sdxl_demo.py --pytorch --output sdxl_pytorch.png
    python examples/apps/sdxl_demo.py --output sdxl_trt.png

Windows ARM64 requires CUDA-enabled ARM64 PyTorch and, for the TensorRT path,
a matching Torch-TensorRT-RTX build. ModelOpt and xformers are not required.
"""

import argparse
import json
from contextlib import ExitStack, contextmanager
from pathlib import Path
from statistics import mean, median
from time import perf_counter


class Timings:
    def __init__(self, synchronize=lambda: None):
        self.synchronize = synchronize
        self.seconds = {}
        self.calls = {}

    @contextmanager
    def measure(self, name):
        self.synchronize()
        start = perf_counter()
        try:
            yield
        finally:
            self.synchronize()
            self.seconds[name] = self.seconds.get(name, 0.0) + perf_counter() - start
            self.calls[name] = self.calls.get(name, 0) + 1

    @contextmanager
    def method(self, obj, method_name, label):
        original = getattr(obj, method_name)
        had_override = method_name in vars(obj)
        previous = vars(obj).get(method_name)

        def timed(*args, **kwargs):
            with self.measure(label):
                return original(*args, **kwargs)

        # MutableTorchTensorRTModule.__setattr__ otherwise forwards changes to
        # the source UNet and marks its weights as changed.
        object.__setattr__(obj, method_name, timed)
        try:
            yield
        finally:
            if had_override:
                object.__setattr__(obj, method_name, previous)
            else:
                object.__delattr__(obj, method_name)


def generate(pipe, args, torch, backend):
    timings = Timings(lambda: torch.cuda.synchronize("cuda:0"))
    with ExitStack() as stack:
        stack.enter_context(timings.method(pipe, "encode_prompt", "text_encoding"))
        stack.enter_context(timings.method(pipe.vae, "decode", "vae_decode"))
        if backend == "tensorrt":
            stack.enter_context(timings.method(pipe.unet, "compile", "compile"))
            # Mutable modules dispatch through _forward rather than forward.
            unet_method = "_forward"
        else:
            unet_method = "forward"
        stack.enter_context(timings.method(pipe.unet, unet_method, "unet"))
        with timings.measure("total"), torch.no_grad():
            image = pipe(
                prompt=args.prompt,
                negative_prompt=args.negative_prompt,
                height=args.height,
                width=args.width,
                num_inference_steps=args.num_inference_steps,
                generator=torch.Generator(device="cuda:0").manual_seed(args.seed),
            ).images[0]

    seconds = timings.seconds
    compile_seconds = seconds.get("compile", 0.0)
    unet_seconds = seconds.get("unet", 0.0)
    result = {
        "total_s": seconds["total"],
        "compile_s": compile_seconds,
        "compile_calls": timings.calls.get("compile", 0),
        "text_encoding_s": seconds.get("text_encoding", 0.0),
        "unet_s": max(0.0, unet_seconds - compile_seconds),
        "unet_calls": timings.calls.get("unet", 0),
        "vae_decode_s": seconds.get("vae_decode", 0.0),
        "other_s": max(
            0.0,
            seconds["total"]
            - seconds.get("text_encoding", 0.0)
            - unet_seconds
            - seconds.get("vae_decode", 0.0),
        ),
    }
    return image, result


def print_run(label, result):
    print(
        f"{label}: total {result['total_s']:.3f}s | "
        f"compile/export {result['compile_s']:.3f}s "
        f"({result['compile_calls']} calls) | "
        f"text {result['text_encoding_s']:.3f}s | "
        f"UNet excluding compile {result['unet_s']:.3f}s "
        f"({result['unet_calls']} calls) | "
        f"VAE {result['vae_decode_s']:.3f}s | other {result['other_s']:.3f}s",
        flush=True,
    )


def benchmark(pipe, args, torch, backend):
    print(f"\nRunning {backend} with seed {args.seed}", flush=True)
    image, first = generate(pipe, args, torch, backend)
    print_run(f"{backend} first generation", first)
    output = args.output
    if args.compare:
        output = output.with_name(f"{output.stem}_{backend}{output.suffix}")
    output.parent.mkdir(parents=True, exist_ok=True)
    start = perf_counter()
    image.save(output)
    save_seconds = perf_counter() - start
    print(f"Saved image to {output.resolve()} ({save_seconds:.3f}s)", flush=True)

    for i in range(args.warmup):
        _, result = generate(pipe, args, torch, backend)
        print_run(f"{backend} extra warmup {i + 1}", result)

    runs = []
    for i in range(args.runs):
        _, result = generate(pipe, args, torch, backend)
        runs.append(result)
        print_run(f"{backend} measured generation {i + 1}", result)

    result = {
        "first": first,
        "runs": runs,
        "mean_s": mean(run["total_s"] for run in runs),
        "median_s": median(run["total_s"] for run in runs),
        "output": str(output),
        "save_s": save_seconds,
    }
    print(
        f"{backend} warmed generation: mean {result['mean_s']:.3f}s, "
        f"median {result['median_s']:.3f}s ({args.runs} runs)",
        flush=True,
    )
    if any(run["compile_calls"] for run in runs):
        print("Compilation occurred in measured runs; inspect compile/export timings.")
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="stabilityai/stable-diffusion-xl-base-1.0")
    parser.add_argument("--prompt", default="a majestic castle in the clouds")
    parser.add_argument("--negative_prompt", default="blurry, low quality")
    parser.add_argument("--output", type=Path, default=Path("sdxl.png"))
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--pytorch", action="store_true", help="Run without importing Torch-TensorRT"
    )
    mode.add_argument(
        "--compare", action="store_true", help="Benchmark PyTorch, then TensorRT"
    )
    parser.add_argument(
        "--runs", type=int, default=1, help="Measured generations after the first"
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=0,
        help="Extra unmeasured generations after the first",
    )
    parser.add_argument("--timings_json", type=Path, help="Save timing results as JSON")
    args = parser.parse_args()
    if any(size <= 0 or size % 8 for size in (args.height, args.width)):
        parser.error("--height and --width must be positive multiples of 8")
    if args.num_inference_steps <= 0:
        parser.error("--num_inference_steps must be positive")
    if args.runs <= 0 or args.warmup < 0:
        parser.error("--runs must be positive and --warmup must be nonnegative")
    return args


def main(args):
    setup = Timings()
    with setup.measure("pipeline_import"):
        import torch
        from diffusers import StableDiffusionXLPipeline

    if not torch.cuda.is_available():
        raise RuntimeError(
            "SDXL requires a CUDA-enabled PyTorch installation and NVIDIA GPU. "
            "On Windows ARM64, use native ARM64 CUDA PyTorch."
        )

    print(f"Loading {args.model} in FP16", flush=True)
    with setup.measure("model_load"):
        pipe = StableDiffusionXLPipeline.from_pretrained(
            args.model,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
            add_watermarker=False,
        )
    with setup.measure("move_to_cuda"):
        pipe.to("cuda:0")
        torch.cuda.synchronize("cuda:0")
    pipe.unet.eval()
    pipe.vae.enable_tiling()

    results = {}
    if args.pytorch or args.compare:
        results["pytorch"] = benchmark(pipe, args, torch, "pytorch")

    if not args.pytorch:
        with setup.measure("tensorrt_import"):
            import torch_tensorrt

        with setup.measure("tensorrt_wrapper"):
            pipe.unet = torch_tensorrt.MutableTorchTensorRTModule(
                pipe.unet,
                immutable_weights=False,
                min_block_size=1,
                use_fp32_acc=True,
                truncate_double=True,
                offload_module_to_cpu=True,
            )
        print("The first TensorRT generation will compile the UNet", flush=True)
        results["tensorrt"] = benchmark(pipe, args, torch, "tensorrt")

    print("\nSetup timings (separate from generation):")
    for label, seconds in setup.seconds.items():
        print(f"  {label}: {seconds:.3f}s")
    if args.compare:
        speedup = results["pytorch"]["median_s"] / results["tensorrt"]["median_s"]
        print(f"Warmed end-to-end speedup (PyTorch / TensorRT): {speedup:.2f}x")
    if args.timings_json:
        report = {
            "settings": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            "setup_s": setup.seconds,
            "results": results,
        }
        args.timings_json.parent.mkdir(parents=True, exist_ok=True)
        args.timings_json.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main(parse_args())
