"""
.. _quantize_vit_fp8:

Quantize and Compile ViT with FP8
==================================

This example demonstrates post-training FP8 quantization of a Hugging Face ViT
model with NVIDIA ModelOpt, then compilation with the Torch-TensorRT Dynamo
backend. There are two methods to compile the model with Torch-TensorRT:
1. IAttention path: converts attention to TRT IAttention Layer
2. Decomposed path: uses original attention implementation, which decomposes attention into multiple layers.

Requirements:

* NVIDIA GPU with FP8 support (Hopper or newer)
* ``nvidia-modelopt`` that supports the Hugging Face quantization recipes
* ``transformers`` to load the ViT model
* ``torch-tensorrt>=2.13.0`` which converts attention to TRT IAttention Layer

"""

# %%
# Imports
# ^^^^^^^

import tempfile
from collections import OrderedDict

import modelopt.torch.quantization as mtq
import numpy as np
import tensorrt as trt
import torch
import torch_tensorrt
from modelopt.recipe import load_recipe
from modelopt.torch.quantization.utils import export_torch_mode
from transformers import ViTForImageClassification

# %%
# Optional layer profiler and benchmark function (Python TensorRT runtime)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


def _format_report(timings: OrderedDict[str, list[float]], n_iters: int) -> str:
    """Format layer timings into a trtexec-style profiling summary."""
    if not timings:
        return "(no profiling data collected)"

    total_time = sum(sum(ts) for ts in timings.values())

    lines: list[str] = []
    lines.append(f"=== Profile ({n_iters} iterations) ===")
    lines.append(
        f"{'Time(ms)':>12s} {'Avg.(ms)':>12s} {'Median(ms)':>12s} {'Time(%)':>9s}   Layer"
    )

    for layer_name, ts in timings.items():
        arr = np.array(ts)
        t_sum = arr.sum()
        t_avg = arr.mean()
        t_med = float(np.median(arr))
        t_pct = 100.0 * t_sum / total_time if total_time > 0 else 0.0
        lines.append(
            f"{t_sum:12.2f} {t_avg:12.4f} {t_med:12.4f} {t_pct:8.1f}   {layer_name}"
        )

    return "\n".join(lines)


class TorchTRTLayerProfiler(trt.IProfiler):
    """Collect per-layer timings across many runs via ``trt.IProfiler``."""

    def __init__(self):
        super().__init__()
        self._timings: OrderedDict[str, list[float]] = OrderedDict()
        self._iterations: int = 0
        self._last_layer: str | None = None

    def report_layer_time(self, layer_name: str, ms: float) -> None:
        if layer_name not in self._timings:
            self._timings[layer_name] = []
        self._timings[layer_name].append(ms)

        if self._last_layer is None or layer_name == next(iter(self._timings)):
            if self._last_layer is not None:
                self._iterations += 1
        self._last_layer = layer_name

    def reset(self) -> None:
        self._timings.clear()
        self._iterations = 0
        self._last_layer = None

    def report(self) -> str:
        n_iters = self._iterations
        first_layer_count = len(next(iter(self._timings.values()), []))
        if first_layer_count > n_iters:
            n_iters = first_layer_count
        text = _format_report(self._timings, n_iters)
        print(text)
        return text


def benchmark_tensorrt(
    trt_compiled,
    input_tensor: torch.Tensor,
    *,
    enable_profiling: bool = False,
    num_warmup: int = 10,
    num_runs: int = 50,
):
    with torch.no_grad():
        for _ in range(num_warmup):
            trt_compiled(input_tensor)
        torch.cuda.synchronize()

        with torch_tensorrt.runtime.enable_cudagraphs(trt_compiled) as cg_model:
            start_events = [
                torch.cuda.Event(enable_timing=True) for _ in range(num_runs)
            ]
            end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_runs)]
            results = []
            for i in range(num_runs):
                start_events[i].record()
                result = cg_model(input_tensor)
                results.append(result.logits.sum().item())
                end_events[i].record()
            torch.cuda.synchronize()
            latencies = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]

        latencies = np.array(latencies)
        print(f"\nLatency over {num_runs} runs (ms):")
        print(f"  Mean:   {np.mean(latencies):.2f}")
        print(f"  Median: {np.median(latencies):.2f}")
        print(f"  Min:    {np.min(latencies):.2f}")
        print(f"  Max:    {np.max(latencies):.2f}")
        print(f"  P90:    {np.percentile(latencies, 90):.2f}")
        print(f"  P95:    {np.percentile(latencies, 95):.2f}")
        print(f"  P99:    {np.percentile(latencies, 99):.2f}")
        print(f"  Avg logits sum: {np.mean(results):.2f}")

        if enable_profiling and hasattr(trt_compiled, "engine"):
            profiler = TorchTRTLayerProfiler()
            trt_compiled.engine.context.profiler = profiler
            torch_results = []
            for _ in range(num_runs):
                result = trt_compiled(input_tensor)
                torch_results.append(result.logits.sum().item())
            torch.cuda.synchronize()
            print(f"  Avg logits sum (no cudagraphs): {np.mean(torch_results):.2f}")
            profiler.report()


# %%
# Configuration and calibration data
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

BATCH_SIZE = 128
input_tensor = torch.rand((BATCH_SIZE, 3, 224, 224), dtype=torch.float16).cuda()

calibration_dataloader = [
    torch.rand((BATCH_SIZE, 3, 224, 224), dtype=torch.float16) for _ in range(2)
]

# %%
# Define function to quantize ViT to FP8 using ModelOpt
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


def quantize_model(model: torch.nn.Module) -> torch.nn.Module:
    # ModelOpt's HF plugin expects attention modules to expose ``config``.
    for mod in model.modules():
        if mod.__class__.__name__ == "ViTAttention":
            mod.config = model.config

    def calibration_loop(model):
        for batch in calibration_dataloader:
            model(batch.cuda())

    # Define quant_cfg to specify which layers to quantize.
    quant_cfg = {
        "quant_cfg": {
            "*": {"enable": False},
            "*weight_quantizer": {"num_bits": (4, 3), "axis": None},
            "*input_quantizer": {"num_bits": (4, 3), "axis": None},
            "*output_quantizer": {"enable": False},
            "*[qkv]_bmm_quantizer": {"num_bits": (4, 3), "axis": None},
            "*softmax_quantizer": {"num_bits": (4, 3), "axis": None},
            "*bmm2_output_quantizer": {"num_bits": (4, 3), "axis": None},
        },
        "algorithm": "max",
    }
    # You can use the following recipe if you have nvidia-modelopt >= 0.46.0
    # recipe = load_recipe("huggingface/vit/ptq/fp8")
    # quant_cfg = recipe.quantize.model_dump()

    mtq.quantize(model, quant_cfg, forward_loop=calibration_loop)
    return model


# %%
# Define function to compile with Torch-TensorRT
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


def compile_tensorrt(
    model: torch.nn.Module, dynamic: bool = False, decompose_attention: bool = False
):
    if dynamic:
        inputs = [
            torch_tensorrt.Input(
                min_shape=(1, 3, 224, 224),
                opt_shape=(BATCH_SIZE, 3, 224, 224),
                max_shape=(1024, 3, 224, 224),
                dtype=torch.float16,
            )
        ]
    else:
        inputs = [torch.rand((BATCH_SIZE, 3, 224, 224), dtype=torch.float16).cuda()]

    with tempfile.TemporaryDirectory() as debug_dir:
        with export_torch_mode(), torch_tensorrt.dynamo.Debugger(
            log_level="info",
            save_layer_info=True,
            logging_dir=debug_dir,
        ):
            trt_compiled = torch_tensorrt.compile(
                model,
                ir="dynamo",
                inputs=inputs,
                use_explicit_typing=True,
                truncate_double=True,
                use_python_runtime=True,
                decompose_attention=decompose_attention,
            )

    return trt_compiled


# %%
# Instantiate ViT model and quantize it
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Both "eager" and "sdpa" attn_implementation work for IAttention path, but only "eager" works for decomposed path.
# For demonstration purposes, we use "eager" attn_implementation here so that both paths can share the same model.


model = (
    ViTForImageClassification.from_pretrained(
        "google/vit-large-patch16-224",
        hidden_act="gelu_fast",
        attn_implementation="eager",
    )
    .eval()
    .half()
    .cuda()
)

quantized_model = quantize_model(model)


# %%
# Compile quantized model with Torch-TensorRT: IAttention path
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


trt_compiled_iattention = compile_tensorrt(
    quantized_model,
    decompose_attention=False,  # False for IAttention path
)

# %%
# IAttention path benchmark inference
benchmark_tensorrt(
    trt_compiled_iattention,
    input_tensor,
)


# %%
# Compile quantized model with Torch-TensorRT: Decomposed path
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

trt_compiled_decomposed = compile_tensorrt(
    quantized_model,
    decompose_attention=True,  # True for decomposed path
)

# %%
# Decomposed path benchmark inference
benchmark_tensorrt(
    trt_compiled_decomposed,
    input_tensor,
)
