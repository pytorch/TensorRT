import copy
import os
import timeit

import huggingface_hub
import modelopt.torch.quantization as mtq
import numpy as np
import torch
from huggingface_hub import snapshot_download
from modelopt.torch.quantization.utils import export_torch_mode
from modelopt.torch.utils.dataset_utils import (
    create_forward_loop,
    get_dataset_dataloader,
)
from safetensors import safe_open
from transformers import StoppingCriteriaList
from transformers.generation.stopping_criteria import (
    EosTokenCriteria,
    MaxLengthCriteria,
)


def export_llm(model, inputs, min_seq_len=1, max_seq_len=16):
    """
    Exports the LLM model into an ExportedProgram with dynamic shapes.
    In the case of guard failures due to some PyTorch kernel implements, we also
    try to re-export the graph by expressing them as runtime assert nodes
    """
    with torch.no_grad():
        with export_torch_mode():
            # max=1024 has contraint violation error. https://github.com/pytorch/pytorch/issues/125604
            seq_len = torch.export.Dim("seq_len", min=min_seq_len, max=max_seq_len)
            position_ids = torch.arange(inputs.shape[1]).unsqueeze(0).to(inputs.device)
            try:
                print("Trying to export the model using torch.export.export()..")
                # strict=False only enables aotautograd tracing and excludes dynamo.
                ep = torch.export.export(
                    model,
                    args=(inputs,),
                    kwargs={"position_ids": position_ids},
                    dynamic_shapes=({1: seq_len}, {1: seq_len}),
                    strict=False,
                )
            except:
                print(
                    "Trying torch.export._trace._export to trace the graph since torch.export.export() failed"
                )
                # This API is used to express the constraint violation guards as asserts in the graph.
                ep = torch.export._trace._export(
                    model,
                    args=(inputs,),
                    kwargs={"position_ids": position_ids},
                    dynamic_shapes=({1: seq_len}, {1: seq_len}),
                    strict=False,
                    allow_complex_guards_as_runtime_asserts=True,
                )

    return ep


def get_zeroed_static_cache_inputs(model: torch.fx.GraphModule):
    """
    Extracts and returns zeroed static KV cache tensors from a torch.fx.GraphModule. This should only be used for static cache_v1 and static cache_v2.

    This function identifies placeholder nodes in the graph that represent KV cache tensors,
    and creates zeroed tensors with the same shape, dtype, and device as the original placeholders.

    Args:
        model (torch.fx.GraphModule): The exported model graph containing KV cache placeholders

    Returns:
        tuple: A tuple of zeroed tensors corresponding to the KV cache placeholders in the graph
    """
    # placeholder nodes are expected to be in the following order:
    # input_ids, kv_cache_key, kv_cache_value, start_idx, end_idx
    placeholder_nodes = [node for node in model.graph.nodes if node.op == "placeholder"]
    # The first two inputs are input_ids, position_ids. The last two inputs are start_idx, end_idx. In between are the KV cache tensors.
    kv_cache_inputs = placeholder_nodes[2:-2]
    zeroed_kv_cache_inputs = []
    for input in kv_cache_inputs:
        zeroed_kv_cache_inputs.append(
            torch.zeros(
                input.meta["val"].shape,
                dtype=input.meta["val"].dtype,
                device=torch.device("cuda:0"),
            )
        )

    return tuple(zeroed_kv_cache_inputs)


def get_zeroed_dynamic_cache_inputs(model: torch.fx.GraphModule):
    """
    Extracts and returns zeroed KV cache tensors from a torch.fx.GraphModule. This should only be used for dynamic cache.

    This function identifies placeholder nodes in the graph that represent KV cache tensors,
    and creates zeroed tensors with the same shape, dtype, and device as the original placeholders.

    Args:
        model (torch.fx.GraphModule): The exported model graph containing KV cache placeholders

    Returns:
        tuple: A tuple of zeroed tensors corresponding to the KV cache placeholders in the graph
    """
    # placeholder nodes are expected to be in the following order:
    # input_ids, kv_cache_key, kv_cache_value, start_idx, end_idx
    placeholder_nodes = [node for node in model.graph.nodes if node.op == "placeholder"]
    # The first two inputs are input_ids, position_ids. The last input is is_generate. In between are the KV cache tensors.
    kv_cache_inputs = placeholder_nodes[2:-1]
    zeroed_kv_cache_inputs = []
    for input in kv_cache_inputs:
        zeroed_kv_cache_inputs.append(
            torch.zeros(
                input.meta["val"].shape,
                dtype=input.meta["val"].dtype,
                device=torch.device("cuda:0"),
            )
        )

    return tuple(zeroed_kv_cache_inputs)


def generate(model, input_seq, max_output_seq_length, eos_token_id, benchmark=True):
    """
    Greedy decoding of the model. This generates up to max_tokens.
    """
    stopping_criteria = StoppingCriteriaList(
        [
            MaxLengthCriteria(max_length=max_output_seq_length),
            EosTokenCriteria(eos_token_id=eos_token_id),
        ]
    )
    isl = input_seq.shape[1]
    osl = max_output_seq_length - isl

    num_tokens_generated = 0
    while num_tokens_generated < osl:
        position_ids = torch.arange(input_seq.shape[1]).unsqueeze(0).cuda()
        outputs = model(input_seq, position_ids=position_ids)
        logits = outputs.logits
        next_token_logits = logits[:, -1, :]
        next_tokens = torch.argmax(next_token_logits, dim=-1)
        input_seq = torch.cat([input_seq, next_tokens[:, None]], dim=-1)
        num_tokens_generated += 1
        # TODO: Handle batch in this check
        if not benchmark and stopping_criteria(input_seq, logits).item():
            break

    return input_seq


def generate_with_static_cache(model, input_seq, max_output_seq_length, eos_token_id):
    """
    Greedy decoding of the model with static KV cache.
    """
    start_idx = 0
    end_idx = input_seq.shape[1]
    position_ids = torch.arange(input_seq.shape[1]).unsqueeze(0).cuda()
    output_seq = input_seq.clone()
    # TODO: Confirm this: When end_idx = max_output_seq_length-1, number of tokens generated = OSL
    num_tokens_generated = 0
    kv_cache = get_zeroed_static_cache_inputs(model)
    while end_idx < max_output_seq_length:
        position_ids = (
            torch.tensor([[start_idx]], dtype=torch.int64).cuda()
            if input_seq.shape[1] == 1
            else position_ids
        )
        input_signature = (input_seq, position_ids, *kv_cache, start_idx, end_idx)
        logits_keys_values = model(*input_signature)
        num_tokens_generated += 1
        logits = logits_keys_values[0]
        kv_cache = logits_keys_values[1:]
        next_token_logits = logits[:, -1, :]
        next_tokens = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        output_seq = torch.cat([output_seq, next_tokens], dim=-1)
        input_seq = next_tokens
        start_idx = end_idx
        end_idx = start_idx + 1
    return output_seq


def generate_with_dynamic_cache(model, input_seq, max_output_seq_length, eos_token_id):
    """
    Greedy decoding of the model with dynamic KV cache.
    """
    position_ids = torch.arange(input_seq.shape[1]).unsqueeze(0).cuda()
    output_seq = input_seq.clone()
    num_output_tokens = max_output_seq_length - input_seq.shape[1]
    num_tokens_generated = 0
    kv_cache = get_zeroed_dynamic_cache_inputs(model)
    last_position_id = position_ids[-1, -1].item()
    breakpoint()
    while num_tokens_generated < num_output_tokens:
        is_generate = False if input_seq.shape[1] > 1 else True
        position_ids = (
            torch.tensor([[last_position_id + 1]], dtype=torch.int64).cuda()
            if input_seq.shape[1] == 1
            else position_ids
        )
        input_signature = (input_seq, position_ids, *kv_cache, is_generate)
        logits_keys_values = model(*input_signature)
        num_tokens_generated += 1
        logits = logits_keys_values[0]
        kv_cache = logits_keys_values[1:]
        next_token_logits = logits[:, -1, :]
        next_tokens = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        output_seq = torch.cat([output_seq, next_tokens], dim=-1)
        input_seq = next_tokens
        last_position_id += 1
    return output_seq


def time_generate(
    generate_fn, model, inputs, output_seq_length, eos_token_id, iterations=10
):
    """
    Measure the time for generating a sentence over certain number of iterations
    """
    timings = []
    for _ in range(iterations):
        start_time = timeit.default_timer()
        _ = generate_fn(model, inputs, output_seq_length, eos_token_id)
        torch.cuda.synchronize()
        end_time = timeit.default_timer()
        timings.append(end_time - start_time)

    return timings


def record_stats(backend, timings, precision, batch_size=1, compile_time_s=None):
    """
    Records different timing stats and adds it to the result
    """
    times = np.array(timings)
    speeds = batch_size / times
    time_mean = np.mean(times).item()
    time_med = np.median(times).item()
    time_99th = np.percentile(times, 99).item()
    time_std = np.std(times, ddof=0).item()
    speed_mean = np.mean(speeds).item()
    speed_med = np.median(speeds).item()

    stats = {
        "Backend": backend,
        "Precision": precision,
        "Batch size": batch_size,
        "Median(FPS)": speed_med,
        "Mean(FPS)": speed_mean,
        "Median-Latency(ms)": time_med * 1000,
        "Mean-Latency(ms)": time_mean * 1000,
        "Latency-StdDev(ms)": time_std * 1000,
        "Compile Time(s)": compile_time_s,
    }
    return stats


def quantize_model(model, args, tokenizer):
    calib_dataloader = get_dataset_dataloader(
        tokenizer=tokenizer,
        batch_size=32,
        num_samples=512,
        device=torch.device("cuda:0"),
    )
    if args.qformat == "fp8":
        quant_cfg = mtq.FP8_DEFAULT_CFG
    else:
        raise RuntimeError("Unsupported quantization format")
    calibrate_loop = create_forward_loop(dataloader=calib_dataloader)

    model = mtq.quantize(model, quant_cfg, forward_loop=calibrate_loop)
    if args.debug:
        mtq.print_quant_summary(model)

    return model


quantize_op = torch.ops.tensorrt.quantize_op


class TensorRTQuantizedLinear(torch.nn.Module):
    """
    TensorRT quantized linear layer that applies FP8 quantization to both input and weight.

    This class implements a quantized linear layer that:
    1. Applies QDQ (Quantize-Dequantize) to input tensor using FP8 E4M3 format
    2. Applies QDQ (Quantize-Dequantize) to weight tensor using FP8 E4M3 format
    3. Performs linear operation with dequantized tensors

    The quantize_op function handles the complete QDQ pattern internally,
    returning dequantized tensors that maintain precision while reducing memory usage.
    """

    def __init__(self, original_linear: torch.nn.Linear, input_amax, weight_amax):
        """
        Initialize quantized linear layer.

        Args:
            original_linear: Original PyTorch linear layer to quantize
            input_amax: Maximum absolute value for input quantization scaling
            weight_amax: Maximum absolute value for weight quantization scaling
        """
        super().__init__()

        # Store reference to original linear layer for weight access
        self.original_linear = original_linear

        # Copy bias from original layer if it exists
        if original_linear.bias is not None:
            self.bias = torch.nn.Parameter(original_linear.bias.clone()).cuda()
        else:
            self.bias = None

        # These control the quantization scaling for input and weight tensors
        self.input_amax = torch.nn.Parameter(input_amax).cuda()
        self.weight_amax = torch.nn.Parameter(weight_amax).cuda()

    def forward(self, x):
        """
        Forward pass with FP8 QDQ quantization.

        Args:
            x: Input tensor of shape (batch_size, in_features)

        Returns:
            Output tensor of shape (batch_size, out_features)
        """
        # Step 1: Apply QDQ to input tensor using FP8 E4M3 format
        # quantize_op performs: Quantize -> Dequantize, returning dequantized tensor
        x_dequantized = quantize_op(
            x,
            self.input_amax,
            num_bits=8,
            exponent_bits=4,
            unsigned=False,
            narrow_range=False,
        )

        # Step 2: Apply QDQ to weight tensor using FP8 E4M3 format
        # quantize_op performs: Quantize -> Dequantize, returning dequantized tensor
        weight_dequantized = quantize_op(
            self.original_linear.weight,
            self.weight_amax,
            num_bits=8,
            exponent_bits=4,
            unsigned=False,
            narrow_range=False,
        )

        # Step 3: Perform linear operation with dequantized tensors
        # Both tensors are now dequantized, maintaining precision while reducing memory
        return torch.nn.functional.linear(x_dequantized, weight_dequantized, self.bias)


def convert_linear_to_tensorrt_quantized(model, model_name):
    """
    Convert linear layers in a model to TensorRT quantized versions using FP8 quantization.

    This function:
    1. Loads quantization scales from Hugging Face model files
    2. Replaces standard linear layers with TensorRTQuantizedLinear layers
    3. Applies FP8 E4M3 quantization to both input and weight tensors

    Args:
        model: PyTorch model to quantize
        model_name: Path to Hugging Face model or model identifier

    Returns:
        Model with quantized linear layers
    """
    # Determine if model_name is a local directory or needs to be downloaded
    if os.path.isdir(model_name):
        hf_folder = model_name
    else:
        # Download model from Hugging Face Hub
        hf_folder = snapshot_download(
            model_name,
            local_files_only=huggingface_hub.constants.HF_HUB_OFFLINE,
            ignore_patterns=["original/**/*"],
            revision=None,
        )

    # Load all tensors from SafeTensors files
    scale_tensors = {}
    for file in os.listdir(hf_folder):
        if file.endswith(".safetensors"):
            with safe_open(
                os.path.join(hf_folder, file), framework="pt", device="cpu"
            ) as f:
                tensor_names = f.keys()
                for name in tensor_names:
                    if name.endswith(".weight_scale") or name.endswith(".input_scale"):
                        scale_tensors[name] = f.get_tensor(name)

    # Iterate through all modules in the model
    for name, module in model.named_modules():
        # Check if the module is a linear layer
        target = torch.nn.modules.linear.Linear
        if isinstance(module, target):
            # Construct names for quantization scale tensors
            # These follow the naming convention: module_name.weight_scale and module_name.input_scale
            weight_scale_name = name + ".weight_scale"
            input_scale_name = name + ".input_scale"

            # Verify that required scale tensors exist in the loaded data
            if weight_scale_name not in scale_tensors:
                print(f"Weight scale tensor {weight_scale_name} not found")
                continue
            if input_scale_name not in scale_tensors:
                print(f"Input scale tensor {input_scale_name} not found")
                continue

            # Calculate amax values for quantization
            # FP8 E4M3 format has a maximum representable value of 448.0
            weight_scale = scale_tensors.pop(weight_scale_name)
            weight_amax = weight_scale * 448.0
            input_amax = scale_tensors.pop(input_scale_name) * 448.0

            # Apply dequantization to the original quantized weight using the scale
            # This ensures the weight is in the correct range for the quantized layer
            module.weight.data = module.weight.to(torch.float32) * weight_scale

            # Create the quantized linear layer with calculated amax values
            quantized_module = TensorRTQuantizedLinear(module, input_amax, weight_amax)

            # Replace the original module with the quantized version
            # Extract parent module name and child module name
            parent_name = ".".join(name.split(".")[:-1])
            child_name = name.split(".")[-1]

            if parent_name:
                # Get the parent module and replace the child
                parent_module = model.get_submodule(parent_name)
                setattr(parent_module, child_name, quantized_module)
            else:
                # If no parent, replace at model level
                setattr(model, child_name, quantized_module)

    if len(scale_tensors) > 0:
        print(f"Warning: {len(scale_tensors)} scale tensors not used")
    return model
