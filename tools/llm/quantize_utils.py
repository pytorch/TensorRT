import json
import logging
import os

import huggingface_hub
import torch
import torch_tensorrt
from huggingface_hub import snapshot_download

logger = logging.getLogger(__name__)

try:
    import modelopt.torch.quantization as mtq  # noqa: F401f

    assert torch.ops.tensorrt.quantize_op.default
except Exception:
    logger.warning("Unable to import quantization op. Please install modelopt library")

from modelopt.core.torch.quantization.qtensor.nvfp4_tensor import NVFP4QTensor
from modelopt.torch.quantization.config import QuantizerAttributeConfig
from modelopt.torch.quantization.nn.modules.tensor_quantizer import TensorQuantizer
from modelopt.torch.utils.dataset_utils import (
    create_forward_loop,
    get_dataset_dataloader,
)
from safetensors import safe_open


def quantize_model(model, args, tokenizer):
    """
    Quantize a PyTorch model using ModelOpt post-training quantization (PTQ).

    This function applies quantization to reduce model precision for faster inference
    while maintaining acceptable accuracy. It uses calibration data generated from
    the provided tokenizer to determine optimal quantization parameters.

    Supported quantization formats:
        - fp8: 8-bit floating point quantization
        - nvfp4: 4-bit NVIDIA floating point quantization
    Args:
        model: PyTorch model to quantize. Must be in evaluation mode.
        args: Command line arguments containing quant_format and debug
        tokenizer: Hugging Face tokenizer for creating calibration data

    Returns:
        Quantized model
    """
    # Create calibration dataloader for quantization
    calib_dataloader = get_dataset_dataloader(
        tokenizer=tokenizer,
        batch_size=32,
        num_samples=512,
        device="cuda:0",
    )

    calibrate_loop = create_forward_loop(dataloader=calib_dataloader)
    model = torch_tensorrt.dynamo.quantize(
        model, args.quant_format, calibrate_loop, debug=args.debug
    )

    return model


class TensorRTQuantizedLinear(torch.nn.Module):
    """
    TensorRT quantized linear layer that applies quantization to both input and weight tensors.
    """

    def __init__(
        self, original_linear: torch.nn.Linear, input_amax, weight_amax, quant_cfg
    ):
        """
        Initialize quantized linear layer.

        Args:
            original_linear: Original PyTorch linear layer to quantize
            input_amax: Maximum absolute value for input quantization scaling
            weight_amax: Maximum absolute value for weight quantization scaling
            quant_cfg: Quantization configuration for TensorQuantizer
        """
        super().__init__()

        # Store reference to original linear layer for weight access
        self.original_linear = original_linear

        # Copy bias from original layer if it exists
        if original_linear.bias is not None:
            self.bias = torch.nn.Parameter(original_linear.bias.clone()).cuda()
        else:
            self.bias = None

        # Create quantizers for input and weight tensors
        self.input_quantizer = TensorQuantizer(
            quant_attribute_cfg=quant_cfg, amax=input_amax
        )
        self.weight_quantizer = TensorQuantizer(
            quant_attribute_cfg=quant_cfg, amax=weight_amax
        )

    def forward(self, input):
        input = self.input_quantizer(input)
        weight = self.weight_quantizer(self.original_linear.weight)
        return torch.nn.functional.linear(input, weight, self.bias)


def load_quantization_config(model_name):
    """
    Load quantization configuration from a Hugging Face model.
    Args:
        model_name (str): Local directory path or model identifier
    Returns:
        dict or None: Quantization configuration. None if no config found.
    """
    # Determine if model_name is a local directory or needs to be downloaded
    if os.path.isdir(model_name):
        model_path = model_name
    else:
        # Download model from Hugging Face Hub
        model_path = snapshot_download(
            model_name,
            local_files_only=huggingface_hub.constants.HF_HUB_OFFLINE,
            ignore_patterns=["original/**/*"],
            revision=None,
        )
    hf_quant_config = None
    # Load and parse quantization configuration
    hf_quant_config_path = f"{model_path}/hf_quant_config.json"
    if os.path.exists(hf_quant_config_path):
        with open(hf_quant_config_path, "r") as f:
            hf_quant_config = json.load(f)
            hf_quant_config = hf_quant_config["quantization"]
            hf_quant_config["model_path"] = model_path

    return hf_quant_config


def convert_linear_to_tensorrt_quantized(model, hf_quant_config):
    """
    Convert linear layers in a model to TensorRT quantized versions from pre-quantized weights.

    This function is specifically designed for Hugging Face quantized models and only
    applies quantization to linear operations. It loads pre-quantized models from
    Hugging Face format and replaces standard linear layers with TensorRTQuantizedLinear
    layers. It supports both FP8 and NVFP4 quantization formats.

    The function:
    1. Loads quantization scales from Hugging Face model files (SafeTensors)
    2. Replaces standard linear layers with TensorRTQuantizedLinear layers
    3. Applies appropriate quantization based on the model's quantization format

    Note: This function only quantizes linear operations and is intended for use
    with pre-quantized Hugging Face models that have been quantized using ModelOpt.

    Args:
        model: PyTorch model to quantize
        hf_quant_config: Quantization configuration

    Returns:
        Model with quantized linear layers

    Raises:
        RuntimeError: If quantization config is not found or unsupported format
    """
    model_path = hf_quant_config["model_path"]
    # Load all tensors from SafeTensors files
    tensors = {}
    for file in os.listdir(model_path):
        if file.endswith(".safetensors"):
            with safe_open(
                os.path.join(model_path, file), framework="pt", device="cpu"
            ) as f:
                tensor_names = f.keys()
                for name in tensor_names:
                    tensors[name] = f.get_tensor(name)

    hf_quant_algo = hf_quant_config.get("quant_algo", None)
    if hf_quant_algo != "FP8" and hf_quant_algo != "NVFP4":
        raise RuntimeError("Only FP8 or NVFP4 quantization is supported")

    # Iterate through all modules in the model
    for name, module in model.named_modules():
        # Check if the module is a linear layer
        target = torch.nn.modules.linear.Linear
        if isinstance(module, target):
            # Construct names for quantization scale tensors
            # These follow the naming convention: module_name.weight_scale and module_name.input_scale
            weight_scale_name = name + ".weight_scale"
            input_scale_name = name + ".input_scale"

            if weight_scale_name not in tensors:
                logger.warning(f"Weight scale tensor {weight_scale_name} not found")
                continue
            if input_scale_name not in tensors:
                logger.warning(f"Input scale tensor {input_scale_name} not found")
                continue

            if hf_quant_algo == "FP8":
                # FP8 E4M3 format has a maximum representable value of 448.0
                # Scale the quantization parameters accordingly
                weight_scale = tensors.pop(weight_scale_name)
                weight_amax = weight_scale * 448.0
                input_amax = tensors.pop(input_scale_name) * 448.0

                # Dequantize the weight using the scale factor
                dequantized_weight_data = module.weight.to(torch.float32) * weight_scale

                # Configure quantizer for FP8 format (4 exponent bits, 3 mantissa bits)
                quantizer_attribute_config = QuantizerAttributeConfig(
                    num_bits=(4, 3), axis=None
                )

            elif hf_quant_algo == "NVFP4":
                # NVFP4 format requires additional scale tensor and different configuration
                weight_name = name + ".weight"
                weight_scale2_name = name + ".weight_scale_2"
                weight_scale = tensors.pop(weight_scale_name)
                input_scale = tensors.pop(input_scale_name)
                weight_scale2 = tensors.pop(weight_scale2_name)

                # Calculate amax values with additional scaling factor for NVFP4
                input_amax = input_scale * 448.0 * 6.0
                weight_amax = weight_scale2 * 448.0 * 6.0

                # Handle NVFP4 tensor format
                weight_data = tensors.pop(weight_name)
                original_shape = list(weight_data.shape)
                original_shape[-1] *= 2  # NVFP4 packs 2 values per element
                nvfp4_tensor = NVFP4QTensor(
                    torch.Size(original_shape), torch.float32, weight_data
                )

                # Dequantize using both scales and block size configuration
                dequantized_weight_data = nvfp4_tensor.dequantize(
                    scale=weight_scale, double_scale=weight_scale2, block_sizes={-1: 16}
                )

                # Configure quantizer for NVFP4 format with dynamic block quantization
                quantizer_attribute_config = QuantizerAttributeConfig(
                    num_bits=(2, 1),
                    axis=None,
                    block_sizes={-1: 16, "type": "dynamic", "scale_bits": (4, 3)},
                    enable=True,
                )

            # Restore the weight to its original full-precision format so that QDQ nodes
            # can be properly inserted and optimized during TensorRT compilation
            module.weight.data = dequantized_weight_data

            # Create the quantized linear layer with calculated amax values
            quantized_module = TensorRTQuantizedLinear(
                module, input_amax, weight_amax, quantizer_attribute_config
            )

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

    # Log any unused tensors for debugging
    if len(tensors) > 0:
        logger.debug(f"{len(tensors)} tensors not used")
        for key in tensors:
            logger.debug(f"    {key}")
    return model
