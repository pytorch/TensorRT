import json
import logging
import os

import huggingface_hub
import torch
from huggingface_hub import snapshot_download

logger = logging.getLogger(__name__)

try:
    import modelopt.torch.quantization as mtq  # noqa: F401f

    assert torch.ops.tensorrt.quantize_op.default
except Exception:
    logger.warning("Unable to import quantization op. Please install modelopt library")

from modelopt.torch.quantization.config import QuantizerAttributeConfig
from modelopt.torch.quantization.nn.modules.tensor_quantizer import TensorQuantizer
from modelopt.torch.quantization.qtensor.nvfp4_tensor import NVFP4QTensor
from modelopt.torch.utils.dataset_utils import (
    create_forward_loop,
    get_dataset_dataloader,
)
from safetensors import safe_open

# FP8 E4M3 format has a maximum representable value of 448.0
MAX_BOUND_FP8 = 448.0
# Additional scaling factor for NVFP4
MAX_BOUND_NVFP4 = 6.0
# INT8 has a maximum representable value of 127.0
MAX_BOUND_INT8 = 127.0

# currently there is no pre-quantized int8 modelopt quantize models in huggingface hub
# so the exact string format in the hf_quant_config.json is not defined
INT8_WEIGHT_ONLY = "int8_weight_only"
INT8_SMOOTHQUANT = "smoothquant"
FP8 = "FP8"
NVFP4 = "NVFP4"
supported_modelopt_quant_algo_strs = [INT8_WEIGHT_ONLY, INT8_SMOOTHQUANT, FP8, NVFP4]


def dequantize_int8(quantized_data, scale, dtype):
    """
    Dequantize INT8 data to floating point.

    Args:
        quantized_data: INT8 quantized data
        scale: Scale factor (already divided by MAX_BOUND_INT8, i.e., amax / 127.0)
              Can be scalar, 1D tensor for per-channel, or 2D tensor
        dtype: Target dtype for dequantized weight (torch.float16, torch.bfloat16, etc.)
        
    Returns:
        Dequantized tensor in the specified dtype
    """
    # Scale is already the actual scale factor (amax / 127.0), use it directly
    if isinstance(scale, torch.Tensor):
        if scale.numel() == 1:
            # Per-tensor quantization
            scale_value = scale.item()
            dequantized_weight = quantized_data.float() * scale_value
        else:
            # Per-channel quantization
            if scale.dim() == 1:
                scale = scale.view(-1, 1)
            dequantized_weight = quantized_data.float() * scale
    else:
        # Scalar scale
        scale_value = float(scale)
        dequantized_weight = quantized_data.float() * scale_value
    
    return dequantized_weight.to(dtype).requires_grad_(False)


def quantize_model(model, args, tokenizer):
    """
    Quantize a PyTorch model using ModelOpt post-training quantization (PTQ).

    This function applies quantization to reduce model precision for faster inference
    while maintaining acceptable accuracy. It uses calibration data generated from
    the provided tokenizer to determine optimal quantization parameters.

    Supported quantization formats:
        - int8: INT8 quantization  
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

    if args.quant_format == "int8":
        if args.quant_algo == "smoothquant":
            if args.weight_only:
                raise RuntimeError("SmoothQuant is supported for weight-and-activation quantization, weight-only flag should not be set")
            quant_cfg = mtq.INT8_SMOOTHQUANT_CFG
        elif args.weight_only:
            quant_cfg = mtq.INT8_WEIGHT_ONLY_CFG
        else:
            raise RuntimeError(f"Unsupported args.quant_algo: {args.quant_algo} and args.weight_only: {args.weight_only} for int8 quantization")
    elif args.quant_format == "fp8":
        quant_cfg = mtq.FP8_DEFAULT_CFG
    elif args.quant_format == "nvfp4":
        quant_cfg = mtq.NVFP4_DEFAULT_CFG
    else:
        raise RuntimeError("Unsupported quantization format")
    calibrate_loop = create_forward_loop(dataloader=calib_dataloader)

    model = mtq.quantize(model, quant_cfg, forward_loop=calibrate_loop)
    if args.debug:
        mtq.print_quant_summary(model)

    return model


class TensorRTQuantizedLinear(torch.nn.Module):
    """
    TensorRT quantized linear layer that applies quantization to both input and weight tensors.
    """

    def __init__(
        self, original_linear: torch.nn.Linear, input_amax, weight_amax, quant_cfg, input_pre_quant_scale=None
    ):
        """
        Initialize quantized linear layer.

        Args:
            original_linear: Original PyTorch linear layer to quantize
            input_amax: Maximum absolute value for input quantization scaling
            weight_amax: Maximum absolute value for weight quantization scaling
            quant_cfg: Quantization configuration for TensorQuantizer
            input_pre_quant_scale: Optional per-channel pre-quantization scale (for SmoothQuant)
        """
        super().__init__()

        # Store reference to original linear layer for weight access
        self.original_linear = original_linear
        if input_amax is not None:
            # Create quantizers for input and weight tensors
            self.input_quantizer = TensorQuantizer(
                quant_attribute_cfg=quant_cfg, amax=input_amax
            )
            # Set pre_quant_scale if provided (for SmoothQuant per-channel smoothing)
            if input_pre_quant_scale is not None:
                self.input_quantizer.pre_quant_scale = input_pre_quant_scale
                self.input_quantizer.axis = None
        else:
            self.input_quantizer = TensorQuantizer(
                quant_attribute_cfg=QuantizerAttributeConfig(disable=True)
            )
        if weight_amax is None:
            raise RuntimeError("Weight amax is required")
        self.weight_quantizer = TensorQuantizer(
            quant_attribute_cfg=quant_cfg, amax=weight_amax
        )

    def forward(self, input):
        input = self.input_quantizer(input)    
        weight = self.weight_quantizer(self.original_linear.weight)
        return torch.nn.functional.linear(input, weight, self.original_linear.bias)


def load_int8_prequantized_model(model_path, model_precision, hf_quant_config):
    """
    Load a int8 pre-quantized model with int8 weights.
    
    This function handles loading models that have int8 quantized weights,
    which can't be loaded directly by from_pretrained. It also converts
    linear layers to TensorRT quantized versions.
    
    Args:
        model_path: Path to the model directory
        model_precision: Model precision (FP16, BF16, FP32)
        hf_quant_config: Quantization configuration dict
    Returns:
        Model with quantized linear layers applied, ready for inference
        
    Raises:
        RuntimeError: If quant_format is specified (pre-quantized models can't be re-quantized)
    """
    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoModelForCausalLM
    from safetensors import safe_open
    
    # Load config
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    
    # Create model structure with empty weights
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(
            config,
            trust_remote_code=True,
        )
    
    model = model.eval()
    
    # Determine weight dtype
    if model_precision == "FP16":
        weight_dtype = torch.float16
    elif model_precision == "BF16":
        weight_dtype = torch.bfloat16
    else:
        weight_dtype = torch.float32
    
    # Load all tensors from SafeTensors files
    model_path_full = hf_quant_config["model_path"]
    tensors = {}
    for file in os.listdir(model_path_full):
        if file.endswith(".safetensors"):
            with safe_open(
                os.path.join(model_path_full, file), framework="pt", device="cpu"
            ) as f:
                tensor_names = f.keys()
                for name in tensor_names:
                    tensors[name] = f.get_tensor(name)
    
    # Load weights into model, handling int8 weights by dequantizing them
    hf_quant_algo = hf_quant_config.get("quant_algo", None)
    state_dict = {}
    
    # First, collect all weight names that need to be loaded
    weight_names_to_load = set()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            weight_name = f"{name}.weight"
            weight_names_to_load.add(weight_name)
         
    # Process weights: 
    for weight_name in weight_names_to_load:
        # Check for dequantized float weight first (saved for from_pretrained compatibility)
        if weight_name in tensors and weight_name not in state_dict:
            weight = tensors[weight_name]
            if weight.dtype == torch.int8:
                name = weight_name.replace(".weight", "")
                weight_scale_name = f"{name}.weight_scale"
                
                if weight_scale_name in tensors:
                    weight_scale = tensors[weight_scale_name]
                    dequantized_weight = dequantize_int8(weight, weight_scale, weight_dtype)
                    state_dict[weight_name] = dequantized_weight.requires_grad_(False)
                else:
                    logger.warning(f"No scale found for int8 weight {weight_name}, skipping")
            else:
                # Regular float weight
                state_dict[weight_name] = weight.to(weight_dtype).requires_grad_(False)

    # Load other parameters (bias, layer norm, etc.) - skip scales and int8 weights
    for key, value in tensors.items():
        if key in state_dict:
            continue  # Already loaded
        # if key.endswith("_scale"):
        #     continue  # Skip quantization metadata
        # Load other parameters
        if value.dtype in (torch.float32, torch.float16, torch.bfloat16):
            logger.info(f"Loading {key} with dtype {value.dtype} to state_dict")
            state_dict[key] = value.to(weight_dtype).requires_grad_(False)
        else:
            # For non-float tensors, ensure requires_grad is False
            if isinstance(value, torch.Tensor):
                state_dict[key] = value.requires_grad_(False)
            else:
                state_dict[key] = value
    
    # Load state dict into model
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys:
        logger.warning(f"Missing keys when loading int8 pre-quantized model: {missing_keys[:5]}...")
    if unexpected_keys:
        # it is expected, because the model is not expecting to have the scale in the state_dict keys
        logger.warning(f"Unexpected keys when loading int8 pre-quantized model: {unexpected_keys[:5]}...")
    
    # Move model from meta device to CUDA
    # Use to_empty() to properly handle the transition from meta tensors
    model = model.to_empty(device="cuda")
    # Reload state dict to CUDA device
    model.load_state_dict(state_dict, strict=False)
    

    # Print confirmation message
    print(
        f"Model is {hf_quant_config['quant_algo']} pre-quantized hf model. Quantized linear layers are applied"
    )
    
    return model


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


def convert_linear_to_tensorrt_quantized(model, model_precision, hf_quant_config):
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
    if hf_quant_algo not in supported_modelopt_quant_algo_strs:
        raise RuntimeError(f"{hf_quant_algo} is not supported, supported algorithms are: {supported_modelopt_quant_algo_strs}")

    if model_precision == "FP16":
        weight_dtype = torch.float16
    elif model_precision == "BF16":
        weight_dtype = torch.bfloat16
    else:
        weight_dtype = torch.float32

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
            if input_scale_name not in tensors and hf_quant_algo in [FP8, NVFP4, INT8_SMOOTHQUANT]:
                logger.warning(f"Input scale tensor {input_scale_name} not found")
                continue

            # Initialize input_pre_quant_scale (only used for SmoothQuant)
            input_pre_quant_scale = None

            if hf_quant_algo == FP8:
                # Scale the quantization parameters accordingly
                weight_scale = tensors.pop(weight_scale_name)
                weight_amax = weight_scale * MAX_BOUND_FP8
                input_amax = tensors.pop(input_scale_name) * MAX_BOUND_FP8

                # Dequantize the weight using the scale factor
                dequantized_weight_data = module.weight.to(weight_dtype) * weight_scale

                # Configure quantizer for FP8 format (4 exponent bits, 3 mantissa bits)
                quantizer_attribute_config = QuantizerAttributeConfig(
                    num_bits=(4, 3), axis=None
                )

            elif hf_quant_algo == NVFP4:
                # NVFP4 format requires additional scale tensor and different configuration
                weight_name = name + ".weight"
                weight_scale2_name = name + ".weight_scale_2"
                weight_scale = tensors.pop(weight_scale_name)
                input_scale = tensors.pop(input_scale_name)
                weight_scale2 = tensors.pop(weight_scale2_name)

                # Calculate amax values with additional scaling factor for NVFP4
                input_amax = input_scale * MAX_BOUND_FP8 * MAX_BOUND_NVFP4
                weight_amax = weight_scale2 * MAX_BOUND_FP8 * MAX_BOUND_NVFP4

                # Handle NVFP4 tensor format
                weight_data = tensors.pop(weight_name)
                original_shape = list(weight_data.shape)
                original_shape[-1] *= 2  # NVFP4 packs 2 values per element
                nvfp4_tensor = NVFP4QTensor(
                    torch.Size(original_shape), weight_dtype, weight_data
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
            elif hf_quant_algo == INT8_SMOOTHQUANT:
                # Get the device of the module to ensure amax values are on the same device
                module_device = next(module.parameters()).device
                
                weight_scale = tensors.pop(weight_scale_name)
                input_scale = tensors.pop(input_scale_name)
                
                # Convert weight_scale to weight_amax and move to device
                if isinstance(weight_scale, torch.Tensor):
                    weight_amax = (weight_scale * MAX_BOUND_INT8).to(module_device)
                else:
                    weight_amax = torch.tensor(weight_scale * MAX_BOUND_INT8, device=module_device, dtype=torch.float32)
                
                # Convert input_scale to input_amax
                # Input/activation quantization should be per-tensor for SmoothQuant
                if isinstance(input_scale, torch.Tensor):
                    if input_scale.numel() > 1:
                        logger.warning(
                            f"Input scale for {name} is per-channel (shape={input_scale.shape}), "
                            "but SmoothQuant activations should be per-tensor. Using max value."
                        )
                        input_scale = input_scale.max()
                    input_amax = (input_scale * MAX_BOUND_INT8).to(module_device)
                else:
                    # Scalar value, convert to tensor
                    input_amax = torch.tensor(input_scale * MAX_BOUND_INT8, device=module_device, dtype=torch.float32)
                
                # Load pre_quant_scale if it exists (SmoothQuant per-channel smoothing factor)
                input_pre_quant_scale_name = name + ".input_pre_quant_scale"
                if input_pre_quant_scale_name in tensors:
                    input_pre_quant_scale = tensors.pop(input_pre_quant_scale_name).to(module_device)
                    logger.debug(f"Loaded pre_quant_scale for {name}: shape={input_pre_quant_scale.shape}")
                
                # int8 pre-quantized model has already been dequantized during the first load, so we don't need to dequantize again
                dequantized_weight_data = module.weight.to(weight_dtype)
                
                # Determine quantization axis based on scale shape
                # For INT8 SmoothQuant:
                # - Weight quantization can be per-channel (axis=0) or per-tensor (axis=None)
                # - Input/activation quantization should be per-tensor (axis=None)
                # Note: TensorRTQuantizedLinear uses the same config for both input and weight quantizers.
                # If weights are per-channel, we set axis=0, which will also apply to inputs (not ideal but
                # ModelOpt's TensorQuantizer should handle per-tensor amax correctly even with axis=0 config).
                if isinstance(weight_scale, torch.Tensor) and weight_scale.numel() > 1:
                    # Per-channel weight quantization (axis=0 for output channels)
                    # Verify it matches the output channel dimension
                    if weight_scale.shape[0] == module.weight.shape[0]:
                        weight_axis = 0
                    else:
                        logger.warning(
                            f"Weight scale shape {weight_scale.shape} doesn't match weight output channels "
                            f"{module.weight.shape[0]} for {name}. Using per-tensor quantization."
                        )
                        weight_axis = None
                else:
                    # Per-tensor weight quantization
                    weight_axis = None
                
                quantizer_attribute_config = QuantizerAttributeConfig(
                    num_bits=8, axis=weight_axis
                )

            # Restore the weight to its original full-precision format so that QDQ nodes
            # can be properly inserted and optimized during TensorRT compilation
            module.weight.data = dequantized_weight_data

            # Create the quantized linear layer with calculated amax values
            # Pass pre_quant_scale for SmoothQuant if available
            quantized_module = TensorRTQuantizedLinear(
                module, input_amax, weight_amax, quantizer_attribute_config,
                input_pre_quant_scale=input_pre_quant_scale if hf_quant_algo == INT8_SMOOTHQUANT else None
            )
            breakpoint()

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
    
    # Ensure model is on CUDA (it should already be from load_int8_prequantized_model)
    # Only move if not already on CUDA
    if next(model.parameters()).device.type != "cuda":
        model = model.to("cuda")
    
    return model