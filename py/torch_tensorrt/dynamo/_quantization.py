import logging
from typing import Any, Callable

import torch

logger = logging.getLogger(__name__)


def quantize(
    model: torch.nn.Module,
    quant_format: str,
    calibrate_loop: Callable[[], Any],
    debug: bool = False,
) -> torch.nn.Module:
    try:
        import modelopt.torch.quantization as mtq

        assert torch.ops.tensorrt.quantize_op.default
    except Exception:
        logger.warning(
            "Unable to import quantization op. Please install modelopt library"
        )

    if quant_format == "fp8":
        quant_cfg = mtq.FP8_DEFAULT_CFG
    elif quant_format == "nvfp4":
        quant_cfg = mtq.NVFP4_DEFAULT_CFG
    else:
        raise RuntimeError("Unsupported quantization format")

    quantized_model = mtq.quantize(model, quant_cfg, forward_loop=calibrate_loop)
    if debug:
        mtq.print_quant_summary(quantized_model)

    return quantized_model
