"""Helpers for TorchAO static FP8 (activation + weight) quantization examples.

Static FP8 calibrates activation and weight ranges with observers, then rewrites
Linear layers to explicit ``quantize_affine_float8_non_decomposed`` /
``dequantize_affine_float8_non_decomposed`` so Torch-TensorRT can emit
``IQuantizeLayer`` / ``IDequantizeLayer`` pairs.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor
from torchao.core.config import AOBaseConfig
from torchao.quantization import quantize_
from torchao.quantization.granularity import PerAxis, PerTensor
from torchao.quantization.observer import AffineQuantizedMinMaxObserver
from torchao.quantization.quant_api import _replace_with_custom_fn_if_matches_filter
from torchao.quantization.quant_primitives import (
    MappingType,
    _dequantize_affine_float8_non_decomposed,
    _quantize_affine_float8_non_decomposed,
)
from torchao.quantization.transform_module import register_quantize_module_handler


class ObservedLinear(torch.nn.Linear):
    """Linear that records activation and weight ranges during calibration."""

    def __init__(
        self,
        in_features,
        out_features,
        act_obs,
        weight_obs,
        bias=True,
        device=None,
        dtype=None,
    ):
        super().__init__(in_features, out_features, bias, device, dtype)
        self.act_obs = act_obs
        self.weight_obs = weight_obs

    def forward(self, input: Tensor):
        observed_input = self.act_obs(input)
        observed_weight = self.weight_obs(self.weight)
        return F.linear(observed_input, observed_weight, self.bias)

    @classmethod
    def from_float(cls, float_linear, act_obs, weight_obs):
        observed = cls(
            float_linear.in_features,
            float_linear.out_features,
            act_obs,
            weight_obs,
            bias=float_linear.bias is not None,
            device=float_linear.weight.device,
            dtype=float_linear.weight.dtype,
        )
        observed.weight = float_linear.weight
        observed.bias = float_linear.bias
        return observed


class QuantizedLinearQDQ(torch.nn.Module):
    """Linear with explicit FP8 Q/DQ on activations and a pre-quantized FP8 weight."""

    def __init__(
        self,
        act_obs,
        weight_obs,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        target_dtype: torch.dtype,
    ):
        super().__init__()
        assert target_dtype == torch.float8_e4m3fn
        self.act_scale, _ = act_obs.calculate_qparams()
        weight_scale, _ = weight_obs.calculate_qparams()
        self.target_dtype = target_dtype
        self.bias = bias
        self.output_dtype = weight.dtype
        weight_scale_2d = (
            weight_scale.view(-1, 1) if weight_scale.dim() == 1 else weight_scale
        )
        self.register_buffer(
            "weight_fp8",
            _quantize_affine_float8_non_decomposed(
                weight, weight_scale_2d, target_dtype
            ),
        )
        self.register_buffer("weight_scale", weight_scale_2d)

    def forward(self, input: Tensor):
        input_fp8 = _quantize_affine_float8_non_decomposed(
            input, self.act_scale, self.target_dtype
        )
        input_hp = _dequantize_affine_float8_non_decomposed(
            input_fp8, self.act_scale, self.output_dtype
        )
        weight_hp = _dequantize_affine_float8_non_decomposed(
            self.weight_fp8, self.weight_scale, self.output_dtype
        )
        return F.linear(input_hp, weight_hp, self.bias)

    @classmethod
    def from_observed(cls, observed_linear, target_dtype):
        return cls(
            observed_linear.act_obs,
            observed_linear.weight_obs,
            observed_linear.weight,
            observed_linear.bias,
            target_dtype,
        )


@dataclass
class StaticQuantConfigQDQ(AOBaseConfig):
    target_dtype: torch.dtype


@register_quantize_module_handler(StaticQuantConfigQDQ)
def _apply_static_quant_qdq_transform(module, config):
    return QuantizedLinearQDQ.from_observed(module, config.target_dtype)


def insert_observers_(model, act_obs, weight_obs):
    def replacement_fn(m):
        return ObservedLinear.from_float(
            m, copy.deepcopy(act_obs), copy.deepcopy(weight_obs)
        )

    _replace_with_custom_fn_if_matches_filter(
        model,
        replacement_fn,
        lambda m, fqn: isinstance(m, torch.nn.Linear),
    )


def create_fp8_observers():
    common_kwargs = dict(
        mapping_type=MappingType.SYMMETRIC,
        target_dtype=torch.float8_e4m3fn,
        eps=torch.finfo(torch.float32).eps,
        scale_dtype=torch.float32,
        zero_point_dtype=torch.float32,
    )
    act_obs = AffineQuantizedMinMaxObserver(granularity=PerTensor(), **common_kwargs)
    weight_obs = AffineQuantizedMinMaxObserver(
        granularity=PerAxis(axis=0), **common_kwargs
    )
    return act_obs, weight_obs


def quantize_static_fp8(
    model: torch.nn.Module,
    example_inputs,
    calibration_steps: int = 10,
) -> torch.nn.Module:
    """Calibrate Linear layers and replace them with explicit FP8 Q/DQ modules."""
    act_obs, weight_obs = create_fp8_observers()
    insert_observers_(model, act_obs, weight_obs)
    with torch.no_grad():
        for _ in range(calibration_steps):
            model(*example_inputs)
    quantize_(
        model,
        StaticQuantConfigQDQ(torch.float8_e4m3fn),
        lambda m, fqn: isinstance(m, ObservedLinear),
    )
    return model
