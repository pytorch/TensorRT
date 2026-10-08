"""Nemotron MoE mixer wrapper and ``torch.ops.trt`` NVFP4 MoE custom ops.

Same pattern as ``attention.py``: eager stub + fake for Dynamo, converter
inserts Edge-LLM ``Nvfp4MoePlugin`` / ``NvFP4MoEPluginGeforce``.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

_NVFP4_ACTIVATION_RELU2 = 4
_NVFP4_ROUTING_MODE_SIGMOID_GROUP_TOPK = 1
_NVFP4_MOE_BACKEND_AUTO = 0
_NVFP4_MOE_IO_DTYPE_FP16 = 1
_NVFP4_MOE_MAX_ROUTED_ROWS_AUTO = 0

_NVFP4_MOE_TARGET_ENV = "EDGELLM_NVFP4_MOE_TARGET"
_NVFP4_MOE_SM12X_ALIASES = frozenset(("sm12x", "sm120", "sm121", "geforce"))
_NVFP4_MOE_SM110_ALIASES = frozenset(
    ("sm100", "sm101", "sm110", "blackwell_dc", "thor", "")
)


def _has_torch_op(namespace: str, name: str) -> bool:
    return hasattr(torch.ops, namespace) and hasattr(
        getattr(torch.ops, namespace), name
    )


def use_geforce_nvfp4_moe() -> bool:
    """True when exporting ``NvFP4MoEPluginGeforce`` (SM12x). Default is SM110."""
    val = os.environ.get(_NVFP4_MOE_TARGET_ENV, "sm110").strip().lower()
    if val in _NVFP4_MOE_SM12X_ALIASES:
        return True
    if val in _NVFP4_MOE_SM110_ALIASES:
        return False
    raise ValueError(
        f"{_NVFP4_MOE_TARGET_ENV}={val!r} is not recognized. "
        "Use sm100/sm110 (Nvfp4MoePlugin) or sm12x (NvFP4MoEPluginGeforce)."
    )


def _nvfp4_moe_stub(
    router_logits: torch.Tensor,
    hidden_states: torch.Tensor,
    fc1_qweights: torch.Tensor,
    fc1_blocks_scale: torch.Tensor,
    fc1_alpha: torch.Tensor,
    fc2_qweights: torch.Tensor,
    fc2_blocks_scale: torch.Tensor,
    fc2_alpha: torch.Tensor,
    input_global_scale: torch.Tensor,
    down_input_scale: torch.Tensor,
    e_score_correction_bias: torch.Tensor,
    num_experts: int,
    top_k: int,
    hidden_size: int,
    moe_inter_size: int,
    activation_type: int,
    n_group: int,
    topk_group: int,
    norm_topk_prob: int,
    routed_scaling_factor: float,
    routing_mode: int,
    backend: int,
    io_dtype: int,
    max_routed_rows: int,
) -> torch.Tensor:
    del router_logits, fc1_qweights, fc1_blocks_scale, fc1_alpha
    del fc2_qweights, fc2_blocks_scale, fc2_alpha
    del input_global_scale, down_input_scale, e_score_correction_bias
    del num_experts, top_k, hidden_size, moe_inter_size, activation_type
    del n_group, topk_group, norm_topk_prob, routed_scaling_factor
    del routing_mode, backend, io_dtype, max_routed_rows
    return torch.zeros_like(hidden_states)


def register_moe_plugin_ops() -> None:
    """Register ``trt::nvfp4_moe_plugin`` and ``trt::nvfp4_moe_plugin_geforce``."""
    if _has_torch_op("trt", "nvfp4_moe_plugin"):
        return

    @torch.library.custom_op("trt::nvfp4_moe_plugin", mutates_args=())
    def nvfp4_moe_plugin(
        router_logits: torch.Tensor,
        hidden_states: torch.Tensor,
        fc1_qweights: torch.Tensor,
        fc1_blocks_scale: torch.Tensor,
        fc1_alpha: torch.Tensor,
        fc2_qweights: torch.Tensor,
        fc2_blocks_scale: torch.Tensor,
        fc2_alpha: torch.Tensor,
        input_global_scale: torch.Tensor,
        down_input_scale: torch.Tensor,
        e_score_correction_bias: torch.Tensor,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        moe_inter_size: int,
        activation_type: int,
        n_group: int,
        topk_group: int,
        norm_topk_prob: int,
        routed_scaling_factor: float,
        routing_mode: int,
        backend: int,
        io_dtype: int,
        max_routed_rows: int,
    ) -> torch.Tensor:
        return _nvfp4_moe_stub(
            router_logits,
            hidden_states,
            fc1_qweights,
            fc1_blocks_scale,
            fc1_alpha,
            fc2_qweights,
            fc2_blocks_scale,
            fc2_alpha,
            input_global_scale,
            down_input_scale,
            e_score_correction_bias,
            num_experts,
            top_k,
            hidden_size,
            moe_inter_size,
            activation_type,
            n_group,
            topk_group,
            norm_topk_prob,
            routed_scaling_factor,
            routing_mode,
            backend,
            io_dtype,
            max_routed_rows,
        )

    @nvfp4_moe_plugin.register_fake
    def _(
        router_logits,
        hidden_states,
        fc1_qweights,
        fc1_blocks_scale,
        fc1_alpha,
        fc2_qweights,
        fc2_blocks_scale,
        fc2_alpha,
        input_global_scale,
        down_input_scale,
        e_score_correction_bias,
        num_experts,
        top_k,
        hidden_size,
        moe_inter_size,
        activation_type,
        n_group,
        topk_group,
        norm_topk_prob,
        routed_scaling_factor,
        routing_mode,
        backend,
        io_dtype,
        max_routed_rows,
    ):
        del router_logits, fc1_qweights, fc1_blocks_scale, fc1_alpha
        del fc2_qweights, fc2_blocks_scale, fc2_alpha
        del input_global_scale, down_input_scale, e_score_correction_bias
        del num_experts, top_k, hidden_size, moe_inter_size, activation_type
        del n_group, topk_group, norm_topk_prob, routed_scaling_factor
        del routing_mode, backend, io_dtype, max_routed_rows
        return torch.empty_like(hidden_states)

    @torch.library.custom_op("trt::nvfp4_moe_plugin_geforce", mutates_args=())
    def nvfp4_moe_plugin_geforce(
        router_logits: torch.Tensor,
        hidden_states: torch.Tensor,
        fc1_qweights: torch.Tensor,
        fc1_blocks_scale: torch.Tensor,
        fc1_alpha: torch.Tensor,
        fc2_qweights: torch.Tensor,
        fc2_blocks_scale: torch.Tensor,
        fc2_alpha: torch.Tensor,
        input_global_scale: torch.Tensor,
        down_input_scale: torch.Tensor,
        e_score_correction_bias: torch.Tensor,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        moe_inter_size: int,
        activation_type: int,
        n_group: int,
        topk_group: int,
        norm_topk_prob: int,
        routed_scaling_factor: float,
        routing_mode: int,
        backend: int,
        io_dtype: int,
        max_routed_rows: int,
    ) -> torch.Tensor:
        return _nvfp4_moe_stub(
            router_logits,
            hidden_states,
            fc1_qweights,
            fc1_blocks_scale,
            fc1_alpha,
            fc2_qweights,
            fc2_blocks_scale,
            fc2_alpha,
            input_global_scale,
            down_input_scale,
            e_score_correction_bias,
            num_experts,
            top_k,
            hidden_size,
            moe_inter_size,
            activation_type,
            n_group,
            topk_group,
            norm_topk_prob,
            routed_scaling_factor,
            routing_mode,
            backend,
            io_dtype,
            max_routed_rows,
        )

    @nvfp4_moe_plugin_geforce.register_fake
    def _(
        router_logits,
        hidden_states,
        fc1_qweights,
        fc1_blocks_scale,
        fc1_alpha,
        fc2_qweights,
        fc2_blocks_scale,
        fc2_alpha,
        input_global_scale,
        down_input_scale,
        e_score_correction_bias,
        num_experts,
        top_k,
        hidden_size,
        moe_inter_size,
        activation_type,
        n_group,
        topk_group,
        norm_topk_prob,
        routed_scaling_factor,
        routing_mode,
        backend,
        io_dtype,
        max_routed_rows,
    ):
        del router_logits, fc1_qweights, fc1_blocks_scale, fc1_alpha
        del fc2_qweights, fc2_blocks_scale, fc2_alpha
        del input_global_scale, down_input_scale, e_score_correction_bias
        del num_experts, top_k, hidden_size, moe_inter_size, activation_type
        del n_group, topk_group, norm_topk_prob, routed_scaling_factor
        del routing_mode, backend, io_dtype, max_routed_rows
        return torch.empty_like(hidden_states)


class PluginNemotronMoE(nn.Module):
    """Wrap ``NemotronHMoE``: native router + shared expert, plugin routed experts.

    Requires packed NVFP4 buffers (``prepare_for_export`` or an already-packed
    NVFP4 checkpoint). ReLU2 + sigmoid-group top-k, matching Nemotron-3-30B-A3B.
    """

    def __init__(self, original: nn.Module, config):
        super().__init__()
        self.gate = original.gate
        self.shared_experts = original.shared_experts
        self.fc1_latent_proj = getattr(original, "fc1_latent_proj", nn.Identity())
        self.fc2_latent_proj = getattr(original, "fc2_latent_proj", nn.Identity())
        self._hf_experts = original.experts

        self.n_routed_experts = int(config.n_routed_experts)
        self.num_experts_per_tok = int(config.num_experts_per_tok)
        self.hidden_size = int(config.hidden_size)
        self.routed_hidden_size = int(
            getattr(config, "moe_latent_size", None) or config.hidden_size
        )
        self.moe_intermediate_size = int(config.moe_intermediate_size)
        self.group_size = int(
            getattr(getattr(config, "quant", None), "group_size", 16) or 16
        )

        self.n_group = int(
            getattr(self.gate, "n_group", getattr(self.gate, "num_group", 1))
        )
        self.topk_group = int(self.gate.topk_group)
        self.norm_topk_prob = int(bool(self.gate.norm_topk_prob))
        self.routed_scaling_factor = float(self.gate.routed_scaling_factor)

        self._padded_hidden_size = self.routed_hidden_size
        self._padded_moe_intermediate_size = self.moe_intermediate_size
        self._export_ready = False

        if hasattr(original, "fc1_qweights"):
            self.fc1_qweights = original.fc1_qweights
            self.fc1_blocks_scale = original.fc1_blocks_scale
            self.fc1_alpha = original.fc1_alpha
            self.fc2_qweights = original.fc2_qweights
            self.fc2_blocks_scale = original.fc2_blocks_scale
            self.fc2_alpha = original.fc2_alpha
            self.input_global_scale = original.input_global_scale
            self.down_input_scale = original.down_input_scale
            self._e_score_correction_bias_fp32 = original._e_score_correction_bias_fp32
            self._padded_hidden_size = int(original._padded_hidden_size)
            self._padded_moe_intermediate_size = int(
                original._padded_moe_intermediate_size
            )
            self._export_ready = True

    def prepare_for_export(self) -> None:
        from tensorrt_edgellm.checkpoint.repacking import repack_nvfp4_moe_experts

        experts = self._hf_experts
        if not isinstance(experts, nn.ModuleList):
            raise TypeError(
                "HF NemotronHExperts stores 3D fp16 tensors, not per-expert "
                "NVFP4 Linears. Load NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4 "
                "(or run ModelOpt NVFP4 quant) before the MoE plugin path."
            )

        hidden_align = 256 if use_geforce_nvfp4_moe() else 1
        (
            fc1_q,
            fc1_scale,
            fc1_alpha,
            fc2_q,
            fc2_scale,
            fc2_alpha,
            padded_inter,
            padded_h,
        ) = repack_nvfp4_moe_experts(
            experts,
            self.routed_hidden_size,
            self.moe_intermediate_size,
            self.group_size,
            hidden_size_alignment=hidden_align,
        )
        device = self.gate.weight.device
        self.register_buffer("fc1_qweights", fc1_q.to(device).contiguous())
        self.register_buffer("fc1_blocks_scale", fc1_scale.to(device).contiguous())
        self.register_buffer("fc1_alpha", fc1_alpha.to(device).contiguous())
        self.register_buffer("fc2_qweights", fc2_q.to(device).contiguous())
        self.register_buffer("fc2_blocks_scale", fc2_scale.to(device).contiguous())
        self.register_buffer("fc2_alpha", fc2_alpha.to(device).contiguous())
        self.register_buffer(
            "input_global_scale",
            torch.ones(self.n_routed_experts, dtype=torch.float32, device=device),
        )
        self.register_buffer(
            "down_input_scale",
            torch.ones(self.n_routed_experts, dtype=torch.float32, device=device),
        )
        self.register_buffer(
            "_e_score_correction_bias_fp32",
            self.gate.e_score_correction_bias.data.to(torch.float32).to(device),
        )
        self._padded_moe_intermediate_size = padded_inter
        self._padded_hidden_size = padded_h
        self._export_ready = True

    def _shared_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        h = self.shared_experts.up_proj(hidden_states)
        r = F.relu(h)
        return self.shared_experts.down_proj(r * r)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if not self._export_ready:
            raise RuntimeError("PluginNemotronMoE.prepare_for_export() was not called")

        router_logits = F.linear(
            hidden_states.view(-1, self.hidden_size), self.gate.weight
        ).float()
        routed = self.fc1_latent_proj(hidden_states)
        plugin_hidden = routed
        if self._padded_hidden_size != self.routed_hidden_size:
            plugin_hidden = F.pad(
                routed, (0, self._padded_hidden_size - self.routed_hidden_size)
            )

        moe_op = (
            torch.ops.trt.nvfp4_moe_plugin_geforce.default
            if use_geforce_nvfp4_moe()
            else torch.ops.trt.nvfp4_moe_plugin.default
        )
        moe_out = moe_op(
            router_logits,
            plugin_hidden,
            self.fc1_qweights,
            self.fc1_blocks_scale,
            self.fc1_alpha,
            self.fc2_qweights,
            self.fc2_blocks_scale,
            self.fc2_alpha,
            self.input_global_scale,
            self.down_input_scale,
            self._e_score_correction_bias_fp32,
            self.n_routed_experts,
            self.num_experts_per_tok,
            self._padded_hidden_size,
            self._padded_moe_intermediate_size,
            _NVFP4_ACTIVATION_RELU2,
            self.n_group,
            self.topk_group,
            self.norm_topk_prob,
            self.routed_scaling_factor,
            _NVFP4_ROUTING_MODE_SIGMOID_GROUP_TOPK,
            _NVFP4_MOE_BACKEND_AUTO,
            _NVFP4_MOE_IO_DTYPE_FP16,
            _NVFP4_MOE_MAX_ROUTED_ROWS_AUTO,
        )
        if self._padded_hidden_size != self.routed_hidden_size:
            moe_out = moe_out[..., : self.routed_hidden_size]
        moe_out = self.fc2_latent_proj(moe_out)
        return moe_out + self._shared_forward(hidden_states)
