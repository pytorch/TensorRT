from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_tensorrt.hf.exporters.models.common.patches import ActionStepEncoderPatch


class TRTFixedCategorySpecificLinearPatch(nn.Module):
    """Freeze one GR00T embodiment-specific Linear into a normal Linear.

    GR00T stores one weight matrix per robot embodiment and selects it with
    embodiment_id at runtime. For TensorRT deployment we compile one robot at a
    time, so this wrapper picks that robot's weights once in __init__ and the
    forward path becomes a plain static F.linear.
    """

    def __init__(self, layer: nn.Module, embodiment_id: torch.Tensor):
        super().__init__()

        cat_id = int(embodiment_id.flatten()[0].item())

        # Original: [num_embodiments, input_dim, output_dim]
        # using cat_id selects the weight matrix for one embodiment/robot -> [input_dim, output_dim]
        # nn.functional.linear expects -> weight: [output_dim, input_dim] so we transpose
        weight = layer.W[cat_id].transpose(0, 1).contiguous()
        bias = layer.b[cat_id].contiguous()

        # detach() breaks any autograd link to the original multi-embodiment
        # parameter; clone() gives this fixed wrapper independent storage for
        # the selected slice. This copy happens once during wrapper creation,
        # not in forward, and lets TensorRT see normal immutable weights.
        self.weight = nn.Parameter(weight.detach().clone(), requires_grad=False)
        self.bias = nn.Parameter(bias.detach().clone(), requires_grad=False)
        self.out_features = int(bias.shape[0])

    def forward(self, x):
        # x: [B, T, input_dim]
        # out: [B, T, output_dim]
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        x = x.reshape(batch_size * seq_len, x.shape[-1])
        x = F.linear(x, self.weight, self.bias)
        return x.reshape(batch_size, seq_len, self.out_features)


class TRTDynamicCategorySpecificLinearPatch(nn.Module):
    """TensorRT-friendly dynamic version of GR00T CategorySpecificLinear.

    Unlike the fixed wrapper, this keeps the full embodiment weight bank and
    uses runtime embodiment_id values to gather W/b for each batch item. The
    math stays equivalent to GR00T category-specific linear:
    x [B,T,in] @ W[embodiment] [B,in,out] + b[embodiment].
    """

    def __init__(self, layer: nn.Module):
        super().__init__()

        # Keep the full embodiment weight bank.
        # W: [num_embodiments, input_dim, output_dim]
        # b: [num_embodiments, output_dim]
        self.W = layer.W
        self.b = layer.b

    def forward(self, x, cat_ids):
        # x:       [B, T, input_dim]
        # cat_ids: [B]

        cat_ids = cat_ids.to(dtype=torch.long)

        # selected_w: [B, input_dim, output_dim]
        # selected_b: [B, output_dim]
        selected_w = torch.index_select(self.W, dim=0, index=cat_ids).to(dtype=x.dtype)
        selected_b = torch.index_select(self.b, dim=0, index=cat_ids).to(dtype=x.dtype)

        # out: [B, T, output_dim]
        out = torch.bmm(x, selected_w)

        # bias: [B, 1, output_dim], broadcast over T
        return out + selected_b.unsqueeze(1)


class TRTDynamicCategorySpecificMLPPatch(nn.Module):
    """Dynamic two-layer category-specific MLP used by GR00T.

    GR00T state encoders and action decoders are CategorySpecificMLP modules:
    each contains layer1/layer2 CategorySpecificLinear layers. This wrapper
    preserves runtime embodiment selection for both layers.
    """

    def __init__(self, mlp: nn.Module):
        super().__init__()
        self.layer1 = TRTDynamicCategorySpecificLinearPatch(mlp.layer1)
        self.layer2 = TRTDynamicCategorySpecificLinearPatch(mlp.layer2)

    def forward(self, x, embodiment_id):
        hidden = F.relu(self.layer1(x, embodiment_id))
        return self.layer2(hidden, embodiment_id)


class TRTGrootActionEncoderPatch(nn.Module):
    def __init__(self, action_encoder: nn.Module, embodiment_id: torch.Tensor):
        super().__init__()
        self.W1 = TRTFixedCategorySpecificLinearPatch(action_encoder.W1, embodiment_id)
        self.W2 = TRTFixedCategorySpecificLinearPatch(action_encoder.W2, embodiment_id)
        self.W3 = TRTFixedCategorySpecificLinearPatch(action_encoder.W3, embodiment_id)
        self.pos_encoding = action_encoder.pos_encoding

    def forward(self, actions, timesteps, embodiment_id):
        batch_size, action_horizon, _ = actions.shape

        if timesteps.dim() == 1 and timesteps.shape[0] == batch_size:
            timesteps = timesteps.unsqueeze(1).expand(-1, action_horizon)
        else:
            raise ValueError("Expected `timesteps` to have shape (B,).")

        action_emb = self.W1(actions)
        timestep_emb = self.pos_encoding(timesteps).to(dtype=action_emb.dtype)
        hidden = torch.cat([action_emb, timestep_emb], dim=-1)
        hidden = F.silu(self.W2(hidden))
        return self.W3(hidden)


class TRTDynamicGrootActionEncoderPatch(nn.Module):
    """Dynamic GR00T noisy-action encoder.

    The original action encoder uses three embodiment-specific linear layers
    around the action embedding, timestep positional embedding, and SiLU block.
    This wrapper keeps embodiment_id dynamic while spelling the category-specific
    pieces as index_select + bmm so Torch-TRT can lower them reliably.
    """

    def __init__(self, action_encoder: nn.Module):
        super().__init__()
        self.W1 = TRTDynamicCategorySpecificLinearPatch(action_encoder.W1)
        self.W2 = TRTDynamicCategorySpecificLinearPatch(action_encoder.W2)
        self.W3 = TRTDynamicCategorySpecificLinearPatch(action_encoder.W3)
        self.pos_encoding = action_encoder.pos_encoding

    def forward(self, actions, timesteps, embodiment_id):
        batch_size, action_horizon, _ = actions.shape

        timesteps = timesteps.unsqueeze(1).expand(-1, action_horizon)

        action_emb = self.W1(actions, embodiment_id)
        timestep_emb = self.pos_encoding(timesteps).to(dtype=action_emb.dtype)

        hidden = torch.cat([action_emb, timestep_emb], dim=-1)
        hidden = F.silu(self.W2(hidden, embodiment_id))
        return self.W3(hidden, embodiment_id)


class GrootDiTStepEncoderPatch(ActionStepEncoderPatch):
    def __init__(self, action_head, embodiment_id: torch.Tensor | None = None):
        super().__init__()
        if embodiment_id is None:
            self.state_encoder = action_head.state_encoder
            self.action_encoder = action_head.action_encoder
        else:
            # Keep embodiment_id as a runtime input while replacing GR00T's
            # category-specific modules with Torch-TRT-friendly dynamic wrappers.
            self.state_encoder = TRTDynamicCategorySpecificMLPPatch(
                action_head.state_encoder
            )
            self.action_encoder = TRTDynamicGrootActionEncoderPatch(
                action_head.action_encoder
            )
        self.future_tokens = action_head.future_tokens
        self.position_embedding = getattr(action_head, "position_embedding", None)
        self.add_pos_embed = action_head.config.add_pos_embed

    def forward(self, actions, timestep, vl_embs, state, embodiment_id):
        state_features = self.state_encoder(state, embodiment_id)
        action_features = self.action_encoder(actions, timestep, embodiment_id)

        if self.add_pos_embed:
            pos_ids = torch.arange(
                action_features.shape[1],
                dtype=torch.long,
                device=action_features.device,
            )
            action_features = action_features + self.position_embedding(
                pos_ids
            ).unsqueeze(0)

        future_tokens = self.future_tokens.weight.unsqueeze(0).expand(
            vl_embs.shape[0],
            -1,
            -1,
        )

        sa_embs = torch.cat(
            (state_features, future_tokens, action_features),
            dim=1,
        )

        expert_args = ()
        expert_kwargs = {
            "hidden_states": sa_embs,
            "encoder_hidden_states": vl_embs,
            "timestep": timestep,
        }

        decoder_args = (embodiment_id,)
        decoder_kwargs = {}

        return expert_args, expert_kwargs, decoder_args, decoder_kwargs


class ContextProjectionPatch(nn.Module):
    """eagle_linear -> vlln -> vl_self_attention (matches eager context path)."""

    def __init__(self, eagle_linear, vlln, vl_self_attention):
        super().__init__()
        self.eagle_linear = eagle_linear
        self.vlln = vlln
        self.vl_self_attention = vl_self_attention

    def forward(self, hidden_states: torch.Tensor):
        context_embs = self.eagle_linear(hidden_states)

        vlln_weight = getattr(self.vlln, "weight", None)
        if vlln_weight is not None:
            context_embs = context_embs.to(dtype=vlln_weight.dtype)

        context_embs = self.vlln(context_embs)
        context_embs = self.vl_self_attention(context_embs)
        return context_embs
