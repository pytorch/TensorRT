import torch
import torch.nn as nn
import torch_tensorrt
import copy, requests
from PIL import Image
from typing import Optional

from transformers import AutoModel, AutoProcessor
from transformers.models.siglip import modeling_siglip as ms
from transformers.models.qwen2 import modeling_qwen2 as mq

# -----------------------------------------------------------------------------
# 0)  Export-friendly attention patches
# -----------------------------------------------------------------------------

def _patch_siglip_attention():
    """Swap SiglipAttention.forward with an sdpa-only version no Flash-Attn kernels."""

    def patched_attention_forward(self, hidden_states, attention_mask=None, output_attentions=False):
        B, S, _ = hidden_states.shape
        # 1. projections
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        # 2. reshape (B, nH, S, dH)
        q = q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        # 3. scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if attention_mask is not None:
            scores = scores + attention_mask  # additive mask
        probs = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
        if self.training and self.dropout > 0:
            probs = torch.nn.functional.dropout(probs, p=self.dropout, training=True)
        ctx = torch.matmul(probs, v)
        ctx = ctx.transpose(1, 2).contiguous().view(B, S, self.embed_dim)
        ctx = self.out_proj(ctx)
        return (ctx, None) if output_attentions else (ctx, None)

    ms.SiglipAttention.forward = patched_attention_forward


# -----------------------------------------------------------------------------
# Qwen2 export-only attention patch (manual SDPA) – ENABLED again
# -----------------------------------------------------------------------------

def _patch_qwen2_attention():
    """Swap Qwen2Attention.forward with simplified SDPA version (export-friendly)."""

    def patched_qwen2_attention_forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value=None,
        cache_position=None,
        **kwargs,
    ):
        B, S, _ = hidden_states.shape

        # 1. linear projections
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # 2. reshape to (B, nH, S, dH)
        q = q.reshape(B, S, self.config.num_attention_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(B, S, self.config.num_key_value_heads, self.head_dim)
        k = k.repeat_interleave(self.num_key_value_groups, dim=2).transpose(1, 2)
        v = v.reshape(B, S, self.config.num_key_value_heads, self.head_dim)
        v = v.repeat_interleave(self.num_key_value_groups, dim=2).transpose(1, 2)

        # 3. rotary positional embedding
        cos, sin = position_embeddings
        q, k = mq.apply_rotary_pos_emb(q, k, cos, sin)

        # 4. scaled dot-product
        attn = (q @ k.transpose(-2, -1)) * self.scaling
        if attention_mask is not None:
            mask = (1.0 - attention_mask[:, None, None, :]).to(q.dtype) * (-65504.0)
            attn = attn + mask

        # 5. softmax fp32 → fp16
        attn = torch.softmax(attn, dim=-1, dtype=torch.float32).to(q.dtype)
        if self.training and self.attention_dropout > 0:
            attn = torch.nn.functional.dropout(attn, p=self.attention_dropout, training=True)

        # 6. value aggregation & projection
        ctx = attn @ v  # (B, nH, S, dH)
        ctx = ctx.transpose(1, 2).contiguous().reshape(B, S, self.config.hidden_size)
        ctx = self.o_proj(ctx)

        return ctx, None

    mq.Qwen2Attention.forward = patched_qwen2_attention_forward


# -----------------------------------------------------------------------------
# 1)  Load base model & processor
# -----------------------------------------------------------------------------

def load_base(device="cuda:0"):
    _patch_siglip_attention()
    _patch_qwen2_attention()

    model_id = "nvidia/Eagle2-2B"
    model = (
        AutoModel.from_pretrained(model_id, trust_remote_code=True, torch_dtype=torch.float16)
        .to(device)
        .eval()
    )
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, use_fast=True)
    # left padded for generation
    if hasattr(processor, "tokenizer"):
        processor.tokenizer.padding_side = "left"

    return model, processor


# -----------------------------------------------------------------------------
# 2)  Torch-TensorRT compile helpers
# -----------------------------------------------------------------------------

class VisionWrapper(nn.Module):
    """Return only last_hidden_state tensor for simpler TRT signature."""

    def __init__(self, vision_model):
        super().__init__()
        self.vision_model = vision_model

    def forward(self, pixel_values):
        out = self.vision_model(pixel_values=pixel_values, output_hidden_states=False, return_dict=True)
        return out.last_hidden_state if hasattr(out, "last_hidden_state") else out


class LMNoCache(nn.Module):
    def __init__(self, lm):
        super().__init__()
        self.lm = lm

    def forward(self, inputs_embeds, attention_mask, position_ids=None, cache_position=None):
        return self.lm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache_position=cache_position,
            use_cache=False
        )


# -----------------------------------------------------------------------------
# 3)  Compile sub-modules with TRT
# -----------------------------------------------------------------------------

def compile_submodules(base_model, device="cuda:0"):
    vision_model = base_model.vision_model
    mlp1 = base_model.mlp1
    language_model = base_model.language_model

    # ---------------- Vision ----------------
    B = torch.export.Dim("batch", min=1, max=8)
    dummy_pixels = torch.randn(4, 3, 448, 448, dtype=torch.float16, device=device)  # always use torch.float16
    dyn_shapes_vis = {"pixel_values": {0: B}}

    vis_wrapper = VisionWrapper(vision_model).to(device)
    with torch.inference_mode():
        exported_vis = torch.export.export(vis_wrapper, (dummy_pixels,), dynamic_shapes=dyn_shapes_vis, strict=False)
    trt_vis = torch_tensorrt.dynamo.compile(
        exported_vis,
        inputs=[dummy_pixels],
        enabled_precisions={torch.float16},
        device=device,
        truncate_double=True,
    )

    # ---------------- MLP1 projector ----------------
    with torch.inference_mode():
        vit_out = vis_wrapper(dummy_pixels)
    h = w = int(vit_out.shape[1] ** 0.5)
    embeds = vit_out.reshape(vit_out.shape[0], h, w, -1)
    embeds = base_model.pixel_shuffle(embeds, scale_factor=base_model.downsample_ratio)
    embeds = embeds.reshape(embeds.shape[0], -1, embeds.shape[-1])

    B2 = torch.export.Dim("batch2", min=1, max=8)
    S2 = torch.export.Dim("seq2", min=1, max=2048)
    dyn_shapes_mlp = {"x": {0: B2, 1: S2}}
    class _MLPWrap(nn.Module):
        def __init__(self, mlp):
            super().__init__()
            self.mlp = mlp
        def forward(self, x):
            return self.mlp(x)
    mlp_wrap = _MLPWrap(mlp1).to(device)
    with torch.inference_mode():
        exported_mlp = torch.export.export(mlp_wrap, (embeds,), dynamic_shapes=dyn_shapes_mlp, strict=False)
    trt_mlp1 = torch_tensorrt.dynamo.compile(
        exported_mlp,
        inputs=[embeds],
        enabled_precisions={torch.float16},
        device=device,
        truncate_double=True,
    )

    # ---------------- Language model ----------------
    hidden_size = language_model.config.hidden_size
    dummy_seq = 13  # 8*k -3 pattern (k=2)
    dummy_batch = 2
    dummy_embeds = torch.randn(dummy_batch, dummy_seq, hidden_size, dtype=torch.float16, device=device)
    # 2D padding mask (int64) – Qwen2 expects (B, S)
    dummy_mask = torch.ones(dummy_batch, dummy_seq, dtype=torch.float16, device=device)
    # Create dummy position_ids and cache_position
    dummy_position_ids = torch.arange(1, dummy_seq + 1, device=device).unsqueeze(0).expand(dummy_batch, -1)
    dummy_cache_position = torch.arange(dummy_seq, device=device)

    B3 = torch.export.Dim("batch3", min=1, max=4)
    K = torch.export.Dim("_k", min=1, max=512)
    seq_sym = 8 * K - 3
    # attention_mask now 4-D : (B, 1, S, S)
    dyn_shapes_lm = {
        "inputs_embeds": {0: B3, 1: seq_sym},
        "attention_mask": {0: B3, 1: seq_sym},
        "position_ids": {0: B3, 1: seq_sym},
        "cache_position": {0: seq_sym},
    }

    lm_wrap = LMNoCache(language_model).to(device).eval()
    with torch.inference_mode():
        exported_lm = torch.export.export(
            lm_wrap, 
            (dummy_embeds, dummy_mask, dummy_position_ids, dummy_cache_position), 
            dynamic_shapes=dyn_shapes_lm, 
            strict=False
        )
    trt_lm = torch_tensorrt.dynamo.compile(
        exported_lm,
        inputs=[dummy_embeds, dummy_mask, dummy_position_ids, dummy_cache_position],
        enabled_precisions={torch.float16},
        device=device,
        truncate_double=True,
    )

    return trt_vis, trt_mlp1, trt_lm


# -----------------------------------------------------------------------------
# 4)  Glue: wrap TRT modules back into full model
# -----------------------------------------------------------------------------

class TRTExtractFeature(nn.Module):
    def __init__(self, trt_vis, trt_mlp, original_model):
        super().__init__()
        self.trt_vis = trt_vis
        self.trt_mlp = trt_mlp
        self.pixel_shuffle = original_model.pixel_shuffle
        self.downsample_ratio = original_model.downsample_ratio

    def forward(self, pixel_values):
        with torch.inference_mode():
            # Convert dtype to torch.float16 as required by TensorRT engine
            if pixel_values.dtype != torch.float16:
                pixel_values = pixel_values.to(torch.float16)
                
            vit = self.trt_vis(pixel_values)
            h = w = int(vit.shape[1] ** 0.5)
            vit = vit.reshape(vit.shape[0], h, w, -1)
            vit = self.pixel_shuffle(vit, scale_factor=self.downsample_ratio)
            vit = vit.reshape(vit.shape[0], -1, vit.shape[-1])
            vit = self.trt_mlp(vit)
            return vit

class TRTLanguageWrapper(nn.Module):
    """Language model wrapper for Eagle2 compiled with TensorRT"""
    
    def __init__(self, trt_lm, original_lm):
        super().__init__()
        self.trt_lm = trt_lm
        self._original_lm = original_lm  # reference original language model
        
        # store direct embedding objects
        self._input_embeddings = original_lm.get_input_embeddings()
        self._output_embeddings = original_lm.get_output_embeddings()
        
        # keep original attributes
        self.config = original_lm.config
        self.main_input_name = original_lm.main_input_name
        self.generation_config = getattr(original_lm, "generation_config", None)
        
        # Qwen2ForCausalLM inherits GenerationMixin; keep important methods
        if hasattr(original_lm, "_apply_logits_warper"):
            self._apply_logits_warper = original_lm._apply_logits_warper
        if hasattr(original_lm, "_prepare_attention_mask_for_generation"):
            self._prepare_attention_mask_for_generation = original_lm._prepare_attention_mask_for_generation
    
    def forward(self, 
                input_ids=None, 
                inputs_embeds=None, 
                attention_mask=None, 
                position_ids=None,
                cache_position=None,
                past_key_values=None,
                **kwargs):
        """Forward pass using TensorRT engine"""
        # Convert embeddings if needed
        if inputs_embeds is None and input_ids is not None:
            inputs_embeds = self._input_embeddings(input_ids)
        
        # Create position_ids if needed – Qwen2 uses 1-indexed ids
        if position_ids is None and inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
            if cache_position is None:
                past_length = 0
                if past_key_values is not None:
                    if isinstance(past_key_values, tuple) and len(past_key_values) > 0:
                        past_length = past_key_values[0][0].shape[2]  # KV cache size
                    elif hasattr(past_key_values, "get_seq_length"):
                        past_length = past_key_values.get_seq_length()
                
                cache_position = torch.arange(
                    past_length, past_length + seq_length, 
                    device=inputs_embeds.device
                )
            
            position_ids = cache_position.unsqueeze(0).expand(batch_size, -1) + 1  # Qwen2 positions are 1-indexed
        
        # Call TRT engine – pass 2-D int64 mask as-is
        outputs = self.trt_lm(
            inputs_embeds,
            attention_mask,
            position_ids,
            cache_position,
        )
        
        # use_cache=True인 경우 더미 캐시 반환
        if "use_cache" in kwargs and kwargs["use_cache"]:
            outputs["past_key_values"] = None
        
        return outputs
    
    # Expose original language model methods
    def get_input_embeddings(self):
        return self._input_embeddings
    
    def get_output_embeddings(self):
        return self._output_embeddings
    
    def prepare_inputs_for_generation(self, *args, **kwargs):
        """Delegate to the original language model's implementation."""
        return self._original_lm.prepare_inputs_for_generation(*args, **kwargs)


# -----------------------------------------------------------------------------
# 5)  End-to-end integration & test
# -----------------------------------------------------------------------------

def build_trt_model(device="cuda:0"):
    base_model, processor = load_base(device)
    trt_vis, trt_mlp1, trt_lm = compile_submodules(base_model, device)

    # 원본 모델을 복제하고 TensorRT 모듈로 대체
    base_model.config.use_cache = False
    trt_model = copy.deepcopy(base_model)
    
    # TRT 모듈 준비 및 교체
    trt_extract = TRTExtractFeature(trt_vis, trt_mlp1, base_model)
    trt_model.extract_feature = trt_extract
    
    trt_model.language_model = TRTLanguageWrapper(trt_lm, base_model.language_model)
    
    # 새로운 forward 메서드 정의 - PaliGemma와 유사하게 GenerationMixin 호환성 강화
    def paligemma_style_forward(self, pixel_values=None, input_ids=None, attention_mask=None, 
                               position_ids=None, image_flags=None, past_key_values=None, 
                               labels=None, use_cache=None, output_attentions=None, 
                               output_hidden_states=None, return_dict=None, num_tiles_list=None, **kwargs):
        """PaliGemma-style forward implementation for GenerationMixin compatibility"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # Extract visual features
        if pixel_values is not None:
            vit_embeds = self.extract_feature(pixel_values)
            
            # Handle image tokens
            if image_flags is not None:
                image_flags = image_flags.squeeze(-1)
                vit_embeds = vit_embeds[image_flags == 1]
        
            # Prepare embeddings – access embedding layer directly
            input_embeds = self.language_model.get_input_embeddings()(input_ids)
            B, N, C = input_embeds.shape
            input_embeds = input_embeds.reshape(B * N, C)
            
            input_ids_flat = input_ids.reshape(B * N)
            selected = (input_ids_flat == self.image_token_index)
            
            # replace while preserving dtype/device and avoiding shape mismatch
            try:
                input_embeds[selected] = 0.0 * input_embeds[selected] + vit_embeds.reshape(-1, C).to(input_embeds.dtype)
            except Exception as e:
                vit_flat = vit_embeds.reshape(-1, C).to(input_embeds.dtype)
                n_token = selected.sum()
                input_embeds[selected] = 0.0 * input_embeds[selected] + vit_flat[:n_token]
                
            input_embeds = input_embeds.reshape(B, N, C)
        else:
            # Text-only input
            input_embeds = None
        
        # attention_mask: convert dtype to float16 but keep 2-D shape
        if attention_mask is not None and attention_mask.dtype != torch.float16:
            attention_mask = attention_mask.to(torch.float16)
        
        # Forward pass through language model
        outputs = self.language_model(
            input_ids=None if input_embeds is not None else input_ids,
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache_position=kwargs.get("cache_position", None),
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        
        return outputs
    
    # Replace model forward method
    import types
    trt_model.forward = types.MethodType(paligemma_style_forward, trt_model)
    
    print("TensorRT-integrated model built")
    return trt_model, processor


# -----------------------------------------------------------------------------
# 6)  CLI test – image caption
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    device = "cuda:0"
    torch.cuda.set_device(0)
    trt_model, processor = build_trt_model(device)

    # sample input
    url = "https://cdn.pixabay.com/photo/2019/08/08/23/33/car-4393990_1280.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    prompt = "Describe this image."

    messages = [{
        "role": "user",
        "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}],
    }]

    text_list = [processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)]
    image_inputs, video_inputs = processor.process_vision_info(messages)
    model_inputs = processor(text=text_list, images=image_inputs, videos=video_inputs, return_tensors="pt", padding=True).to(device)
    
    # Convert pixel_values to float16 for TensorRT compatibility
    if "pixel_values" in model_inputs and model_inputs["pixel_values"].dtype != torch.float16:
        model_inputs["pixel_values"] = model_inputs["pixel_values"].to(torch.float16)
    
    # Remove image_sizes parameter (unused by GenerationMixin)
    if "image_sizes" in model_inputs:
        print("Removing 'image_sizes' parameter as it's not used by generation")
        model_inputs.pop("image_sizes")
    
    # Remove unnecessary keys not used by GenerationMixin
    if "image_flags" in model_inputs:
        model_inputs.pop("image_flags")
    
    trt_model.eval()
    
    print("\nStarting generation ...")
    with torch.inference_mode():
        outputs = trt_model.generate(
            **model_inputs,
            max_new_tokens=64,
            do_sample=False,
            use_cache=False,
        )
    print("Generated:", processor.batch_decode(outputs, skip_special_tokens=True)[0])


