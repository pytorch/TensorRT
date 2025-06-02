import torch
import torch.nn as nn
import torch_tensorrt
import copy, requests
from PIL import Image
from typing import Optional
from transformers import AutoModel, AutoProcessor, GenerationMixin
from transformers.models.siglip import modeling_siglip as ms
from transformers.models.qwen2 import modeling_qwen2 as mq
from transformers.modeling_outputs import CausalLMOutputWithPast

# 0) Export-friendly attention patches

def _patch_siglip_attention():
    """Swap SiglipAttention.forward with an sdpa-only version no Flash-Attn kernels."""
    def patched_attention_forward(self, hidden_states, attention_mask=None, output_attentions=False):
        B, S, _ = hidden_states.shape
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        q = q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        # Compute scores in fp32 to avoid half-precision overflow/underflow
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        # scores = torch.matmul(q.float(), k.transpose(-2, -1).float()) * self.scale
        print("scores.mean siglip:", scores.mean().item(), "scores.std siglip:", scores.std().item())
        if attention_mask is not None:
            # cast mask to fp32 so dtype matches scores before the addition
            scores = scores + attention_mask
            print("masked_scores.mean siglip:", scores.mean().item())
        print("masked scores.mean siglip:", scores.mean().item(), "masked scores.std siglip:", scores.std().item())
        # scores = scores - scores.max(dim=-1, keepdim=True).values # numerically stable softmax float16
        print("stable masked scores.mean siglip:", scores.mean().item(), "stable masked scores.std siglip:", scores.std().item())
        probs = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
        print("probs.max siglip:", probs.max().item(), "probs.sum siglip:", probs.sum().item())
        if self.training and self.dropout > 0:
            probs = torch.nn.functional.dropout(probs, p=self.dropout, training=True)
        ctx = torch.matmul(probs, v)
        ctx = ctx.transpose(1, 2).contiguous().view(B, S, self.embed_dim)
        ctx = self.out_proj(ctx)
        return (ctx, None) if output_attentions else (ctx, None)
    ms.SiglipAttention.forward = patched_attention_forward

# 1) Load base model & processor

def load_base(device="cuda:0"):
    _patch_siglip_attention()
    # _patch_qwen2_attention()
    model_id = "nvidia/Eagle2-2B"
    model = (
        AutoModel.from_pretrained(model_id, trust_remote_code=True, torch_dtype=torch.float16) # torch.float16
        .to(device)
        .eval()
    )
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, use_fast=True)
    if hasattr(processor, "tokenizer"):
        processor.tokenizer.padding_side = "left"
    return model, processor

# 2) Torch-TensorRT compile helpers

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

    def forward(self, inputs_embeds, position_ids=None, cache_position=None):
        # Ensure inputs are in float16 and on the correct device
        inputs_embeds = inputs_embeds.to(dtype=torch.float32, device=self.lm.device) 

        if position_ids is not None:
            position_ids = position_ids.to(device=self.lm.device)
        if cache_position is not None:
            cache_position = cache_position.to(device=self.lm.device)

        outputs = self.lm(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            cache_position=cache_position,
            use_cache=False
        )
        # Robustly extract logits
        if isinstance(outputs, torch.Tensor):
            logits = outputs
        elif isinstance(outputs, (tuple, list)) and len(outputs) > 0:
            logits = outputs[0]  # Assuming logits are the first element
        elif hasattr(outputs, "logits"):
            logits = outputs.logits
        else:
            raise TypeError(f"Unexpected output type from language model: {type(outputs)}")
        return logits

# 3) Compile sub-modules with TRT

def compile_submodules(base_model, device="cuda:0"):
    vision_model = base_model.vision_model
    mlp1 = base_model.mlp1
    language_model = base_model.language_model

    B = torch.export.Dim("batch", min=1, max=8)
    dummy_pixels = torch.randn(4, 3, 448, 448, dtype=torch.float16, device=device)
    dyn_shapes_vis = {"pixel_values": {0: B}}
    vis_wrapper = VisionWrapper(vision_model).to(device)
    with torch.inference_mode():
        exported_vis = torch.export.export(vis_wrapper, (dummy_pixels,), dynamic_shapes=dyn_shapes_vis, strict=False)
    trt_vis = torch_tensorrt.dynamo.compile(
        exported_vis,
        inputs=[dummy_pixels],
        enabled_precisions={torch.float32},
        device=device,
        # truncate_double=True,
        # disable_tf32=True,
        # use_explicit_typing=True,
        # use_fp32_acc=True,
    )

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
        enabled_precisions={torch.float32},
        device=device,
        # truncate_double=True,
        # disable_tf32=True,
        # use_explicit_typing=True,
        # use_fp32_acc=True,
    )

    hidden_size = language_model.config.hidden_size
    dummy_seq = 13
    dummy_batch = 2
    dummy_embeds = torch.randn(dummy_batch, dummy_seq, hidden_size, dtype=torch.float16, device=device)
    dummy_position_ids = torch.arange(1, dummy_seq + 1, device=device).unsqueeze(0).expand(dummy_batch, -1)
    dummy_cache_position = torch.arange(dummy_seq, device=device)
    B3 = torch.export.Dim("batch3", min=1, max=4)
    # Use a single symbolic dimension for sequence length rather than a complicated expression that can
    # confuse shape unification during tracing.
    S3 = torch.export.Dim("seq_lm", min=1, max=2048)
    dyn_shapes_lm = {
        "inputs_embeds": {0: B3, 1: S3},
        "position_ids": {0: B3, 1: S3},
        "cache_position": {0: S3},
    }
    lm_wrap = LMNoCache(language_model).to(device).eval()
    with torch.inference_mode():
        exported_lm = torch.export.export(
            lm_wrap,
            (dummy_embeds, dummy_position_ids, dummy_cache_position),
            dynamic_shapes=dyn_shapes_lm,
            strict=False
        )
    trt_lm = torch_tensorrt.dynamo.compile(
        exported_lm,
        inputs=[dummy_embeds, dummy_position_ids, dummy_cache_position],
        enabled_precisions={torch.float32},
        device=device,
        # truncate_double=True,
        # disable_tf32=True,
        # use_explicit_typing=True,
        # use_fp32_acc=True,
    )
    return trt_vis, trt_mlp1, trt_lm

# 4) Glue: wrap TRT modules back into full model

class TRTExtractFeature(nn.Module):
    def __init__(self, trt_vis, trt_mlp, original_model):
        super().__init__()
        self.trt_vis = trt_vis
        self.trt_mlp = trt_mlp
        self.pixel_shuffle = original_model.pixel_shuffle
        self.downsample_ratio = original_model.downsample_ratio
    def forward(self, pixel_values):
        with torch.inference_mode():
            if pixel_values.dtype != torch.float16:
                pixel_values = pixel_values.to(torch.float16)
            vit = self.trt_vis(pixel_values)
            # vit = self.trt_vis(pixel_values.to(torch.float16))   ######
            print("vit.shape:", vit.shape, "vit.mean:", vit.mean().item(), "vit.std:", vit.std().item())

            h = w = int(vit.shape[1] ** 0.5)
            vit = vit.reshape(vit.shape[0], h, w, -1)
            vit = self.pixel_shuffle(vit, scale_factor=self.downsample_ratio)
            vit = vit.reshape(vit.shape[0], -1, vit.shape[-1])
            vit = self.trt_mlp(vit)
            print("vit_embeds.shape:", vit.shape, "vit_embeds.mean:", vit.mean().item(), "vit_embeds.std:", vit.std().item())

            return vit

class TRTLanguageWrapper(nn.Module, GenerationMixin):
    """Language model wrapper for Eagle2 compiled with TensorRT"""
    def __init__(self, trt_lm, original_lm):
        super().__init__()
        self.trt_lm = trt_lm
        self._original_lm = original_lm
        self._input_embeddings = original_lm.get_input_embeddings()
        self._output_embeddings = original_lm.get_output_embeddings()
        self.config = original_lm.config
        self.main_input_name = original_lm.main_input_name
        self.generation_config = getattr(original_lm, "generation_config", None)
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
                use_cache=False,
                **kwargs):
        """Forward pass using TensorRT engine, returning CausalLMOutputWithPast for compatibility"""
        # device = next(self.trt_lm.parameters()).device if hasattr(self.trt_lm, "parameters") else "cuda:0"

        # Prepare inputs
        if inputs_embeds is None and input_ids is not None:
            inputs_embeds = self._input_embeddings(input_ids).to(dtype=torch.float16, device=device)
            # inputs_embeds = self._input_embeddings(input_ids).to(dtype=torch.float32, device=device)
        else:
            inputs_embeds = inputs_embeds.to(dtype=torch.float16, device=device)
            # inputs_embeds = inputs_embeds.to(dtype=torch.float32, device=device)

        if position_ids is None and inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
            if cache_position is None:
                past_length = 0
                if past_key_values is not None:
                    if isinstance(past_key_values, tuple) and len(past_key_values) > 0:
                        past_length = past_key_values[0][0].shape[2]
                    elif hasattr(past_key_values, "get_seq_length"):
                        past_length = past_key_values.get_seq_length()
                cache_position = torch.arange(
                    past_length, past_length + seq_length,
                    device=device
                )
            position_ids = cache_position.unsqueeze(0).expand(batch_size, -1).to(device=device)

        # Execute TensorRT engine
        logits = self.trt_lm(
            inputs_embeds,
            position_ids,
            cache_position,
        )

        # Ensure logits is a tensor
        if not isinstance(logits, torch.Tensor):
            raise TypeError(f"Expected logits to be a tensor, but got {type(logits)}")

        # Ensure logits are in float16
        # logits = logits.to(dtype=torch.float16)

        return CausalLMOutputWithPast(
            logits=logits,
            past_key_values=None if not use_cache else past_key_values,
        )

    def get_input_embeddings(self):
        return self._input_embeddings

    def get_output_embeddings(self):
        return self._output_embeddings

    def prepare_inputs_for_generation(self, *args, **kwargs):
        return self._original_lm.prepare_inputs_for_generation(*args, **kwargs)
# 5) End-to-end integration & test

def build_trt_model(device="cuda:0"):
    base_model, processor = load_base(device)
    trt_vis, trt_mlp1, trt_lm = compile_submodules(base_model, device)
    base_model.config.use_cache = False
    trt_model = copy.deepcopy(base_model)
    trt_extract = TRTExtractFeature(trt_vis, trt_mlp1, base_model)
    trt_model.extract_feature = trt_extract
    trt_model.language_model = TRTLanguageWrapper(trt_lm, base_model.language_model)

    def paligemma_style_forward(self, pixel_values=None, input_ids=None, attention_mask=None,
                                position_ids=None, image_flags=None, past_key_values=None,
                                labels=None, use_cache=None, output_attentions=None,
                                output_hidden_states=None, return_dict=None, num_tiles_list=None, **kwargs):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        print("attention_mask dtype:", attention_mask.dtype, "attention_mask.shape:", attention_mask.shape) # , "attention_mask.mean:", attention_mask.mean().item(), "attention_mask.std:", attention_mask.std().item())
        if pixel_values is not None:
            print("pixel_values.shape:", pixel_values.shape, "mean:", pixel_values.mean().item(), "std:", pixel_values.std().item())
            vit_embeds = self.extract_feature(pixel_values)
            # vit_embeds = nn.LayerNorm(vit_embeds.shape[-1],
            #                           eps=1e-5, elementwise_affine=False
            #                           ).to(vit_embeds.dtype)(vit_embeds)
            if torch.isnan(vit_embeds).any():
                print("2 Warning: NaN detected in vit_embeds!")
            else:
                print("2 vit_embeds are valid.")

            print("vit_embeds.shape:", vit_embeds.shape, "mean:", vit_embeds.mean().item(), "std:", vit_embeds.std().item())
            input_embeds = self.language_model.get_input_embeddings()(input_ids)

            if torch.isnan(input_embeds).any():
                print("1 Warning: NaN detected in input embeds!")
            else:
                print("1 Input embeds are valid.")

            print("vit σ:", vit_embeds.std().item(),
                  "txt σ:", input_embeds.std().item())
            
            B, N, C = input_embeds.shape
            input_embeds = input_embeds.reshape(B * N, C)
            input_ids_flat = input_ids.reshape(B * N)
            selected = (input_ids_flat == self.image_token_index)
            print("selected.sum():", selected.sum().item())
            try:
                input_embeds[selected] = 0.0 * input_embeds[selected] + vit_embeds.reshape(-1, C).to(input_embeds.dtype)
                print("input_embeds[selected].mean:", input_embeds[selected].mean().item())
            except Exception as e:
                vit_flat = vit_embeds.reshape(-1, C).to(input_embeds.dtype)
                n_token = selected.sum()
                print("Embedding replacement failed:", e)
                input_embeds[selected] = 0.0 * input_embeds[selected] + vit_flat[:n_token]
            input_embeds = input_embeds.reshape(B, N, C)
        else:
            input_embeds = self.language_model.get_input_embeddings()(input_ids)
        # if attention_mask is not None and attention_mask.dtype != torch.float16:
        #     attention_mask = attention_mask.to(torch.float16) # torch.float16
        # if attention_mask is not None:
        #     attention_mask = attention_mask.to(torch.bool)

        gen_kwargs = {
            'attention_mask': attention_mask,
            'position_ids': position_ids,
            'past_key_values': past_key_values,
            'use_cache': use_cache if use_cache is not None else False,
            'output_attentions': output_attentions,
            'output_hidden_states': output_hidden_states,
            'return_dict': return_dict,
        }
        gen_kwargs.update({k: v for k, v in kwargs.items() if k != 'inputs_embeds'})

        print("position_ids:", position_ids)

        if torch.isnan(input_embeds).any():
            print("2 Warning: NaN detected in input embeds!")
        else:
            print("2 Input embeds are valid.")

        outputs = self.language_model(inputs_embeds=input_embeds, **gen_kwargs)
        
        print("logits.shape:", outputs.logits.shape, "predicted_token:", outputs.logits[:, -1, :].argmax(-1).item())

        print("logits max:", outputs.logits.max().item(), "logits min:", outputs.logits.min().item())
        print("top 5 token probs:", torch.softmax(outputs.logits[:, -1, :], dim=-1).topk(5))

        return outputs

    import types
    trt_model.forward = types.MethodType(paligemma_style_forward, trt_model)
    print("TensorRT-integrated model built")
    return base_model, trt_model, processor

# 6) CLI test – image caption
if __name__ == "__main__":
    device = "cuda:0"
    torch.cuda.set_device(0)
    base_model, trt_model, processor = build_trt_model(device)
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
    if "pixel_values" in model_inputs and model_inputs["pixel_values"].dtype != torch.float16:
        model_inputs["pixel_values"] = model_inputs["pixel_values"].to(torch.float16) # torch.float16
    if "image_sizes" in model_inputs:
        print("Removing 'image_sizes' parameter as it's not used by generation")
        model_inputs.pop("image_sizes")
    if "image_flags" in model_inputs:
        model_inputs.pop("image_flags")

    # with torch.inference_mode():
    #     outputs = base_model.generate(**model_inputs, max_new_tokens=64, use_cache=False)
    # print("Generated text (torch):", processor.batch_decode(outputs, skip_special_tokens=True)[0])
    # Debug input
    print("input_ids:", model_inputs['input_ids'])
    print("image_token_index:", trt_model.config.image_token_index)
    print("Number of image tokens:", (model_inputs['input_ids'] == trt_model.config.image_token_index).sum().item())
    # trt_model.eval()
    # logits = trt_model(**model_inputs).logits
    # last_logits = logits[:, -1, :]
    # print("Last logits mean:", last_logits.mean().item())
    # print("Last logits std:", last_logits.std().item())
    # print("Last logits max:", last_logits.max().item())
    # print("Predicted token:", last_logits.argmax(-1).item())

    print("Token ID 0 decodes to:", processor.tokenizer.decode([0]))
    print("Vocabulary sample:", {i: processor.tokenizer.decode([i]) for i in range(5)})

    print("\nStarting generation ...")
    with torch.inference_mode():
        outputs = trt_model.generate(
            **model_inputs,
            max_new_tokens=64,  # Test one token first
            do_sample=False,
            use_cache=False,
        )
    print("Generated token ID:", outputs[:, -1].item())
    print("Generated text:", processor.batch_decode(outputs, skip_special_tokens=True)[0])