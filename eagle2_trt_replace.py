import torch
import torch.nn as nn
import torch_tensorrt
import copy, requests
from PIL import Image
from typing import Optional
from transformers import AutoModel, AutoProcessor, GenerationMixin

from transformers.models.siglip import modeling_siglip as ms
import transformers.models.qwen2.modeling_qwen2 as mq

ms.ALL_ATTENTION_FUNCTIONS["flash_attention_2"] = ms.ALL_ATTENTION_FUNCTIONS["sdpa"]
mq.ALL_ATTENTION_FUNCTIONS["flash_attention_2"] = mq.ALL_ATTENTION_FUNCTIONS["sdpa"]

from transformers.modeling_outputs import CausalLMOutputWithPast

# 1) Load base model & processor
def load_base(device="cuda:1"):
    model_id = "nvidia/Eagle2-2B"
    model = (
        AutoModel.from_pretrained(model_id, trust_remote_code=True, torch_dtype=torch.float16)
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

    def forward(self, inputs_embeds, position_ids=None, cache_position=None): #, position_ids=None, cache_position=None):
        # Ensure inputs are in float16 and on the correct device
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

def compile_submodules(base_model, device="cuda:1"):
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


    S3 = torch.export.Dim("seq_lm", min=1, max=2560)
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
            # print("vit.shape:", vit.shape, "vit.mean:", vit.mean().item(), "vit.std:", vit.std().item())

            h = w = int(vit.shape[1] ** 0.5)
            vit = vit.reshape(vit.shape[0], h, w, -1)
            vit = self.pixel_shuffle(vit, scale_factor=self.downsample_ratio)
            vit = vit.reshape(vit.shape[0], -1, vit.shape[-1])
            vit = self.trt_mlp(vit)
            # print("vit_embeds.shape:", vit.shape, "vit_embeds.mean:", vit.mean().item(), "vit_embeds.std:", vit.std().item())

            return vit

class TRTLanguageWrapper(nn.Module, GenerationMixin):
    """Language model wrapper for Eagle2 compiled with TensorRT"""
    def __init__(self, trt_lm, original_lm, device="cuda:1"):
        super().__init__()
        self.trt_lm = trt_lm
        self._original_lm = original_lm
        # Attributes GenerationMixin expects
        self.config = original_lm.config
        self.main_input_name = original_lm.main_input_name
        self.generation_config = getattr(original_lm, "generation_config", None)

        self.device = torch.device(device)

        self._input_embeddings = original_lm.get_input_embeddings()
        self._output_embeddings = original_lm.get_output_embeddings()
        if hasattr(original_lm, "_apply_logits_warper"):
            self._apply_logits_warper = original_lm._apply_logits_warper
        if hasattr(original_lm, "_prepare_attention_mask_for_generation"):
            self._prepare_attention_mask_for_generation = original_lm._prepare_attention_mask_for_generation

        # Some GenerationMixin helpers inspect this attribute directly.
        self._supports_cache_class = getattr(original_lm, "_supports_cache_class", False)

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
            # inputs_embeds = self._input_embeddings(input_ids).to(dtype=torch.float16, device=self.device)
            inputs_embeds = self._input_embeddings(input_ids).to(dtype=torch.float16, device=device)
        else:
            # inputs_embeds = inputs_embeds.to(dtype=torch.float16, device=self.device)
            inputs_embeds = inputs_embeds.to(dtype=torch.float16, device=device)

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
                    device=self.device
                )
            position_ids = cache_position.unsqueeze(0).expand(batch_size, -1).to(device=self.device)

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

def build_trt_model(device="cuda:1"):
    base_model, processor = load_base(device)
    trt_vis, trt_mlp1, trt_lm = compile_submodules(base_model, device)
    base_model.config.use_cache = False
    trt_model = copy.deepcopy(base_model)
    trt_extract = TRTExtractFeature(trt_vis, trt_mlp1, base_model)
    trt_model.extract_feature = trt_extract
    trt_model.language_model = TRTLanguageWrapper(trt_lm, base_model.language_model, device)

    def paligemma_style_forward(
        self,
        pixel_values=None,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        """Forward pass that extracts visual features once and re-uses them on later decoding steps."""

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 1) Visual features (ViT) – run once then cache
        vit_embeds = None
        if pixel_values is not None:
            print("extracting features from image")
            vit_embeds = self.extract_feature(pixel_values)
            self._cached_visual_features = vit_embeds
        else:
            vit_embeds = getattr(self, "_cached_visual_features", None)

        # 2) Text token embeddings
        input_embeds = self.language_model.get_input_embeddings()(input_ids)

        # 3) Replace [IMG] token positions with image embeddings (if any)
        if vit_embeds is not None:
            B, N, C = input_embeds.shape
            flat_emb = input_embeds.view(B * N, C)
            mask = (input_ids.view(B * N) == self.image_token_index)
            try:
                flat_emb[mask] = vit_embeds.reshape(-1, C).to(flat_emb.dtype)[: mask.sum()]
            except Exception:
                # Fallback in unlikely size-mismatch cases
                flat_emb[mask] = vit_embeds.reshape(-1, C)[: mask.sum()].to(flat_emb.dtype)
            input_embeds = flat_emb.view(B, N, C)

        # 4) Delegate to language model
        lm_kwargs = {
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "use_cache": use_cache if use_cache is not None else False,
            "output_attentions": output_attentions,
            "output_hidden_states": output_hidden_states,
            "return_dict": return_dict,
        }
        # forward additional kwargs transparently
        lm_kwargs.update({k: v for k, v in kwargs.items() if k != "inputs_embeds"})

        return self.language_model(inputs_embeds=input_embeds, **lm_kwargs)

    import types
    trt_model.forward = types.MethodType(paligemma_style_forward, trt_model)

    # ------------------------------------------------------------------
    # NEW: ensure `pixel_values` is only used at the very first decoding
    #       step by stripping it from `model_kwargs` after step-0. This
    #       prevents the ViT path from running on every autoregressive
    #       iteration.
    # ------------------------------------------------------------------
    def _update_model_kwargs_for_generation(self, outputs, model_kwargs, is_encoder_decoder=False, num_new_tokens=1):
        """Wrap the default GenerationMixin helper but drop `pixel_values`."""
        # Call the vanilla helper from GenerationMixin to handle cache & masks
        model_kwargs = GenerationMixin._update_model_kwargs_for_generation(
            self, outputs, model_kwargs, is_encoder_decoder=is_encoder_decoder, num_new_tokens=num_new_tokens
        )
        # After the first step, `pixel_values` is no longer needed.
        model_kwargs.pop("pixel_values", None)
        return model_kwargs

    # Bind the method to the TRTT-wrapped model instance.
    trt_model._update_model_kwargs_for_generation = types.MethodType(_update_model_kwargs_for_generation, trt_model)

    print("TensorRT-integrated model built")
    return base_model, trt_model, processor

# 6) CLI benchmark – Torch vs TensorRT (128 ISL tokens → 128 OSL tokens)
if __name__ == "__main__":
    device = "cuda:1"
    torch.cuda.set_device(device)

    # Build models (Torch reference & TensorRT-optimised)
    base_model, trt_model, processor = build_trt_model(device)

    url = "https://cdn.pixabay.com/photo/2019/08/08/23/33/car-4393990_1280.jpg"
    image = Image.open(requests.get(url, stream=True).raw)

    prompt_tokens = ["token"] * 230 # when 230 tokens, ISL = 2048  (image 1792 + text 256). There are 26 additional overhead tokens
    prompt_text = " ".join(prompt_tokens)

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt_text},
        ],
    }]

    # Apply chat template & process vision info
    text_list = [processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)]
    image_inputs, video_inputs = processor.process_vision_info(messages)

    # Tokenise → dict of tensors
    model_inputs = processor(
        text=text_list,
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt",
        padding=True,
    ).to(device)

    input_ids = model_inputs["input_ids"]
    img_tok_id = base_model.image_token_index      # [IMG] token id

    n_total = input_ids.shape[1]                  # ISL
    n_img   = (input_ids == img_tok_id).sum()     # image token count
    n_txt   = n_total - n_img                     # text token count
    print(f"ISL = {n_total}  (image {n_img} + text {n_txt})")

    # Ensure pixel_values are fp16 and drop unused keys
    if "pixel_values" in model_inputs and model_inputs["pixel_values"].dtype != torch.float16:
        model_inputs["pixel_values"] = model_inputs["pixel_values"].to(torch.float16)
    for _drop_key in ("image_sizes", "image_flags"):
        model_inputs.pop(_drop_key, None)

    # Shared generation kwargs (KV-cache disabled)
    gen_kwargs = dict(max_new_tokens=128, do_sample=False, use_cache=False, eos_token_id=None, early_stopping=False )

    def _benchmark(model, label):
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        with torch.inference_mode():
            out = model.generate(**model_inputs, **gen_kwargs)
        end.record()
        torch.cuda.synchronize()

        runtime_s = start.elapsed_time(end) / 1000.0  # milliseconds → seconds
        text_out = processor.batch_decode(out, skip_special_tokens=True)[0]
        print(f"[{label}] runtime: {runtime_s:.3f} s | last token id: {out[:, -1].item()}")
        return out, text_out, runtime_s

    print("\nRunning baseline (pure Torch) …")
    torch_outputs, torch_text, torch_time = 0, 0, 0
    # torch_outputs, torch_text, torch_time = _benchmark(base_model, "Torch")

    print("\nRunning TensorRT-optimised model …")
    trt_model.generation_config.eos_token_id = None
    trt_outputs, trt_text, trt_time = _benchmark(trt_model, "TensorRT")

    # ------------------------------------------------------------------
    # Verify correctness and report speed-up
    # ------------------------------------------------------------------
    tokens_equal = False # torch.equal(torch_outputs, trt_outputs)
    text_equal = False # (torch_text == trt_text)

    print("\n=== Verification ===")
    print(f"Token IDs identical: {tokens_equal}")
    print(f"Decoded text identical: {text_equal}")

    speedup = 0 # torch_time / trt_time if trt_time > 0 else float('inf')
    print("\n=== Timing ===")
    print(f"Torch time      : {torch_time:.3f} s")
    print(f"TensorRT time   : {trt_time:.3f} s")
    print(f"Speed-up (×)    : {speedup:.2f}")