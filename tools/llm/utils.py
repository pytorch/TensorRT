import copy
import timeit

import numpy as np
import torch
from transformers import StoppingCriteriaList
from transformers.generation.stopping_criteria import (
    EosTokenCriteria,
    MaxLengthCriteria,
)


_VISION_MODULE_ATTRS = ("visual", "vision_model", "vision_tower", "vision_encoder")


def _is_windowed_rope_vision_module(module: torch.nn.Module) -> bool:
    return (
        hasattr(module, "patch_embed")
        and hasattr(module, "blocks")
        and hasattr(module, "rot_pos_emb")
        and hasattr(module, "get_window_index")
    )


def _is_merged_windowed_rope_vision_module(module: torch.nn.Module) -> bool:
    return _is_windowed_rope_vision_module(module) and hasattr(module, "merger")


def _is_tiled_aspect_ratio_vision_module(module: torch.nn.Module) -> bool:
    return (
        hasattr(module, "patch_embedding")
        and hasattr(module, "global_transformer")
        and hasattr(module, "pre_tile_positional_embedding")
    )


def _is_native_vit_vision_module(module: torch.nn.Module) -> bool:
    has_patch_embedding = any(
        hasattr(module, attr_name)
        for attr_name in ("patch_embed", "patch_embedding", "embeddings")
    )
    has_transformer = any(
        hasattr(module, attr_name)
        for attr_name in ("blocks", "encoder", "transformer", "global_transformer")
    )
    return has_patch_embedding and has_transformer


def _is_compiled_vit_plugin_adapter(module: torch.nn.Module) -> bool:
    return hasattr(module, "compiled_visual") and hasattr(module, "input_contract")


def _is_vision_module(module: torch.nn.Module) -> bool:
    return (
        _is_windowed_rope_vision_module(module)
        or _is_tiled_aspect_ratio_vision_module(module)
        or _is_native_vit_vision_module(module)
        or _is_compiled_vit_plugin_adapter(module)
    )


def _contains_vision_module(module: torch.nn.Module) -> bool:
    for attr_name in _VISION_MODULE_ATTRS:
        if isinstance(getattr(module, attr_name, None), torch.nn.Module):
            return True

    for _, child in module.named_modules():
        if child is module:
            continue
        if _is_vision_module(child):
            return True
    return False


_LANGUAGE_MODULE_ATTRS = (
    "language_model",
    "text_model",
    "llm",
    "decoder",
    "model",
)


def get_language_model(model: torch.nn.Module) -> torch.nn.Module:
    for attr_name in _LANGUAGE_MODULE_ATTRS:
        candidate = getattr(model, attr_name, None)
        if not isinstance(candidate, torch.nn.Module):
            continue
        if attr_name == "model" and _contains_vision_module(candidate):
            continue
        return candidate

    for module_name, module in model.named_modules():
        if module is model:
            continue
        if module_name.rsplit(".", 1)[-1] not in _LANGUAGE_MODULE_ATTRS:
            continue
        if _contains_vision_module(module):
            continue
        return module

    raise ValueError(
        "Cannot find a language-model submodule. Expected a language/text "
        "model leaf module that does not also contain the vision tower."
    )


def get_vision_model(model: torch.nn.Module) -> torch.nn.Module:
    visual = getattr(model, "visual", None)
    if isinstance(visual, torch.nn.Module):
        return visual

    for parent_attr in ("model", "language_model"):
        parent = getattr(model, parent_attr, None)
        if not isinstance(parent, torch.nn.Module):
            continue
        visual = getattr(parent, "visual", None)
        if isinstance(visual, torch.nn.Module):
            return visual

    for _, module in model.named_modules():
        if module is model:
            continue
        if _is_merged_windowed_rope_vision_module(module):
            return module

    for attr_name in _VISION_MODULE_ATTRS:
        if attr_name == "visual":
            continue
        candidate = getattr(model, attr_name, None)
        if isinstance(candidate, torch.nn.Module):
            return candidate

    for parent_attr in ("model", "language_model"):
        parent = getattr(model, parent_attr, None)
        if not isinstance(parent, torch.nn.Module):
            continue
        for attr_name in _VISION_MODULE_ATTRS:
            candidate = getattr(parent, attr_name, None)
            if isinstance(candidate, torch.nn.Module):
                return candidate

    for _, module in model.named_modules():
        if module is model:
            continue
        if _is_vision_module(module):
            return module

    raise ValueError(
        "Cannot find a vision-model submodule. Expected a Hugging Face vision "
        "tower alias or a module with ViT-like patch embedding and transformer "
        "blocks/encoder."
    )


def extract_vision_tensor(vision_output) -> torch.Tensor:
    """Normalize Hugging Face vision outputs to a tensor of image embeddings."""
    if isinstance(vision_output, torch.Tensor):
        tensor = vision_output
    elif hasattr(vision_output, "last_hidden_state"):
        tensor = vision_output.last_hidden_state
    elif hasattr(vision_output, "pooler_output"):
        tensor = vision_output.pooler_output
    elif isinstance(vision_output, (tuple, list)):
        tensor = vision_output[0]
    else:
        raise TypeError(
            "Vision model returned an unsupported output type: "
            f"{type(vision_output).__name__}"
        )

    if tensor.dim() == 3 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)
    return tensor


def _find_qwen_visual_with_merger(model: torch.nn.Module):
    candidates = []
    for parent in (model, getattr(model, "model", None), getattr(model, "language_model", None)):
        if isinstance(parent, torch.nn.Module):
            candidates.append(getattr(parent, "visual", None))

    try:
        candidates.append(get_vision_model(model))
    except ValueError:
        pass

    for candidate in candidates:
        if isinstance(candidate, torch.nn.Module) and hasattr(candidate, "merger"):
            return candidate

    for _, module in model.named_modules():
        if hasattr(module, "merger"):
            return module
    return None


def _maybe_merge_qwen_image_embeds(
    model: torch.nn.Module,
    image_embeds: torch.Tensor,
    expected_tokens: int | None,
) -> torch.Tensor:
    if expected_tokens is None or image_embeds.shape[0] == expected_tokens:
        return image_embeds
    if image_embeds.dim() != 2:
        return image_embeds

    visual = _find_qwen_visual_with_merger(model)
    if visual is None:
        return image_embeds

    spatial_merge_unit = getattr(visual, "spatial_merge_unit", None)
    if spatial_merge_unit is not None:
        expected_raw_tokens = expected_tokens * int(spatial_merge_unit)
        if image_embeds.shape[0] != expected_raw_tokens:
            return image_embeds

    try:
        merged = visual.merger(image_embeds)
    except Exception:
        return image_embeds

    return merged if isinstance(merged, torch.Tensor) else image_embeds


def get_qwen_image_embeds(
    model: torch.nn.Module,
    pixel_values: torch.Tensor,
    image_grid_thw: torch.Tensor,
    expected_tokens: int | None = None,
) -> torch.Tensor:
    """
    Return Qwen image embeddings at the token level expected by the LM.

    Prefer the full Hugging Face VLM helper when available, since it owns the
    exact visual merge/projector path. Fall back to the resolved visual module
    for compiled adapters and older model implementations.
    """
    get_image_features = getattr(model, "get_image_features", None)
    if callable(get_image_features):
        call_attempts = (
            lambda: get_image_features(
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
            ),
            lambda: get_image_features(pixel_values, image_grid_thw),
            lambda: get_image_features(pixel_values),
        )
        for call in call_attempts:
            try:
                image_embeds = extract_vision_tensor(call())
                return _maybe_merge_qwen_image_embeds(
                    model, image_embeds, expected_tokens
                )
            except TypeError:
                continue

    image_embeds = extract_vision_tensor(
        get_vision_model(model)(pixel_values, image_grid_thw)
    )
    return _maybe_merge_qwen_image_embeds(model, image_embeds, expected_tokens)


def _get_qwen_rope_owner(model: torch.nn.Module):
    if hasattr(model, "get_rope_index"):
        return model
    parent = getattr(model, "model", None)
    if hasattr(parent, "get_rope_index"):
        return parent
    return None


def _get_qwen_config_attr(model: torch.nn.Module, attr_name: str):
    for owner in (model, getattr(model, "model", None)):
        config = getattr(owner, "config", None)
        value = getattr(config, attr_name, None)
        if value is not None:
            return value
    return None


def get_qwen_mm_token_type_ids(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
) -> torch.Tensor:
    """
    Build Qwen multimodal token type ids from special image/video tokens.

    Qwen uses 0 for text tokens, 1 for image tokens, and 2 for video tokens.
    """
    mm_token_type_ids = torch.zeros_like(input_ids, dtype=torch.int32)

    image_token_id = _get_qwen_config_attr(model, "image_token_id")
    if image_token_id is not None:
        mm_token_type_ids = torch.where(
            input_ids == int(image_token_id),
            torch.ones_like(mm_token_type_ids),
            mm_token_type_ids,
        )

    video_token_id = _get_qwen_config_attr(model, "video_token_id")
    if video_token_id is not None:
        mm_token_type_ids = torch.where(
            input_ids == int(video_token_id),
            torch.full_like(mm_token_type_ids, 2),
            mm_token_type_ids,
        )

    return mm_token_type_ids


def get_qwen_position_ids(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    image_grid_thw: torch.Tensor | None = None,
    attention_mask: torch.Tensor | None = None,
    video_grid_thw: torch.Tensor | None = None,
    second_per_grid_ts: torch.Tensor | None = None,
    mm_token_type_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Return Qwen2.5-VL multimodal RoPE position ids when the model exposes
    get_rope_index, otherwise fall back to plain text position ids.
    """
    rope_owner = _get_qwen_rope_owner(model)
    if rope_owner is None:
        return torch.arange(
            input_ids.shape[1], dtype=torch.long, device=input_ids.device
        ).unsqueeze(0)

    kwargs = {"input_ids": input_ids}
    if image_grid_thw is not None:
        kwargs["image_grid_thw"] = image_grid_thw
    if video_grid_thw is not None:
        kwargs["video_grid_thw"] = video_grid_thw
    if second_per_grid_ts is not None:
        kwargs["second_per_grid_ts"] = second_per_grid_ts
    if attention_mask is not None:
        kwargs["attention_mask"] = attention_mask
    if mm_token_type_ids is None:
        mm_token_type_ids = get_qwen_mm_token_type_ids(model, input_ids)
    kwargs["mm_token_type_ids"] = mm_token_type_ids

    try:
        position_ids = rope_owner.get_rope_index(**kwargs)
    except TypeError:
        try:
            position_ids = rope_owner.get_rope_index(
                input_ids,
                image_grid_thw,
                video_grid_thw,
                second_per_grid_ts,
                attention_mask,
                mm_token_type_ids,
            )
        except TypeError:
            position_ids = rope_owner.get_rope_index(
                input_ids,
                image_grid_thw,
                video_grid_thw,
                attention_mask,
                mm_token_type_ids,
            )

    if isinstance(position_ids, tuple):
        position_ids = position_ids[0]
    return position_ids.to(device=input_ids.device, dtype=torch.long)


def export_llm(model, inputs, min_seq_len=1, max_seq_len=16, position_ids=None):
    """
    Exports the LLM model into an ExportedProgram with dynamic shapes.
    In the case of guard failures due to some PyTorch kernel implements, we also
    try to re-export the graph by expressing them as runtime assert nodes
    """
    with torch.no_grad():
        # max=1024 has contraint violation error. https://github.com/pytorch/pytorch/issues/125604
        seq_len = torch.export.Dim("seq_len", min=min_seq_len, max=max_seq_len)
        if position_ids is None:
            position_ids = torch.arange(inputs.shape[1]).unsqueeze(0).to(inputs.device)
        position_seq_dim = position_ids.dim() - 1
        try:
            print("Trying to export the model using torch.export.export()..")
            # strict=False only enables aotautograd tracing and excludes dynamo.
            ep = torch.export.export(
                model,
                args=(inputs,),
                kwargs={"position_ids": position_ids},
                dynamic_shapes=({1: seq_len}, {position_seq_dim: seq_len}),
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
                dynamic_shapes=({1: seq_len}, {position_seq_dim: seq_len}),
                strict=False,
                prefer_deferred_runtime_asserts_over_guards=True,
            )

    return ep


def get_zeroed_static_cache_inputs(
    model: "torch.fx.GraphModule",
    device: str = "cuda:0",
    has_position_ids: bool = True,
):
    """
    Extracts and returns zeroed static KV cache tensors from a torch.fx.GraphModule. This should only be used for static cache_v1 and static cache_v2.

    This function identifies placeholder nodes in the graph that represent KV cache tensors,
    and creates zeroed tensors with the same shape, dtype, and device as the original placeholders.

    Args:
        model (torch.fx.GraphModule): The exported model graph containing KV cache placeholders
        device (str): Device to create the zeroed tensors on.
        has_position_ids (bool): Whether position_ids is present as an input. Default: True

    Returns:
        tuple: A tuple of zeroed tensors corresponding to the KV cache placeholders in the graph
    """
    # placeholder nodes are expected to be in the following order:
    # input_ids, position_ids, kv_cache_key, kv_cache_value, ..., start_idx, end_idx
    placeholder_nodes = [node for node in model.graph.nodes if node.op == "placeholder"]

    # By default, assume input_ids and position_ids are present as the first two inputs.
    # If has_position_ids is False, only input_ids is present.
    if has_position_ids:
        kv_start = 2
    else:
        kv_start = 1
    # The last two inputs are start_idx, end_idx.
    kv_end = -2

    kv_cache_inputs = placeholder_nodes[kv_start:kv_end]
    zeroed_kv_cache_inputs = []
    for input in kv_cache_inputs:
        zeroed_kv_cache_inputs.append(
            torch.zeros(
                input.meta["val"].shape,
                dtype=input.meta["val"].dtype,
                device=torch.device(device),
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


def generate(
    model,
    input_seq,
    max_output_seq_length,
    eos_token_id,
    benchmark=True,
    dynamic_seqlen_range=None,
):
    """
    Greedy decoding of the model. This generates up to max_tokens.

    Args:
        dynamic_seqlen_range: Optional (min, max) tuple.  When set, marks
            dimension 1 of input_seq as dynamic before each model call so that
            torch.compile + TensorRT builds a single engine covering the full
            range instead of recompiling per sequence length.  Pass
            ``(1, max_output_seq_length)`` to cover every possible step.
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
        if dynamic_seqlen_range is not None:
            # Mark the sequence-length dimension as dynamic so that the
            # torch.compile cache hits the same TRT engine across all steps
            # instead of recompiling for every new sequence length.
            # Note: we intentionally omit min/max bounds here.  Passing them
            # triggers a compile-time guard-range check that fails for models
            # whose attention math contains modulo operations (e.g. Qwen SDPA
            # block-size padding produces s1*(8+s1-s1%8)>1 guards that the
            # symbolic solver can't verify without concrete values).  Without
            # bounds, dynamo traces symbolically and TRT infers the profile
            # from the first concrete shape it sees.
            torch._dynamo.mark_dynamic(input_seq, 1)
        position_ids = torch.arange(
            input_seq.shape[1], device=input_seq.device
        ).unsqueeze(0)
        if dynamic_seqlen_range is not None:
            torch._dynamo.mark_dynamic(position_ids, 1)
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
        "Model Precision": precision,
        "Batch size": batch_size,
        "Median(FPS)": speed_med,
        "Mean(FPS)": speed_mean,
        "Median-Latency(ms)": time_med * 1000,
        "Mean-Latency(ms)": time_mean * 1000,
        "Latency-StdDev(ms)": time_std * 1000,
        "Compile Time(s)": compile_time_s,
    }
    return stats


def _prepare_mm_inputs(
    model,
    pixel_values: torch.Tensor | None,
    input_ids: torch.Tensor,
    emb_layer: torch.nn.Embedding,
    with_timing: bool = False,
):
    """
    Prepares multimodal inputs for Eagle2-style VLMs by encoding images and merging with text embeddings.
    Optionally times the vision and MLP parts.
    """
    vision_time = 0.0
    mlp_time = 0.0
    vit_embeds = None

    if pixel_values is not None:
        if with_timing:
            vision_start = torch.cuda.Event(enable_timing=True)
            vision_end = torch.cuda.Event(enable_timing=True)
            mlp_start = torch.cuda.Event(enable_timing=True)
            mlp_end = torch.cuda.Event(enable_timing=True)

            vision_start.record()
            vit_out = model.vision_model(pixel_values)
            vision_end.record()
            torch.cuda.synchronize()
            vision_time = vision_start.elapsed_time(vision_end)
        else:
            vit_out = model.vision_model(pixel_values)

        vit_embeds = (
            vit_out.last_hidden_state
            if hasattr(vit_out, "last_hidden_state")
            else vit_out
        )

        if with_timing:
            mlp_start.record()

        h = w = int(vit_embeds.shape[1] ** 0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
        vit_embeds = model.pixel_shuffle(
            vit_embeds, scale_factor=model.downsample_ratio
        )
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
        vit_embeds = model.mlp1(vit_embeds)

        if with_timing:
            mlp_end.record()
            torch.cuda.synchronize()
            mlp_time = mlp_start.elapsed_time(mlp_end)

    seq_tokens = input_ids.clone()
    seq_embeds = emb_layer(seq_tokens)

    if vit_embeds is not None:
        B, N, C = seq_embeds.shape
        flat_emb = seq_embeds.view(B * N, C)
        mask = seq_tokens.view(B * N) == model.image_token_index
        try:
            flat_emb[mask] = vit_embeds.reshape(-1, C).to(flat_emb.dtype)[: mask.sum()]
        except Exception:
            # Fallback in unlikely size-mismatch cases
            flat_emb[mask] = vit_embeds.reshape(-1, C)[: mask.sum()].to(flat_emb.dtype)
        seq_embeds = flat_emb.view(B, N, C)

    if with_timing:
        return seq_tokens, seq_embeds, vision_time, mlp_time
    else:
        return seq_tokens, seq_embeds


def generate_mm(
    model,
    pixel_values: torch.Tensor | None,
    input_ids: torch.Tensor,
    eos_token_id: int,
    emb_layer: torch.nn.Embedding,
    max_new_tokens: int = 64,
    device: str = "cuda:0",
    with_timing: bool = False,
):
    """Greedy decode for Eagle2-style VLM, with optional detailed timing.

    Parameters
    ----------
    model : nn.Module
        Must expose vision_model, mlp1, language_model, pixel_shuffle, downsample_ratio, image_token_index.
    pixel_values : Tensor | None
        Input image batch (B,C,H,W) or None.
    input_ids : LongTensor  (B, N_prompt)
        Text prompt token ids including [IMG] placeholder(s).
    eos_token_id : int
        Stop generation when all sequences emit EOS.
    emb_layer : nn.Embedding
        Embedding layer for input_ids.
    max_new_tokens : int
        Maximum number of new tokens to generate.
    with_timing : bool
        If True, returns detailed timing information.

    Returns
    -------
    if with_timing is False:
        torch.LongTensor: Generated token sequence (only new tokens).
    if with_timing is True:
        tuple: (
            seq_tokens: Full generated token sequence,
            step_times: List of latencies for each generation step,
            overall_time: Total generation time,
            vision_time: Vision encoder latency,
            mlp_time: MLP latency
        )
    """
    if with_timing:
        overall_start = torch.cuda.Event(enable_timing=True)
        overall_end = torch.cuda.Event(enable_timing=True)
        lm_start = torch.cuda.Event(enable_timing=True)
        lm_end = torch.cuda.Event(enable_timing=True)
        overall_start.record()

    # --- Input preparation ---
    if with_timing:
        seq_tokens, seq_embeds, vision_time, mlp_time = _prepare_mm_inputs(
            model, pixel_values, input_ids, emb_layer, with_timing=True
        )
    else:
        seq_tokens, seq_embeds = _prepare_mm_inputs(
            model, pixel_values, input_ids, emb_layer, with_timing=False
        )

    # ───────────────────────────────── Greedy loop ───────────────────────────────────────────────────
    step_times = []
    generated = 0

    while generated < max_new_tokens:
        if with_timing:
            lm_start.record()

        cur_embeds = seq_embeds
        position_ids = (
            torch.arange(cur_embeds.shape[1]).unsqueeze(0).to(cur_embeds.device)
        )
        with torch.no_grad():
            logits = model.language_model(
                inputs_embeds=cur_embeds, position_ids=position_ids
            )
            if hasattr(logits, "logits"):
                logits = logits.logits

        next_tok = torch.argmax(logits[:, -1, :], dim=-1)

        if with_timing:
            lm_end.record()
            torch.cuda.synchronize()
            step_times.append(lm_start.elapsed_time(lm_end))

        seq_tokens = torch.cat([seq_tokens, next_tok[:, None]], dim=-1)
        seq_embeds = torch.cat([seq_embeds, emb_layer(next_tok)[:, None, :]], dim=1)

        generated += 1
        if (next_tok == eos_token_id).all():
            break

    if with_timing:
        overall_end.record()
        torch.cuda.synchronize()
        overall_time = overall_start.elapsed_time(overall_end)
        return (
            seq_tokens[:, input_ids.shape[1] :],
            step_times,
            overall_time,
            vision_time,
            mlp_time,
        )
    else:
        return seq_tokens[:, input_ids.shape[1] :]


@torch.inference_mode()
def generate_mm_with_static_cache(
    model,
    pixel_values: torch.Tensor | None,
    input_ids: torch.Tensor,
    eos_token_id: int,
    emb_layer: torch.nn.Embedding,
    max_new_tokens: int = 64,
    device: str = "cuda:0",
    with_timing: bool = False,
):
    """
    Greedy Decoder for multimodal VLM (using static KV-cache v1), with optional timing.
    Basic structure is identical to LM version (generate_with_static_cache) but
    * Input is `inputs_embeds`
    * Vision tokens are sent together only in the first step
    """
    if with_timing:
        overall_start = torch.cuda.Event(enable_timing=True)
        overall_end = torch.cuda.Event(enable_timing=True)
        lm_start = torch.cuda.Event(enable_timing=True)
        lm_end = torch.cuda.Event(enable_timing=True)
        overall_start.record()
        vision_time, mlp_time = 0.0, 0.0

    if with_timing:
        seq_tokens, seq_embeds, vision_time, mlp_time = _prepare_mm_inputs(
            model, pixel_values, input_ids, emb_layer, with_timing=True
        )
    else:
        seq_tokens, seq_embeds = _prepare_mm_inputs(
            model, pixel_values, input_ids, emb_layer, with_timing=False
        )

    # ───────────────────── KV-cache initialization ─────────────────────
    kv_cache = get_zeroed_static_cache_inputs(
        model.language_model, device=device, has_position_ids=True
    )
    start_idx = 0
    end_idx = seq_embeds.size(1)
    generated = 0
    max_total_len = end_idx + max_new_tokens
    output_tokens = seq_tokens.clone()
    step_times = []

    # ───────────────────── Greedy loop ───────────────────────
    while output_tokens.size(1) < max_total_len:
        if with_timing:
            lm_start.record()

        cur_embeds = seq_embeds if generated == 0 else seq_embeds[:, -1:, :]

        if generated == 0:
            position_ids = (
                torch.arange(cur_embeds.shape[1]).unsqueeze(0).to(cur_embeds.device)
            )
        else:
            position_ids = torch.tensor([[start_idx]], dtype=torch.int64).to(
                cur_embeds.device
            )

        input_signature = (
            cur_embeds,
            position_ids,
            *kv_cache,
            start_idx,
            end_idx,
        )

        logits_and_kv = model.language_model(*input_signature)
        logits, kv_cache = logits_and_kv[0], logits_and_kv[1:]

        next_tok = logits[:, -1, :].argmax(dim=-1)
        output_tokens = torch.cat([output_tokens, next_tok[:, None]], dim=-1)

        next_embed = emb_layer(next_tok)[:, None, :]
        seq_embeds = next_embed

        generated += 1
        start_idx = end_idx
        end_idx += 1

        if with_timing:
            lm_end.record()
            torch.cuda.synchronize()
            step_times.append(lm_start.elapsed_time(lm_end))

        if (next_tok == eos_token_id).all():
            break

    if with_timing:
        overall_end.record()
        torch.cuda.synchronize()
        overall_time = overall_start.elapsed_time(overall_end)
        return (
            output_tokens[:, input_ids.shape[1] :],
            step_times,
            overall_time,
            vision_time,
            mlp_time,
        )
    else:
        return output_tokens[:, input_ids.shape[1] :]


def _prepare_qwen_mm_inputs(
    model,
    pixel_values: torch.Tensor | None,
    input_ids: torch.Tensor,
    image_grid_thw: torch.Tensor,
    emb_layer: torch.nn.Embedding,
    with_timing: bool = False,
):
    """
    Prepares multimodal inputs for Qwen2.5-VL by encoding images and merging with text embeddings.
    Optionally times the vision part.
    """
    vision_time = 0.0
    image_embeds = None
    seq_tokens = input_ids.clone()
    seq_embeds = emb_layer(seq_tokens)
    image_mask = seq_tokens == model.config.image_token_id
    num_image_tokens = image_mask.sum().item()

    if pixel_values is not None:
        if with_timing:
            vision_start = torch.cuda.Event(enable_timing=True)
            vision_end = torch.cuda.Event(enable_timing=True)
            vision_start.record()

        image_embeds = get_qwen_image_embeds(
            model,
            pixel_values,
            image_grid_thw,
            expected_tokens=num_image_tokens,
        )

        if with_timing:
            vision_end.record()
            torch.cuda.synchronize()
            vision_time = vision_start.elapsed_time(vision_end)

    if image_embeds is not None:
        if num_image_tokens != image_embeds.shape[0]:
            raise ValueError(
                "Number of image tokens "
                f"({num_image_tokens}) does not match number of image embeddings "
                f"({image_embeds.shape[0]}). Image embedding shape: "
                f"{tuple(image_embeds.shape)}."
            )
        mask_expanded = image_mask.unsqueeze(-1).expand_as(seq_embeds)
        seq_embeds = seq_embeds.masked_scatter(
            mask_expanded, image_embeds.to(seq_embeds.dtype)
        )

    if with_timing:
        return seq_tokens, seq_embeds, vision_time
    else:
        return seq_tokens, seq_embeds


def generate_mm_qwen2_5_vl(
    model,
    pixel_values: torch.Tensor | None,
    input_ids: torch.Tensor,
    image_grid_thw: torch.Tensor,
    attention_mask: torch.Tensor | None,
    eos_token_id: int,
    emb_layer: torch.nn.Embedding,
    max_new_tokens: int = 64,
    with_timing: bool = False,
):
    """
    Custom generation function for the Qwen2_5_VLForConditionalGeneration model, with optional timing.
    """
    language_model = get_language_model(model)

    if with_timing:
        overall_start = torch.cuda.Event(enable_timing=True)
        overall_end = torch.cuda.Event(enable_timing=True)
        lm_start = torch.cuda.Event(enable_timing=True)
        lm_end = torch.cuda.Event(enable_timing=True)
        overall_start.record()

    if with_timing:
        seq_tokens, seq_embeds, vision_time = _prepare_qwen_mm_inputs(
            model,
            pixel_values,
            input_ids,
            image_grid_thw,
            emb_layer,
            with_timing=True,
        )
    else:
        seq_tokens, seq_embeds = _prepare_qwen_mm_inputs(
            model,
            pixel_values,
            input_ids,
            image_grid_thw,
            emb_layer,
            with_timing=False,
        )

    step_times = []
    generated = 0
    seq_attention_mask = (
        attention_mask.clone()
        if attention_mask is not None
        else torch.ones_like(seq_tokens, dtype=torch.long)
    )
    while generated < max_new_tokens:
        if with_timing:
            lm_start.record()

        position_ids = get_qwen_position_ids(
            model,
            seq_tokens,
            image_grid_thw=image_grid_thw,
            attention_mask=seq_attention_mask,
        )

        with torch.no_grad():
            outputs = language_model(
                inputs_embeds=seq_embeds,
                position_ids=position_ids,
            )
            hidden_states = (
                outputs
                if isinstance(outputs, torch.Tensor)
                else outputs.last_hidden_state
            )

        logits = model.lm_head(hidden_states[:, -1, :])
        next_tok = torch.argmax(logits, dim=-1)

        if with_timing:
            lm_end.record()
            torch.cuda.synchronize()
            step_times.append(lm_start.elapsed_time(lm_end))

        seq_tokens = torch.cat([seq_tokens, next_tok[:, None]], dim=1)
        seq_attention_mask = torch.cat(
            [
                seq_attention_mask,
                torch.ones(
                    seq_attention_mask.shape[0],
                    1,
                    dtype=seq_attention_mask.dtype,
                    device=seq_attention_mask.device,
                ),
            ],
            dim=1,
        )
        next_emb = emb_layer(next_tok)[:, None, :]
        seq_embeds = torch.cat([seq_embeds, next_emb], dim=1)

        generated += 1
        if (next_tok == eos_token_id).all():
            break

    if with_timing:
        overall_end.record()
        torch.cuda.synchronize()
        overall_time = overall_start.elapsed_time(overall_end)
        return (
            seq_tokens[:, input_ids.shape[1] :],
            step_times,
            overall_time,
            vision_time,
            0.0,
        )
    else:
        return seq_tokens[:, input_ids.shape[1] :]


def generate_mm_qwen2_5_vl_with_static_cache(
    model,
    pixel_values: torch.Tensor | None,
    input_ids: torch.Tensor,
    image_grid_thw: torch.Tensor,
    attention_mask: torch.Tensor | None,
    eos_token_id: int,
    emb_layer: torch.nn.Embedding,
    max_new_tokens: int = 64,
    device: str = "cuda:0",
    with_timing: bool = False,
) -> torch.LongTensor:
    """
    Greedy Decoder for Qwen-2.5-VL using static KV-cache, with optional timing.
    """
    language_model = get_language_model(model)

    if with_timing:
        overall_start = torch.cuda.Event(enable_timing=True)
        overall_end = torch.cuda.Event(enable_timing=True)
        lm_start = torch.cuda.Event(enable_timing=True)
        lm_end = torch.cuda.Event(enable_timing=True)
        overall_start.record()

    if with_timing:
        seq_tokens, seq_embeds, vision_time = _prepare_qwen_mm_inputs(
            model,
            pixel_values,
            input_ids,
            image_grid_thw,
            emb_layer,
            with_timing=True,
        )
    else:
        seq_tokens, seq_embeds = _prepare_qwen_mm_inputs(
            model,
            pixel_values,
            input_ids,
            image_grid_thw,
            emb_layer,
            with_timing=False,
        )

    kv_cache = get_zeroed_static_cache_inputs(
        language_model, device=device, has_position_ids=True
    )
    start_idx = 0
    end_idx = seq_embeds.size(1)
    generated = 0
    max_total_len = end_idx + max_new_tokens
    output_tokens = seq_tokens.clone()
    step_times = []
    seq_attention_mask = (
        attention_mask.clone()
        if attention_mask is not None
        else torch.ones_like(output_tokens, dtype=torch.long)
    )

    while output_tokens.size(1) < max_total_len:
        if with_timing:
            lm_start.record()

        cur_embeds = seq_embeds if generated == 0 else seq_embeds[:, -1:, :]

        full_position_ids = get_qwen_position_ids(
            model,
            output_tokens,
            image_grid_thw=image_grid_thw,
            attention_mask=seq_attention_mask,
        )
        position_ids = full_position_ids if generated == 0 else full_position_ids[..., -1:]

        input_signature = (
            cur_embeds,
            position_ids,
            *kv_cache,
            start_idx,
            end_idx,
        )

        outputs_and_kv = language_model(*input_signature)
        hidden_states, kv_cache = outputs_and_kv[0], outputs_and_kv[1:]

        logits = model.lm_head(hidden_states[:, -1, :])
        next_tok = logits.argmax(dim=-1)
        output_tokens = torch.cat([output_tokens, next_tok[:, None]], dim=-1)

        next_embed = emb_layer(next_tok)[:, None, :]
        seq_embeds = next_embed
        seq_attention_mask = torch.cat(
            [
                seq_attention_mask,
                torch.ones(
                    seq_attention_mask.shape[0],
                    1,
                    dtype=seq_attention_mask.dtype,
                    device=seq_attention_mask.device,
                ),
            ],
            dim=1,
        )

        generated += 1
        start_idx = end_idx
        end_idx += 1

        if with_timing:
            lm_end.record()
            torch.cuda.synchronize()
            step_times.append(lm_start.elapsed_time(lm_end))

        if (next_tok == eos_token_id).all():
            break

    if with_timing:
        overall_end.record()
        torch.cuda.synchronize()
        overall_time = overall_start.elapsed_time(overall_end)
        # For Qwen, there is no separate MLP part like in Eagle, so mlp_time is 0.
        return (
            output_tokens[:, input_ids.shape[1] :],
            step_times,
            overall_time,
            vision_time,
            0.0,
        )
    else:
        return output_tokens[:, input_ids.shape[1] :]
