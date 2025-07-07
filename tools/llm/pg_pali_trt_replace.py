import torch
import torch.nn as nn
import torch_tensorrt
import torch.nn.functional as F
from typing import Optional
from transformers import PaliGemmaForConditionalGeneration
from transformers.image_utils import load_image
import copy
import time
from torchtrt_ext import register_sdpa


if __name__ == "__main__":
    # Specify the GPU to use
    device = "cuda:1"  # Use GPU 1
    torch.cuda.set_device(1)

    ####################################################################
    # 1) Load the base model
    ####################################################################
    base_model_id = "google/paligemma2-3b-pt-224"
    base_model = PaliGemmaForConditionalGeneration.from_pretrained(
        base_model_id, torch_dtype=torch.float16
    ).to(device).eval()

    # Original sub-modules
    vision_tower_orig = base_model.vision_tower
    multi_modal_orig = base_model.multi_modal_projector
    language_model_orig = base_model.language_model

    ####################################################################
    # 2) Torch-TensorRT compile: Vision Tower
    ####################################################################
    # Dummy inputs and dynamic shapes (simply setting batch dimension as dynamic)
    dummy_pixel_values = torch.randn(2, 3, 224, 224, dtype=torch.float16, device=device)

    B = torch.export.Dim("batch_dim", min=1, max=4)
    dynamic_shapes_vision = {
        "pixel_values": {0: B}
    }

    with torch.inference_mode():
        exported_vision = torch.export.export(
            vision_tower_orig,
            args=(),
            kwargs={"pixel_values": dummy_pixel_values},
            dynamic_shapes=dynamic_shapes_vision,
            strict=False
        )

    # with torch_tensorrt.logging.debug():
    with torch_tensorrt.logging.debug():
        trt_vision_tower = torch_tensorrt.dynamo.compile(
            exported_vision,
            inputs=[dummy_pixel_values],
            enabled_precisions={torch.float32},
            truncate_double=True,
            device=device,
            disable_tf32=True,
            use_explicit_typing=True,
            use_fp32_acc=True,
        )

    ####################################################################
    # 3) Torch-TensorRT compile: Multi-Modal Projector
    ####################################################################
    # Vision tower output last_hidden_state -> [batch, seq_len, vision_config.hidden_size]
    seq_len = 196  # Standard vision encoder output sequence length
    hidden_size_vision = base_model.config.vision_config.hidden_size
    dummy_image_hidden = torch.randn(
        2, seq_len, hidden_size_vision,
        dtype=torch.float16, device=device
    )

    B2 = torch.export.Dim("batch_dim2", min=1, max=4)
    S2 = torch.export.Dim("seq_dim2", min=1, max=512)
    dynamic_shapes_proj = {
        "image_features": {
            0: B2,
            1: S2
        }
    }

    with torch.inference_mode():
        exported_proj = torch.export.export(
            multi_modal_orig,
            args=(),
            kwargs={"image_features": dummy_image_hidden},
            dynamic_shapes=dynamic_shapes_proj,
            strict=False
        )

    # with torch_tensorrt.logging.debug():
    with torch_tensorrt.logging.debug():
        trt_multi_modal_proj = torch_tensorrt.dynamo.compile(
            exported_proj,
            inputs=[dummy_image_hidden],
            enabled_precisions={torch.float32},
            truncate_double=True,
            device=device,
            disable_tf32=True,
            use_explicit_typing=True,
            use_fp32_acc=True,
        )

    ####################################################################
    # 4) Torch-TensorRT compile: Language Model
    ####################################################################
    # Prepare dummy inputs
    hidden_size_text = base_model.config.text_config.hidden_size

    # Set seq_len in the form of (8*N-3) - matching the actual model behavior
    B = torch.export.Dim("batch_dim", min=1, max=4)
    __seq_dim = torch.export.Dim("__seq_dim", min=1, max=64)  # (max=33: covers up to length 261)
    seq_dim = 8 * __seq_dim - 3
    ext_dim = torch.export.Dim("ext_dim", min=5, max=500)  # Extended sequence length

    # Compile with 4D attention mask
    dummy_seq_len = 13
    dummy_ext_seq_len = 113  # Extended length to match difference (actually 361 - 261 = 100 difference)

    dummy_embeds = torch.randn(
        (2, dummy_seq_len, hidden_size_text),
        device=device,
        dtype=torch.float16
    )
    dummy_attention_mask = torch.ones(
        2, 1, dummy_seq_len, dummy_ext_seq_len,
        device=device,
        dtype=torch.float16  # Note: actual type is float16
    )

    # Define dynamic shapes
    dynamic_shapes_lm = {
        "inputs_embeds": {
            0: B,
            1: seq_dim
        },
        "attention_mask": {
            0: B,
            1: 1,
            2: seq_dim,
            3: ext_dim  # Extended length range
        }
    }

    # LMNoCache wrapper class modified to handle 4D masks
    class LMNoCache(nn.Module):
        def __init__(self, lm: nn.Module):
            super().__init__()
            self.lm = lm

        def forward(self, inputs_embeds: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
            return self.lm(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,  # Pass 4D mask directly
                use_cache=False
            )

    wrapped_lm = LMNoCache(language_model_orig).to(device).eval()

    with torch.inference_mode():
        exported_program = torch.export.export(
            wrapped_lm,
            args=(dummy_embeds, dummy_attention_mask),
            dynamic_shapes=dynamic_shapes_lm,
            strict=False
        )

    # Only perform compilation
    # with torch_tensorrt.logging.debug():
    with torch_tensorrt.logging.debug():
        trt_language_model = torch_tensorrt.dynamo.compile(
            exported_program,
            inputs=[dummy_embeds, dummy_attention_mask],
            enabled_precisions={torch.float32},
            truncate_double=True,
            device=device,
            disable_tf32=True,
            use_explicit_typing=True,
            use_fp32_acc=True,
        )

    ####################################################################
    # 5) Replace base_model modules with TRT optimized versions
    ####################################################################
    # Clone the original model
    base_model.config.use_cache = False
    trt_integrated_model = copy.deepcopy(base_model)
    
    # Replace with TRT optimized modules
    trt_integrated_model.vision_tower = trt_vision_tower
    trt_integrated_model.multi_modal_projector = trt_multi_modal_proj
    
    # Define TRTLanguageModelWrapper (with data type conversion logic already added)
    class TRTLanguageModelWrapper(nn.Module):
        def __init__(self, trt_lm, original_language_model):
            super().__init__()
            self.trt_lm = trt_lm
            self.original_language_model = original_language_model
            self.original_embeddings = original_language_model.get_input_embeddings()
            
        def forward(self, **kwargs):
            inputs_embeds = kwargs.get('inputs_embeds')
            attention_mask = kwargs.get('attention_mask')
            
            # # Debugging logs
            # print(f"Final inputs_embeds shape: {inputs_embeds.shape}")
            # print(f"Final attention_mask shape: {attention_mask.shape}")
            # print(f"Final attention_mask type: {attention_mask.dtype}")
                
            return self.trt_lm(inputs_embeds, attention_mask)
        
        def get_input_embeddings(self):
            return self.original_embeddings
            
        def prepare_inputs_for_generation(self, *args, **kwargs):
            # Call the prepare_inputs_for_generation method of the original LM
            return self.original_language_model.prepare_inputs_for_generation(*args, **kwargs)

    # Use the wrapper
    trt_integrated_model.language_model = TRTLanguageModelWrapper(
        trt_language_model, 
        language_model_orig
    )
    
    ####################################################################
    # 6) Test and compare with actual sample inputs
    ####################################################################
    from transformers import PaliGemmaProcessor

    # Set up test image and prompt
    url = "https://cdn.pixabay.com/photo/2019/08/08/23/33/car-4393990_1280.jpg"
    image = load_image(url)
    processor = PaliGemmaProcessor.from_pretrained(base_model_id)
    prompt = "caption the image"

    model_inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
    
    # Print input shapes
    print(f"input_ids shape: {model_inputs['input_ids'].shape}")
    print(f"pixel_values shape: {model_inputs['pixel_values'].shape}")

    # Define inference time measurement function
    def measure_inference_time(model_fn, num_warmup=20, num_actual=100):
        """
        Measure model inference time
        """
        # Warmup - synchronization only needed once at the end
        print(f"Warming up for {num_warmup} iterations...")
        for _ in range(num_warmup):
            model_fn()
        torch.cuda.synchronize()  # Synchronize only once after warmup is complete
        
        # Measurement
        timings = []
        print(f"Measuring for {num_actual} iterations...")
        
        for _ in range(num_actual):
            start_time = time.time()
            model_fn()
            torch.cuda.synchronize()  # Wait for GPU operations to complete
            end_time = time.time()
            timings.append((end_time - start_time) * 1000)  # Convert to ms
        
        # Calculate statistics
        timings = torch.tensor(timings)
        mean_time = torch.mean(timings).item()
        std_time = torch.std(timings).item()
        min_time = torch.min(timings).item()
        max_time = torch.max(timings).item()
        
        return {
            "mean_ms": mean_time,
            "std_ms": std_time,
            "min_ms": min_time,
            "max_ms": max_time,
            "raw_timings": timings.tolist()
        }

    # PyTorch base model inference function
    def run_base_model():
        with torch.inference_mode():
            outputs = base_model(
                input_ids=model_inputs["input_ids"],
                pixel_values=model_inputs["pixel_values"].half(),
                attention_mask=model_inputs["attention_mask"],
            )
        return outputs

    # TensorRT optimized model inference function
    def run_trt_model():
        with torch.inference_mode():
            outputs = trt_integrated_model(
                input_ids=model_inputs["input_ids"],
                pixel_values=model_inputs["pixel_values"].half(),
                attention_mask=model_inputs["attention_mask"],
            )
        return outputs

    # Run performance measurement
    print("\n=== Performance Measurement ===")

    print("\nMeasuring TensorRT optimized model...")
    trt_timing = measure_inference_time(run_trt_model)

    print("Measuring base PyTorch model...")
    base_timing = measure_inference_time(run_base_model)


    # Print results
    print("\n=== Performance Results ===")
    print(f"Base PyTorch model: {base_timing['mean_ms']:.2f} ± {base_timing['std_ms']:.2f} ms")
    print(f"Raw Base model timings (ms): {base_timing['raw_timings']}")
    print(f"TensorRT model: {trt_timing['mean_ms']:.2f} ± {trt_timing['std_ms']:.2f} ms")
    print(f"Raw TRT timings (ms): {trt_timing['raw_timings']}")
    print(f"Speedup: {base_timing['mean_ms'] / trt_timing['mean_ms']:.2f}x")

    # PyTorch base model results
    with torch.inference_mode():
        base_outputs = base_model(
            input_ids=model_inputs["input_ids"],
            pixel_values=model_inputs["pixel_values"].half(),
            attention_mask=model_inputs["attention_mask"],
        )
    
    # TensorRT optimized model results
    with torch.inference_mode():
        trt_outputs = trt_integrated_model(
            input_ids=model_inputs["input_ids"],
            pixel_values=model_inputs["pixel_values"].half(),
            attention_mask=model_inputs["attention_mask"],
        )
    
    baseline_fp32 = base_outputs.logits.float()
    trt_fp32 = trt_outputs.logits.float()

    # Cosine similarity calculation function
    def compute_cosine_similarity(tensor1, tensor2):
        tensor1_flat = tensor1.reshape(-1)
        tensor2_flat = tensor2.reshape(-1)
        return F.cosine_similarity(tensor1_flat.unsqueeze(0), tensor2_flat.unsqueeze(0), dim=1).item()
    
    # Calculate cosine similarity for logits
    cosine_sim_fp16 = compute_cosine_similarity(base_outputs.logits, trt_outputs.logits)
    cosine_sim_fp32 = compute_cosine_similarity(baseline_fp32, trt_fp32)
    print(f"Cosine similarity between base and TRT model logits fp16: {cosine_sim_fp16:.6f}")
    print(f"Cosine similarity between base and TRT model logits fp32: {cosine_sim_fp32:.6f}")
    
    # Calculate absolute error
    abs_diff = (base_outputs.logits - trt_outputs.logits).abs()
    max_diff = abs_diff.max().item()
    mean_diff = abs_diff.mean().item()
    print(f"Max absolute difference: {max_diff:.6f}")
    print(f"Mean absolute difference: {mean_diff:.6f}")

    ####################################################################
    # 7) Text generation and comparison (using generate instead of argmax)
    ####################################################################

    def compare_token_by_token_generation(base_model, trt_model, model_inputs, processor, max_tokens=5):
        """Compare generation process token by token, using the generate function"""
        
        # Basic settings and input preparation
        input_ids = model_inputs["input_ids"].clone()
        pixel_values = model_inputs["pixel_values"].half()
        attention_mask = model_inputs["attention_mask"].clone()
        input_len = input_ids.shape[-1]
        
        # Compare initial logits
        with torch.inference_mode():
            base_outputs = base_model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                return_dict=True
            )
            
            trt_outputs = trt_model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                return_dict=True
            )
        
        base_logits = base_outputs.logits[:, -1, :].float()
        trt_logits = trt_outputs.logits[:, -1, :].float()
        
        # Calculate cosine similarity
        cos_sim = F.cosine_similarity(base_logits.flatten(), trt_logits.flatten(), dim=0)
        
        print(f"\n===== Initial Token Comparison =====")
        print(f"Logits cosine similarity: {cos_sim.item():.6f}")
        
        # Calculate probability distribution differences
        base_probs = F.softmax(base_logits, dim=-1)
        trt_probs = F.softmax(trt_logits, dim=-1)
        prob_diff = (base_probs - trt_probs).abs()
        max_diff = prob_diff.max().item()
        print(f"Maximum probability difference: {max_diff:.6f}")
        
        # Compare token generation step by step
        base_curr_ids = input_ids.clone()
        trt_curr_ids = input_ids.clone()
        
        for i in range(max_tokens):
            print(f"\n===== Generating Token {i+1} =====")
            
            # Base model: generate only one token
            with torch.inference_mode():
                base_output = base_model.generate(
                    input_ids=base_curr_ids,
                    pixel_values=pixel_values,
                    attention_mask=torch.ones_like(base_curr_ids),  # Adjust if necessary
                    max_new_tokens=1,
                    do_sample=False,
                    return_dict_in_generate=True,
                    output_scores=True
                )
                base_next_token = base_output.sequences[0, -1].unsqueeze(0)
                base_curr_ids = base_output.sequences
                
                # Extract logits/scores from the Base model
                base_token_score = base_output.scores[0].float()  # Logits for the last token
            
            # TRT model: generate only one token
            with torch.inference_mode():
                trt_output = trt_model.generate(
                    input_ids=trt_curr_ids,
                    pixel_values=pixel_values,
                    attention_mask=torch.ones_like(trt_curr_ids),  # Adjust if necessary
                    max_new_tokens=1,
                    do_sample=False,
                    return_dict_in_generate=True,
                    output_scores=True
                )
                trt_next_token = trt_output.sequences[0, -1].unsqueeze(0)
                trt_curr_ids = trt_output.sequences
                
                # Extract logits/scores from the TRT model
                trt_token_score = trt_output.scores[0].float()  # Logits for the last token
            
            # Compare and output results
            base_token_text = processor.decode([base_next_token.item()])
            trt_token_text = processor.decode([trt_next_token.item()])
            
            print(f"Base model token: {base_token_text} (ID: {base_next_token.item()})")
            print(f"TRT model token: {trt_token_text} (ID: {trt_next_token.item()})")
            print(f"Tokens match: {(base_next_token == trt_next_token).item()}")
            
            # Compare logit similarities
            cos_sim = F.cosine_similarity(
                base_token_score.flatten(), 
                trt_token_score.flatten(),
                dim=0
            )
            print(f"Logits cosine similarity: {cos_sim.item():.6f}")
            
            # Detailed analysis for token mismatches
            if base_next_token != trt_next_token:
                base_top5 = base_token_score.topk(5, dim=-1)
                trt_top5 = trt_token_score.topk(5, dim=-1)
                
                print("Base model top-5:", [processor.decode([t.item()]) for t in base_top5.indices[0]])
                print("TRT model top-5:", [processor.decode([t.item()]) for t in trt_top5.indices[0]])
                
                # Compare probability values
                base_probs = F.softmax(base_token_score, dim=-1)
                trt_probs = F.softmax(trt_token_score, dim=-1)
                
                # Check tokens with large probability differences
                for b_idx, t_idx in zip(base_top5.indices[0], trt_top5.indices[0]):
                    b_idx, t_idx = b_idx.item(), t_idx.item()
                    base_prob = base_probs[0, b_idx].item()
                    trt_prob = trt_probs[0, t_idx].item()
                    
                    print(f"  Top base token '{processor.decode([b_idx])}': Base={base_prob:.6f}")
                    print(f"  Top TRT token '{processor.decode([t_idx])}': TRT={trt_prob:.6f}")
        
        # Final generation results
        base_text = processor.decode(base_curr_ids[0, input_len:], skip_special_tokens=True)
        trt_text = processor.decode(trt_curr_ids[0, input_len:], skip_special_tokens=True)
        
        print("\n===== Final Generated Sequences =====")
        print(f"Base model: {base_text}")
        print(f"TRT model: {trt_text}")
        
        return base_curr_ids, trt_curr_ids

    compare_token_by_token_generation(base_model, trt_integrated_model, model_inputs, processor)


    ####################################################################
    # Module-by-module performance measurement
    ####################################################################
    def measure_modules_performance():
        print("\n=== Module-by-Module Performance Comparison ===")
        
        # 1. Measure Vision Tower performance
        def run_vision_tower_base():
            with torch.inference_mode():
                return vision_tower_orig(pixel_values=model_inputs["pixel_values"].half())
        
        def run_vision_tower_trt():
            with torch.inference_mode():
                return trt_vision_tower(model_inputs["pixel_values"].half())
        
        print("\n--- Vision Tower Performance ---")
        print("Measuring base vision tower...")
        vision_base_timing = measure_inference_time(run_vision_tower_base)
        
        print("\nMeasuring TensorRT vision tower...")
        vision_trt_timing = measure_inference_time(run_vision_tower_trt)
        
        vision_speedup = vision_base_timing['mean_ms'] / vision_trt_timing['mean_ms']
        print(f"Vision Tower - Base: {vision_base_timing['mean_ms']:.2f} ± {vision_base_timing['std_ms']:.2f} ms")
        print(f"Vision Tower - TRT: {vision_trt_timing['mean_ms']:.2f} ± {vision_trt_timing['std_ms']:.2f} ms")
        print(f"Vision Tower - Speedup: {vision_speedup:.2f}x")
        
        # 2. Cache Vision Tower output (for next stage input)
        with torch.inference_mode():
            vision_base_output = vision_tower_orig(pixel_values=model_inputs["pixel_values"].half())
            vision_trt_output = trt_vision_tower(model_inputs["pixel_values"].half())
            
        # 3. Measure Multi-Modal Projector performance
        def run_mm_projector_base():
            with torch.inference_mode():
                return multi_modal_orig(image_features=vision_base_output.last_hidden_state)
        
        def run_mm_projector_trt():
            with torch.inference_mode():
                return trt_multi_modal_proj(vision_trt_output.last_hidden_state)
        
        print("\n--- Multi-Modal Projector Performance ---")
        print("Measuring base multi-modal projector...")
        mm_base_timing = measure_inference_time(run_mm_projector_base)
        
        print("\nMeasuring TensorRT multi-modal projector...")
        mm_trt_timing = measure_inference_time(run_mm_projector_trt)
        
        mm_speedup = mm_base_timing['mean_ms'] / mm_trt_timing['mean_ms']
        print(f"Multi-Modal Projector - Base: {mm_base_timing['mean_ms']:.2f} ± {mm_base_timing['std_ms']:.2f} ms")
        print(f"Multi-Modal Projector - TRT: {mm_trt_timing['mean_ms']:.2f} ± {mm_trt_timing['std_ms']:.2f} ms")
        print(f"Multi-Modal Projector - Speedup: {mm_speedup:.2f}x")
        
        # 4. Cache Multi-Modal Projector output
        with torch.inference_mode():
            mm_base_output = multi_modal_orig(image_features=vision_base_output.last_hidden_state)
            mm_trt_output = trt_multi_modal_proj(vision_trt_output.last_hidden_state)
        
        # 5. Measure Language Model performance - prepare inputs
        # Tokenized text input
        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs["attention_mask"]
        
        # Calculate text embeddings
        with torch.inference_mode():
            input_embeds = base_model.language_model.get_input_embeddings()(input_ids)
        
        # 5.1 Language Model only measurement (starting from input embeddings)
        def run_language_model_base():
            with torch.inference_mode():
                # Combining image features with inputs_embeds is handled inside the actual model
                return language_model_orig(
                    inputs_embeds=input_embeds,
                    attention_mask=attention_mask
                )
        
        def run_language_model_trt():
            with torch.inference_mode():
                # 1. Convert mask data type
                fp16_mask = attention_mask.to(dtype=torch.float16)
                
                # 2. Convert mask dimensions: 2D -> 4D
                batch_size, seq_len = fp16_mask.shape
                ext_seq_len = seq_len + 100  # Add the same difference as during dummy compilation to the original sequence length
                
                # Create 4D mask
                mask_4d = torch.ones(
                    batch_size, 1, seq_len, ext_seq_len,
                    dtype=torch.float16, device=input_embeds.device
                )
                
                # 3. Pass inputs to TensorRT
                return trt_language_model(
                    input_embeds.to(dtype=torch.float16),  # Explicitly float16
                    mask_4d
                )
        
        print("\n--- Language Model Performance ---")
        print("Measuring base language model...")
        lm_base_timing = measure_inference_time(run_language_model_base)
        
        print("\nMeasuring TensorRT language model...")
        lm_trt_timing = measure_inference_time(run_language_model_trt)
        
        lm_speedup = lm_base_timing['mean_ms'] / lm_trt_timing['mean_ms']
        print(f"Language Model - Base: {lm_base_timing['mean_ms']:.2f} ± {lm_base_timing['std_ms']:.2f} ms")
        print(f"Language Model - TRT: {lm_trt_timing['mean_ms']:.2f} ± {lm_trt_timing['std_ms']:.2f} ms")
        print(f"Language Model - Speedup: {lm_speedup:.2f}x")
        
        # 6. Overall results summary
        print("\n=== Module Performance Summary ===")
        print(f"Vision Tower:        Base={vision_base_timing['mean_ms']:.2f}ms, TRT={vision_trt_timing['mean_ms']:.2f}ms, Speedup={vision_speedup:.2f}x")
        print(f"Multi-Modal Proj:    Base={mm_base_timing['mean_ms']:.2f}ms, TRT={mm_trt_timing['mean_ms']:.2f}ms, Speedup={mm_speedup:.2f}x")
        print(f"Language Model:      Base={lm_base_timing['mean_ms']:.2f}ms, TRT={lm_trt_timing['mean_ms']:.2f}ms, Speedup={lm_speedup:.2f}x")
        print(f"Total - Component:   Base={vision_base_timing['mean_ms'] + mm_base_timing['mean_ms'] + lm_base_timing['mean_ms']:.2f}ms, TRT={vision_trt_timing['mean_ms'] + mm_trt_timing['mean_ms'] + lm_trt_timing['mean_ms']:.2f}ms")
        print(f"Total - End-to-End:  Base={base_timing['mean_ms']:.2f}ms, TRT={trt_timing['mean_ms']:.2f}ms, Speedup={base_timing['mean_ms'] / trt_timing['mean_ms']:.2f}x")
        print(f"Overhead:            Base={base_timing['mean_ms'] - (vision_base_timing['mean_ms'] + mm_base_timing['mean_ms'] + lm_base_timing['mean_ms']):.2f}ms, TRT={trt_timing['mean_ms'] - (vision_trt_timing['mean_ms'] + mm_trt_timing['mean_ms'] + lm_trt_timing['mean_ms']):.2f}ms")

    measure_modules_performance()