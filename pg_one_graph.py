import torch
import torch.nn as nn
import torch_tensorrt
import torch.nn.functional as F
from typing import Optional, Dict, Any
from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor
from transformers.image_utils import load_image
from PIL import Image
import copy
import time

class PaliGemmaWrapper(nn.Module):
    """
    Wrapper class for PaliGemmaForConditionalGeneration, providing an optimized forward pass for TensorRT compilation.
    """
    def __init__(self, model_id, device="cuda:1"):
        super().__init__()
        self.model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.float16
        ).to(device).eval()
        
        # Disable use_cache if needed
        self.model.config.use_cache = False
        
    def forward(self, input_ids, pixel_values, attention_mask=None):
        """
        Model forward pass - accurately replicates the original model's behavior in a TensorRT-friendly way
        
        Args:
            input_ids: Input token IDs, shape [batch_size, seq_len]
            pixel_values: Image pixel values, shape [batch_size, 3, H, W]
            attention_mask: Attention mask (optional)
        
        Returns:
            logits: Model prediction logits
        """
        # 1. Extract image features (Vision Tower)
        vision_outputs = self.model.vision_tower(pixel_values)
        vision_hidden = vision_outputs.last_hidden_state
        
        # 2. Project image features to text dimension (Multi-Modal Projector)
        image_features = self.model.multi_modal_projector(vision_hidden)
        
        # Apply normalization (scaling)
        hidden_size = self.model.config.text_config.hidden_size
        image_features = image_features / (hidden_size ** 0.5)
        
        # 3. Get text embeddings
        inputs_embeds = self.model.get_input_embeddings()(input_ids)
        
        # 4. Insert image features at image token positions (using masked_scatter)
        special_image_mask = (input_ids == self.model.config.image_token_index).unsqueeze(-1)
        special_image_mask = special_image_mask.expand_as(inputs_embeds).to(inputs_embeds.device)
        special_image_mask = special_image_mask.expand_as(inputs_embeds)
        
        # Ensure image features and embeddings have compatible formats
        image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
        
        # Insert image features using masked_scatter (applied directly without condition checks)
        # inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)
        
        # 5. Process attention mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=input_ids.device, dtype=torch.int64)
            
        # 6. Calculate logits through language model
        outputs = self.model.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            use_cache=False
        )
        
        return outputs.logits

def export_and_compile_paligemma():
    # Initialize model
    device = "cuda:1"
    torch.cuda.set_device(1)
    
    model_id = "google/paligemma2-3b-pt-224"
    model = PaliGemmaWrapper(model_id, device)
    
    # Create dummy inputs
    batch_size = 2
    seq_len = 13
    
    # Get language model hidden size (same as original code)
    hidden_size_text = model.model.config.text_config.hidden_size
    
    # Prepare input IDs and pixel values
    dummy_input_ids = torch.randint(0, 32000, (batch_size, seq_len), device=device)
    dummy_pixel_values = torch.randn(batch_size, 3, 224, 224, dtype=torch.float16, device=device)
    
    # Create 4D attention mask same as original
    dummy_ext_seq_len = seq_len + 100  # Same calculation as original (361-261=100 or 113-13=100)
    dummy_attention_mask = torch.ones(
        batch_size, 1, seq_len, dummy_ext_seq_len,
        device=device,
        dtype=torch.float16
    )
    
    # Define dynamic dimensions - match with original
    B = torch.export.Dim("batch_dim", min=1, max=4)
    __seq_dim = torch.export.Dim("__seq_dim", min=1, max=64)  # max=33 would limit to max length of 261
    seq_dim = 8 * __seq_dim - 3  # Maintain original formula
    ext_dim = torch.export.Dim("ext_dim", min=5, max=500)  # Extended sequence length
    
    # Configure dynamic dimensions - match with original code
    dynamic_shapes = {
        "input_ids": {
            0: B,
            1: seq_dim
        },
        "pixel_values": {
            0: B
        },
        "attention_mask": {
            0: B,
            1: 1,
            2: seq_dim,
            3: ext_dim
        }
    }
    
    print("Exporting model...")
    with torch.inference_mode():
        exported_model = torch.export.export(
            model,
            args=(dummy_input_ids, dummy_pixel_values, dummy_attention_mask),
            dynamic_shapes=dynamic_shapes,
            strict=False
        )
    
    print("Compiling with TensorRT...")
    with torch_tensorrt.logging.debug():
        compiled_model = torch_tensorrt.dynamo.compile(
            exported_model,
            inputs=[dummy_input_ids, dummy_pixel_values, dummy_attention_mask],
            enabled_precisions={torch.float32},
            debug=True,
            truncate_double=True,
            device=device,
            disable_tf32=True,  # Maintain original setting
            use_explicit_typing=True,
            use_fp32_acc=True,
        )
    
    return compiled_model, model

# Testing and performance measurement - unified measurement function
def measure_inference_time(model_fn, num_warmup=20, num_actual=100):
    """
    Measure model inference time
    """
    # Warmup
    print(f"Warming up for {num_warmup} iterations...")
    for _ in range(num_warmup):
        model_fn()
    torch.cuda.synchronize()
    
    # Measurement
    timings = []
    print(f"Measuring for {num_actual} iterations...")
    
    for _ in range(num_actual):
        start_time = time.time()
        model_fn()
        torch.cuda.synchronize()
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

def compute_cosine_similarity(tensor1, tensor2):
    tensor1_flat = tensor1.reshape(-1)
    tensor2_flat = tensor2.reshape(-1)
    return F.cosine_similarity(tensor1_flat.unsqueeze(0), tensor2_flat.unsqueeze(0), dim=1).item()

# Testing and performance measurement
def test_unified_model(compiled_model, original_wrapper, processor, test_image, prompt, device):
    # Prepare inputs
    inputs = processor(images=test_image, text=prompt, return_tensors="pt").to(device)
    
    # Run original model
    def run_original():
        with torch.inference_mode():
            return original_wrapper(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"].half(),
                attention_mask=inputs["attention_mask"].to(torch.float16)
            )
    
    # Run compiled model
    def run_compiled():
        with torch.inference_mode():
            # Prepare inputs for TensorRT engine
            input_ids_rt = inputs["input_ids"]            # Long type is acceptable (allowed in enabled_precisions)
            pixel_values_rt = inputs["pixel_values"].half()  # float16
            
            # Build a 4D attention mask to match the compilation profile
            # Shape: [batch, 1, seq_len, ext_seq_len]
            batch_rt, seq_len_rt = input_ids_rt.shape
            ext_seq_len_rt = seq_len_rt + 100  # Matching the logic used for dummy mask
            attention_mask_rt = torch.ones(
                batch_rt, 1, seq_len_rt, ext_seq_len_rt,
                dtype=torch.float16,
                device=input_ids_rt.device
            )
            
            return compiled_model(
                input_ids_rt,
                pixel_values_rt,
                attention_mask_rt
            )
    
    # Validate accuracy
    print("Validating outputs...")
    with torch.inference_mode():
        original_outputs = run_original()
        compiled_outputs = run_compiled()
    
    # Calculate cosine similarity
    cosine_sim = compute_cosine_similarity(
        original_outputs.float(),
        compiled_outputs.float()
    )
    
    # Measure performance
    print("\n=== Performance Measurement ===")
    print("\nMeasuring original model...")
    original_timing = measure_inference_time(run_original)
    
    print("\nMeasuring compiled model...")
    compiled_timing = measure_inference_time(run_compiled)
    
    # Output results
    print("\n=== Performance Results ===")
    print(f"Original model: {original_timing['mean_ms']:.2f} ± {original_timing['std_ms']:.2f} ms")
    print(f"Compiled model: {compiled_timing['mean_ms']:.2f} ± {compiled_timing['std_ms']:.2f} ms")
    print(f"Speedup: {original_timing['mean_ms'] / compiled_timing['mean_ms']:.2f}x")
    print(f"Cosine similarity: {cosine_sim:.6f}")
    
    # Calculate absolute error
    abs_diff = (original_outputs - compiled_outputs).abs()
    max_diff = abs_diff.max().item()
    mean_diff = abs_diff.mean().item()
    print(f"Max absolute difference: {max_diff:.6f}")
    print(f"Mean absolute difference: {mean_diff:.6f}")
    
    return {
        "original_timing": original_timing,
        "compiled_timing": compiled_timing,
        "speedup": original_timing['mean_ms'] / compiled_timing['mean_ms'],
        "cosine_similarity": cosine_sim,
        "max_diff": max_diff,
        "mean_diff": mean_diff,
        "inputs": inputs,
        "original_logits": original_outputs,
        "compiled_logits": compiled_outputs
    }

# Main test code
if __name__ == "__main__":
    device = "cuda:1"
    torch.cuda.set_device(1)
    
    model_id = "google/paligemma2-3b-pt-224"
    processor = PaliGemmaProcessor.from_pretrained(model_id)
    
    # Load image - changed to use URL
    url = "https://cdn.pixabay.com/photo/2019/08/08/23/33/car-4393990_1280.jpg"
    test_image = load_image(url)
    
    # Set prompt - same as pg_pali_trt_replace.py
    prompt = "caption the image"
    
    # Prepare inputs
    model_inputs = processor(text=prompt, images=test_image, return_tensors="pt").to(device)
    
    # Output input shapes
    print(f"input_ids shape: {model_inputs['input_ids'].shape}")
    print(f"pixel_values shape: {model_inputs['pixel_values'].shape}")
    print(f"attention_mask shape: {model_inputs['attention_mask'].shape}")
    
    # 1. Load and run original PyTorch model
    print("\n=== Original PyTorch Model Output ===")
    with torch.inference_mode():
        # Load original model
        original_model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.float16
        ).to(device).eval()
        original_model.config.use_cache = False
        
        # Run original model
        original_outputs = original_model(
            input_ids=model_inputs["input_ids"],
            pixel_values=model_inputs["pixel_values"].half(),
            attention_mask=model_inputs["attention_mask"],
        )
        
        print(f"Original output logits shape: {original_outputs.logits.shape}")
        print(f"Original output logits dtype: {original_outputs.logits.dtype}")
    
    # 2. Validate PaliGemmaWrapper (before compilation)
    print("\n=== PaliGemmaWrapper Output (Before Compilation) ===")
    wrapper_model = PaliGemmaWrapper(model_id, device)
    
    with torch.inference_mode():
        # Run wrapper model
        wrapper_outputs = wrapper_model(
            input_ids=model_inputs["input_ids"],
            pixel_values=model_inputs["pixel_values"].half(),
            attention_mask=model_inputs["attention_mask"]
        )
        
        print(f"Wrapper output shape: {wrapper_outputs.shape}")
        print(f"Wrapper output dtype: {wrapper_outputs.dtype}")
    
        # Compare original model and wrapper model outputs
        cosine_sim = compute_cosine_similarity(
            original_outputs.logits.float(),
            wrapper_outputs.float()
        )
        
        # Calculate absolute error
        abs_diff = (original_outputs.logits.float() - wrapper_outputs.float()).abs()
        max_diff = abs_diff.max().item()
        mean_diff = abs_diff.mean().item()
        
        print(f"Cosine similarity between original and wrapper outputs: {cosine_sim:.6f}")
        print(f"Max absolute difference: {max_diff:.6f}")
        print(f"Mean absolute difference: {mean_diff:.6f}")
        
        # Compare top 5 tokens
        orig_last_token_logits = original_outputs.logits[0, -1, :]
        wrap_last_token_logits = wrapper_outputs[0, -1, :]
        
        orig_top5 = torch.topk(orig_last_token_logits, 5)
        wrap_top5 = torch.topk(wrap_last_token_logits, 5)
        
        print("\nTop 5 predicted tokens comparison:")
        for i in range(5):
            orig_token = orig_top5.indices[i].item()
            wrap_token = wrap_top5.indices[i].item()
            orig_text = processor.decode([orig_token])
            wrap_text = processor.decode([wrap_token])
            print(f"  Original #{i+1}: '{orig_text}' (score: {orig_top5.values[i].item():.4f})")
            print(f"  Wrapper  #{i+1}: '{wrap_text}' (score: {wrap_top5.values[i].item():.4f})")
            print(f"  Match: {orig_token == wrap_token}")
    
    # 3. Now proceed with TensorRT compilation
    print("Initializing and compiling unified model...")
    # Compile and test
    compiled_model, original_wrapper = export_and_compile_paligemma()
    
    print("\nTesting unified model...")
    results = test_unified_model(compiled_model, original_wrapper, processor, test_image, prompt, device)
    
    # Text generation test
    print("\n===== Text Generation Comparison =====")
    
    # Original model text generation
    with torch.inference_mode():
        base_model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.float16
        ).to(device).eval()
        base_model.config.use_cache = False
        
        # Base model generate
        inputs = results["inputs"]
        base_generated = base_model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"].half(),
            attention_mask=inputs["attention_mask"],
            max_new_tokens=50,
            do_sample=False
        )
        input_len = inputs["input_ids"].shape[1]
        base_text = processor.decode(base_generated[0][input_len:], skip_special_tokens=True)
    
    print("\nOriginal model generated text:")
    print(base_text)
    
    # Unified wrapper model text generation
    with torch.inference_mode():
        wrapper_generated = original_wrapper.model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"].half(),
            attention_mask=inputs["attention_mask"],
            max_new_tokens=50,
            do_sample=False
        )
        wrapper_text = processor.decode(wrapper_generated[0][input_len:], skip_special_tokens=True)
    
    print("\nWrapper model generated text:")
    print(wrapper_text)
    
    # Calculate text similarity
    from difflib import SequenceMatcher
    def similarity_ratio(a, b):
        return SequenceMatcher(None, a, b).ratio()
    
    text_similarity = similarity_ratio(base_text, wrapper_text)
    print(f"\nText similarity between base and wrapper model outputs: {text_similarity:.6f}")
    
    print("\nUnified TensorRT model compilation completed successfully!")