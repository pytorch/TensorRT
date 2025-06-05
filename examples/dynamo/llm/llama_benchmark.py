import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import timeit

USE_CACHE = False
# MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
MODEL_NAME = "Qwen/Qwen3-0.6B"
MAX_NEW_TOKENS = 128


def main():
    # Initialize model and tokenizer
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        use_cache=False,
        device_map="auto"
    )
    # model.generation_config.cache_implementation = "static"
    # model.forward = torch.compile(model.forward)
    
    # Prepare input prompt
    word = "What"
    # Tokenize the word
    word_ids = tokenizer(word, return_tensors="pt").input_ids[0]  # Get the first (and only) sequence
    # Repeat the token 2048 times
    input_ids = word_ids.repeat(1024).unsqueeze(0).to(model.device)  # Add batch dimension and move to device
    print(f"Input tensor shape: {input_ids.shape}")

    # # Warm-up pass
    print("Running warm-up pass...")
    output_ids = model.generate(
        input_ids,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        use_cache=USE_CACHE
    )
    
    # Benchmark loop
    print("Running benchmark...")
    num_iterations = 10
    total_time = 0
    timings = []
    
    for i in range(num_iterations):
        start_time = timeit.default_timer()
        output_ids = model.generate(
            input_ids,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=USE_CACHE
        )
        end_time = timeit.default_timer()
        generation_time = end_time - start_time
        total_time += generation_time
        timings.append(generation_time)
        
        # Decode and print first iteration output
        # if i == 0:
        #     output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        #     print("\nFirst generation output:")
        #     print(output_text)
    
    # Calculate and print statistics
    average_time = total_time / num_iterations
    print(f"\nPerformance Statistics:")
    print(f"Average generation time over {num_iterations} iterations: {average_time*1000:.2f} milliseconds")
    print(f"Average tokens per second: {100/average_time:.2f}")
    print("\nIndividual timings (ms):")
    for i, t in enumerate(timings):
        print(f"Iteration {i+1}: {t*1000:.2f}")

if __name__ == "__main__":
    main() 