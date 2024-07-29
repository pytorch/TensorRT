#!/bin/bash

batch_sizes=(1 16 64 256)
backends=("torch" "dynamo" "inductor")

# GPT2 model context length = 1024
echo "Benchmarking GPT2 model"
isl=(128 256)
osl=(256 512)
for bs in ${batch_sizes[@]}
  do
    for backend in ${backends[@]}
        do
        for i in ${!isl[@]};
          do
          echo "Benchmarking GPT2 model for bs ${bs} with ISL ${isl[i]}, OSL ${osl[i]} and backend ${backend}"
          python perf_run.py --model_torch "gpt2" \
                            --is_text_llm \
                            --precision fp16 \
                            --inputs "(${bs}, ${isl[i]})@int64" \
                            --output_sequence_length ${osl[i]} \
                            --batch_size ${bs} \
                            --truncate \
                            --backends ${backend} \
                            --report "gpt2_perf_bs${bs}_backend_${backend}_isl${isl[i]}_osl${osl[i]}.csv"
          done
        done
  done
# Clear HF cache
rm -rf ~/.cache/huggingface/hub/


# "meta-llama/Llama-2-7b-chat-hf" model, context length = 4K
echo "Benchmarking Llama2-7B model"
isl=(128 256)
osl=(256 512)
for bs in ${batch_sizes[@]}
  do
    for backend in ${backends[@]}
        do
        for i in ${!isl[@]};
          do
          echo "Benchmarking Llama2-7B model for bs ${bs} with ISL ${isl[i]}, OSL ${osl[i]} and backend ${backend}"
          python perf_run.py --model_torch "meta-llama/Llama-2-7b-chat-hf" \
                            --is_text_llm \
                            --precision fp16 \
                            --inputs "(${bs}, ${isl[i]})@int64" \
                            --output_sequence_length ${osl[i]} \
                            --batch_size ${bs} \
                            --truncate \
                            --backends ${backend} \
                            --report "llama2_7b_perf_bs${bs}_backend_${backend}_isl${isl[i]}_osl${osl[i]}.csv"
          done
        done
  done
# Clear HF cache
rm -rf ~/.cache/huggingface/hub/

# "meta-llama/Meta-Llama-3-8B" model, context length = 8K
echo "Benchmarking Llama3 model"
isl=(128 256)
osl=(256 512)
for bs in ${batch_sizes[@]}
  do
    for backend in ${backends[@]}
        do
        for i in ${!isl[@]};
          do
          echo "Benchmarking Llama3-8B model for bs ${bs} with ISL ${isl[i]}, OSL ${osl[i]} and backend ${backend}"
          python perf_run.py --model_torch "meta-llama/Meta-Llama-3-8B" \
                            --is_text_llm \
                            --precision fp16 \
                            --inputs "(${bs}, ${isl[i]})@int64" \
                            --output_sequence_length ${osl[i]} \
                            --batch_size ${bs} \
                            --truncate \
                            --backends ${backend} \
                            --report "llama3_8b_perf_bs${bs}_backend_${backend}_isl${isl[i]}_osl${osl[i]}.csv"
          done
        done
  done
# Clear HF cache
rm -rf ~/.cache/huggingface/hub/

# "meta-llama/Meta-Llama-3.1-8B-Instruct" model, context length = 128K
echo "Benchmarking Llama3.1 model"
isl=(128 256)
osl=(256 512)
for bs in ${batch_sizes[@]}
  do
    for backend in ${backends[@]}
        do
        for i in ${!isl[@]};
          do
          echo "Benchmarking Llama-3.1-8B-Instruct model for bs ${bs} with ISL ${isl[i]}, OSL ${osl[i]} and backend ${backend}"
          python perf_run.py --model_torch "meta-llama/Meta-Llama-3.1-8B-Instruct" \
                            --is_text_llm \
                            --precision fp16 \
                            --inputs "(${bs}, ${isl[i]})@int64" \
                            --output_sequence_length ${osl[i]} \
                            --batch_size ${bs} \
                            --truncate \
                            --backends ${backend} \
                            --report "llama3.1_8b_instruct_perf_bs${bs}_backend_${backend}_isl${isl[i]}_osl${osl[i]}.csv"
          done
        done
  done
# Clear HF cache
rm -rf ~/.cache/huggingface/hub/

# "apple/DCLM-7B" model, context length = 2K
echo "Benchmarking apple/DCLM-7B model"
isl=(128 256)
osl=(256 512)
for bs in ${batch_sizes[@]}
  do
    for backend in ${backends[@]}
        do
        for i in ${!isl[@]};
          do
          echo "Benchmarking apple/DCLM-7B model for bs ${bs} with ISL ${isl[i]}, OSL ${osl[i]} and backend ${backend}"
          python perf_run.py --model_torch "apple/DCLM-7B" \
                            --is_text_llm \
                            --precision fp16 \
                            --inputs "(${bs}, ${isl[i]})@int64" \
                            --output_sequence_length ${osl[i]} \
                            --batch_size ${bs} \
                            --truncate \
                            --backends ${backend} \
                            --report "apple_dclm_7b_perf_bs${bs}_backend_${backend}_isl${isl[i]}_osl${osl[i]}.csv"
          done
        done
  done
# Clear HF cache
rm -rf ~/.cache/huggingface/hub/

# "mistralai/Mistral-7B-Instruct-v0.3" model, context length 32k
echo "Benchmarking mistralai/Mistral-7B-Instruct-v0.3 model"
isl=(128 256)
osl=(256 512)
for bs in ${batch_sizes[@]}
  do
    for backend in ${backends[@]}
        do
        for i in ${!isl[@]};
          do
          echo "Benchmarking mistralai/Mistral-7B-Instruct-v0.3 model for bs ${bs} with ISL ${isl[i]}, OSL ${osl[i]} and backend ${backend}"
          python perf_run.py --model_torch "mistralai/Mistral-7B-Instruct-v0.3" \
                            --is_text_llm \
                            --precision fp16 \
                            --inputs "(${bs}, ${isl[i]})@int64" \
                            --output_sequence_length ${osl[i]} \
                            --batch_size ${bs} \
                            --truncate \
                            --backends ${backend} \
                            --report "mistralai_7b_v0.3_perf_bs${bs}_backend_${backend}_isl${isl[i]}_osl${osl[i]}.csv"
          done
        done
  done
# Clear HF cache
rm -rf ~/.cache/huggingface/hub/

# "microsoft/Phi-3-mini-4k-instruct" model, context length = 4K
echo "Benchmarking microsoft/Phi-3-mini-4k-instruct model"
isl=(128 256)
osl=(256 512)
for bs in ${batch_sizes[@]}
  do
    for backend in ${backends[@]}
        do
        for i in ${!isl[@]};
          do
          echo "Benchmarking microsoft/Phi-3-mini-4k-instruct model for bs ${bs} with ISL ${isl[i]}, OSL ${osl[i]} and backend ${backend}"
          python perf_run.py --model_torch "microsoft/Phi-3-mini-4k-instruct" \
                            --is_text_llm \
                            --precision fp16 \
                            --inputs "(${bs}, ${isl[i]})@int64" \
                            --output_sequence_length ${osl[i]} \
                            --batch_size ${bs} \
                            --truncate \
                            --backends ${backend} \
                            --report "phi3_mini_4k_perf_bs${bs}_backend_${backend}_isl${isl[i]}_osl${osl[i]}.csv"
          done
        done
  done
# Clear HF cache
rm -rf ~/.cache/huggingface/hub/

# Collect and concatenate all results
echo "Concatenating all results"
python accumulate_results.py