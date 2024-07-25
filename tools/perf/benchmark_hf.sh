#!/bin/bash

batch_sizes=(1 16 64 256)
backends=("torch")
models=(
  "gpt2"
  # "meta-llama/Meta-Llama-3-8B"
#   "mistralai/Mistral-7B-Instruct-v0.3"
#   "meta-llama/Meta-Llama-3-8B-Instruct"
#   "microsoft/Phi-3-vision-128k-instruct"
#   "microsoft/Phi-3-mini-4k-instruct"
#   "mistralai/Mistral-7B-v0.3"
#   "shenzhi-wang/Llama3-8B-Chinese-Chat"
#   "microsoft/Phi-3-mini-128k-instruct"
#   "failspy/Llama-3-8B-Instruct-MopeyMule"
#   "openchat/openchat-3.6-8b-20240522"
)

# GPT2 model
echo "Benchmarking GPT2 model"
isl=(128 256)
osl=(256 512)
for bs in ${batch_sizes[@]}
  do
    for backend in ${backends[@]}
        do
        for i in ${!isl[@]};
          do
          echo "Benchmarking GPT2 model for bs ${bs} with ISL ${isl}, OSL ${osl} and backend ${backend}"
          python perf_run.py --model_torch "gpt2" \
                            --is_text_llm \
                            --precision fp16 \
                            --inputs "(${bs}, ${isl})@int64" \
                            --output_sequence_length ${osl} \
                            --batch_size ${bs} \
                            --truncate \
                            --backends ${backend} \
                            --report "gpt2_perf_bs${bs}_backend_${backend}_isl${isl}_osl${osl}.csv"
          done
        done
  done
# Clear HF cache
rm -rf ~/.cache/huggingface/hub/

# "meta-llama/Meta-Llama-3-8B" model
echo "Benchmarking Llama3 model"
isl=(128 256)
osl=(256 512)
for bs in ${batch_sizes[@]}
  do
    for backend in ${backends[@]}
        do
        for i in ${!isl[@]};
          do
          echo "Benchmarking Llama3-8B model for bs ${bs} with ISL ${isl}, OSL ${osl} and backend ${backend}"
          python perf_run.py --model_torch "meta-llama/Meta-Llama-3-8B" \
                            --is_text_llm \
                            --precision fp16 \
                            --inputs "(${bs}, ${isl})@int64" \
                            --output_sequence_length ${osl} \
                            --batch_size ${bs} \
                            --truncate \
                            --backends ${backend} \
                            --report "llama3_8b_perf_bs${bs}_backend_${backend}_isl${isl}_osl${osl}.csv"
          done
        done
  done
# Clear HF cache
rm -rf ~/.cache/huggingface/hub/


# Collect and concatenate all results
echo "Concatenating all results"
python accumulate_results.py