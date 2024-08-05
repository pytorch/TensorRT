#!/bin/bash
batch_size=$1
backend=$2
model_name=$3
isl=$4
osl=$5
precision=$6
iterations=$7
modified_model_name=$(echo "$model_name" | sed 's/\//-/g')
echo "Benchmarking ${model_name} model for bs ${batch_size} with ISL ${isl[i]}, OSL ${osl[i]} and backend ${backend} for ${iterations} iterations"
python perf_run.py --model_torch ${model_name} \
                --is_text_llm \
                --precision ${precision} \
                --inputs "(${batch_size}, ${isl[i]})@int64" \
                --output_sequence_length ${osl[i]} \
                --batch_size ${batch_size} \
                --truncate \
                --backends ${backend} \
                --iterations ${iterations} \
                --report "${modified_model_name}_perf_bs${bs}_backend_${backend}_isl${isl[i]}_osl${osl[i]}.csv"

# Clear HF cache
rm -rf ~/.cache/huggingface/hub/
