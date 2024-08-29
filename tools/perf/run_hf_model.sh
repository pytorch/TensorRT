#!/bin/bash
batch_size=$1
backend=$2
model_name=$3
isl=$4
osl=$5
precision=$6
iterations=$7
modified_model_name=$(echo "$model_name" | sed 's/\//-/g')
echo "Benchmarking ${model_name} model for bs ${batch_size} with ISL ${isl}, OSL ${osl} and backend ${backend} for ${iterations} iterations"
python perf_run.py --model_torch ${model_name} \
                --is_text_llm \
                --precision ${precision} \
                --inputs "(${batch_size}, ${isl})@int64" \
                --output_sequence_length ${osl} \
                --batch_size ${batch_size} \
                --truncate \
                --backends ${backend} \
                --iterations ${iterations} \
                --report "${modified_model_name}_perf_bs${batch_size}_backend_${backend}_isl${isl}_osl${osl}.csv"

# Move the report file to the mounted volume in the docker 
mv "${modified_model_name}_perf_bs${batch_size}_backend_${backend}_isl${isl}_osl${osl}.csv" /work

# Clear HF cache
rm -rf ~/.cache/huggingface/hub/
