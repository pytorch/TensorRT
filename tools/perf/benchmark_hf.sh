#!/bin/bash

batch_sizes=(1 16 64 256)
backends=("torch" "torch_compile" "inductor")
models=(
  "gpt2"
  "microsoft/phi-2"
)
input_sequence_lengths=(128)
output_sequence_lengths=(1)


for model in ${models[@]}
do
  for bs in ${batch_sizes[@]}
  do
    for backend in ${backends[@]}
    do
      for isl in ${input_sequence_lengths[@]}
      do
        for osl in ${output_sequence_lengths[@]}
        do
          model_stringified=$(echo ${model} | tr / _)
          echo "Benchmarking ${model} model for bs ${bs} with ISL ${isl} and backend ${backend}"
          (python perf_run.py --model_hf ${model} \
                            --precision fp16 \
                            --input_sequence_length ${isl} \
                            --output_sequence_length ${osl} \
                            --batch_size ${bs} \
                            --truncate \
                            --backends ${backend} \
                            --report "${model_stringified}_perf_bs${bs}_backend_${backend}_isl${isl}_osl${osl}.csv"
          ) 2>&1 | tee "${model_stringified}_perf_bs${bs}_backend_${backend}_isl${isl}_osl${osl}.log"
        done
      done
    done
  done
  rm -rf ~/.cache/huggingface/hub/
done


# Collect and concatenate all results
echo "Concatenating all results"
python accumulate_results.py
