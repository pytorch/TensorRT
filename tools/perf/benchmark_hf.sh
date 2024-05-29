#!/bin/bash

batch_sizes=(1 16 64 256)
backends=("torch" "torch_compile" "inductor")
INPUT_PROMPT="Hi there, can you hear me?"


echo "Benchmarking gpt2 model"
for bs in ${batch_sizes[@]}
do
  for backend in ${backends[@]}
  do
    python perf_run.py --model_hf gpt2 \
                       --precision fp16 \
                       --inputs="${INPUT_PROMPT}" \
                       --batch_size ${bs} \
                       --truncate \
                       --backends ${backend} \
                       --report "gpt2_perf_bs${bs}_backend_${backend}.csv"
  done
done

echo "Benchmarking google/gemma-2b model"
for bs in ${batch_sizes[@]}
do
  for backend in ${backends[@]}
  do
    python perf_run.py --model_hf google/gemma-2b \
                       --precision fp16 \
                       --inputs="${INPUT_PROMPT}" \
                       --batch_size ${bs} \
                       --truncate \
                       --backends ${backend} \
                       --report "gemma_2b_perf_bs${bs}_backend_${backend}.csv"
  done
done


# Collect and concatenate all results
echo "Concatenating all results"
python accumulate_results.py
