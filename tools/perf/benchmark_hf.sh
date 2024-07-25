#!/bin/bash

batch_sizes=(1 16 64 256)
backends=("torch" "torch_compile" "inductor")
INPUT_PROMPT="Hi there, can you hear me?"

echo "Benchmarking GPT2 model"
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

echo "Benchmarking Llama-7B model"
for bs in ${batch_sizes[@]}
do
  for backend in ${backends[@]}
  do
    python perf_run.py --model_hf /opt/Llama-2-7b-hf \
                       --precision fp16 \
                       --inputs="${INPUT_PROMPT}" \
                       --batch_size ${bs} \
                       --truncate \
                       --backends ${backend} \
                       --report "llama_7b_perf_bs${bs}_backend_${backend}.csv"
  done
done

echo "Benchmarking mistralai/Mistral-7B-Instruct-v0.2 model"
for bs in ${batch_sizes[@]}
do
  for backend in ${backends[@]}
  do
    python perf_run.py --model_hf mistralai/Mistral-7B-Instruct-v0.2 \
                       --precision fp16 \
                       --inputs="${INPUT_PROMPT}" \
                       --batch_size ${bs} \
                       --truncate \
                       --backends ${backend} \
                       --report "mistral_7b_instruct_perf_bs${bs}_backend_${backend}.csv"
  done
done

echo "Benchmarking NexaAIDev/Octopus-v2 model"
for bs in ${batch_sizes[@]}
do
  for backend in ${backends[@]}
  do
    python perf_run.py --model_hf NexaAIDev/Octopus-v2 \
                       --precision fp16 \
                       --inputs="${INPUT_PROMPT}" \
                       --batch_size ${bs} \
                       --truncate \
                       --backends ${backend} \
                       --report "octopus_v2_perf_bs${bs}_backend_${backend}.csv"
  done
done

echo "Benchmarking google/codegemma-7b model"
for bs in ${batch_sizes[@]}
do
  for backend in ${backends[@]}
  do
    python perf_run.py --model_hf google/codegemma-7b \
                       --precision fp16 \
                       --inputs="${INPUT_PROMPT}" \
                       --batch_size ${bs} \
                       --truncate \
                       --backends ${backend} \
                       --report "codegemma_7b_perf_bs${bs}_backend_${backend}.csv"
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

echo "Benchmarking google/recurrentgemma-2b model"
for bs in ${batch_sizes[@]}
do
  for backend in ${backends[@]}
  do
    python perf_run.py --model_hf google/recurrentgemma-2b \
                       --precision fp16 \
                       --inputs="${INPUT_PROMPT}" \
                       --batch_size ${bs} \
                       --truncate \
                       --backends ${backend} \
                       --report "recurrentgemma_2b_perf_bs${bs}_backend_${backend}.csv"
  done
done

echo "Benchmarking databricks/dbrx-instruct model"
for bs in ${batch_sizes[@]}
do
  for backend in ${backends[@]}
  do
    python perf_run.py --model_hf databricks/dbrx-instruct \
                       --precision fp16 \
                       --inputs="${INPUT_PROMPT}" \
                       --batch_size ${bs} \
                       --truncate \
                       --backends ${backend} \
                       --report "dbrx_instruct_perf_bs${bs}_backend_${backend}.csv"
  done
done


echo "Benchmarking ai21labs/Jamba-v0.1 model"
for bs in ${batch_sizes[@]}
do
  for backend in ${backends[@]}
  do
    python perf_run.py --model_hf ai21labs/Jamba-v0.1 \
                       --precision fp16 \
                       --inputs="${INPUT_PROMPT}" \
                       --batch_size ${bs} \
                       --truncate \
                       --backends ${backend} \
                       --report "Jamba_perf_bs${bs}_backend_${backend}.csv"
  done
done


# Collect and concatenate all results
echo "Concatenating all results"
python accumulate_results.py