#!/bin/bash
# Usage : bash run_hf_model.sh <batch_size> <backend> <model_name> <isl> <osl> <precision> <iterations>

# GPT2 model torch backend
bash run_hf_model.sh 1 "torch" "gpt2" 128 256 "fp16" 1
bash run_hf_model.sh 16 "torch" "gpt2" 128 256 "fp16" 1
bash run_hf_model.sh 64 "torch" "gpt2" 128 256 "fp16" 1
bash run_hf_model.sh 256 "torch" "gpt2" 128 256 "fp16" 1

# GPT2 model dynamo backend
bash run_hf_model.sh 1 "dynamo" "gpt2" 128 256 "fp16" 1
bash run_hf_model.sh 16 "dynamo" "gpt2" 128 256 "fp16" 1
bash run_hf_model.sh 64 "dynamo" "gpt2" 128 256 "fp16" 1
bash run_hf_model.sh 256 "dynamo" "gpt2" 128 256 "fp16" 1

# GPT2 model inductor backend
bash run_hf_model.sh 1 "inductor" "gpt2" 128 256 "fp16" 1
bash run_hf_model.sh 16 "inductor" "gpt2" 128 256 "fp16" 1
bash run_hf_model.sh 64 "inductor" "gpt2" 128 256 "fp16" 1
bash run_hf_model.sh 256 "inductor" "gpt2" 128 256 "fp16" 1

