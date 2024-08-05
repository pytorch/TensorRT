#!/bin/bash
# Usage : bash run_hf_model.sh <batch_size> <backend> <model_name> <isl> <osl> <precision> <iterations>

# "mistralai/Mistral-7B-Instruct-v0.3" model torch backend
# isl, osl = 128, 256
bash run_hf_model.sh 1 "torch" "mistralai/Mistral-7B-Instruct-v0.3" 128 256 "fp16" 3
bash run_hf_model.sh 16 "torch" "mistralai/Mistral-7B-Instruct-v0.3" 128 256 "fp16" 3
bash run_hf_model.sh 64 "torch" "mistralai/Mistral-7B-Instruct-v0.3" 128 256 "fp16" 3
bash run_hf_model.sh 256 "torch" "mistralai/Mistral-7B-Instruct-v0.3" 128 256 "fp16" 3
# isl, osl = 128, 2176
bash run_hf_model.sh 1 "torch" "mistralai/Mistral-7B-Instruct-v0.3" 128 2176 "fp16" 3
bash run_hf_model.sh 16 "torch" "mistralai/Mistral-7B-Instruct-v0.3" 128 2176 "fp16" 3
bash run_hf_model.sh 64 "torch" "mistralai/Mistral-7B-Instruct-v0.3" 128 2176 "fp16" 3
bash run_hf_model.sh 256 "torch" "mistralai/Mistral-7B-Instruct-v0.3" 128 2176 "fp16" 3
# isl, osl = 2048, 2176
bash run_hf_model.sh 1 "torch" "mistralai/Mistral-7B-Instruct-v0.3" 2048 2176 "fp16" 3
bash run_hf_model.sh 16 "torch" "mistralai/Mistral-7B-Instruct-v0.3" 2048 2176 "fp16" 3
bash run_hf_model.sh 64 "torch" "mistralai/Mistral-7B-Instruct-v0.3" 2048 2176 "fp16" 3
bash run_hf_model.sh 256 "torch" "mistralai/Mistral-7B-Instruct-v0.3" 2048 2176 "fp16" 3
# isl, osl = 2048, 4096
bash run_hf_model.sh 1 "torch" "mistralai/Mistral-7B-Instruct-v0.3" 2048 4096 "fp16" 3
bash run_hf_model.sh 16 "torch" "mistralai/Mistral-7B-Instruct-v0.3" 2048 4096 "fp16" 3
bash run_hf_model.sh 64 "torch" "mistralai/Mistral-7B-Instruct-v0.3" 2048 4096 "fp16" 3
bash run_hf_model.sh 256 "torch" "mistralai/Mistral-7B-Instruct-v0.3" 2048 4096 "fp16" 3

# "mistralai/Mistral-7B-Instruct-v0.3" model dynamo backend
# isl, osl = 128, 256
bash run_hf_model.sh 1 "dynamo" "mistralai/Mistral-7B-Instruct-v0.3" 128 256 "fp16" 3
bash run_hf_model.sh 16 "dynamo" "mistralai/Mistral-7B-Instruct-v0.3" 128 256 "fp16" 3
bash run_hf_model.sh 64 "dynamo" "mistralai/Mistral-7B-Instruct-v0.3" 128 256 "fp16" 3
bash run_hf_model.sh 256 "dynamo" "mistralai/Mistral-7B-Instruct-v0.3" 128 256 "fp16" 3
# isl, osl = 128, 2176
bash run_hf_model.sh 1 "dynamo" "mistralai/Mistral-7B-Instruct-v0.3" 128 2176 "fp16" 3
bash run_hf_model.sh 16 "dynamo" "mistralai/Mistral-7B-Instruct-v0.3" 128 2176 "fp16" 3
bash run_hf_model.sh 64 "dynamo" "mistralai/Mistral-7B-Instruct-v0.3" 128 2176 "fp16" 3
bash run_hf_model.sh 256 "dynamo" "mistralai/Mistral-7B-Instruct-v0.3" 128 2176 "fp16" 3
# isl, osl = 2048, 2176
bash run_hf_model.sh 1 "dynamo" "mistralai/Mistral-7B-Instruct-v0.3" 2048 2176 "fp16" 3
bash run_hf_model.sh 16 "dynamo" "mistralai/Mistral-7B-Instruct-v0.3" 2048 2176 "fp16" 3
bash run_hf_model.sh 64 "dynamo" "mistralai/Mistral-7B-Instruct-v0.3" 2048 2176 "fp16" 3
bash run_hf_model.sh 256 "dynamo" "mistralai/Mistral-7B-Instruct-v0.3" 2048 2176 "fp16" 3
# isl, osl = 2048, 4096
bash run_hf_model.sh 1 "dynamo" "mistralai/Mistral-7B-Instruct-v0.3" 2048 4096 "fp16" 3
bash run_hf_model.sh 16 "dynamo" "mistralai/Mistral-7B-Instruct-v0.3" 2048 4096 "fp16" 3
bash run_hf_model.sh 64 "dynamo" "mistralai/Mistral-7B-Instruct-v0.3" 2048 4096 "fp16" 3
bash run_hf_model.sh 256 "dynamo" "mistralai/Mistral-7B-Instruct-v0.3" 2048 4096 "fp16" 3

# "mistralai/Mistral-7B-Instruct-v0.3" model inductor backend
# isl, osl = 128, 256
bash run_hf_model.sh 1 "inductor" "mistralai/Mistral-7B-Instruct-v0.3" 128 256 "fp16" 3
bash run_hf_model.sh 16 "inductor" "mistralai/Mistral-7B-Instruct-v0.3" 128 256 "fp16" 3
bash run_hf_model.sh 64 "inductor" "mistralai/Mistral-7B-Instruct-v0.3" 128 256 "fp16" 3
bash run_hf_model.sh 256 "inductor" "mistralai/Mistral-7B-Instruct-v0.3" 128 256 "fp16" 3
# isl, osl = 128, 2176
bash run_hf_model.sh 1 "inductor" "mistralai/Mistral-7B-Instruct-v0.3" 128 2176 "fp16" 3
bash run_hf_model.sh 16 "inductor" "mistralai/Mistral-7B-Instruct-v0.3" 128 2176 "fp16" 3
bash run_hf_model.sh 64 "inductor" "mistralai/Mistral-7B-Instruct-v0.3" 128 2176 "fp16" 3
bash run_hf_model.sh 256 "inductor" "mistralai/Mistral-7B-Instruct-v0.3" 128 2176 "fp16" 3
# isl, osl = 2048, 2176
bash run_hf_model.sh 1 "inductor" "mistralai/Mistral-7B-Instruct-v0.3" 2048 2176 "fp16" 3
bash run_hf_model.sh 16 "inductor" "mistralai/Mistral-7B-Instruct-v0.3" 2048 2176 "fp16" 3
bash run_hf_model.sh 64 "inductor" "mistralai/Mistral-7B-Instruct-v0.3" 2048 2176 "fp16" 3
bash run_hf_model.sh 256 "inductor" "mistralai/Mistral-7B-Instruct-v0.3" 2048 2176 "fp16" 3
# isl, osl = 2048, 4096
bash run_hf_model.sh 1 "inductor" "mistralai/Mistral-7B-Instruct-v0.3" 2048 4096 "fp16" 3
bash run_hf_model.sh 16 "inductor" "mistralai/Mistral-7B-Instruct-v0.3" 2048 4096 "fp16" 3
bash run_hf_model.sh 64 "inductor" "mistralai/Mistral-7B-Instruct-v0.3" 2048 4096 "fp16" 3
bash run_hf_model.sh 256 "inductor" "mistralai/Mistral-7B-Instruct-v0.3" 2048 4096 "fp16" 3