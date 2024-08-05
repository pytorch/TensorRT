#!/bin/bash
# Usage : bash run_hf_model.sh <batch_size> <backend> <model_name> <isl> <osl> <precision> <iterations>

# "meta-llama/Meta-Llama-3.1-8B-Instruct" model torch backend
# isl, osl = 128, 256
bash run_hf_model.sh 1 "torch" "meta-llama/Meta-Llama-3.1-8B-Instruct" 128 256 "fp16" 3
bash run_hf_model.sh 16 "torch" "meta-llama/Meta-Llama-3.1-8B-Instruct" 128 256 "fp16" 3
bash run_hf_model.sh 64 "torch" "meta-llama/Meta-Llama-3.1-8B-Instruct" 128 256 "fp16" 3
bash run_hf_model.sh 256 "torch" "meta-llama/Meta-Llama-3.1-8B-Instruct" 128 256 "fp16" 3
# isl, osl = 128, 2176
bash run_hf_model.sh 1 "torch" "meta-llama/Meta-Llama-3.1-8B-Instruct" 128 2176 "fp16" 3
bash run_hf_model.sh 16 "torch" "meta-llama/Meta-Llama-3.1-8B-Instruct" 128 2176 "fp16" 3
bash run_hf_model.sh 64 "torch" "meta-llama/Meta-Llama-3.1-8B-Instruct" 128 2176 "fp16" 3
bash run_hf_model.sh 256 "torch" "meta-llama/Meta-Llama-3.1-8B-Instruct" 128 2176 "fp16" 3
# isl, osl = 2048, 2176
bash run_hf_model.sh 1 "torch" "meta-llama/Meta-Llama-3.1-8B-Instruct" 2048 2176 "fp16" 3
bash run_hf_model.sh 16 "torch" "meta-llama/Meta-Llama-3.1-8B-Instruct" 2048 2176 "fp16" 3
bash run_hf_model.sh 64 "torch" "meta-llama/Meta-Llama-3.1-8B-Instruct" 2048 2176 "fp16" 3
bash run_hf_model.sh 256 "torch" "meta-llama/Meta-Llama-3.1-8B-Instruct" 2048 2176 "fp16" 3
# isl, osl = 2048, 4096
bash run_hf_model.sh 1 "torch" "meta-llama/Meta-Llama-3.1-8B-Instruct" 2048 4096 "fp16" 3
bash run_hf_model.sh 16 "torch" "meta-llama/Meta-Llama-3.1-8B-Instruct" 2048 4096 "fp16" 3
bash run_hf_model.sh 64 "torch" "meta-llama/Meta-Llama-3.1-8B-Instruct" 2048 4096 "fp16" 3
bash run_hf_model.sh 256 "torch" "meta-llama/Meta-Llama-3.1-8B-Instruct" 2048 4096 "fp16" 3

# "meta-llama/Meta-Llama-3.1-8B-Instruct" model dynamo backend
# isl, osl = 128, 256
bash run_hf_model.sh 1 "dynamo" "meta-llama/Meta-Llama-3.1-8B-Instruct" 128 256 "fp16" 3
bash run_hf_model.sh 16 "dynamo" "meta-llama/Meta-Llama-3.1-8B-Instruct" 128 256 "fp16" 3
bash run_hf_model.sh 64 "dynamo" "meta-llama/Meta-Llama-3.1-8B-Instruct" 128 256 "fp16" 3
bash run_hf_model.sh 256 "dynamo" "meta-llama/Meta-Llama-3.1-8B-Instruct" 128 256 "fp16" 3
# isl, osl = 128, 2176
bash run_hf_model.sh 1 "dynamo" "meta-llama/Meta-Llama-3.1-8B-Instruct" 128 2176 "fp16" 3
bash run_hf_model.sh 16 "dynamo" "meta-llama/Meta-Llama-3.1-8B-Instruct" 128 2176 "fp16" 3
bash run_hf_model.sh 64 "dynamo" "meta-llama/Meta-Llama-3.1-8B-Instruct" 128 2176 "fp16" 3
bash run_hf_model.sh 256 "dynamo" "meta-llama/Meta-Llama-3.1-8B-Instruct" 128 2176 "fp16" 3
# isl, osl = 2048, 2176
bash run_hf_model.sh 1 "dynamo" "meta-llama/Meta-Llama-3.1-8B-Instruct" 2048 2176 "fp16" 3
bash run_hf_model.sh 16 "dynamo" "meta-llama/Meta-Llama-3.1-8B-Instruct" 2048 2176 "fp16" 3
bash run_hf_model.sh 64 "dynamo" "meta-llama/Meta-Llama-3.1-8B-Instruct" 2048 2176 "fp16" 3
bash run_hf_model.sh 256 "dynamo" "meta-llama/Meta-Llama-3.1-8B-Instruct" 2048 2176 "fp16" 3
# isl, osl = 2048, 4096
bash run_hf_model.sh 1 "dynamo" "meta-llama/Meta-Llama-3.1-8B-Instruct" 2048 4096 "fp16" 3
bash run_hf_model.sh 16 "dynamo" "meta-llama/Meta-Llama-3.1-8B-Instruct" 2048 4096 "fp16" 3
bash run_hf_model.sh 64 "dynamo" "meta-llama/Meta-Llama-3.1-8B-Instruct" 2048 4096 "fp16" 3
bash run_hf_model.sh 256 "dynamo" "meta-llama/Meta-Llama-3.1-8B-Instruct" 2048 4096 "fp16" 3

# "meta-llama/Meta-Llama-3.1-8B-Instruct" model inductor backend
# isl, osl = 128, 256
bash run_hf_model.sh 1 "inductor" "meta-llama/Meta-Llama-3.1-8B-Instruct" 128 256 "fp16" 3
bash run_hf_model.sh 16 "inductor" "meta-llama/Meta-Llama-3.1-8B-Instruct" 128 256 "fp16" 3
bash run_hf_model.sh 64 "inductor" "meta-llama/Meta-Llama-3.1-8B-Instruct" 128 256 "fp16" 3
bash run_hf_model.sh 256 "inductor" "meta-llama/Meta-Llama-3.1-8B-Instruct" 128 256 "fp16" 3
# isl, osl = 128, 2176
bash run_hf_model.sh 1 "inductor" "meta-llama/Meta-Llama-3.1-8B-Instruct" 128 2176 "fp16" 3
bash run_hf_model.sh 16 "inductor" "meta-llama/Meta-Llama-3.1-8B-Instruct" 128 2176 "fp16" 3
bash run_hf_model.sh 64 "inductor" "meta-llama/Meta-Llama-3.1-8B-Instruct" 128 2176 "fp16" 3
bash run_hf_model.sh 256 "inductor" "meta-llama/Meta-Llama-3.1-8B-Instruct" 128 2176 "fp16" 3
# isl, osl = 2048, 2176
bash run_hf_model.sh 1 "inductor" "meta-llama/Meta-Llama-3.1-8B-Instruct" 2048 2176 "fp16" 3
bash run_hf_model.sh 16 "inductor" "meta-llama/Meta-Llama-3.1-8B-Instruct" 2048 2176 "fp16" 3
bash run_hf_model.sh 64 "inductor" "meta-llama/Meta-Llama-3.1-8B-Instruct" 2048 2176 "fp16" 3
bash run_hf_model.sh 256 "inductor" "meta-llama/Meta-Llama-3.1-8B-Instruct" 2048 2176 "fp16" 3
# isl, osl = 2048, 4096
bash run_hf_model.sh 1 "inductor" "meta-llama/Meta-Llama-3.1-8B-Instruct" 2048 4096 "fp16" 3
bash run_hf_model.sh 16 "inductor" "meta-llama/Meta-Llama-3.1-8B-Instruct" 2048 4096 "fp16" 3
bash run_hf_model.sh 64 "inductor" "meta-llama/Meta-Llama-3.1-8B-Instruct" 2048 4096 "fp16" 3
bash run_hf_model.sh 256 "inductor" "meta-llama/Meta-Llama-3.1-8B-Instruct" 2048 4096 "fp16" 3