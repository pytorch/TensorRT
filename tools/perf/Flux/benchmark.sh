#!/bin/bash

nvidia-smi --query-gpu=index,utilization.gpu,utilization.memory,memory.used --format=csv,nounits -lms 1000 >> fp4_nvidia-smi.log &
NVIDIA_SMI_PID=$!
python  tools/perf/Flux/flux_quantization.py --dtype fp4 2>&1 | tee fp4.log
python  tools/perf/Flux/flux_quantization.py --dtype fp4 --sdpa 2>&1 | tee fp4_sdpa.log
#python  tools/perf/Flux/flux_quantization.py --dtype fp4 --sdpa --mha 2>&1 | tee fp4_sdpa_mha.log
kill $NVIDIA_SMI_PID
#scp fp4*.log lanl@d5c2237-lcedt.dyn.nvidia.com:/home/lanl/git/script/flux/gb100/0611/
sleep 10


nvidia-smi --query-gpu=index,utilization.gpu,utilization.memory,memory.used --format=csv,nounits -lms 1000 >> fp8_nvidia-smi.log &
NVIDIA_SMI_PID=$!
python  tools/perf/Flux/flux_quantization.py --dtype fp8  2>&1 | tee fp8.log
python  tools/perf/Flux/flux_quantization.py --dtype fp8 --sdpa 2>&1 | tee fp8_sdpa.log
#python  tools/perf/Flux/flux_quantization.py --dtype fp8 --sdpa --mha 2>&1 | tee fp8_sdpa_mha.log
kill $NVIDIA_SMI_PID
# scp fp8*.log lanl@d5c2237-lcedt.dyn.nvidia.com:/home/lanl/git/script/flux/gb100/0611/
sleep 10


nvidia-smi --query-gpu=index,utilization.gpu,utilization.memory,memory.used --format=csv,nounits -lms 1000 >> fp16_nvidia-smi.log &
NVIDIA_SMI_PID=$!
python  tools/perf/Flux/flux_quantization.py --dtype fp16  2>&1 | tee fp16.log
python  tools/perf/Flux/flux_quantization.py --dtype fp16 --sdpa 2>&1 | tee fp16_sdpa.log
kill $NVIDIA_SMI_PID
# scp fp16*.log lanl@d5c2237-lcedt.dyn.nvidia.com:/home/lanl/git/script/flux/gb100/0611/
sleep 10

# nvidia-smi --query-gpu=index,utilization.gpu,utilization.memory,memory.used --format=csv,nounits -lms 1000 >> bf16_nvidia-smi.log &
# NVIDIA_SMI_PID=$!
# python  tools/perf/Flux/flux_quantization.py --dtype bf16 2>&1 | tee bf16.log
# kill $NVIDIA_SMI_PID
# sleep 10

# nvidia-smi --query-gpu=index,utilization.gpu,utilization.memory,memory.used --format=csv,nounits -lms 1000 >> fp32_nvidia-smi.log &
# NVIDIA_SMI_PID=$!
# python  tools/perf/Flux/flux_quantization.py --dtype fp32 2>&1 | tee fp32.log
# kill $NVIDIA_SMI_PID
# sleep 10