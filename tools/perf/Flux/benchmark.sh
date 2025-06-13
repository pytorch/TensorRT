#TODO: Enter the HF Token
huggingface-cli login --token HF_TOKEN

nvidia-smi --query-gpu=index,utilization.gpu,utilization.memory,temperature.gpu,temperature.memory,power.draw,clocks.sm,clocks.mem,memory.total,memory.used --format=csv,nounits -lms 500 >> fp8_gpu_utilization.txt &
NVIDIA_SMI_PID=$!
python flux_perf.py --dtype fp8 --low_vram_mode> fp8_benchmark.txt
kill $NVIDIA_SMI_PID


