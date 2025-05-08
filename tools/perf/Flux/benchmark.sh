#TODO: Enter the HF Token
huggingface-cli login --token HF_TOKEN

nvidia-smi --query-gpu=index,utilization.gpu,utilization.memory,temperature.gpu,temperature.memory,power.draw,clocks.sm,clocks.mem,memory.total,memory.used --format=csv,nounits -lms 5000 >> fp8_gpu_utilization.txt &
NVIDIA_SMI_PID=$!
python flux_quantization.py --dtype fp8 > fp8_benchmark.txt
kill $NVIDIA_SMI_PID
sleep 10

nvidia-smi --query-gpu=index,utilization.gpu,utilization.memory,temperature.gpu,temperature.memory,power.draw,clocks.sm,clocks.mem,memory.total,memory.used --format=csv,nounits -lms 5000 >> int8_gpu_utilization.txt &
NVIDIA_SMI_PID=$!
python flux_quantization.py --dtype int8 > int8_benchmark.txt
kill $NVIDIA_SMI_PID
sleep 10

nvidia-smi --query-gpu=index,utilization.gpu,utilization.memory,temperature.gpu,temperature.memory,power.draw,clocks.sm,clocks.mem,memory.total,memory.used --format=csv,nounits -lms 5000 >> fp16_gpu_utilization.txt &
NVIDIA_SMI_PID=$!
python flux_perf.py > fp16_benchmark.txt
kill $NVIDIA_SMI_PID
sleep 10