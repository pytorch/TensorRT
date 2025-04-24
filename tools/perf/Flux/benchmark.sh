#TODO: Enter the HF Token
huggingface-cli login --token HF_TOKEN

python flux_quantization.py --dtype fp8 > fp8_benchmark.txt
python flux_quantization.py --dtype int8 > int8_benchmark.txt