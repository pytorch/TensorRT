#!/bin/bash
# global parameters
precision="fp16"
iterations=1
backends=("dynamo" "inductor")
batch_sizes=(1 16)
hf_token="<Enter HF token>"
image_name="<docker image name>"

# Stage 1 : GPT2 experiment
models=("gpt2")
isl=(128)
osl=(256)
for model in ${models[@]}
    do
        for bs in ${batch_sizes[@]}
            do
                for backend in ${backends[@]}
                    do
                        for i in ${!isl[@]};
                            do
                                docker run --rm -it --gpus 0 --shm-size=10.24g --ulimit stack=67108864 -v "$PWD:/work" --ipc=host ${image_name} /bin/bash -c "cd /opt/torch_tensorrt/tools/perf; HF_TOKEN="${hf_token}" bash run_hf_model.sh "${bs}" "$backend" "$model" "${isl[i]}" "${osl[i]}" "${precision}" "${iterations}"; exit"
                            done
                    done
            done
    done
# Clear HF cache
rm -rf ~/.cache/huggingface/hub/

# Stage 2 : non-GPT2 experiments
isl=(128 128)
osl=(256 2176)
models=("meta-llama/Meta-Llama-3.1-8B-Instruct" "meta-llama/Llama-2-7b-chat-hf" "mistralai/Mistral-7B-Instruct-v0.3")
backends=("dynamo" "inductor")
for model in ${models[@]}
    do
        for bs in ${batch_sizes[@]}
            do
                for backend in ${backends[@]}
                    do
                        for i in ${!isl[@]};
                            do
                                docker run --rm -it --gpus 0 --shm-size=10.24g --ulimit stack=67108864 -v "$PWD:/work" --ipc=host ${image_name} /bin/bash -c "cd /opt/torch_tensorrt/tools/perf; HF_TOKEN="${hf_token}" bash run_hf_model.sh "${bs}" "$backend" "$model" "${isl[i]}" "${osl[i]}" "${precision}" "${iterations}"; exit"
                            done
                    done
            done
    done
# Clear HF cache
rm -rf ~/.cache/huggingface/hub/