#!/bin/bash

MODELS_DIR="models"

# Download the Torchscript models
python hub.py

batch_sizes=(1 2 4 8 16 32 64 128 256)

#Benchmark VGG16 model
echo "Benchmarking VGG16 model"
for bs in ${batch_sizes[@]}
do
  python perf_run.py --model ${MODELS_DIR}/vgg16_scripted.jit.pt \
                     --precision fp32,fp16 --inputs="(${bs}, 3, 224, 224)" \
                     --batch_size ${bs} \
                     --backends torch,torch_tensorrt,tensorrt \
                     --report "vgg_perf_bs${bs}.txt"
done

# Benchmark Resnet50 model
echo "Benchmarking Resnet50 model"
for bs in ${batch_sizes[@]}
do
  python perf_run.py --model ${MODELS_DIR}/resnet50_scripted.jit.pt \
                     --precision fp32,fp16 --inputs="(${bs}, 3, 224, 224)" \
                     --batch_size ${bs} \
                     --backends torch,torch_tensorrt,tensorrt \
                     --report "rn50_perf_bs${bs}.txt"
done

# Benchmark VIT model
echo "Benchmarking VIT model"
for bs in ${batch_sizes[@]}
do
  python perf_run.py --model ${MODELS_DIR}/vit_scripted.jit.pt \
                     --precision fp32,fp16 --inputs="(${bs}, 3, 224, 224)" \
                     --batch_size ${bs} \
                     --backends torch,torch_tensorrt,tensorrt \
                     --report "vit_perf_bs${bs}.txt"
done

# Benchmark EfficientNet-B0 model
echo "Benchmarking EfficientNet-B0 model"
for bs in ${batch_sizes[@]}
do
  python perf_run.py --model ${MODELS_DIR}/efficientnet_b0_scripted.jit.pt \
                     --precision fp32,fp16 --inputs="(${bs}, 3, 224, 224)" \
                     --batch_size ${bs} \
                     --backends torch,torch_tensorrt,tensorrt \
                     --report "eff_b0_perf_bs${bs}.txt"
done

# Benchmark BERT model
echo "Benchmarking Huggingface BERT base model"
for bs in ${batch_sizes[@]}
do
  python perf_run.py --model ${MODELS_DIR}/bert_base_uncased_traced.jit.pt \
                     --precision fp32 --inputs="(${bs}, 128)@int32;(${bs}, 128)@int32" \
                     --batch_size ${bs} \
                     --backends torch,torch_tensorrt \
                     --truncate \
                     --report "bert_base_perf_bs${bs}.txt"
done
