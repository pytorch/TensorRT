#!/bin/bash

# Download the Torchscript models
python hub.py

batch_sizes=(1 2 4 8 16 32 64 128 256)

# # Benchmark VGG16 model
# echo "Benchmarking VGG16 model"
# for bs in 1 2
# do
#   python perf_run.py --model models/vgg16_scripted.jit.pt \
#                      --precision fp32,fp16 --inputs="(${bs}, 3, 224, 224)" \
#                      --batch_size ${bs} \
#                      --backends torch,torch_tensorrt,tensorrt \
#                      --report "vgg_perf_bs${bs}.txt"
# done
#
# # Benchmark Resnet50 model
# echo "Benchmarking Resnet50 model"
# for bs in 1 2
# do
#   python perf_run.py --model models/resnet50_scripted.jit.pt \
#                      --precision fp32,fp16 --inputs="(${bs}, 3, 224, 224)" \
#                      --batch_size ${bs} \
#                      --backends torch,torch_tensorrt,tensorrt \
#                      --report "rn50_perf_bs${bs}.txt"
# done
#
# # Benchmark VIT model
# echo "Benchmarking VIT model"
# for bs in 1 2
# do
#   python perf_run.py --model models/vit_scripted.jit.pt \
#                      --precision fp32,fp16 --inputs="(${bs}, 3, 224, 224)" \
#                      --batch_size ${bs} \
#                      --backends torch,torch_tensorrt,tensorrt \
#                      --report "vit_perf_bs${bs}.txt"
# done
#
# # Benchmark EfficientNet-B0 model
# echo "Benchmarking EfficientNet-B0 model"
# for bs in 1 2
# do
#   python perf_run.py --model models/efficientnet_b0_scripted.jit.pt \
#                      --precision fp32,fp16 --inputs="(${bs}, 3, 224, 224)" \
#                      --batch_size ${bs} \
#                      --backends torch,torch_tensorrt,tensorrt \
#                      --report "eff_b0_perf_bs${bs}.txt"
# done

# Benchmark BERT model
for bs in 1
do
  python perf_run.py --model models/bert_base_uncased_traced.jit.pt \
                     --precision fp32 --inputs="(${bs}, 128)@int32;(${bs}, 128)@int32" \
                     --batch_size ${bs} \
                     --backends torch_tensorrt \
                     --truncate \
                     --report "bert_base_perf_bs${bs}.txt"
done
