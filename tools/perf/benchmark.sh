#!/bin/bash

MODELS_DIR="models"

# Download the Torchscript models
python hub.py

batch_sizes=(1 2 4 8 16 32 64 128 256)
large_model_batch_sizes=(1 2 4 8 16 32 64)


# Benchmark VGG16 model
echo "Benchmarking VGG16 model"
for bs in ${batch_sizes[@]}
do
  python perf_run.py --model ${MODELS_DIR}/vgg16_scripted.jit.pt \
                     --model_torch ${MODELS_DIR}/vgg16_pytorch.pt \
                     --precision fp32,fp16 --inputs="(${bs}, 3, 224, 224)" \
                     --batch_size ${bs} \
                     --truncate \
                     --backends torch,ts_trt,dynamo,torch_compile,inductor \
                     --report "vgg16_perf_bs${bs}.txt"
done

# Benchmark AlexNet model
echo "Benchmarking AlexNet model"
for bs in ${batch_sizes[@]}
do
  python perf_run.py --model ${MODELS_DIR}/alexnet_scripted.jit.pt \
                     --model_torch ${MODELS_DIR}/alexnet_pytorch.pt \
                     --precision fp32,fp16 --inputs="(${bs}, 3, 227, 227)" \
                     --batch_size ${bs} \
                     --truncate \
                     --backends torch,ts_trt,dynamo,torch_compile,inductor \
                     --report "alexnet_perf_bs${bs}.txt"
done

# Benchmark Resnet50 model
echo "Benchmarking Resnet50 model"
for bs in ${batch_sizes[@]}
do
  python perf_run.py --model ${MODELS_DIR}/resnet50_scripted.jit.pt \
                     --model_torch ${MODELS_DIR}/resnet50_pytorch.pt \
                     --precision fp32,fp16 --inputs="(${bs}, 3, 224, 224)" \
                     --batch_size ${bs} \
                     --truncate \
                     --backends torch,ts_trt,dynamo,torch_compile,inductor \
                     --report "resnet50_perf_bs${bs}.txt"
done

# Benchmark VIT model
echo "Benchmarking VIT model"
for bs in ${batch_sizes[@]}
do
  python perf_run.py --model ${MODELS_DIR}/vit_scripted.jit.pt \
                     --model_torch ${MODELS_DIR}/vit_pytorch.pt \
                     --precision fp32,fp16 --inputs="(${bs}, 3, 224, 224)" \
                     --batch_size ${bs} \
                     --truncate \
                     --backends torch,ts_trt,dynamo,torch_compile,inductor \
                     --report "vit_perf_bs${bs}.txt"
done

# Benchmark VIT Large model
echo "Benchmarking VIT Large model"
for bs in ${large_model_batch_sizes[@]}
do
  python perf_run.py --model ${MODELS_DIR}/vit_large_scripted.jit.pt \
                     --model_torch ${MODELS_DIR}/vit_large_pytorch.pt \
                     --precision fp32,fp16 --inputs="(${bs}, 3, 224, 224)" \
                     --truncate \
                     --batch_size ${bs} \
                     --backends torch,ts_trt,dynamo,torch_compile,inductor \
                     --report "vit_large_perf_bs${bs}.txt"

# Benchmark EfficientNet-B0 model
echo "Benchmarking EfficientNet-B0 model"
for bs in ${batch_sizes[@]}
do
  python perf_run.py --model ${MODELS_DIR}/efficientnet_b0_scripted.jit.pt \
                     --model_torch ${MODELS_DIR}/efficientnet_b0_pytorch.pt \
                     --precision fp32,fp16 --inputs="(${bs}, 3, 224, 224)" \
                     --batch_size ${bs} \
                     --truncate \
                     --backends torch,ts_trt,dynamo,torch_compile,inductor \
                     --report "efficientnet_b0_perf_bs${bs}.txt"
done

# Benchmark Stable Diffusion UNet model
echo "Benchmarking SD UNet model"
for bs in ${large_model_batch_sizes[@]}
do
  python perf_run.py --model_torch ${MODELS_DIR}/sd_unet_pytorch.pt \
                     --precision fp32,fp16 --inputs="(${bs}, 4, 128, 128)@fp16;(${bs})@fp16;(${bs}, 1, 768)@fp16" \
                     --batch_size ${bs} \
                     --backends torch,dynamo,torch_compile,inductor \
                     --truncate \
                     --report "sd_unet_perf_bs${bs}.txt"
done

# Benchmark BERT model
echo "Benchmarking Huggingface BERT base model"
for bs in ${batch_sizes[@]}
do
  python perf_run.py --model ${MODELS_DIR}/bert_base_uncased_traced.jit.pt \
                     --model_torch "bert_base_uncased" \
                     --precision fp32 --inputs="(${bs}, 128)@int32;(${bs}, 128)@int32" \
                     --batch_size ${bs} \
                     --backends torch,ts_trt,dynamo,torch_compile,inductor \
                     --truncate \
                     --report "bert_base_perf_bs${bs}.txt"
done

# Collect and concatenate all results
echo "Concatenating all results"
(echo "Output of All Model Runs"; echo) >> all_outputs.txt;

for i in $(ls *_bs*.txt);
  do (echo $i; cat $i; echo; echo) >> all_outputs.txt;
done
