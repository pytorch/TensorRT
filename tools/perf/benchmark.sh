#!/bin/bash

MODELS_DIR="models"

# Download the Torchscript models
python hub.py

batch_sizes=(1 2 4 8 16 32 64 128 256)
large_model_batch_sizes=(1 2 4 8 16 32 64)
backends=("torch" "ts_trt" "dynamo" "torch_compile" "inductor" "onnx_trt")
backends_no_torchscript=("torch" "dynamo" "torch_compile" "inductor" "onnx_trt")


# Benchmark VGG16 model
echo "Benchmarking VGG16 model"
for bs in ${batch_sizes[@]}
do
  for backend in ${backends[@]}
  do
    python perf_run.py --model ${MODELS_DIR}/vgg16_scripted.jit.pt \
                       --model_torch vgg16 \
                       --precision fp16 --inputs="(${bs}, 3, 224, 224)" \
                       --batch_size ${bs} \
                       --truncate \
                       --backends ${backend} \
                       --report "vgg16_perf_bs${bs}_backend_${backend}.csv"
  done
done

# Benchmark AlexNet model
echo "Benchmarking AlexNet model"
for bs in ${batch_sizes[@]}
do
  for backend in ${backends[@]}
  do
    python perf_run.py --model ${MODELS_DIR}/alexnet_scripted.jit.pt \
                       --model_torch alexnet \
                       --precision fp16 --inputs="(${bs}, 3, 227, 227)" \
                       --batch_size ${bs} \
                       --truncate \
                       --backends ${backend} \
                       --report "alexnet_perf_bs${bs}_backend_${backend}.csv"
  done
done

# Benchmark Resnet50 model
echo "Benchmarking Resnet50 model"
for bs in ${batch_sizes[@]}
do
  for backend in ${backends[@]}
  do
    python perf_run.py --model ${MODELS_DIR}/resnet50_scripted.jit.pt \
                       --model_torch resnet50 \
                       --precision fp16 --inputs="(${bs}, 3, 224, 224)" \
                       --batch_size ${bs} \
                       --truncate \
                       --backends ${backend} \
                       --report "resnet50_perf_bs${bs}_backend_${backend}.csv"
  done
done

# Benchmark VIT model
echo "Benchmarking VIT model"
for bs in ${batch_sizes[@]}
do
  for backend in ${backends[@]}
  do
    python perf_run.py --model ${MODELS_DIR}/vit_scripted.jit.pt \
                       --model_torch vit \
                       --precision fp16 --inputs="(${bs}, 3, 224, 224)" \
                       --batch_size ${bs} \
                       --truncate \
                       --backends ${backend} \
                       --report "vit_perf_bs${bs}_backend_${backend}.csv"
  done
done

# Benchmark VIT Large model
echo "Benchmarking VIT Large model"
for bs in ${large_model_batch_sizes[@]}
do
  for backend in ${backends[@]}
  do
    python perf_run.py --model ${MODELS_DIR}/vit_large_scripted.jit.pt \
                       --model_torch vit_large \
                       --precision fp16 --inputs="(${bs}, 3, 224, 224)" \
                       --batch_size ${bs} \
                       --truncate \
                       --backends ${backend} \
                       --report "vit_large_perf_bs${bs}_backend_${backend}.csv"
  done
done

# Benchmark EfficientNet-B0 model
echo "Benchmarking EfficientNet-B0 model"
for bs in ${batch_sizes[@]}
do
  for backend in ${backends[@]}
  do
    python perf_run.py --model ${MODELS_DIR}/efficientnet_b0_scripted.jit.pt \
                       --model_torch efficientnet_b0 \
                       --precision fp16 --inputs="(${bs}, 3, 224, 224)" \
                       --batch_size ${bs} \
                       --truncate \
                       --backends ${backend} \
                       --report "efficientnet_b0_perf_bs${bs}_backend_${backend}.csv"
  done
done

# Benchmark Stable Diffusion v1.4 UNet model
echo "Benchmarking SD-v1.4 UNet model"
for bs in ${large_model_batch_sizes[@]}
do
  for backend in ${backends_no_torchscript[@]}
  do
    python perf_run.py --model_torch sd1.4_unet \
                       --precision fp16 --inputs="(${bs}, 4, 64, 64);(${bs});(${bs}, 1, 768)" \
                       --batch_size ${bs} \
                       --truncate \
                       --backends ${backend} \
                       --report "sd1.4_unet_perf_bs${bs}_backend_${backend}.csv"
  done
done

# Benchmark Stable Diffusion v2.1 UNet model
echo "Benchmarking SD-v2.1 UNet model"
for bs in ${large_model_batch_sizes[@]}
do
  for backend in ${backends_no_torchscript[@]}
  do
    python perf_run.py --model_torch sd2.1_unet \
                       --precision fp16 --inputs="(${bs}, 4, 64, 64);(${bs});(${bs}, 1, 1024)" \
                       --batch_size ${bs} \
                       --truncate \
                       --backends ${backend} \
                       --report "sd2.1_unet_perf_bs${bs}_backend_${backend}.csv"
  done
done

# Benchmark Stable Diffusion v2.1 VAE decoder model
echo "Benchmarking SD-v2.1 VAE decoder model"
for bs in ${large_model_batch_sizes[@]}
do
  for backend in ${backends_no_torchscript[@]}
  do
    python perf_run.py --model_torch sd2.1_vae_decoder \
                       --precision fp16 --inputs="(${bs}, 4, 64, 64)" \
                       --batch_size ${bs} \
                       --truncate \
                       --backends ${backend} \
                       --report "sd2.1_vae_decoder_perf_bs${bs}_backend_${backend}.csv"
  done
done

# Benchmark BERT model
echo "Benchmarking Huggingface BERT base model"
for bs in ${batch_sizes[@]}
do
  for backend in ${backends[@]}
  do
    python perf_run.py --model ${MODELS_DIR}/bert_base_uncased_traced.jit.pt \
                       --model_torch "bert_base_uncased" \
                       --precision fp16 --inputs="(${bs}, 128)@int32;(${bs}, 128)@int32" \
                       --batch_size ${bs} \
                       --truncate \
                       --backends ${backend} \
                       --report "bert_base_perf_bs${bs}_backend_${backend}.csv"
  done
done

# Collect and concatenate all results
echo "Concatenating all results"
python accumulate_results.py
