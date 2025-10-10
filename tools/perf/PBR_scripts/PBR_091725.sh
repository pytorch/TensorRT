#!/bin/bash

OPT_LEVEL=5
WARMUP_ITER=10
ITERATIONS=20
OUTPUT_FOLDER="outputs"

mkdir -p $OUTPUT_FOLDER

# Format: model_name|batch_size|inputs|precision
model_configs=(
    "resnet50|128|(128, 3, 224, 224)@fp32|fp32"
    "efficientnet_b0|128|(128, 3, 224, 224)@fp32|fp32"
    "monai_unet|128|(128, 32, 224, 224)@fp32|fp32"
    "bert_base_uncased|128|(128, 128)@int32;(128, 128)@int32|fp16"
    "sd2.1_unet|32|(32, 4, 64, 64)@fp16;(32)@fp16;(32, 1, 1024)@fp16|fp16"
    "sd2.1_vae_decoder|32|(32, 4, 64, 64)@fp16|fp16"
    "google_vit|32|(32, 3, 224, 224)@fp16|fp16"
)

for model_config in "${model_configs[@]}"
do
    # Split the delimited string
    IFS='|' read -r model_name batch_size inputs precision <<< "$model_config"
    optimization_level=$OPT_LEVEL
    warmup_iter=$WARMUP_ITER
    iterations=$ITERATIONS
    echo "Benchmarking ${model_name} model for bs=${batch_size}, precision=${precision}, optimization_level=${optimization_level}, warmup_iter=${warmup_iter}, iterations=${iterations}"

    # Torch-TRT, ONNX-TRT to generate TRT engines
    python perf_run.py --backends=onnx_trt,dynamo \
                    --model_torch="${model_name}" \
                    --batch_size="${batch_size}" \
                    --inputs="${inputs}" \
                    --precision="${precision}" \
                    --optimization_level="${optimization_level}" \
                    --enable_cuda_graph \
                    --save_onnx_trt_engine \
                    --save_dynamo_trt_engine \
                    #    --report=${model_name}_perf_bs${batch_size}_optlevel${optimization_level}_${precision}_onnx_trt.csv

    onnx_trt_engine_name=${model_name}-onnx-trt-engine.trt
    dynamo_trt_engine_name=${model_name}-dynamo-trt-engine.trt

    # ONNX-TRT generated TRTEngine + trtexec
    trtexec --loadEngine=${onnx_trt_engine_name} --noTF32 --useCudaGraph --warmUp=${warmup_iter} --iterations=${iterations} 2>&1 | tee $OUTPUT_FOLDER/${model_name}_perf_onnx_trt_engine_with_trtexec.log
    # Torch-TRT generated TRTEngine + trtexec
    trtexec --loadEngine=${dynamo_trt_engine_name} --noTF32 --useCudaGraph --warmUp=${warmup_iter} --iterations=${iterations} 2>&1 | tee $OUTPUT_FOLDER/${model_name}_perf_torch_trt_engine_with_trtexec.log

    # ONNX-TRT generated TRTEngine + CppRuntime
    python hook_in_engine_to_torch_trt.py --model_name="${model_name}" --engine_name="${onnx_trt_engine_name}" --inputs="${inputs}" --batch_size="${batch_size}" --precision="${precision}" --enable_cuda_graph --iterations="${iterations}" --output_folder="$OUTPUT_FOLDER"
    # Torch-TRT generated TRTEngine + CppRuntime
    python hook_in_engine_to_torch_trt.py --model_name="${model_name}" --engine_name="${dynamo_trt_engine_name}" --inputs="${inputs}" --batch_size="${batch_size}" --precision="${precision}" --enable_cuda_graph --iterations="${iterations}" --output_folder="$OUTPUT_FOLDER"

    # ONNX-TRT generated TRTEngine + PythonRuntime
    python hook_in_engine_to_torch_trt.py --model_name="${model_name}" --engine_name="${onnx_trt_engine_name}" --inputs="${inputs}" --batch_size="${batch_size}" --precision="${precision}" --enable_cuda_graph --iterations="${iterations}" --use_python_runtime --output_folder="$OUTPUT_FOLDER"
    # Torch-TRT generated TRTEngine + PythonRuntime
    python hook_in_engine_to_torch_trt.py --model_name="${model_name}" --engine_name="${dynamo_trt_engine_name}" --inputs="${inputs}" --batch_size="${batch_size}" --precision="${precision}" --enable_cuda_graph --iterations="${iterations}" --use_python_runtime --output_folder="$OUTPUT_FOLDER"
done


# Format: model_name|batch_size
model_graph_break_configs=(
    "sd2.1_unet|32"
    "sd2.1_vae_decoder|32"
    "google_vit|32"
)

for model_config in "${model_graph_break_configs[@]}"
do
    # Split the delimited string
    IFS='|' read -r model_name batch_size <<< "$model_config"
    optimization_level=$OPT_LEVEL
    iterations=$ITERATIONS
    echo "Benchmarking ${model_name} model for bs=${batch_size}, optimization_level=${optimization_level}, iterations=${iterations}"

    # Torch-TRT w/o Graph break + CppRuntime
    python multi_backend_benchmark.py --model_name="${model_name}" --batch_size="${batch_size}" --iterations="${iterations}" --optimization_level="${optimization_level}" --output_folder="$OUTPUT_FOLDER"
    # Torch-TRT w/ Graph break (SDPA in inductor, rest in TRT) + CppRuntime
    python multi_backend_benchmark.py --model_name="${model_name}" --batch_size="${batch_size}" --iterations="${iterations}" --optimization_level="${optimization_level}" --output_folder="$OUTPUT_FOLDER" --sdpa_backend="inductor"
    # Torch-TRT w/ Graph break (SDPA in torch_eager, rest in TRT) + CppRuntime
    python multi_backend_benchmark.py --model_name="${model_name}" --batch_size="${batch_size}" --iterations="${iterations}" --optimization_level="${optimization_level}" --output_folder="$OUTPUT_FOLDER" --sdpa_backend="torch_eager"

    # Torch-TRT w/o Graph break + PythonRuntime
    python multi_backend_benchmark.py --model_name="${model_name}" --batch_size="${batch_size}" --iterations="${iterations}" --optimization_level="${optimization_level}" --use_python_runtime --output_folder="$OUTPUT_FOLDER"
    # Torch-TRT w/ Graph break (SDPA in inductor, rest in TRT) + PythonRuntime
    python multi_backend_benchmark.py --model_name="${model_name}" --batch_size="${batch_size}" --iterations="${iterations}" --optimization_level="${optimization_level}" --use_python_runtime --output_folder="$OUTPUT_FOLDER" --sdpa_backend="inductor"
    # Torch-TRT w/ Graph break (SDPA in torch_eager, rest in TRT) + PythonRuntime
    python multi_backend_benchmark.py --model_name="${model_name}" --batch_size="${batch_size}" --iterations="${iterations}" --optimization_level="${optimization_level}" --use_python_runtime --output_folder="$OUTPUT_FOLDER" --sdpa_backend="torch_eager"
done
