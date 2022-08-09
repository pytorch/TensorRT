# Performance Benchmarking

This is a comprehensive Python benchmark suite to run perf runs using different supported backends. Following backends are supported:

1. Torch
2. Torch-TensorRT
3. TensorRT

Note: Please note that for ONNX models, user can convert the ONNX model to TensorRT serialized engine and then use this package.

## Prerequisite

Benchmark scripts depends on following Python packages in addition to requirements.txt packages

1. Torch-TensorRT
2. Torch
3. TensorRT

## Structure

```
./
├── config
│   ├── vgg16_trt.yml
│   └── vgg16.yml
├── models
├── perf_run.py
└── README.md
```

Please save your configuration files at config directory. Similarly, place your model files at models path.

## Usage

To run the benchmark for a given configuration file:

```
python perf_run.py --config=config/vgg16.yml
```

## Configuration

There are two sample configuration files added.

* vgg16.yml demonstrates a configuration with all the supported backends (Torch, Torch-TensorRT, TensorRT)
* vgg16_trt.yml demonstrates how to use an external TensorRT serialized engine file directly.


### Supported fields

| Name | Supported Values | Description |
| --- | --- | --- |
| backend | all, torch, torch_tensorrt, tensorrt | Supported backends for inference. |
| input | - | Input binding names. Expected to list shapes of each input bindings |
| model | - | Configure the model filename and name |
| filename | - | Model file name to load from disk. |
| name | - | Model name |
| runtime | - | Runtime configurations |
| device | 0 | Target device ID to run inference. Range depends on available GPUs |
| precision | fp32, fp16 or half, int8 | Target precision to run inference. int8 cannot be used with 'all' backend |
| calibration_cache | - | Calibration cache file expected for torch_tensorrt runtime in int8 precision |

Note:
1. Please note that torch runtime perf is not supported for int8 yet.
2. Torchscript module filename should end with .jit.pt otherwise it will be treated as a TensorRT engine.



Additional sample use case:

```
backend:
  - torch
  - torch_tensorrt
  - tensorrt
input:
  input0:
    - 3
    - 224
    - 224
  num_inputs: 1
model:
  filename: model.plan
  name: vgg16
runtime:
  device: 0
  precision:
    - fp32
    - fp16
```
