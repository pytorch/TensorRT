# Performance Benchmarking

This is a comprehensive Python benchmark suite to run perf runs using different supported backends. Following backends are supported:

1. Torch
2. Torch-TensorRT
3. FX-TRT
4. TensorRT


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
├── hub.py
├── custom_models.py
├── requirements.txt
├── benchmark.sh
└── README.md
```



* `config` - Directory which contains sample yaml configuration files for VGG network.
* `models` - Model directory
* `perf_run.py` - Performance benchmarking script which supports torch, torch_tensorrt, fx2trt, tensorrt backends
* `hub.py` - Script to download torchscript models for VGG16, Resnet50, EfficientNet-B0, VIT, HF-BERT
* `custom_models.py` - Script which includes custom models other than torchvision and timm (eg: HF BERT)
* `utils.py` - utility functions script
* `benchmark.sh` - This is used for internal performance testing of VGG16, Resnet50, EfficientNet-B0, VIT, HF-BERT.

## Usage

There are two ways you can run a performance benchmark.

### Using YAML config files

To run the benchmark for a given configuration file:

```python
python perf_run.py --config=config/vgg16.yml
```

There are two sample configuration files added.

* vgg16.yml demonstrates a configuration with all the supported backends (Torch, Torch-TensorRT, TensorRT)
* vgg16_trt.yml demonstrates how to use an external TensorRT serialized engine file directly.


### Supported fields

| Name              | Supported Values                     | Description                                                  |
| ----------------- | ------------------------------------ | ------------------------------------------------------------ |
| backend           | all, torch, torch_tensorrt, tensorrt, fx2trt | Supported backends for inference.                            |
| input             | -                                    | Input binding names. Expected to list shapes of each input bindings |
| model             | -                                    | Configure the model filename and name                        |
| model_torch             | -                              | Name of torch model file and name (used for fx2trt) (optional)                  |
| filename          | -                                    | Model file name to load from disk.                           |
| name              | -                                    | Model name                                                   |
| runtime           | -                                    | Runtime configurations                                       |
| device            | 0                                    | Target device ID to run inference. Range depends on available GPUs |
| precision         | fp32, fp16 or half, int8             | Target precision to run inference. int8 cannot be used with 'all' backend |
| calibration_cache | -                                    | Calibration cache file expected for torch_tensorrt runtime in int8 precision |

Additional sample use case:

```
backend:
  - torch
  - torch_tensorrt
  - tensorrt
  - fx2trt
input:
  input0:
    - 3
    - 224
    - 224
  num_inputs: 1
model:
  filename: model.plan
  name: vgg16
model_torch:
  filename: model_torch.pt
  name: vgg16
runtime:
  device: 0
  precision:
    - fp32
    - fp16
```

Note:

1. Please note that measuring INT8 performance is only supported via a `calibration cache` file or QAT mode for `torch_tensorrt` backend.
2. TensorRT engine filename should end with `.plan` otherwise it will be treated as Torchscript module.

### Using CompileSpec options via CLI

Here are the list of `CompileSpec` options that can be provided directly to compile the pytorch module

* `--backends` : Comma separated string of backends. Eg: torch, torch_tensorrt, tensorrt or fx2trt
* `--model` : Name of the model file (Can be a torchscript module or a tensorrt engine (ending in `.plan` extension)). If the backend is `fx2trt`, the input should be a Pytorch module (instead of a torchscript module) and the options for model are (`vgg16` | `resnet50` | `efficientnet_b0`)
* `--model_torch` : Name of the PyTorch model file (optional, only necessary if fx2trt is a chosen backend)
* `--inputs` : List of input shapes & dtypes. Eg: (1, 3, 224, 224)@fp32 for Resnet or (1, 128)@int32;(1, 128)@int32 for BERT
* `--batch_size` : Batch size
* `--precision` : Comma separated list of precisions to build TensorRT engine Eg: fp32,fp16
* `--device` : Device ID
* `--truncate` : Truncate long and double weights in the network in Torch-TensorRT
* `--is_trt_engine` : Boolean flag to be enabled if the model file provided is a TensorRT engine.
* `--report` : Path of the output file where performance summary is written.

Eg:

```
  python perf_run.py --model ${MODELS_DIR}/vgg16_scripted.jit.pt \
                     --model_torch ${MODELS_DIR}/vgg16_torch.pt \
                     --precision fp32,fp16 --inputs="(1, 3, 224, 224)@fp32" \
                     --batch_size 1 \
                     --backends torch,torch_tensorrt,tensorrt,fx2trt \
                     --report "vgg_perf_bs1.txt"
```

### Example models

This tool benchmarks any pytorch model or torchscript module. As an example, we provide VGG16, Resnet50, EfficientNet-B0, VIT, HF-BERT models in `hub.py` that we internally test for performance.
The torchscript modules for these models can be generated by running
```
python hub.py
```
You can refer to `benchmark.sh` on how we run/benchmark these models.
