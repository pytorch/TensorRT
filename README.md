<div align="center">

Torch-TensorRT
===========================
<h4> Easily achieve the best inference performance for any PyTorch model on the NVIDIA platform. </h4>

[![Documentation](https://img.shields.io/badge/docs-master-brightgreen)](https://nvidia.github.io/Torch-TensorRT/)
[![pytorch](https://img.shields.io/badge/PyTorch-2.8-green)](https://download.pytorch.org/whl/nightly/cu128)
[![cuda](https://img.shields.io/badge/CUDA-12.8-green)](https://developer.nvidia.com/cuda-downloads)
[![trt](https://img.shields.io/badge/TensorRT-10.12.0-green)](https://github.com/nvidia/tensorrt-llm)
[![license](https://img.shields.io/badge/license-BSD--3--Clause-blue)](./LICENSE)
[![Linux x86-64 Nightly Wheels](https://github.com/pytorch/TensorRT/actions/workflows/build-test-linux-x86_64.yml/badge.svg?branch=nightly)](https://github.com/pytorch/TensorRT/actions/workflows/build-test-linux-x86_64.yml)
[![Linux SBSA Nightly Wheels](https://github.com/pytorch/TensorRT/actions/workflows/build-test-linux-aarch64.yml/badge.svg?branch=nightly)](https://github.com/pytorch/TensorRT/actions/workflows/build-test-linux-aarch64.yml)
[![Windows Nightly Wheels](https://github.com/pytorch/TensorRT/actions/workflows/build-test-windows.yml/badge.svg?branch=nightly)](https://github.com/pytorch/TensorRT/actions/workflows/build-test-windows.yml)

---
<div align="left">

Torch-TensorRT brings the power of TensorRT to PyTorch. Accelerate inference latency by up to 5x compared to eager execution in just one line of code.
</div></div>

## Installation
Stable versions of Torch-TensorRT are published on PyPI
```bash
pip install torch-tensorrt
```

Nightly versions of Torch-TensorRT are published on the PyTorch package index
```bash
pip install --pre torch-tensorrt --index-url https://download.pytorch.org/whl/nightly/cu128
```

Torch-TensorRT is also distributed in the ready-to-run [NVIDIA NGC PyTorch Container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch) which has all dependencies with the proper versions and example notebooks included.

For more advanced installation  methods, please see [here](https://pytorch.org/TensorRT/getting_started/installation.html)

## Quickstart

### Option 1: torch.compile
You can use Torch-TensorRT anywhere you use `torch.compile`:

```python
import torch
import torch_tensorrt

model = MyModel().eval().cuda() # define your model here
x = torch.randn((1, 3, 224, 224)).cuda() # define what the inputs to the model will look like

optimized_model = torch.compile(model, backend="tensorrt")
optimized_model(x) # compiled on first run

optimized_model(x) # this will be fast!
```

### Option 2: Export
If you want to optimize your model ahead-of-time and/or deploy in a C++ environment, Torch-TensorRT provides an export-style workflow that serializes an optimized module. This module can be deployed in PyTorch or with libtorch (i.e. without a Python dependency).

#### Step 1: Optimize + serialize
```python
import torch
import torch_tensorrt

model = MyModel().eval().cuda() # define your model here
inputs = [torch.randn((1, 3, 224, 224)).cuda()] # define a list of representative inputs here

trt_gm = torch_tensorrt.compile(model, ir="dynamo", inputs=inputs)
torch_tensorrt.save(trt_gm, "trt.ep", inputs=inputs) # PyTorch only supports Python runtime for an ExportedProgram. For C++ deployment, use a TorchScript file
torch_tensorrt.save(trt_gm, "trt.ts", output_format="torchscript", inputs=inputs)
```

#### Step 2: Deploy
##### Deployment in PyTorch:
```python
import torch
import torch_tensorrt

inputs = [torch.randn((1, 3, 224, 224)).cuda()] # your inputs go here

# You can run this in a new python session!
model = torch.export.load("trt.ep").module()
# model = torch_tensorrt.load("trt.ep").module() # this also works
model(*inputs)
```

##### Deployment in C++:
```cpp
#include "torch/script.h"
#include "torch_tensorrt/torch_tensorrt.h"

auto trt_mod = torch::jit::load("trt.ts");
auto input_tensor = [...]; // fill this with your inputs
auto results = trt_mod.forward({input_tensor});
```

## Further resources
- [Double PyTorch Inference Speed for Diffusion Models Using Torch-TensorRT](https://developer.nvidia.com/blog/double-pytorch-inference-speed-for-diffusion-models-using-torch-tensorrt/)
- [Up to 50% faster Stable Diffusion inference with one line of code](https://pytorch.org/TensorRT/tutorials/_rendered_examples/dynamo/torch_compile_stable_diffusion.html#sphx-glr-tutorials-rendered-examples-dynamo-torch-compile-stable-diffusion-py)
- [Optimize LLMs from Hugging Face with Torch-TensorRT](https://docs.pytorch.org/TensorRT/tutorials/compile_hf_models.html#compile-hf-models)
- [Run your model in FP8 with Torch-TensorRT](https://pytorch.org/TensorRT/tutorials/_rendered_examples/dynamo/vgg16_fp8_ptq.html)
- [Accelerated Inference in PyTorch 2.X with Torch-TensorRT](https://www.youtube.com/watch?v=eGDMJ3MY4zk&t=1s)
- [Tools to resolve graph breaks and boost performance]() \[coming soon\]
- [Tech Talk (GTC '23)](https://www.nvidia.com/en-us/on-demand/session/gtcspring23-s51714/)
- [Documentation](https://nvidia.github.io/Torch-TensorRT/)


## Platform Support

| Platform            | Support                                          |
| ------------------- | ------------------------------------------------ |
| Linux AMD64 / GPU   | **Supported**                                    |
| Linux SBSA / GPU    | **Supported**                                    |
| Windows / GPU       | **Supported (Dynamo only)**                      |
| Linux Jetson / GPU | **Source Compilation Supported on JetPack-4.4+**  |
| Linux Jetson / DLA | **Source Compilation Supported on JetPack-4.4+**  |
| Linux ppc64le / GPU | Not supported                                    |

> Note: Refer [NVIDIA L4T PyTorch NGC container](https://ngc.nvidia.com/catalog/containers/nvidia:l4t-pytorch) for PyTorch libraries on JetPack.

### Dependencies

These are the following dependencies used to verify the testcases. Torch-TensorRT can work with other versions, but the tests are not guaranteed to pass.

- Bazel 8.1.1
- Libtorch 2.10.0.dev (latest nightly)
- CUDA 12.8 (CUDA 12.6 on Jetson)
- TensorRT 10.12 (TensorRT 10.3 on Jetson)

## Deprecation Policy

Deprecation is used to inform developers that some APIs and tools are no longer recommended for use. Beginning with version 2.3, Torch-TensorRT has the following deprecation policy:

Deprecation notices are communicated in the Release Notes. Deprecated API functions will have a statement in the source documenting when they were deprecated. Deprecated methods and classes will issue deprecation warnings at runtime, if they are used. Torch-TensorRT provides a 6-month migration period after the deprecation. APIs and tools continue to work during the migration period. After the migration period ends, APIs and tools are removed in a manner consistent with semantic versioning.

## Contributing

Take a look at the [CONTRIBUTING.md](CONTRIBUTING.md)


## License

The Torch-TensorRT license can be found in the [LICENSE](./LICENSE) file. It is licensed with a BSD Style licence
