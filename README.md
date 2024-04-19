<div align="center">

Torch-TensorRT
===========================
<h4> Easily achieve the best inference performance for any PyTorch model on the NVIDIA platform. </h4>

[![Documentation](https://img.shields.io/badge/docs-master-brightgreen)](https://nvidia.github.io/Torch-TensorRT/)
[![pytorch](https://img.shields.io/badge/PyTorch-2.2-green)](https://www.python.org/downloads/release/python-31013/)
[![cuda](https://img.shields.io/badge/cuda-12.1-green)](https://developer.nvidia.com/cuda-downloads)
[![trt](https://img.shields.io/badge/TensorRT-8.6.1-green)](https://github.com/nvidia/tensorrt-llm)
[![license](https://img.shields.io/badge/license-BSD--3--Clause-blue)](./LICENSE)
[![CircleCI](https://circleci.com/gh/pytorch/TensorRT.svg?style=svg)](https://app.circleci.com/pipelines/github/pytorch/TensorRT)

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
pip install --pre torch-tensorrt --index-url https://download.pytorch.org/whl/nightly/cu121
```

Torch-TensorRT is also distributed in the ready-to-run [NVIDIA NGC PyTorch Container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch) which has all dependencies with the proper versions and example notebooks included.

For more advanced installation  methods, please see [here](https://github.com/pytorch/tensorrt/INSTALLATION.md).

## Quickstart

### Option 1: torch.compile
You can use Torch-TensorRT anywhere you use `torch.compile`:

```python
import torch
import torch_tensorrt

model = <YOUR  MODEL HERE>
x = <YOUR INPUT HERE>

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

model = <YOUR  MODEL HERE>
x = <YOUR INPUT HERE>

optimized_model = torch_tensorrt.compile(model, example_inputs)
serialize # fix me
```

#### Step 2: Deploy
##### Deployment in PyTorch:
```python
import torch
import torch_tensorrt

x = <YOUR INPUT HERE>

# fix me
optimized_model = load_model
optimized_model(x) 
```

##### Deployment in C++:
```cpp
#include "torch/script.h"
#include "torch_tensorrt/torch_tensorrt.h"

// to fill
```

## Further resources
- [Optimize models from Hugging Face with Torch-TensorRT]() \[coming soon\]
- [Run your model in FP8 with Torch-TensorRT]() \[coming soon\]
- [Tools to resolve graph breaks and boost performance]() \[coming soon\]
- [Tech Talk (GTC '23)](https://www.nvidia.com/en-us/on-demand/session/gtcspring23-s51714/)
- [Documentation](https://nvidia.github.io/Torch-TensorRT/)


## Platform Support

| Platform            | Support                                          |
| ------------------- | ------------------------------------------------ |
| Linux AMD64 / GPU   | **Supported**                                    |
| Windows / GPU       | **Official support coming soon**                |
| Linux aarch64 / GPU | **Native Compilation Supported on JetPack-4.4+ (use v1.0.0 for the time being)** |
| Linux aarch64 / DLA | **Native Compilation Supported on JetPack-4.4+ (use v1.0.0 for the time being)** |
| Linux ppc64le / GPU | Not supported                                    |

> Note: Refer [NVIDIA L4T PyTorch NGC container](https://ngc.nvidia.com/catalog/containers/nvidia:l4t-pytorch) for PyTorch libraries on JetPack.

### Dependencies

These are the following dependencies used to verify the testcases. Torch-TensorRT can work with other versions, but the tests are not guaranteed to pass.

- Bazel 5.2.0
- Libtorch 2.3.0.dev (latest nightly) (built with CUDA 12.1)
- CUDA 12.1
- cuDNN 8.9.5
- TensorRT 8.6.1

## Deprecation Policy

Deprecation is used to inform developers that some APIs and tools are no longer recommended for use. Beginning with version 2.3, Torch-TensorRT has the following deprecation policy:

Deprecation notices are communicated in the Release Notes. Deprecated API functions will have a statement in the source documenting when they were deprecated. Deprecated methods and classes will issue deprecation warnings at runtime, if they are used. Torch-TensorRT provides a 6-month migration period after the deprecation. APIs and tools continue to work during the migration period. After the migration period ends, APIs and tools are removed in a manner consistent with semantic versioning.

## Contributing

Take a look at the [CONTRIBUTING.md](CONTRIBUTING.md)


## License

The Torch-TensorRT license can be found in the [LICENSE](./LICENSE) file. It is licensed with a BSD Style licence
