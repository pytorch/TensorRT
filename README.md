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

model = MyModel().eval().cuda() # define your model here
x = [torch.randn((1, 3, 224, 224)).cuda()] # define a list of relevant inputs here

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
inputs = [torch.randn((1, 3, 224, 224)).cuda()] # define a list of relevant inputs here

trt_gm = torch_tensorrt.compile(model, ir="dynamo", inputs) 
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

<<<<<<< HEAD
auto trt_mod = torch::jit::load("trt.ts");
auto input_tensor = [...]; // fill this with your inputs
auto results = trt_mod.forward({input_tensor});
=======
...
// Set input datatypes. Allowed options torch::{kFloat, kHalf, kChar, kInt32, kBool}
// Size of input_dtypes should match number of inputs to the network.
// If input_dtypes is not set, default precision follows traditional PyT / TRT rules
auto input = torch_tensorrt::Input(dims, torch::kHalf);
auto compile_settings = torch_tensorrt::ts::CompileSpec({input});
// FP16 execution
compile_settings.enabled_precisions = {torch::kHalf};
// Compile module
auto trt_mod = torch_tensorrt::ts::compile(ts_mod, compile_settings);
// Run like normal
auto results = trt_mod.forward({in_tensor});
// Save module for later
trt_mod.save("trt_torchscript_module.ts");
...
>>>>>>> 1a89aea5b (Fix minor grammatical corrections (#2779))
```

## Further resources
- [Up to 50% faster Stable Diffusion inference with one line of code](https://pytorch.org/TensorRT/tutorials/_rendered_examples/dynamo/torch_compile_stable_diffusion.html#sphx-glr-tutorials-rendered-examples-dynamo-torch-compile-stable-diffusion-py)
- [Optimize LLMs from Hugging Face with Torch-TensorRT]() \[coming soon\]
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
- Libtorch 2.4.0.dev (latest nightly) (built with CUDA 12.1)
- CUDA 12.1
- cuDNN 8.9.5
- TensorRT 10.0.0.6

## Deprecation Policy

Deprecation is used to inform developers that some APIs and tools are no longer recommended for use. Beginning with version 2.3, Torch-TensorRT has the following deprecation policy:

<<<<<<< HEAD
Deprecation notices are communicated in the Release Notes. Deprecated API functions will have a statement in the source documenting when they were deprecated. Deprecated methods and classes will issue deprecation warnings at runtime, if they are used. Torch-TensorRT provides a 6-month migration period after the deprecation. APIs and tools continue to work during the migration period. After the migration period ends, APIs and tools are removed in a manner consistent with semantic versioning.
=======
```
pip install tensorrt torch-tensorrt
```

## Compiling Torch-TensorRT

### Installing Dependencies

#### 0. Install Bazel

If you don't have bazel installed, the easiest way is to install bazelisk using the method of you choosing https://github.com/bazelbuild/bazelisk

Otherwise you can use the following instructions to install binaries https://docs.bazel.build/versions/master/install.html

Finally if you need to compile from source (e.g. aarch64 until bazel distributes binaries for the architecture) you can use these instructions

```sh
export BAZEL_VERSION=<VERSION>
mkdir bazel
cd bazel
curl -fSsL -O https://github.com/bazelbuild/bazel/releases/download/$BAZEL_VERSION/bazel-$BAZEL_VERSION-dist.zip
unzip bazel-$BAZEL_VERSION-dist.zip
bash ./compile.sh
```

You need to start by having CUDA installed on the system, LibTorch will automatically be pulled for you by bazel,
then you have two options.

#### 1. Building using cuDNN & TensorRT tarball distributions

> This is recommended so as to build Torch-TensorRT hermetically and insures any bugs are not caused by version issues

> Make sure when running Torch-TensorRT that these versions of the libraries are prioritized in your `$LD_LIBRARY_PATH`

1. You need to download the tarball distributions of TensorRT and cuDNN from the NVIDIA website.
   - https://developer.nvidia.com/cudnn
   - https://developer.nvidia.com/tensorrt
2. Place these files in a directory (the directories `third_party/dist_dir/[x86_64-linux-gnu | aarch64-linux-gnu]` exist for this purpose)
3. Compile using:

``` shell
bazel build //:libtorchtrt --compilation_mode opt --distdir third_party/dist_dir/[x86_64-linux-gnu | aarch64-linux-gnu]
```

#### 2. Building using locally installed cuDNN & TensorRT

> If you find bugs and you compiled using this method please disclose you used this method in the issue
> (an `ldd` dump would be nice too)

1. Install TensorRT, CUDA and cuDNN on the system before starting to compile.
2. In `WORKSPACE` comment out

```py
# Downloaded distributions to use with --distdir
http_archive(
    name = "cudnn",
    urls = ["<URL>",],

    build_file = "@//third_party/cudnn/archive:BUILD",
    sha256 = "<TAR SHA256>",
    strip_prefix = "cuda"
)

http_archive(
    name = "tensorrt",
    urls = ["<URL>",],

    build_file = "@//third_party/tensorrt/archive:BUILD",
    sha256 = "<TAR SHA256>",
    strip_prefix = "TensorRT-<VERSION>"
)
```

and uncomment

```py
# Locally installed dependencies
new_local_repository(
    name = "cudnn",
    path = "/usr/",
    build_file = "@//third_party/cudnn/local:BUILD"
)

new_local_repository(
   name = "tensorrt",
   path = "/usr/",
   build_file = "@//third_party/tensorrt/local:BUILD"
)
```

3. Compile using:

``` shell
bazel build //:libtorchtrt --compilation_mode opt
```

### FX path (Python only) installation
If the user plans to try FX path (Python only) and would like to avoid bazel build. Please follow the steps below.
``` shell
cd py && python3 setup.py install --fx-only
```

### Debug build

``` shell
bazel build //:libtorchtrt --compilation_mode=dbg
```

### Native compilation on NVIDIA Jetson AGX
We performed end to end testing on Jetson platform using Jetpack SDK 4.6.

``` shell
bazel build //:libtorchtrt --platforms //toolchains:jetpack_4.6
```

> Note: Please refer [installation](docs/tutorials/installation.html) instructions for Pre-requisites

A tarball with the include files and library can then be found in bazel-bin

### Running Torch-TensorRT on a JIT Graph

> Make sure to add LibTorch to your LD_LIBRARY_PATH <br>
> `export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)/bazel-TensorRT/external/libtorch/lib`

``` shell
bazel run //cpp/bin/torchtrtc -- $(realpath <PATH TO GRAPH>) out.ts <input-size>
```

## Compiling the Python Package

To compile the python package for your local machine, just run `python3 setup.py install` in the `//py` directory.
To build wheel files for different python versions, first build the Dockerfile in ``//py`` then run the following
command

```
docker run -it -v$(pwd)/..:/workspace/Torch-TensorRT build_torch_tensorrt_wheel /bin/bash /workspace/Torch-TensorRT/py/build_whl.sh
```

Python compilation expects using the tarball based compilation strategy from above.


## Testing using Python backend

Torch-TensorRT supports testing in Python using [nox](https://nox.thea.codes/en/stable)

To install the nox using python-pip

```
python3 -m pip install --upgrade nox
```

To list supported nox sessions:

```
nox --session -l
```

Environment variables supported by nox

```
PYT_PATH          - To use different PYTHONPATH than system installed Python packages
TOP_DIR           - To set the root directory of the noxfile
USE_CXX11         - To use cxx11_abi (Defaults to 0)
USE_HOST_DEPS     - To use host dependencies for tests (Defaults to 0)
```

Usage example

```
nox --session l0_api_tests
```

Supported Python versions:
```
["3.7", "3.8", "3.9", "3.10"]
```

## How do I add support for a new op...

### In Torch-TensorRT?

Thanks for wanting to contribute! There are two main ways to handle supporting a new op. Either you can write a converter for the op from scratch and register it in the NodeConverterRegistry or if you can map the op to a set of ops that already have converters you can write a graph rewrite pass which will replace your new op with an equivalent subgraph of supported ops. It's preferred to use graph rewriting because then we do not need to maintain a large library of op converters. Also do look at the various op support trackers in the [issues](https://github.com/pytorch/TensorRT/issues) for information on the support status of various operators.

### In my application?

> The Node Converter Registry is not exposed in the top level API but in the internal headers shipped with the tarball.

You can register a converter for your op using the `NodeConverterRegistry` inside your application.

## Structure of the repo

| Component                | Description                                                  |
| ------------------------ | ------------------------------------------------------------ |
| [**core**](core)         | Main JIT ingest, lowering, conversion and runtime implementations |
| [**cpp**](cpp)           | C++ API and CLI source                                       |
| [**examples**](examples) | Example applications to show different features of Torch-TensorRT |
| [**py**](py)             | Python API for Torch-TensorRT                                |
| [**tests**](tests)       | Unit tests for Torch-TensorRT                                |
>>>>>>> 1a89aea5b (Fix minor grammatical corrections (#2779))

## Contributing

Take a look at the [CONTRIBUTING.md](CONTRIBUTING.md)


## License

The Torch-TensorRT license can be found in the [LICENSE](./LICENSE) file. It is licensed with a BSD Style licence
