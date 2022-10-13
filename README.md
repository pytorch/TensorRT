# Torch-TensorRT

[![Documentation](https://img.shields.io/badge/docs-master-brightgreen)](https://nvidia.github.io/Torch-TensorRT/)

> Ahead of Time (AOT) compiling for PyTorch JIT and FX

Torch-TensorRT is a compiler for PyTorch/TorchScript/FX, targeting NVIDIA GPUs via NVIDIA's TensorRT Deep Learning Optimizer and Runtime. Unlike PyTorch's Just-In-Time (JIT) compiler, Torch-TensorRT is an Ahead-of-Time (AOT) compiler, meaning that before you deploy your TorchScript code, you go through an explicit compile step to convert a standard TorchScript or FX program into an module targeting a TensorRT engine. Torch-TensorRT operates as a PyTorch extention and compiles modules that integrate into the JIT runtime seamlessly. After compilation using the optimized graph should feel no different than running a TorchScript module. You also have access to TensorRT's suite of configurations at compile time, so you are able to specify operating precision (FP32/FP16/INT8) and other settings for your module.

Resources:
- [Documentation](https://nvidia.github.io/Torch-TensorRT/)
- [FX path Documentation](https://github.com/pytorch/TensorRT/blob/master/docsrc/tutorials/getting_started_with_fx_path.rst)
- [Torch-TensorRT Explained in 2 minutes!](https://www.youtube.com/watch?v=TU5BMU6iYZ0&ab_channel=NVIDIADeveloper)
- [Comprehensive Discusion (GTC Event)](https://www.nvidia.com/en-us/on-demand/session/gtcfall21-a31107/)
- [Pre-built Docker Container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch). To use this container, make an NGC account and sign in to NVIDIA's registry with an API key. Refer to [this guide](https://docs.nvidia.com/ngc/ngc-catalog-user-guide/index.html#registering-activating-ngc-account) for the same.

## NVIDIA NGC Container
Torch-TensorRT is distributed in the ready-to-run NVIDIA [NGC PyTorch Container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch) starting with 21.11. We recommend using this prebuilt container to experiment & develop with Torch-TensorRT; it has all dependencies with the proper versions as well as example notebooks included.

## Building a docker container for Torch-TensorRT

We provide a `Dockerfile` in `docker/` directory. It expects a PyTorch NGC container as a base but can easily be modified to build on top of any container that provides, PyTorch, CUDA, cuDNN and TensorRT. The dependency libraries in the container can be found in the <a href="https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/index.html">release notes</a>.

Please follow this instruction to build a Docker container.

```bash
docker build --build-arg BASE=<CONTAINER VERSION e.g. 21.11> -f docker/Dockerfile -t torch_tensorrt:latest .
```

In the case of building on top of a custom base container, you first must determine the
version of the PyTorch C++ ABI. If your source of PyTorch is pytorch.org, likely this is the pre-cxx11-abi in which case you must modify `//docker/dist-build.sh` to not build the
C++11 ABI version of Torch-TensorRT.

You can then build the container using:


```bash
docker build --build-arg BASE_IMG=<IMAGE> -f docker/Dockerfile -t torch_tensorrt:latest .
```

If you would like to build outside a docker container, please follow the section [Compiling Torch-TensorRT](#compiling-torch-tensorrt)

## Example Usage

### C++

```c++
#include "torch/script.h"
#include "torch_tensorrt/torch_tensorrt.h"

...
// Set input datatypes. Allowerd options torch::{kFloat, kHalf, kChar, kInt32, kBool}
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
```

### Python

```py
import torch_tensorrt

...

trt_ts_module = torch_tensorrt.compile(torch_script_module,
    inputs = [example_tensor, # Provide example tensor for input shape or...
        torch_tensorrt.Input( # Specify input object with shape and dtype
            min_shape=[1, 3, 224, 224],
            opt_shape=[1, 3, 512, 512],
            max_shape=[1, 3, 1024, 1024],
            # For static size shape=[1, 3, 224, 224]
            dtype=torch.half) # Datatype of input tensor. Allowed options torch.(float|half|int8|int32|bool)
    ],
    enabled_precisions = {torch.half}, # Run with FP16
)

result = trt_ts_module(input_data) # run inference
torch.jit.save(trt_ts_module, "trt_torchscript_module.ts") # save the TRT embedded Torchscript
```

> Notes on running in lower precisions:
>
> - Enabled lower precisions with compile_spec.enabled_precisions
> - The module should be left in FP32 before compilation (FP16 can support half tensor models)
> - Provided input tensors dtype should be the same as module before compilation, regardless of `enabled_precisions`. This can be overrided by setting `Input::dtype`

## Platform Support

| Platform            | Support                                          |
| ------------------- | ------------------------------------------------ |
| Linux AMD64 / GPU   | **Supported**                                    |
| Linux aarch64 / GPU | **Native Compilation Supported on JetPack-4.4+ (use v1.0.0 for the time being)** |
| Linux aarch64 / DLA | **Native Compilation Supported on JetPack-4.4+ (use v1.0.0 for the time being)** |
| Windows / GPU       | **Unofficial Support**                           |
| Linux ppc64le / GPU | -                                                |
| NGC Containers      | **Included in PyTorch NGC Containers 21.11+**   |

> Torch-TensorRT will be included in NVIDIA NGC containers (https://ngc.nvidia.com/catalog/containers/nvidia:pytorch) starting in 21.11.

> Note: Refer NVIDIA NGC container(https://ngc.nvidia.com/catalog/containers/nvidia:l4t-pytorch) for PyTorch libraries on JetPack.

### Dependencies

These are the following dependencies used to verify the testcases. Torch-TensorRT can work with other versions, but the tests are not guaranteed to pass.

- Bazel 5.2.0
- Libtorch 1.12.1 (built with CUDA 11.6)
- CUDA 11.6
- cuDNN 8.4.1
- TensorRT 8.4.3.1

## Prebuilt Binaries and Wheel files

Releases: https://github.com/pytorch/TensorRT/releases

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
> `export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)/bazel-Torch-TensorRT/external/libtorch/lib`

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

Thanks for wanting to contribute! There are two main ways to handle supporting a new op. Either you can write a converter for the op from scratch and register it in the NodeConverterRegistry or if you can map the op to a set of ops that already have converters you can write a graph rewrite pass which will replace your new op with an equivalent subgraph of supported ops. Its preferred to use graph rewriting because then we do not need to maintain a large library of op converters. Also do look at the various op support trackers in the [issues](https://github.com/pytorch/TensorRT/issues) for information on the support status of various operators.

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

## Contributing

Take a look at the [CONTRIBUTING.md](CONTRIBUTING.md)


## License

The Torch-TensorRT license can be found in the LICENSE file. It is licensed with a BSD Style licence
