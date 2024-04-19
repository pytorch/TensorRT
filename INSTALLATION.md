## Pre-built wheels
Stable versions of Torch-TensorRT are published on PyPI
```bash
pip install torch-tensorrt
```

Nightly versions of Torch-TensorRT are published on the PyTorch package index
```bash
pip install --pre torch-tensorrt --index-url https://download.pytorch.org/whl/nightly/cu121
```

Torch-TensorRT is also distributed in the ready-to-run [NVIDIA NGC PyTorch Container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch) which has all dependencies with the proper versions and example notebooks included.

## Building a docker container for Torch-TensorRT

We provide a `Dockerfile` in `docker/` directory. It expects a PyTorch NGC container as a base but can easily be modified to build on top of any container that provides, PyTorch, CUDA, cuDNN and TensorRT. The dependency libraries in the container can be found in the <a href="https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/index.html">release notes</a>.

Please follow this instruction to build a Docker container.

```bash
docker build --build-arg BASE=<CONTAINER VERSION e.g. 21.11> -f docker/Dockerfile -t torch_tensorrt:latest .
```

In the case of building on top of a custom base container, you first must determine the
version of the PyTorch C++ ABI. If your source of PyTorch is pytorch.org, likely this is the pre-cxx11-abi in which case you must modify `//docker/dist-build.sh` to not build the
C++11 ABI version of Torch-TensorRT.

You can then build the container using the build command in the [docker README](docker/README.md#instructions)

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

