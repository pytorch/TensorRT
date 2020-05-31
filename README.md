# TRTorch

> Ahead of Time (AOT) compiling for PyTorch JIT

TRTorch is a compiler for PyTorch/TorchScript, targeting NVIDIA GPUs via NVIDIA's TensorRT Deep Learning Optimizer and Runtime. Unlike PyTorch's Just-In-Time (JIT) compiler, TRTorch is an Ahead-of-Time (AOT) compiler, meaning that before you deploy your TorchScript code, you go through an explicit compile step to convert a standard TorchScript program into an module targeting a TensorRT engine. TRTorch operates as a PyTorch extention and compiles modules that integrate into the JIT runtime seamlessly. After compilation using the optimized graph should feel no different than running a TorchScript module. You also have access to TensorRT's suite of configurations at compile time, so you are able to specify operating precision (FP32/FP16/INT8) and other settings for your module.

More Information / System Architecture:

- [GTC 2020 Talk](https://developer.nvidia.com/gtc/2020/video/s21671)

## Example Usage

### C++
```c++
#include "torch/script.h"
#include "trtorch/trtorch.h"

...
auto compile_settings = trtorch::ExtraInfo(dims);
// FP16 execution
compile_settings.op_precision = torch::kFloat;
// Compile module
auto trt_mod = trtorch::CompileGraph(ts_mod, compile_settings);
// Run like normal
auto results = trt_mod.forward({in_tensor});
// Save module for later
trt_mod.save("trt_torchscript_module.ts");
...
```

### Python
```py
import trtorch

...
compile_settings = {
    "input_shapes": [
        {
            "min": [1, 3, 224, 224],
            "opt": [1, 3, 512, 512],
            "max": [1, 3, 1024, 1024]
        }, # For static size [1, 3, 224, 224]
    ],
    "op_precision": torch.half # Run with FP16
}

trt_ts_module = trtorch.compile(torch_script_module, compile_settings)

input_data = input_data.half()
result = trt_ts_module(input_data)
torch.jit.save(trt_ts_module, "trt_torchscript_module.ts")
```

> Notes on running in lower precisions:
> - Set precision with extra_info.op_precision
> - The module should be left in FP32 before compilation (FP16 can support half tensor models)
> - In FP16 only input tensors should be converted to FP16, other precisions use FP32

## Platform Support

| Platform | Support |
| -------- | ------- |
| Linux AMD64 / GPU   | **Supported** |
| Linux aarch64 / GPU | **Planned/Possible with Native Compiation but untested** |
| Linux aarch64 / DLA | **Planned/Possible with Native Compilation but untested** |
| Windows / GPU       | - |
| Linux ppc64le / GPU | - |

### Dependencies

- Libtorch 1.5.0
- CUDA 10.2
- cuDNN 7.6.5
- TensorRT 7.0.0

## Prebuilt Binaries and Wheel files

Releases: https://github.com/NVIDIA/TRTorch/releases

## Compiling TRTorch

### Installing Dependencies

You need to start by having CUDA installed on the system, Libtorch will automatically be pulled for you by bazel,
then you have two options.

#### 1. Building using cuDNN & TensorRT tarball distributions

> This is recommended so as to build TRTorch hermetically and insures any bugs are not caused by version issues

> Make sure when running TRTorch that these versions of the libraries are prioritized in your `$LD_LIBRARY_PATH`

1. You need to download the tarball distributions of TensorRT and cuDNN from the NVIDIA website.
    - https://developer.nvidia.com/cudnn
    - https://developer.nvidia.com/tensorrt
2. Place these files in a directory (the directories `third_party/distdir/[x86_64-linux-gnu | aarch64-linux-gnu]` exist for this purpose)
3. Compile using:
``` shell
bazel build //:libtrtorch --compilation_mode opt --distdir third_party/distdir/[x86_64-linux-gnu | aarch64-linux-gnu]
```

#### 2. Building using locally installed cuDNN & TensorRT

> If you find bugs and you compiled using this method please disclose it in the issue
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
bazel build //:libtrtorch --compilation_mode opt
```

### Debug build
``` shell
bazel build //:libtrtorch --compilation_mode=dbg
```

A tarball with the include files and library can then be found in bazel-bin

### Running TRTorch on a JIT Graph

> Make sure to add LibTorch to your LD_LIBRARY_PATH <br>
>`export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)/bazel-TRTorch/external/libtorch/lib`


``` shell
bazel run //cpp/trtorchexec -- $(realpath <PATH TO GRAPH>) <input-size>
```

## Compiling the Python Package

To compile the python package for your local machine, just run `python3 setup.py install` in the `//py` directory.
To build wheel files for different python versions, first build the Dockerfile in ``//py`` then run the following
command
```
docker run -it -v$(pwd)/..:/workspace/TRTorch build_trtorch_wheel /bin/bash /workspace/TRTorch/py/build_whl.sh
```
Python compilation expects using the tarball based compilation strategy from above.

## How do I add support for a new op...

### In TRTorch?

Thanks for wanting to contribute! There are two main ways to handle supporting a new op. Either you can write a converter for the op from scratch and register it in the NodeConverterRegistry or if you can map the op to a set of ops that already have converters you can write a graph rewrite pass which will replace your new op with an equivalent subgraph of supported ops. Its preferred to use graph rewriting because then we do not need to maintain a large library of op converters. Also do look at the various op support trackers in the [issues](https://github.com/NVIDIA/TRTorch/issues) for information on the support status of various operators.

### In my application?

> The Node Converter Registry is not exposed in the top level API but in the internal headers shipped with the tarball.

You can register a converter for your op using the `NodeConverterRegistry` inside your application.

## Known Limitations

- You cannot use both Adaptive Pooling in PyTorch and also use TRTorch Dynamic input shape (follow [#49](https://github.com/NVIDIA/TRTorch/issues/49) for the latest on the issue)

## Structure of the repo

| Component     | Description                                                  |
| ------------- | ------------------------------------------------------------ |
| [**core**](core)  | Main JIT ingest, lowering, conversion and execution implementations |
| [**cpp**](cpp)   | C++ specific components including API and example applications |
| [**cpp/api**](cpp/api)   | C++ API for TRTorch |
| [**py**](py)   | Python API for TRTorch |
| [**tests**](tests) | Unit test for TRTorch |

## Contributing

Take a look at the [CONTRIBUTING.md](CONTRIBUTING.md)


## License

The TRTorch license can be found in the LICENSE file. It is licensed with a BSD Style licence
