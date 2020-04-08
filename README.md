# TRTorch

> Ahead of Time (AOT) compiling for PyTorch JIT

TRTorch is a compiler for PyTorch/TorchScript, targeting NVIDIA GPUs via NVIDIA's TensorRT Deep Learning Optimizer and Runtime. Unlike PyTorch's Just-In-Time (JIT) compiler, TRTorch is an Ahead-of-Time (AOT) compiler, meaning that before you deploy your TorchScript code, you go through an explicit compile step to convert a standard TorchScript program into an module targeting a TensorRT engine. TRTorch operates as a PyTorch extention and compiles modules that integrate into the JIT runtime seamlessly. After compilation using the optimized graph should feel no different than running a TorchScript module. You also have access to TensorRT's suite of configurations at compile time, so you are able to specify operating precision (FP32/F16) and other settings for your module.

More Information / System Architecture:

- [GTC 2020 Talk](https://developer.nvidia.com/gtc/2020/video/s21671)

## Example Usage

```c++
#include "torch/script.h"
#include "trtorch/trtorch.h"

...
auto compile_settings = trtorch::ExtraInfo(dims);
// FP16 execution
compile_settings.op_precision = torch::kHalf;
// Compile module
auto trt_mod = trtorch::CompileGraph(ts_mod, compile_settings);
// Run like normal
auto results = trt_mod.forward({in_tensor});
...
```

## Platform Support

| Platform | Support |
| -------- | ------- |
| Linux AMD64 / GPU   | **Supported** |
| Linux aarch64 / GPU | **Planned/Possible with Native Compiation and small modifications to the build system** |
| Linux aarch64 / DLA | **Planned/Possible with Native Compilation but untested** |
| Windows / GPU       | - |
| Linux ppc64le / GPU | - |

### Dependencies

- Libtorch 1.4.0
- CUDA 10.1
- cuDNN 7.6
- TensorRT 6.0.1

## Prebuilt Binaries

Releases: https://github.com/NVIDIA/TRTorch/releases

## Compiling TRTorch

Install TensorRT, CUDA and cuDNN on the system before starting to compile.

``` shell
bazel build //:libtrtorch --compilation_mode=opt
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

## How do I add support for a new op...

### In TRTorch?

Thanks for wanting to contribute! There are two main ways to handle supporting a new op. Either you can write a converter for the op from scratch and register it in the NodeConverterRegistry or if you can map the op to a set of ops that already have converters you can write a graph rewrite pass which will replace your new op with an equivalent subgraph of supported ops. Its preferred to use graph rewriting because then we do not need to maintain a large library of op converters. Also do look at the various op support trackers in the [issues](https://github.com/NVIDIA/TRTorch/issues) for information on the support status of various operators.

### In my application?

> The Node Converter Registry is not exposed in the top level API but you can try using the internal headers shipped with the tarball.

You can register a converter for your op using the NodeConverterRegistry inside your application.

## Structure of the repo

| Component     | Description                                                  |
| ------------- | ------------------------------------------------------------ |
| [**core**](core)  | Main JIT ingest, lowering, conversion and execution implementations |
| [**cpp**](cpp)   | C++ specific components including API and example applications |
| [**cpp/api**](cpp/api)   | C++ API for TRTorch |
| [**tests**](tests) | Unit test for TRTorch |

## Contributing

Take a look at the [CONTRIBUTING.md](CONTRIBUTING.md)


## License

The TRTorch license can be found in the LICENSE file. It is licensed with a BSD Style licence
