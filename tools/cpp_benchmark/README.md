# Benchmarking

This is a quick benchmarking application for Torch-TensorRT. It lets you run supported TorchScript modules both in JIT and TRT and returns the average runtime and throughput.

## Compilation / Usage

Run with bazel:

> Note: Make sure libtorch and TensorRT are in your LD_LIBRARY_PATH before running, if you need a location you can `export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:[WORKSPACE ROOT]/bazel-Torch-TensorRT/external/libtorch/lib:[WORKSPACE ROOT]/bazel-Torch-TensorRT/external/tensorrt/lib`

``` sh
bazel run //tools/cpp_benchmark --cxxopt="-DNDEBUG" --cxxopt="-DJIT" --cxxopt="-DTRT" -- [PATH TO JIT MODULE FILE] [INPUT SIZE (explicit batch)]
```

For example:

``` shell
bazel run //tools/cpp_benchmark  --cxxopt="-DNDEBUG" --cxxopt="-DJIT" --cxxopt="-DTRT" -- $(realpath /tests/models/resnet50.jit.pt) "(32 3 224 224)"
```

### Options

You can run a module with JIT or TRT via Torch-TensorRT in either FP32 or FP16. These options are controlled by preprocessor directives.

- To enable JIT profiling, add the argument `--cxxopt="-DJIT"`

- To enable TRT profiling, add the argument `--cxxopt="-DTRT"`

- To enable FP16 execution, add the argument `--cxxopt="-DHALF"`

- To also save the TRT engine, add the argument `--cxxopt="-DSAVE_ENGINE"`

> It's suggested to also define `--cxxopt="-DNDEBUG"` to supress debug information
