# ptq

This is a short example application that shows how to use TRTorch to perform post-training quantization for a module.

## Compilation

``` shell
bazel build //cpp/ptq --cxxopt="-DNDEBUG"
```

If you want insight into what is going under the hood or need debug symbols

``` shell
bazel build //cpp/ptq --compilation_mode=dbg
```

## Usage

``` shell
ptq
```