# C++ API

Targets in module create the user facing C++ library for the Torch-TensorRT core.

## Building libtorchtrt.so

### Debug build
``` shell
bazel build //cpp/lib:libtorchtrt.so --compilation_mode=dbg
```

### Release build

``` shell
bazel build //cpp/lib:libtorchtrt.so -c opt
```

## Usage

``` c++
#include "torch_tensorrt/torch_tensorrt.h"
```
