# C++ API

Targets in module create the user facing C++ library for the TRTorch core.

## Building libtrtorch.so

### Debug build
``` shell
bazel build //cpp/lib:libtrtorch.so --compilation_mode=dbg
```

### Release build

``` shell
bazel build //cpp/lib:libtrtorch.so -c opt
```

## Usage

``` c++
#include "trtorch/trtorch.h"
```