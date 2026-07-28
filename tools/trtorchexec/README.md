# trtorchexec

This is a short example application that shows how to use TRTorch in a program. It also lets you quickly test if your TorchScript program is currently supported or if you may need to create some converters

## Compilation

``` shell
bazel build //cpp/trtorchexec --cxxopt="-DNDEBUG"
```

If you want insight into what is going under the hood or need debug symbols

``` shell
bazel build //cpp/trtorchexec --compilation_mode=dbg
```

## Usage

``` shell
trtorchexec <path-to-exported-script-module> <input-size>
trtorchexec <path-to-exported-script-module> <min-input-size> <opt-input-size> <max-input-size>
```

ex. `trtorchexec $(realpath tests/models/resnet50.jit.pt) "(32 3 224 224)"`
