# ptq

This is a short example application that shows how to use TRTorch to perform post-training quantization for a module.

## Prerequisites

1. Download CIFAR10 Dataset Binary version
2. Train a network on CIFAR10 (see `training/` for a VGG16 recipie)
3. Export model to torchscript

## Compilation

``` shell
bazel build //cpp/ptq --compilation_mode=opt
```

If you want insight into what is going under the hood or need debug symbols

``` shell
bazel build //cpp/ptq --compilation_mode=dbg
```

## Usage

``` shell
ptq <path-to-module> <path-to-cifar10>
```

## Citations

```
Krizhevsky, A., & Hinton, G. (2009). Learning multiple layers of features from tiny images.
Simonyan, K., & Zisserman, A. (2014). Very deep convolutional networks for large-scale image recognition. arXiv preprint arXiv:1409.1556.
```
