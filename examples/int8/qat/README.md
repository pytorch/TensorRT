# Deploying QAT models using Torch-TensorRT

Quantization Aware training (QAT) simulates quantization during training by quantizing weights and activation layers. This will help reduce the loss in accuracy when we convert the network trained in FP32 to INT8 for faster inference. QAT introduces additional nodes in the graph which will be used to learn the dynamic ranges of weights and activation layers. Typical workflow for training QAT networks is to train a model until convergence and then finetune with the quantization layers.

For more detailed information, please refer to <a href="https://developer.nvidia.com/blog/achieving-fp32-accuracy-for-int8-inference-using-quantization-aware-training-with-tensorrt/">Achieving FP32 Accuracy for INT8 Inference Using Quantization Aware Training with NVIDIA TensorRT</a> blogpost.


## Running the Example Application

This is a short example application that shows how to use Torch-TensorRT to perform inference on a quantization-aware-trained model.

## Prerequisites

1. Download CIFAR10 Dataset Binary version ([https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz](https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz))
2. Train a network on CIFAR10 and perform quantization aware training on it. Refer to `examples/int8/training/vgg16/README.md` for detailed instructions.
   Export the QAT model to Torchscript.
3. Install NVIDIA's <a href="https://github.com/NVIDIA/TensorRT/tree/master/tools/pytorch-quantization">pytorch quantization toolkit</a>
4. TensorRT 8.0.1.6 or above

## Compilation using bazel

``` shell
bazel run //examples/int8/qat --compilation_mode=opt <path-to-module> <path-to-cifar10>
```

If you want insight into what is going under the hood or need debug symbols

``` shell
bazel run //examples/int8/qat --compilation_mode=dbg <path-to-module> <path-to-cifar10>
```

This will build a binary named `qat` in `bazel-out/k8-<opt|dbg>/bin/cpp/int8/qat/` directory. Optionally you can add this to `$PATH` environment variable to run `qat` from anywhere on your system.

## Compilation using Makefile

1) Download releases of <a href="https://pytorch.org">LibTorch</a>, <a href="https://github.com/pytorch/TensorRT/releases">Torch-TensorRT </a>and <a href="https://developer.nvidia.com/nvidia-tensorrt-download">TensorRT</a> and unpack them in the deps directory. Ensure CUDA is installed at `/usr/local/cuda` , if not you need to modify the CUDA include and lib paths in the Makefile.

```sh
cd examples/torch_tensorrt_example/deps
# Download latest Torch-TensorRT release tar file (libtorch_tensorrt.tar.gz) from https://github.com/pytorch/TensorRT/releases
tar -xvzf libtorch_tensorrt.tar.gz
# unzip libtorch downloaded from pytorch.org
unzip libtorch.zip
```

> If cuDNN and TensorRT are not installed on your system / in your LD_LIBRARY_PATH then do the following as well

```sh
cd deps
mkdir cudnn && tar -xvzf <cuDNN TARBALL> --directory cudnn --strip-components=1
mkdir tensorrt && tar -xvzf <TensorRT TARBALL> --directory tensorrt --strip-components=1
cd ..
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)/deps/torch_tensorrt/lib:$(pwd)/deps/libtorch/lib:$(pwd)/deps/tensorrt/lib:$(pwd)/deps/cudnn/lib64:/usr/local/cuda/lib
```

2) Build and run `qat`

We import header files `cifar10.h` and `benchmark.h` from `ROOT_DIR`. `ROOT_DIR` should point to the path where Torch-TensorRT is located `<path_to_torch_tensorrt>`.

By default it is set to `../../../`. If your Torch-TensorRT directory structure is different, please set `ROOT_DIR` accordingly.

```sh
cd examples/int8/qat
# This will generate a ptq binary
make ROOT_DIR=<PATH> CUDA_VERSION=11.1
./qat <path-to-module> <path-to-cifar10>
```

## Usage

``` shell
qat <path-to-module> <path-to-cifar10>
```

## Example Output

```
Accuracy of JIT model on test set: 92.1%
Compiling and quantizing module
Accuracy of quantized model on test set: 91.0044%
Latency of JIT model FP32 (Batch Size 32): 1.73497ms
Latency of quantized model (Batch Size 32): 0.365737ms
```

## Citations

```
Krizhevsky, A., & Hinton, G. (2009). Learning multiple layers of features from tiny images.
Simonyan, K., & Zisserman, A. (2014). Very deep convolutional networks for large-scale image recognition. arXiv preprint arXiv:1409.1556.
```
