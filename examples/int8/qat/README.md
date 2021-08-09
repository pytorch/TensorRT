# Deploying QAT models using TRTorch

Quantization Aware training (QAT) simulates quantization during training by quantizing weights and activation layers. This will help reduce the loss in accuracy when we convert the network trained in FP32 to INT8 for faster inference. QAT introduces additional nodes in the graph which will be used to learn the dynamic ranges of weights and activation layers. Typical workflow for training QAT networks is to train a model until convergence and then finetune with the quantization layers.

For more detailed information, please refer to <a href="https://developer.nvidia.com/blog/achieving-fp32-accuracy-for-int8-inference-using-quantization-aware-training-with-tensorrt/">Achieving FP32 Accuracy for INT8 Inference Using Quantization Aware Training with NVIDIA TensorRT</a> blogpost.


## Running the Example Application

This is a short example application that shows how to use TRTorch to perform inference on a quantization-aware-trained model.

## Prerequisites

1. Download CIFAR10 Dataset Binary version ([https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz](https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz))
2. Train a network on CIFAR10 and perform quantization aware training on it. Refer to `cpp/int8/training/vgg16/README.md` for detailed instructions.
   Export the QAT model to Torchscript.
3. Install NVIDIA's <a href="https://github.com/NVIDIA/TensorRT/tree/master/tools/pytorch-quantization">pytorch quantization toolkit</a>
4. TensorRT 8.0.1.6 or above

## Compilation

``` shell
bazel build //cpp/qat --compilation_mode=opt
```

If you want insight into what is going under the hood or need debug symbols

``` shell
bazel build //cpp/qat --compilation_mode=dbg
```

This will build a binary named `qat` in `bazel-out/k8-<opt|dbg>/bin/cpp/int8/qat/` directory. Optionally you can add this to `$PATH` environment variable to run `qat` from anywhere on your system.

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
