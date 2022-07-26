# ptq

## How to create your own PTQ application

Post Training Quantization (PTQ) is a technique to reduce the required computational resources for inference while still preserving the accuracy of your model by mapping the traditional FP32 activation space to a reduced INT8 space. TensorRT uses a calibration step which executes your model with sample data from the target domain and track the activations in FP32 to calibrate a mapping to INT8 that minimizes the information loss between FP32 inference and INT8 inference.

Users writing TensorRT applications are required to setup a calibrator class which will provide sample data to the TensorRT calibrator. With Torch-TensorRT we look to leverage existing infrastructure in PyTorch to make implementing calibrators easier.

LibTorch provides a `Dataloader` and `Dataset` API which steamlines preprocessing and batching input data. Torch-TensorRT uses Dataloaders as the base of a generic calibrator implementation. So you will be able to reuse or quickly implement a `torch::Dataset` for your target domain, place it in a Dataloader and create a INT8 Calibrator from it which you can provide to Torch-TensorRT to run INT8 Calibration during compliation of your module.

### Code

Here is an example interface of a `torch::Dataset` class for CIFAR10:

```C++
//examples/int8/ptq/datasets/cifar10.h
#pragma once

#include "torch/data/datasets/base.h"
#include "torch/data/example.h"
#include "torch/types.h"

#include <cstddef>
#include <string>

namespace datasets {
// The CIFAR10 Dataset
class CIFAR10 : public torch::data::datasets::Dataset<CIFAR10> {
public:
    // The mode in which the dataset is loaded
    enum class Mode { kTrain, kTest };

    // Loads CIFAR10 from un-tarred file
    // Dataset can be found https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz
    // Root path should be the directory that contains the content of tarball
    explicit CIFAR10(const std::string& root, Mode mode = Mode::kTrain);

    // Returns the pair at index in the dataset
    torch::data::Example<> get(size_t index) override;

    // The size of the dataset
    c10::optional<size_t> size() const override;

    // The mode the dataset is in
    bool is_train() const noexcept;

    // Returns all images stacked into a single tensor
    const torch::Tensor& images() const;

    // Returns all targets stacked into a single tensor
    const torch::Tensor& targets() const;

    // Trims the dataset to the first n pairs
    CIFAR10&& use_subset(int64_t new_size);


private:
    Mode mode_;
    torch::Tensor images_, targets_;
};
} // namespace datasets
```

This class's implementation reads from the binary distribution of the CIFAR10 dataset and builds two tensors which hold the images and labels.

Then we select a subset of the dataset to use for calibration, since we don't need the the full dataset for calibration and calibration does take time, then define the preprocessing to apply to the images in the dataset and  create a Dataloader from the dataset which will batch the data:

```C++
auto calibration_dataset = datasets::CIFAR10(data_dir, datasets::CIFAR10::Mode::kTest)
                                    .use_subset(320)
                                    .map(torch::data::transforms::Normalize<>({0.4914, 0.4822, 0.4465},
                                                                              {0.2023, 0.1994, 0.2010}))
                                    .map(torch::data::transforms::Stack<>());
auto calibration_dataloader = torch::data::make_data_loader(std::move(calibration_dataset),
                                                            torch::data::DataLoaderOptions().batch_size(32)
                                                                                            .workers(2));
```

Next we create a calibrator from the `calibration_dataloader` using the calibrator factory:

```C++
auto calibrator = torch_tensorrt::ptq::make_int8_calibrator(std::move(calibration_dataloader), calibration_cache_file, true);

```

Here we also define a location to write a calibration cache file to which we can use to reuse the calibration data without needing the dataset and whether or not we should use the cache file if it exists. There also exists a `torch_tensorrt::ptq::make_int8_cache_calibrator` factory which creates a calibrator that uses the cache only for cases where you may do engine building on a machine that has limited storage (i.e. no space for a dataset) or to have a simpiler deployment application.

The calibrator factories create a calibrator that inherits from a `nvinfer1::IInt8Calibrator` virtual class (`nvinfer1::IInt8EntropyCalibrator2` by default) which defines the calibration algorithm used when calibrating. You can explicitly make the selection of calibration algorithm like this:

```C++
// MinMax Calibrator is geared more towards NLP tasks
auto calibrator = torch_tensorrt::ptq::make_int8_calibrator<nvinfer1::IInt8MinMaxCalibrator>(std::move(calibration_dataloader), calibration_cache_file, true);
```

Then all thats required to setup the module for INT8 calibration is to set the following compile settings in the `torch_tensorrt::CompileSpec` struct and compiling the module:

```C++
    std::vector<std::vector<int64_t>> input_shape = {{32, 3, 32, 32}};
    /// Configure settings for compilation
    auto compile_spec = torch_tensorrt::CompileSpec({input_shape});
    /// Set enable INT8 precision
    compile_spec.enabled_precisions.insert(torch::kI8);
    /// Use the TensorRT Entropy Calibrator
    compile_spec.ptq_calibrator = calibrator;
    /// Set a larger workspace (you may get better performace from doing so)
    compile_spec.workspace_size = 1 << 28;

    auto trt_mod = torch_tensorrt::CompileGraph(mod, compile_spec);
```

If you have an existing Calibrator implementation for TensorRT you may directly set the `ptq_calibrator` field with a pointer to your calibrator and it will work as well.

From here not much changes in terms of how to execution works. You are still able to fully use Libtorch as the sole interface for inference. Data should remain in FP32 precision when it's passed into `trt_mod.forward`.


## Running the Example Application

This is a short example application that shows how to use Torch-TensorRT to perform post-training quantization for a module.

## Prerequisites

1. Download CIFAR10 Dataset Binary version ([https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz](https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz))
2. Train a network on CIFAR10 (see `examples/int8/training/vgg16/README.md` for a VGG16 recipe)
3. Export model to torchscript

## Compilation using bazel

``` shell
bazel run //examples/int8/ptq --compilation_mode=opt <path-to-module> <path-to-cifar10>
```

If you want insight into what is going under the hood or need debug symbols

``` shell
bazel run //examples/int8/ptq --compilation_mode=dbg <path-to-module> <path-to-cifar10>
```

This will build a binary named `ptq` in `bazel-out/k8-<opt|dbg>/bin/cpp/int8/ptq/` directory. Optionally you can add this to `$PATH` environment variable to run `ptq` from anywhere on your system.

## Compilation using Makefile

1) Download releases of <a href="https://pytorch.org">LibTorch</a>, <a href="https://github.com/pytorch/TensorRT/releases">Torch-TensorRT </a>and <a href="https://developer.nvidia.com/nvidia-tensorrt-download">TensorRT</a> and unpack them in the deps directory.

```sh
cd examples/torch_tensorrtrt_example/deps
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

2) Build and run `ptq`

We import header files `cifar10.h` and `benchmark.h` from `ROOT_DIR`. `ROOT_DIR` should point to the path where Torch-TensorRT is located `<path_to_torch_tensorrt>`.

By default it is set to `../../../`. If your Torch-TensorRT directory structure is different, please set `ROOT_DIR` accordingly.

```sh
cd examples/int8/ptq
# This will generate a ptq binary
make ROOT_DIR=<PATH> CUDA_VERSION=11.1
./ptq <path-to-module> <path-to-cifar10>
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
