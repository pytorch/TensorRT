.. _ptq:

Post Training Quantization (PTQ)
=================================

Post Training Quantization (PTQ) is a technique to reduce the required computational resources for inference
while still preserving the accuracy of your model by mapping the traditional FP32 activation space to a reduced
INT8 space. TensorRT uses a calibration step which executes your model with sample data from the target domain
and track the activations in FP32 to calibrate a mapping to INT8 that minimizes the information loss between
FP32 inference and INT8 inference.

Users writing TensorRT applications are required to setup a calibrator class which will provide sample data to
the TensorRT calibrator. With TRTorch we look to leverage existing infrastructure in PyTorch to make implementing
calibrators easier.

LibTorch provides a ``DataLoader`` and ``Dataset`` API which steamlines preprocessing and batching input data.
These APIs are exposed via both C++ and Python interface which makes it easier for the end user.
For C++ interface, we use ``torch::Dataset`` and ``torch::data::make_data_loader`` objects to construct and perform pre-processing on datasets.
The equivalent functionality in python interface uses ``torch.utils.data.Dataset`` and ``torch.utils.data.DataLoader``.
This section of the PyTorch documentation has more information https://pytorch.org/tutorials/advanced/cpp_frontend.html#loading-data and https://pytorch.org/tutorials/recipes/recipes/loading_data_recipe.html.
TRTorch uses Dataloaders as the base of a generic calibrator implementation. So you will be able to reuse or quickly
implement a ``torch::Dataset`` for your target domain, place it in a DataLoader and create a INT8 Calibrator
which you can provide to TRTorch to run INT8 Calibration during compliation of your module.

.. _writing_ptq_cpp:

How to create your own PTQ application in C++
----------------------------------------

Here is an example interface of a ``torch::Dataset`` class for CIFAR10:

.. code-block:: c++
    :linenos:

    //cpp/ptq/datasets/cifar10.h
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


This class's implementation reads from the binary distribution of the CIFAR10 dataset and builds two tensors which hold the images and labels.

We use a subset of the dataset to use for calibration, since we don't need the the full dataset for effective calibration and calibration does
some take time, then define the preprocessing to apply to the images in the dataset and create a DataLoader from the dataset which will batch the data:

.. code-block:: c++

    auto calibration_dataset = datasets::CIFAR10(data_dir, datasets::CIFAR10::Mode::kTest)
                                        .use_subset(320)
                                        .map(torch::data::transforms::Normalize<>({0.4914, 0.4822, 0.4465},
                                                                                {0.2023, 0.1994, 0.2010}))
                                        .map(torch::data::transforms::Stack<>());
    auto calibration_dataloader = torch::data::make_data_loader(std::move(calibration_dataset),
                                                                torch::data::DataLoaderOptions().batch_size(32)
                                                                                                .workers(2));


Next we create a calibrator from the ``calibration_dataloader`` using the calibrator factory (found in ``trtorch/ptq.h``):

.. code-block:: c++

    #include "trtorch/ptq.h"
    ...

    auto calibrator = trtorch::ptq::make_int8_calibrator(std::move(calibration_dataloader), calibration_cache_file, true);

Here we also define a location to write a calibration cache file to which we can use to reuse the calibration data without needing the dataset and whether or not
we should use the cache file if it exists. There also exists a ``trtorch::ptq::make_int8_cache_calibrator`` factory which creates a calibrator that uses the cache
only for cases where you may do engine building on a machine that has limited storage (i.e. no space for a full dataset) or to have a simpiler deployment application.

The calibrator factories create a calibrator that inherits from a ``nvinfer1::IInt8Calibrator`` virtual class (``nvinfer1::IInt8EntropyCalibrator2`` by default) which
defines the calibration algorithm used when calibrating. You can explicitly make the selection of calibration algorithm like this:

.. code-block:: c++

    // MinMax Calibrator is geared more towards NLP tasks
    auto calibrator = trtorch::ptq::make_int8_calibrator<nvinfer1::IInt8MinMaxCalibrator>(std::move(calibration_dataloader), calibration_cache_file, true);

Then all thats required to setup the module for INT8 calibration is to set the following compile settings in the `trtorch::CompileSpec` struct and compiling the module:

.. code-block:: c++

    std::vector<std::vector<int64_t>> input_shape = {{32, 3, 32, 32}};
    /// Configure settings for compilation
    auto compile_spec = trtorch::CompileSpec({input_shape});
    /// Set operating precision to INT8
    compile_spec.op_precision = torch::kI8;
    /// Use the TensorRT Entropy Calibrator
    compile_spec.ptq_calibrator = calibrator;
    /// Set a larger workspace (you may get better performace from doing so)
    compile_spec.workspace_size = 1 << 28;

    auto trt_mod = trtorch::CompileGraph(mod, compile_spec);

If you have an existing Calibrator implementation for TensorRT you may directly set the ``ptq_calibrator`` field with a pointer to your calibrator and it will work as well.
From here not much changes in terms of how to execution works. You are still able to fully use LibTorch as the sole interface for inference. Data should remain
in FP32 precision when it's passed into `trt_mod.forward`. There exists an example application in the TRTorch demo that takes you from training a VGG16 network on
CIFAR10 to deploying in INT8 with TRTorch here: https://github.com/NVIDIA/TRTorch/tree/master/cpp/ptq

.. _writing_ptq_python:

How to create your own PTQ application in Python
----------------------------------------

TRTorch Python API provides an easy and convenient way to use pytorch dataloaders with TensorRT calibrators. ``DataLoaderCalibrator`` class can be used to create
a TensorRT calibrator by providing desired configuration. The following code demonstrates an example on how to use it

.. code-block:: python

    testing_dataset = torchvision.datasets.CIFAR10(root='./data',
                                                            train=False,
                                                            download=True,
                                                            transform=transforms.Compose([
                                                                transforms.ToTensor(),
                                                                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                                     (0.2023, 0.1994, 0.2010))
                                                            ]))

    testing_dataloader = torch.utils.data.DataLoader(testing_dataset,
                                                          batch_size=1,
                                                          shuffle=False,
                                                          num_workers=1)
    calibrator = trtorch.ptq.DataLoaderCalibrator(testing_dataloader,
                                                  cache_file='./calibration.cache',
                                                  use_cache=False,
                                                  algo_type=trtorch.ptq.CalibrationAlgo.ENTROPY_CALIBRATION_2,
                                                  device=torch.device('cuda:0'))

    compile_spec = {
             "input_shapes": [[1, 3, 32, 32]],
             "op_precision": torch.int8,
             "calibrator": calibrator,
             "device": {
                 "device_type": trtorch.DeviceType.GPU,
                 "gpu_id": 0,
                 "dla_core": 0,
                 "allow_gpu_fallback": False,
                 "disable_tf32": False
             }
         }
    trt_mod = trtorch.compile(model, compile_spec)

In the cases where there is a pre-existing calibration cache file that users want to use, ``CacheCalibrator`` can be used without any dataloaders. The following example demonstrates how
to use ``CacheCalibrator`` to use in INT8 mode.

.. code-block:: python

  calibrator = trtorch.ptq.CacheCalibrator("./calibration.cache")

  compile_settings = {
        "input_shapes": [[1, 3, 32, 32]],
        "op_precision": torch.int8,
        "calibrator": calibrator,
        "max_batch_size": 32,
    }

  trt_mod = trtorch.compile(model, compile_settings)

If you already have an existing calibrator class (implemented directly using TensorRT API), you can directly set the calibrator field to your class which can be very convenient.
For a demo on how PTQ can be performed on a VGG network using TRTorch API, you can refer to https://github.com/NVIDIA/TRTorch/blob/master/tests/py/test_ptq_dataloader_calibrator.py
and https://github.com/NVIDIA/TRTorch/blob/master/tests/py/test_ptq_trt_calibrator.py

Citations
^^^^^^^^^^^

Krizhevsky, A., & Hinton, G. (2009). Learning multiple layers of features from tiny images.

Simonyan, K., & Zisserman, A. (2014). Very deep convolutional networks for large-scale image recognition. arXiv preprint arXiv:1409.1556.