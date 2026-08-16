.. Torch-TensorRT documentation master file, created by
   sphinx-quickstart on Mon May  4 13:43:16 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Torch-TensorRT
==============
Ahead-of-time compilation of TorchScript / PyTorch JIT for NVIDIA GPUs
-----------------------------------------------------------------------
Torch-TensorRT is a compiler for PyTorch/TorchScript, targeting NVIDIA GPUs via NVIDIA's TensorRT Deep Learning Optimizer and Runtime.
Unlike PyTorch's Just-In-Time (JIT) compiler, Torch-TensorRT is an Ahead-of-Time (AOT) compiler, meaning that before you deploy your
TorchScript code, you go through an explicit compile step to convert a standard TorchScript program into an module targeting
a TensorRT engine. Torch-TensorRT operates as a PyTorch extention and compiles modules that integrate into the JIT runtime seamlessly.
After compilation using the optimized graph should feel no different than running a TorchScript module.
You also have access to TensorRT's suite of configurations at compile time, so you are able to specify
operating precision (FP32/FP16/INT8) and other settings for your module.

More Information / System Architecture:

* `GTC 2020 Talk <https://developer.nvidia.com/gtc/2020/video/s21671>`_

Getting Started
----------------
* :ref:`installation`
* :ref:`getting_started`
* :ref:`ptq`
* :ref:`torchtrtc`
* :ref:`use_from_pytorch`
* :ref:`runtime`
* :ref:`using_dla`

.. toctree::
   :caption: Getting Started
   :maxdepth: 1
   :hidden:

   tutorials/installation
   tutorials/getting_started_with_cpp_api
   tutorials/getting_started_with_python_api
   tutorials/creating_torchscript_module_in_python
   tutorials/ptq
   tutorials/torchtrtc
   tutorials/use_from_pytorch
   tutorials/runtime
   tutorials/using_dla

.. toctree::
   :caption: Notebooks
   :maxdepth: 1
   :hidden:

   _notebooks/CitriNet-example
   _notebooks/dynamic-shapes
   _notebooks/EfficientNet-example
   _notebooks/Hugging-Face-BERT
   _notebooks/lenet-getting-started
   _notebooks/Resnet50-example
   _notebooks/ssd-object-detection-demo
   _notebooks/vgg-qat


Python API Documenation
------------------------
* :ref:`torch_tensorrt_py`
* :ref:`torch_tensorrt_logging_py`
* :ref:`torch_tensorrt_ptq_py`
* :ref:`torch_tensorrt_ts_py`

.. toctree::
   :caption: Python API Documenation
   :maxdepth: 0
   :hidden:

   py_api/torch_tensorrt
   py_api/logging
   py_api/ptq
   py_api/ts

C++ API Documenation
----------------------
* :ref:`namespace_torch_tensorrt`
* :ref:`namespace_torch_tensorrt__logging`
* :ref:`namespace_torch_tensorrt__ptq`
* :ref:`namespace_torch_tensorrt__torchscript`


.. toctree::
   :caption: C++ API Documenation
   :maxdepth: 1
   :hidden:

   _cpp_api/torch_tensort_cpp
   _cpp_api/namespace_torch_tensorrt
   _cpp_api/namespace_torch_tensorrt__logging
   _cpp_api/namespace_torch_tensorrt__torchscript
   _cpp_api/namespace_torch_tensorrt__ptq

Contributor Documentation
--------------------------------
* :ref:`system_overview`
* :ref:`writing_converters`
* :ref:`useful_links`

.. toctree::
   :caption: Contributor Documentation
   :maxdepth: 1
   :hidden:

   contributors/system_overview
   contributors/writing_converters
   contributors/useful_links

Indices
----------------
* :ref:`supported_ops`
* :ref:`genindex`
* :ref:`search`

.. toctree::
   :caption: Indices
   :maxdepth: 1
   :hidden:

   indices/supported_ops
