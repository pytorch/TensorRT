.. Torch-TensorRT documentation master file, created by
   sphinx-quickstart on Mon May  4 13:43:16 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Torch-TensorRT
==============

In-framework compilation of PyTorch inference code for NVIDIA GPUs
--------------------------------------------------------------------------
Torch-TensorRT is a inference compiler for PyTorch, targeting NVIDIA GPUs via NVIDIA's TensorRT Deep Learning Optimizer and Runtime.
It supports both just-in-time (JIT) compilation workflows via the ``torch.compile`` interface as well as ahead-of-time (AOT) workflows.
Torch-TensorRT integrates seamlessly into the PyTorch ecosystem supporting hybrid execution of optimized TensorRT code with standard PyTorch code.

More Information / System Architecture:

* `Torch-TensorRT 2.0 <https://www.nvidia.com/en-us/on-demand/session/gtcspring23-s51714/>`_

Getting Started
----------------
* :ref:`installation`
* :ref:`getting_started_with_python_api`
* :ref:`getting_started_cpp`

.. toctree::
   :caption: Getting Started
   :maxdepth: 1
   :hidden:

   getting_started/installation
   getting_started/getting_started_with_python_api
   getting_started/getting_started_with_cpp_api
   getting_started/getting_started_with_windows


User Guide
------------
* :ref:`creating_a_ts_mod`
* :ref:`getting_started_with_fx`
* :ref:`torch_compile`
* :ref:`dynamo_export`
* :ref:`ptq`
* :ref:`runtime`
* :ref:`saving_models`
* :ref:`dynamic_shapes`
* :ref:`use_from_pytorch`
* :ref:`using_dla`

.. toctree::
   :caption: User Guide
   :maxdepth: 1
   :hidden:

   user_guide/creating_torchscript_module_in_python
   user_guide/getting_started_with_fx_path
   user_guide/torch_compile
   user_guide/dynamo_export
   user_guide/ptq
   user_guide/runtime
   user_guide/saving_models
   user_guide/dynamic_shapes
   user_guide/use_from_pytorch
   user_guide/using_dla

Tutorials
------------
* :ref:`torch_tensorrt_tutorials`
* :ref:`serving_torch_tensorrt_with_triton`
* :ref:`notebooks`

.. toctree::
   :caption: Tutorials
   :maxdepth: 3
   :hidden:

   tutorials/serving_torch_tensorrt_with_triton
   tutorials/notebooks
   tutorials/_rendered_examples/dynamo/torch_compile_resnet_example
   tutorials/_rendered_examples/dynamo/torch_compile_transformers_example
   tutorials/_rendered_examples/dynamo/torch_compile_advanced_usage

Python API Documenation
------------------------
* :ref:`torch_tensorrt_py`
* :ref:`torch_tensorrt_logging_py`
* :ref:`torch_tensorrt_ptq_py`
* :ref:`torch_tensorrt_ts_py`
* :ref:`torch_tensorrt_fx_py`

.. toctree::
   :caption: Python API Documenation
   :maxdepth: 0
   :hidden:

   py_api/torch_tensorrt
   py_api/logging
   py_api/ptq
   py_api/ts
   py_api/fx

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

CLI Documentation
---------------------
* :ref:`torchtrtc`

.. toctree::
   :caption: CLI Documenation
   :maxdepth: 0
   :hidden:

   cli/torchtrtc


Contributor Documentation
--------------------------------
* :ref:`system_overview`
* :ref:`writing_converters`
* :ref:`writing_dynamo_aten_lowering_passes`
* :ref:`useful_links`

.. toctree::
   :caption: Contributor Documentation
   :maxdepth: 1
   :hidden:

   contributors/system_overview
   contributors/writing_converters
   contributors/writing_dynamo_aten_lowering_passes
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


Legacy Further Information (TorchScript)
-------------------------------------------

* `Introductory Blog Post <https://developer.nvidia.com/blog/accelerating-inference-up-to-6x-faster-in-pytorch-with-torch-tensorrt/>`_
* `GTC 2020 Talk <https://www.nvidia.com/en-us/on-demand/session/gtcsj20-s21671/>`_
* `GTC 2020 Fall Talk <https://www.nvidia.com/en-us/on-demand/session/gtcfall20-a21864/>`_
* `GTC 2021 Talk <https://www.nvidia.com/en-us/on-demand/session/gtcspring21-s31864/>`_
* `GTC 2021 Fall Talk <https://www.nvidia.com/en-us/on-demand/session/gtcfall21-a31107/>`_
* `PyTorch Ecosystem Day 2021 <https://assets.pytorch.org/pted2021/posters/I6.png>`_
* `PyTorch Developer Conference 2021 <https://s3.amazonaws.com/assets.pytorch.org/ptdd2021/posters/D2.png>`_
* `PyTorch Developer Conference 2022 <https://pytorch.s3.amazonaws.com/posters/ptc2022/C04.pdf>`_