.. _debugging:

Debugging Torch-TensorRT Compilation
====================================


FX Graph Visualization
----------------------

Debug Mode
-------------


Profiling TensorRT Engines
--------------------------

There are some profiling tools built into Torch-TensorRT to measure the performance of TensorRT sub blocks in compiled modules.
This can be used in conjunction with PyTorch profiling tools to get a picture of the performance of your model.
Profiling for any particular sub block can be enabled by the ``enabled_profiling()`` method of any
`` __torch__.classes.tensorrt.Engine`` attribute, or of any ``torch_tensorrt.runtime.TorchTensorRTModule``. The profiler will
dump trace files by default in /tmp, though this path can be customized by either setting the
profile_path_prefix of ``__torch__.classes.tensorrt.Engine`` or as an argument to
torch_tensorrt.runtime.TorchTensorRTModule.enable_precision(profiling_results_dir="").
Traces can be visualized using the Perfetto tool (https://perfetto.dev)

.. image:: /user_guide/images/perfetto.png
   :width: 512px
   :height: 512px
   :scale: 50 %
   :align: right
