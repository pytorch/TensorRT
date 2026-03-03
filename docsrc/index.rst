.. Torch-TensorRT documentation master file.

Torch-TensorRT
==============

**Torch-TensorRT** compiles PyTorch models for NVIDIA GPUs using TensorRT,
delivering significant inference speedups with minimal code changes. It supports
just-in-time compilation via ``torch.compile`` and ahead-of-time export via
``torch.export``, integrating seamlessly with the PyTorch ecosystem.

**New to Torch-TensorRT?** Start with :doc:`getting_started/installation`, then
try the :doc:`getting_started/quick_start` guide.

**Ready to optimize a model?** See :doc:`dynamo/torch_compile` for ``torch.compile``
or :doc:`dynamo/dynamo_export` for ahead-of-time export with ``torch.export``.

Additional resources:

* `Torch-TensorRT 2.0 GTC Talk <https://www.nvidia.com/en-us/on-demand/session/gtcspring23-s51714/>`_
* `GitHub Repository <https://github.com/pytorch/TensorRT>`_

.. toctree::
   :hidden:

   getting_started/installation
   getting_started/quick_start
   user_guide/index
   tutorials/advanced_usage
   tutorials/model_zoo
   py_api/index
   cpp_api
   legacy
   contributing
