.. Torch-TensorRT documentation master file.

Torch-TensorRT
==============

**Torch-TensorRT** compiles PyTorch models for NVIDIA GPUs using TensorRT,
delivering significant inference speedups with minimal code changes. It supports
just-in-time compilation via ``torch.compile`` and ahead-of-time export via
``torch.export``, integrating seamlessly with the PyTorch ecosystem.

**New to Torch-TensorRT?** Start with :doc:`getting_started/installation`, then
try the quick start guide below.

**Ready to optimize a model?** See :doc:`user_guide/compilation/torch_compile` for ``torch.compile``
or :doc:`user_guide/compilation/dynamo_export` for ahead-of-time export with ``torch.export``.

Additional resources:

* `Torch-TensorRT 2.0 GTC Talk <https://www.nvidia.com/en-us/on-demand/session/gtcspring23-s51714/>`_
* `GitHub Repository <https://github.com/pytorch/TensorRT>`_

.. include:: getting_started/quick_start.rst

.. toctree::
   :hidden:
   :caption: Getting Started

   getting_started/installation

.. toctree::
   :hidden:
   :caption: User Guide

   user_guide/index

.. toctree::
   :hidden:
   :caption: Tutorials

   tutorials/advanced_usage
   tutorials/model_zoo

.. toctree::
   :hidden:
   :caption: API Reference

   api_reference

.. toctree::
    :hidden:
    :caption: Debugging

    debugging/index

.. toctree::
   :hidden:
   :caption: Contributing

   contributing

.. toctree::
   :hidden:
   :caption: Legacy

   legacy
