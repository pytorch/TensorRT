Python API
==========

Reference documentation for all public Python modules in Torch-TensorRT.

Core
----

.. toctree::
   :maxdepth: 1

   torch_tensorrt
   dynamo
   logging
   runtime
   kernels
   executorch
   ../indices/supported_ops

Legacy
------

.. note::

   The ``fx`` module is in maintenance mode — for new
   projects use the Dynamo frontend (``torch.compile`` or ``torch.export``).

.. toctree::
   :maxdepth: 1

   fx
