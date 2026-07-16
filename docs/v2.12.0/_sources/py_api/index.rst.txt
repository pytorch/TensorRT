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
   ../cli/torchtrtc
   ../indices/supported_ops

Legacy
------

.. note::

   The ``ts``, ``ptq``, and ``fx`` modules are in maintenance mode — for new
   projects use the Dynamo frontend (``torch.compile`` or ``torch.export``).

.. toctree::
   :maxdepth: 1

   ts
   ptq
   fx
