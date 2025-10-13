Introduction
============

This toolchain captures TensorRT network creation and build parameters at runtime via a shim, then deterministically replays them to reproduce an engine build. Use it to debug or reproduce builds independent of the originating framework.

Prerequisites
-------------

- TensorRT installed (ensure you know the absolute path to its ``lib`` and ``bin`` directories)
- ``libtensorrt_shim.so`` available in your TensorRT ``lib`` directory
- ``tensorrt_player`` available in your TensorRT ``bin`` directory

Quick start: Capture
--------------------

.. code-block:: bash

    TORCHTRT_ENABLE_TENSORRT_API_CAPTURE=1 python test.py

You should see ``shim.json`` and ``shim.bin`` generated in ``/tmp/torch_tensorrt_{current_user}/shim``.

Replay: Build the engine from the capture
-----------------------------------------

Use ``tensorrt_player`` to replay the captured build without the original framework:

.. code-block:: bash

    tensorrt_player -j /absolute/path/to/shim.json -o /absolute/path/to/output_engine

This produces a serialized TensorRT engine at ``output_engine``.

Validate the engine
-------------------

Run the engine with ``trtexec``:

.. code-block:: bash

    trtexec --loadEngine=/absolute/path/to/output_engine

Notes
-----

- Ensure the ``libnvinfer.so`` used by the shim matches the TensorRT version in your environment.
- If multiple TensorRT versions are installed, prefer absolute paths as shown above.
- Currently, it is not supported to capture multiple engines, in case of graph break, only the first engine will be captured.


