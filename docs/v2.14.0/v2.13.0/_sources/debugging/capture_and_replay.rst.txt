Capture and Replay
==================

This toolchain captures TensorRT network creation and build parameters at runtime via a shim, then deterministically replays them to reproduce an engine build. Use it to debug or reproduce builds independent of the originating framework.

Prerequisites
-------------

- TensorRT installed (ensure you know the absolute path to its ``lib`` and ``bin`` directories)
- ``libtensorrt_shim.so`` available in your TensorRT ``lib`` directory
- ``tensorrt_player`` available in your TensorRT ``bin`` directory

Quick start: Capture
--------------------

Example ``test.py``:

.. code-block:: python

    import torch
    import torch_tensorrt as torchtrt
    import torchvision.models as models
    class MyModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv1d(3, 3, 3, padding=1, stride=1, bias=True)

        def forward(self, x):
            return self.conv(x)

    model = MyModule().eval().to("cuda")
    input = torch.randn((1, 3, 3)).to("cuda").to(torch.float32)

    compile_spec = {
        "inputs": [
            torchtrt.Input(
                min_shape=(1, 3, 3),
                opt_shape=(2, 3, 3),
                max_shape=(3, 3, 3),
                dtype=torch.float32,
            )
        ],
        "min_block_size": 1,
        "cache_built_engines": False,
        "reuse_cached_engines": False,
        "use_python_runtime": True,
    }

    try:
        with torchtrt.dynamo.Debugger(
            "graphs",
            logging_dir="debuglogs",
        ):
            trt_mod = torchtrt.compile(model, **compile_spec)

    except Exception as e:
        raise e

    print("done.....")

.. code-block:: bash

    TORCHTRT_ENABLE_TENSORRT_API_CAPTURE=1 python test.py

When ``TORCHTRT_ENABLE_TENSORRT_API_CAPTURE=1`` is set, capture and replay files are automatically saved under ``debuglogs/capture_replay/`` (i.e., the ``capture_replay`` subdirectory of ``logging_dir``). You should see ``capture.json`` and associated ``.bin`` files generated there.

Replay: Build the engine from the capture
-----------------------------------------

Use ``tensorrt_player`` to replay the captured build without the original framework:

.. code-block:: bash

    tensorrt_player -j debuglogs/capture_replay/capture.json -o /absolute/path/to/output_engine

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
