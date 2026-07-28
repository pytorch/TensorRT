.. _cuda_graphs_design:

CUDAGraphs
===========

.. note::

   This page documents the design for CUDAGraphs support in Torch-TensorRT.
   Original design discussion:
   `RFC #2736 <https://github.com/pytorch/TensorRT/discussions/2736>`_.

Goal
----

Reduce inference latency by hiding CUDA kernel launch overhead using
`CUDA Graphs <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-graphs>`_.
A CUDA Graph records a sequence of GPU operations and replays them as a single
batched launch, eliminating the per-kernel CPU overhead that can dominate for small
models or short-batch workloads.

User API
---------

.. code-block:: python

    import torch_tensorrt

    # torch.export + dynamo.compile path
    trt_gm = torch_tensorrt.compile(
        model,
        arg_inputs=inputs,
        use_cuda_graph=True,
    )

    # torch.export path — explicit CUDAGraphs module
    import torch_tensorrt.runtime as runtime
    with runtime.enable_cudagraphs(trt_gm) as cudagraphs_module:
        output = cudagraphs_module(*inputs)

Limitations
-----------

* CUDAGraphs require **static input shapes**. A graph recorded for one input
  shape is invalid for a different shape and must be re-recorded.
* Memory addresses must be stable across replay. Input and output buffers must be
  allocated before recording and reused on every replay (no allocation inside the
  graph).
* Not compatible with models that have **data-dependent shapes** (DDS ops such as
  ``nonzero``).
* The ``torch.compile`` (JIT) path supports recompilation on shape changes via
  PyTorch guards; ``torch.export`` (AOT) requires a static shape or explicit
  per-shape re-recording.

Internal Implementation
------------------------

Recording Strategy
^^^^^^^^^^^^^^^^^^

Recording is done *post engine build* as a wrapper over the execute-engine call.
For each TRT subgraph the runtime:

1. Performs a **warmup run** outside the graph to initialize internal TRT state.
2. Begins CUDA stream capture with ``cudaStreamBeginCapture``.
3. Calls ``execute_async_v3`` (the TensorRT asynchronous execution call).
4. Ends capture with ``cudaStreamEndCapture`` to obtain a ``cudaGraph_t``.
5. Instantiates the graph with ``cudaGraphInstantiate``.

Subsequent inference launches the instantiated graph instead of calling
``execute_async_v3`` directly:

.. code-block:: python

    # Pseudocode reflecting the recording pattern from TRT docs
    from cuda import cudart

    err, stream = cudart.cudaStreamCreate()

    # Warmup: required to update TRT internal state for the input shape
    context.execute_async_v3(stream)

    # Capture
    cudart.cudaStreamBeginCapture(stream, cudart.cudaStreamCaptureModeGlobal)
    context.execute_async_v3(stream)
    err, graph = cudart.cudaStreamEndCapture(stream)
    err, instance = cudart.cudaGraphInstantiate(graph, 0)

    # Replay
    for _ in range(iterations):
        cudart.cudaGraphLaunch(instance, stream)
        cudart.cudaStreamSynchronize(stream)

Graph Storage
^^^^^^^^^^^^^

Each runtime module (both C++ ``TorchTensorRTModule`` and Python
``PythonTorchTensorRTModule``) stores a ``cudaGraphExec_t`` instance. When
``use_cuda_graph=True`` is set at compile time the runtime records one graph
per engine for the first input shape encountered.

For ``ir="torch_compile"`` (JIT path), guard-triggered recompilation provides
dynamic shape support: a new graph is recorded whenever a guard fails and the
model is retraced. This mirrors PyTorch Inductor's ``mode="reduce-overhead"``.

``_CudaGraphsTorchTensorRTModule``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``_CudaGraphsTorchTensorRTModule`` is a dedicated Python runtime wrapper that
combines TRT execution with CUDAGraph management. It:

* Pre-allocates a static output buffer sized for the compiled output shapes.
* Records the full forward pass (across all TRT subgraphs and interleaved
  PyTorch ops) into a single ``torch.cuda.CUDAGraph``.
* On each forward call copies inputs into the pre-allocated input buffer and
  replays the graph.

This gives the lowest-latency execution path for static-shape models.

Related
-------

* :ref:`execution` — runtime module overview.
* `Example: torch_export_cudagraphs.py <https://github.com/pytorch/TensorRT/blob/main/examples/dynamo/torch_export_cudagraphs.py>`_
