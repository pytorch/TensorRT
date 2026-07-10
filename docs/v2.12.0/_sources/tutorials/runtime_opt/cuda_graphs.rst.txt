.. _cuda_graphs:

CUDAGraphs and the Output Allocator
=====================================

Torch-TensorRT provides two runtime features that can significantly reduce per-request
inference latency for steady-state workloads: **CUDA Graphs** and the
**Dynamic Output Allocator**.

----

CUDA Graphs
-----------

CUDA Graphs capture a sequence of GPU operations into a replayable graph. On replay,
the entire sequence is submitted in a single kernel launch rather than individual
dispatches, eliminating CPU-side dispatch overhead and improving GPU utilization.

Enabling CUDAGraphs
^^^^^^^^^^^^^^^^^^^^

The canonical way to enable CUDA graphs is the ``enable_cudagraphs`` context manager:

.. code-block:: python

    import torch
    import torch_tensorrt

    trt_gm = torch_tensorrt.compile(model, ir="dynamo", arg_inputs=inputs)

    with torch.no_grad():
        with torch_tensorrt.runtime.enable_cudagraphs(trt_gm) as cg_module:
            # First call: warms up and records the CUDA graph
            output = cg_module(*inputs)
            # Subsequent calls: replay the captured graph (fast path)
            output = cg_module(*inputs)
    # Graph recording is torn down on context exit; trt_gm is restored

The context manager automatically selects one of two capture modes based on the
compiled model:

* **Per-subgraph mode** (no graph breaks): CUDA graph capture is applied to each
  individual TRT submodule. ``cg_module`` is the same ``GraphModule`` object with
  the per-subgraph flag enabled.

* **Whole-graph mode** (model has PyTorch fallback subgraphs / graph breaks): The
  entire forward pass — TRT subgraphs *and* PyTorch subgraphs between them — is
  captured as a single CUDA graph via ``CudaGraphsTorchTensorRTModule``. This
  eliminates inter-subgraph dispatch overhead even when the model is partially
  executed in PyTorch.

  .. code-block:: python

      # Force a graph break so the model has a PyTorch fallback subgraph
      opt_with_break = torch_tensorrt.compile(
          model, ir="dynamo", arg_inputs=[input],
          torch_executed_ops={"torch.ops.aten.mul.Tensor"},
          min_block_size=1,
      )

      with torch_tensorrt.runtime.enable_cudagraphs(opt_with_break) as cg_module:
          # cg_module is a CudaGraphsTorchTensorRTModule wrapping opt_with_break
          output = cg_module(input)

You can also enable CUDA graphs globally for the session (without a context manager):

.. code-block:: python

    torch_tensorrt.runtime.set_cudagraphs_mode(True)
    output = trt_gm(*inputs)
    torch_tensorrt.runtime.set_cudagraphs_mode(False)

Prefer the context manager over ``set_cudagraphs_mode`` — it guarantees the mode is
restored even if an exception occurs.

How Recording Works
^^^^^^^^^^^^^^^^^^^^

1. **Warm-up**: Three forward passes on a side CUDA stream. This forces memory
   allocations and kernel initializations to happen *before* recording, so they are
   excluded from the graph.

2. **Input shape tracking**: A shape key is computed from all input shapes. If the
   key changes between calls, the captured graph is reset and re-recorded for the new
   shapes.

3. **Replay**: On a shape-key cache hit, input tensors are copied into pre-allocated
   buffers and the captured graph is replayed in a single submission.

Limitations
^^^^^^^^^^^

* **Dynamic shapes**: CUDAGraphs require fixed tensor addresses and sizes. If your
  input shapes change between requests, the graph is re-recorded for each new shape.
  For variable-batch workloads, consider bucketing inputs by shape or using the
  ``DynamicOutputAllocator`` instead.

* **Data-dependent-shape ops**: Operations like ``nonzero`` and ``unique`` produce
  outputs whose size is unknown at graph-capture time. These require the
  ``DynamicOutputAllocator`` and are incompatible with full CUDA graph capture unless
  they are partitioned into a separate PyTorch subgraph.

* **Weight streaming**: If ``enable_weight_streaming=True``, the graph is re-recorded
  whenever weights are streamed (flagged by ``is_weight_streaming_set``).

* **Not serializable**: ``CudaGraphsTorchTensorRTModule`` is a runtime wrapper and
  cannot be saved via ``torch_tensorrt.save()``. Save the underlying ``trt_gm`` first,
  load it, then wrap.

----

DynamicOutputAllocator
-----------------------

Some TRT ops produce outputs whose shape depends on runtime data — TensorRT calls
these **data-dependent shape (DDS)** operations. Examples: ``aten.nonzero``,
``aten.unique``, ``aten.nms``.

For these ops, TRT cannot pre-allocate output buffers of the correct size. The
``DynamicOutputAllocator`` solves this by implementing TRT's ``IOutputAllocator``
interface: TRT calls back into the allocator at runtime to request a buffer of the
correct size, and the allocator provides a freshly-allocated CUDA tensor.

When Is It Used?
^^^^^^^^^^^^^^^^^

Automatically — you do not need to configure it manually. When any converter in the
graph sets ``requires_output_allocator=True`` in its ``ConverterSupport``, the
``TRTInterpreter`` sets ``ctx.requires_output_allocator = True`` on the
``ConversionContext``. The runtime module then uses the ``DynamicOutputAllocator``
for that engine.

.. code-block:: python

    # Check if a compiled module uses the output allocator
    for name, submodule in trt_gm.named_children():
        if hasattr(submodule, "requires_output_allocator"):
            print(f"{name}: requires_output_allocator = {submodule.requires_output_allocator}")

Writing a Converter That Requires the Output Allocator
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Set ``requires_output_allocator=True`` in the decorator:

.. code-block:: python

    @dynamo_tensorrt_converter(
        torch.ops.aten.nonzero.default,
        requires_output_allocator=True,
        supports_dynamic_shapes=True,
    )
    def aten_ops_nonzero(ctx, target, args, kwargs, name):
        ...

Performance Implications
^^^^^^^^^^^^^^^^^^^^^^^^^

The ``DynamicOutputAllocator`` performs a CUDA memory allocation on every forward
pass for each DDS output. This adds a small latency cost compared to pre-allocated
buffers. If DDS ops are in your hot path:

* Use ``torch_executed_ops`` to force the DDS op to run in PyTorch where NumPy-style
  dynamic allocation is cheap.
* Cache output tensor handles across calls if the output size is bounded in practice.

----

Choosing Between Approaches
-----------------------------

.. list-table::
   :widths: 25 25 25 25
   :header-rows: 1

   * - Scenario
     - CUDAGraphs
     - Output Allocator
     - Recommendation
   * - Fixed-shape inference, latency critical
     - Ideal
     - Not needed
     - Enable CUDAGraphs
   * - Variable batch sizes
     - Re-records per shape
     - Not needed
     - Bucket inputs + CUDAGraphs, or no CUDAGraphs
   * - Graph contains ``nonzero`` / ``unique``
     - Incompatible
     - Required automatically
     - Let the allocator run, disable CUDAGraphs for that subgraph
   * - Maximize throughput, not latency
     - Marginal benefit
     - Not needed
     - Skip, focus on ``optimization_level``
