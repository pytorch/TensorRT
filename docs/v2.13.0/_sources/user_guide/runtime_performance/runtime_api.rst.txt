.. _runtime_api:

Runtime API
===========

``torch_tensorrt.runtime`` exposes a set of context managers and utility functions for
controlling TRT engine execution behavior **after** compilation. These APIs let you
opt in to CUDA graph capture, pre-allocated output buffers, and weight streaming
without recompiling the engine.

----

``enable_cudagraphs``
---------------------

Wraps a compiled ``torch.fx.GraphModule`` so that forward calls are captured and
replayed as a CUDA graph. See :ref:`cuda_graphs` for a full explanation of the two
modes (whole-graph vs per-subgraph).

.. code-block:: python

    import torch_tensorrt

    trt_model = torch_tensorrt.compile(model, ir="dynamo", arg_inputs=inputs)

    with torch_tensorrt.runtime.enable_cudagraphs(trt_model) as cg_model:
        # First call records the graph; subsequent calls replay it
        output = cg_model(*inputs)
        output = cg_model(*inputs)  # fast replay
    # CUDA graph recording is torn down on exit; trt_model is restored

**Mode selection** is automatic: if the model contains PyTorch fallback subgraphs,
the whole graph is captured using ``CudaGraphsTorchTensorRTModule`` (whole-graph mode);
if the model is pure TRT, per-subgraph CUDA graph capture is used.

Whole-graph CUDA graph capture requires fixed input shapes. If your model uses
data-dependent-shape ops, use ``enable_output_allocator`` instead (incompatible with
CUDA graphs).

----

``enable_pre_allocated_outputs``
----------------------------------

Allocates output tensors **once** at the start of the context and reuses them for
every subsequent forward call. This eliminates the overhead of output buffer allocation
on the critical path and is useful for latency-sensitive inference loops.

.. code-block:: python

    with torch_tensorrt.runtime.enable_pre_allocated_outputs(trt_model) as pre_model:
        for batch in dataloader:
            output = pre_model(batch.cuda())
            # output is valid until the next call
    # Output pre-allocation is released on exit

.. warning::

   The output tensors are **overwritten in-place** on each call. Copy them before the
   next forward pass if you need to retain the values:

   .. code-block:: python

       with torch_tensorrt.runtime.enable_pre_allocated_outputs(trt_model) as pre_model:
           result = pre_model(*inputs).clone()  # clone before next call

----

``enable_output_allocator``
-----------------------------

Activates TRT's dynamic output allocator for models with **data-dependent output
shapes** — ops like ``nonzero``, ``unique``, ``masked_select``, or ``nms`` whose
output size is not known at compile time.

.. code-block:: python

    with torch_tensorrt.runtime.enable_output_allocator(trt_model) as dds_model:
        # Works with variable-length outputs
        indices = dds_model(mask_tensor)

On entry, ``use_output_allocator`` is enabled on each TRT submodule; on exit it is
disabled and the module reverts to standard static-allocation execution.

.. note::

   ``enable_output_allocator`` is **incompatible with CUDA graphs** — do not nest
   it inside ``enable_cudagraphs``.

----

``weight_streaming``
--------------------

See :ref:`resource_management` for a complete guide. In brief:

.. code-block:: python

    with torch_tensorrt.runtime.weight_streaming(trt_model) as ctx:
        # Limit GPU memory used for weights to 1 GiB
        ctx.device_budget = 1 * 1024**3
        output = trt_model(*inputs)
    # Budget is restored to the original value on exit

``weight_streaming`` requires the model to be compiled with
``enable_weight_streaming=True``.

----

``set_cudagraphs_mode`` / ``get_cudagraphs_mode``
--------------------------------------------------

Low-level global CUDA graph mode control (useful for testing or profiling):

.. code-block:: python

    from torch_tensorrt.runtime import (
        set_cudagraphs_mode,
        get_cudagraphs_mode,
        get_whole_cudagraphs_mode,
    )

    # Enable per-subgraph CUDA graphs globally
    set_cudagraphs_mode(True)
    print(get_cudagraphs_mode())       # True
    print(get_whole_cudagraphs_mode()) # False

    # Enable whole-graph CUDA graphs via the context manager
    with torch_tensorrt.runtime.enable_cudagraphs(trt_model) as cg_model:
        print(get_whole_cudagraphs_mode())  # True (if model has fallback subgraphs)

Prefer ``enable_cudagraphs`` over manual ``set_cudagraphs_mode`` calls — the context
manager handles mode restoration on exit automatically.

----

``set_multi_device_safe_mode``
-------------------------------

Enables a thread-safe mode for running the same compiled TRT module on multiple CUDA
devices concurrently. When enabled, the runtime serializes device context switches
before each engine execution:

.. code-block:: python

    torch_tensorrt.runtime.set_multi_device_safe_mode(True)

    # Now safe to call trt_model from multiple threads on different devices
    output = trt_model(*inputs)

    torch_tensorrt.runtime.set_multi_device_safe_mode(False)

This incurs a small overhead per forward call. Only enable it when genuinely running
across multiple devices from the same Python process.
