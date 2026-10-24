.. _dynamic_memory_allocation_design:

Dynamic Memory Allocation
==========================

.. note::

   This page documents the design for dynamically allocated engine memory in
   Torch-TensorRT.
   Original design discussion:
   `RFC #3714 <https://github.com/pytorch/TensorRT/discussions/3714>`_.

Goal
----

Some TRT engines consume significantly more GPU memory than the equivalent
PyTorch module. When multiple TRT-accelerated submodules are loaded simultaneously
(e.g. in a diffusers pipeline with UNet, VAE, and text encoder), the total
resident GPU memory can exceed the device limit even if each module runs
sequentially.

The solution is **dynamic memory allocation**: instead of allocating device memory
for an engine when it is loaded, allocation is deferred to execution time. In the
C++ runtime, the memory is released when the call returns, so engines that run one
after another do not hold activation memory at the same time. See Limitations for
the cases that keep memory between calls.

User API
---------

A context manager sets the allocation strategy for all TRT engines within scope:

.. code-block:: python

    import torch_tensorrt

    with torch_tensorrt.runtime.enable_dynamic_engine_context(trt_model):
        output = trt_model(*inputs)

Alternatively, the strategy can be set per-module:

.. code-block:: python

    trt_model.set_resource_allocation_strategy("dynamic")

Two strategies are available:

* ``"static"`` (default) — device memory is allocated when the engine is loaded
  (via ``createExecutionContext``). Memory is held for the lifetime of the engine.
* ``"dynamic"`` — device memory is allocated at execution time instead of load time
  (via ``createExecutionContextWithoutDeviceMemory`` + manual device-memory
  assignment). See Limitations for how long each runtime keeps it.

Internal Implementation
------------------------

C++ Runtime
^^^^^^^^^^^^

The ``TRTEngine`` C++ class manages an ``IExecutionContext`` per engine. When the
strategy is switched to dynamic the existing context is destroyed and a new
context is created *without* device memory:

.. code-block:: cpp

    void TRTEngine::set_resource_allocation_strategy(
            ResourceAllocationStrategy new_strategy) {
        if (new_strategy != resource_allocation_strategy_) {
            resource_allocation_strategy_ = new_strategy;
            if (new_strategy == ResourceAllocationStrategy::kDynamic) {
                exec_ctx_ = engine_->createExecutionContextWithoutDeviceMemory();
            } else {
                exec_ctx_ = engine_->createExecutionContext();
            }
        }
    }

During ``execute_engine``, when dynamic allocation is active, a temporary
``torch::Tensor`` of type ``uint8`` provides the required device memory:

.. code-block:: cpp

    void execute_engine(...) {
        torch::Tensor dynamic_workspace;
        if (engine.resource_allocation_strategy == kDynamic) {
            dynamic_workspace = torch::empty(
                engine.device_memory_size,
                torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA)
            );
            exec_ctx_->setDeviceMemory(dynamic_workspace.data_ptr());
        }
        // ... run inference ...
        // dynamic_workspace is freed here, unless a captured CUDA graph keeps it
    }

Python Exposure
^^^^^^^^^^^^^^^^

The ``_ResourceAllocator`` Python module wraps the C++ setting and provides the
context manager surface. The ``TorchTensorRTModule`` exposes
``set_resource_allocation_strategy`` through its TorchBind interface.

Limitations
-----------

* Dynamic allocation does **not** reduce the peak memory of a single engine
  during inference — it only reduces the memory that is *resident* when the
  engine is idle.
* The per-inference allocation/free overhead is small but non-zero; avoid dynamic
  allocation for latency-critical paths where ``static`` would fit in memory.
* With CUDA graphs on, the C++ runtime keeps an engine's memory for as long as its
  captured graph, because a replay writes to the memory the graph was recorded with.
* The Python runtime keeps each engine's memory from its last call, with or without
  CUDA graphs, until a later call replaces it.
* So in both cases, several engines can hold their memory at the same time.

Related
-------

* :ref:`execution` — runtime module architecture.
* `Example: dynamic_memory_allocation.py <https://github.com/pytorch/TensorRT/blob/main/examples/dynamo/dynamic_memory_allocation.py>`_
