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
for an engine when it is loaded, allocation is deferred to execution time. The
memory is released immediately after inference completes, so only one engine holds
GPU activation memory at a time.

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
* ``"dynamic"`` — device memory is allocated on each forward pass
  (via ``createExecutionContextWithoutDeviceMemory`` + manual device-memory
  assignment), and released immediately after the call returns.

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
        // dynamic_workspace freed here (goes out of scope)
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

Related
-------

* :ref:`execution` — runtime module architecture.
* `Example: dynamic_memory_allocation.py <https://github.com/pytorch/TensorRT/blob/main/examples/dynamo/dynamic_memory_allocation.py>`_
