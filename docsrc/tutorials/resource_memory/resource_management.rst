.. _resource_management:

Resource Management
===================

Overview
--------

Efficient control of CPU and GPU memory is essential for successful model compilation, 
especially when working with large models such as LLMs or diffusion models. 
Uncontrolled memory growth can cause compilation failures or process termination. 
This guide describes the symptoms of excessive memory usage and provides methods 
to reduce both CPU and GPU memory consumption.

Memory Usage Control
--------------------

Peak memory during compilation, as multiples of the model's weight size, measured on a
3 GiB fp16 model. The GPU column includes the model's own copy. TensorRT's builder also
needs about 2 GB of host working memory that does not grow with the model.

.. list-table::
   :header-rows: 1

   * - Setting
     - Peak CPU memory
     - Peak GPU memory
   * - Default
     - ~1x
     - ~2x
   * - ``offload_module_to_cpu=True``
     - ~2x
     - ~1x

Once compilation finishes, the compiled module holds only the TensorRT engine, about 1x
the weight size on the GPU. Delete the original PyTorch model if you no longer need it.
Models whose lowering creates new weight tensors (for example by folding transposes or
casts of weights) need more.

CPU Memory
^^^^^^^^^^

**Common symptoms of high CPU memory usage:**

- Program freeze  
- Process terminated by the operating system  

**Ways to lower CPU memory usage:**

1. **Keep memory trimming enabled**

   On Linux, Torch-TensorRT returns the memory the TensorRT builder frees to the
   operating system after each engine build (glibc ``malloc_trim``). Without it the
   process keeps on the order of gigabytes of freed builder memory for the rest of its
   life. Trimming is on by default; to turn it off, set:

   .. code-block:: bash

      export TORCHTRT_ENABLE_BUILDER_MALLOC_TRIM=0

2. **Disable CPU offloading**

   In compilation settings, set:

   .. code-block:: python

      offload_module_to_cpu = False

   The weights then stay on the GPU instead of moving to the CPU. TensorRT still builds
   from a host copy of them, but that copy is freed as soon as the engine is built.

GPU Memory
^^^^^^^^^^

**Common symptoms of high GPU memory usage:**

- CUDA out-of-memory errors
- TensorRT compilation errors

**Ways to lower GPU memory usage:**

1. **Enable offloading to CPU**

   In compilation settings, set:

   .. code-block:: python

      offload_module_to_cpu = True

   This moves the model's weights to CPU memory before the engine builds, a block at a
   time, and TensorRT reads them there in place. Peak GPU memory drops to about the size
   of the engine (**1x**), while CPU memory holds the weights for the rest of
   compilation (about **2x** at peak, while the engine is handed to the runtime).

Unified Memory Systems
^^^^^^^^^^^^^^^^^^^^^^

On systems where the CPU and GPU share physical memory, such as Jetson or DGX Spark,
CPU and GPU allocations come out of the same pool, so budget for their sum. Counting
the model itself, compilation peaks at about **3.8x** the weight size with default
settings and about **3.2x** with ``offload_module_to_cpu=True``. Moving the weights to
the CPU frees no memory on these systems, but TensorRT then builds from them directly
instead of from a second copy.

----

Runtime Weight Streaming
------------------------

Weight streaming allows a compiled TRT engine to use **less GPU VRAM at inference time**
by streaming model weights from CPU memory to the GPU on demand. This is useful for very
large models (LLMs, diffusion models) that exceed available VRAM.

**Enable during compilation:**

.. code-block:: python

    trt_model = torch_tensorrt.compile(
        model,
        ir="dynamo",
        arg_inputs=inputs,
        enable_weight_streaming=True,
    )

**Adjust the GPU memory budget at runtime:**

Use ``torch_tensorrt.runtime.weight_streaming`` as a context manager to set how much GPU
memory the engine is allowed to use for weights. Setting a smaller budget forces more
streaming from CPU:

.. code-block:: python

    import torch_tensorrt

    # Allocate 2 GiB on GPU for weights; the rest streams from CPU
    with torch_tensorrt.runtime.weight_streaming(trt_model) as ctx:
        ctx.device_budget = 2 * 1024**3  # bytes
        output = trt_model(*inputs)
    # Budget is reset to the original value on exit

**Query available budget information:**

.. code-block:: python

    with torch_tensorrt.runtime.weight_streaming(trt_model) as ctx:
        # Total streamable bytes across all TRT submodules
        print(f"Total streamable: {ctx.total_device_budget} bytes")
        # Automatically selected optimal budget
        auto_budget = ctx.get_automatic_weight_streaming_budget()
        print(f"Auto budget: {auto_budget} bytes")
        ctx.device_budget = auto_budget
        output = trt_model(*inputs)

.. note::

   Weight streaming requires ``enable_weight_streaming=True`` at compile time. If the
   model was not compiled with this flag, ``ctx.total_device_budget`` will be ``0`` and
   setting ``device_budget`` will raise a ``RuntimeError``.

----

Dynamic Resource Allocation
----------------------------

By default, TRT submodules allocate GPU memory **statically** at module initialization.
The ``ResourceAllocationStrategy`` context manager temporarily switches all TRT
submodules in a compiled graph module to **dynamic** allocation — resources are allocated
and freed per forward call rather than held for the module lifetime.

This can reduce peak GPU memory when running multiple compiled models concurrently, at
the cost of slightly higher per-call latency:

.. code-block:: python

    from torch_tensorrt.dynamo.runtime import ResourceAllocationStrategy

    trt_model = torch_tensorrt.compile(model, ir="dynamo", arg_inputs=inputs)

    with ResourceAllocationStrategy(trt_model, dynamically_allocate_resources=True):
        output = trt_model(*inputs)
    # Submodules revert to static allocation on exit

Use ``dynamically_allocate_resources=False`` to force static allocation inside the
context (the opposite direction — useful for profiling or benchmarking).

----

Compiling from Memory-Mapped Weights
------------------------------------

To compile, a model's weights normally have to be loaded into host memory first, all of them, before
the first engine is built. ``torch_tensorrt.dynamo.tools.map_weights`` instead points the model's weights at
zero-copy views of a safetensors file:

- ``torch.export`` reads only shapes, so a weight's pages become resident only when TensorRT reads them,
  while it builds the engine that uses the weight.
- With ``release_weights_after_build=True``, the pages are dropped again once the last engine that reads
  the weight is built. The tensor stays valid; its pages are re-read from the file if it is used again.

.. code-block:: python

    import torch
    import torch_tensorrt
    from torch_tensorrt.dynamo.tools import map_weights, save_weights_for_mapping

    # Once, wherever the model loads: write its weights in the dtypes it runs in.
    save_weights_for_mapping(model, "weights.safetensors")

    # To compile: build the model without allocating weights, and map them.
    with torch.device("meta"):
        model = MyModel().eval()
    mapping = map_weights(model, "weights.safetensors", strict=True)
    print(f"{mapping.mapped_bytes / 2**30:.2f} GiB mapped")

    example = torch.randn(1, 3, 224, 224)  # CPU: the mapped weights are CPU tensors
    exported = torch.export.export(model, (example,))
    trt_model = torch_tensorrt.dynamo.compile(
        exported, inputs=[example.cuda()], release_weights_after_build=True
    )

**What to know:**

- **Compile on the CPU.** Mapped weights are CPU tensors, so keep the model and its example inputs on
  the CPU for ``torch.export``. TensorRT takes weights as host pointers and reads each one in place:
  there's no GPU copy and no host copy of the weights.
- **Checkpoints.** Any safetensors file maps if its tensors have the names, dtypes and shapes the model
  uses; a mismatch raises. ``save_weights_for_mapping`` is for checkpoints whose loader casts, for
  example stored in float32 and run in bfloat16. It writes the weights as loaded, a tied weight once.
- **Missing names** are listed in ``mapping.missing`` and left as they were (still on ``meta`` if the
  model was built there). ``strict=True`` raises instead for any name in the model's ``state_dict``;
  non-persistent buffers, such as rotary frequencies, aren't required, so recompute those or write them
  with ``save_weights_for_mapping``. On any error the model is left unchanged.
- **The file must stay in place** while the model or its weights are in use. ``map_weights`` maps it
  copy-on-write and never writes to it. A write to a released weight is lost when its pages are dropped.
- **Releasing pages uses** ``madvise(MADV_DONTNEED)`` **(POSIX).** Elsewhere the mapping works, but the
  pages stay resident until the file is unmapped.

Combined with CPU compilation and the resource partitioner, the host's peak stays near the largest
single engine build instead of the whole model. For pi0.5 (6.7 GiB of bfloat16 weights), host RSS before
compilation went from 8.7 to 2.9 GiB. See :ref:`compile_from_mapped_weights` for a runnable example.


