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

Some memory stays allocated between calls. With CUDA graphs on, the C++ runtime keeps
each engine's memory for as long as its captured graph, because a replay writes to the
memory the graph was recorded with. The Python runtime keeps each engine's memory from
its last call, with or without CUDA graphs, until a later call replaces it or a settings
change drops it.

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
- The pages are dropped again once the last engine that reads the weight is built. A mapped weight stays
  valid; its pages are read back if it is used again.

``torch_tensorrt.dynamo.low_memory`` does all of it in one block: it builds the model on ``meta``, so
building it allocates no weights, maps them, and drops their pages when the block ends, unless the block
sets ``release_weights_after_build=False``. It turns the resource partitioner on, so the model is built
as several engines, each small enough to build within ``cpu_memory_budget`` (or the available memory
without one), so only about one engine's weights are resident at a time. A part of the model that runs in
PyTorch gets its weights on the GPU after compilation (``offload_module_to_cpu``, on in the block):

.. code-block:: python

    import torch
    import torch_tensorrt

    # Map the checkpoint's weights, and build engines that fit an 8 GiB host memory budget.
    ctx = torch_tensorrt.dynamo.low_memory(
        lambda: MyModel().eval(), "model.safetensors", cpu_memory_budget=8 << 30
    )
    with ctx as model:
        print(f"{ctx.report.mapped_bytes / 2**30:.2f} GiB mapped")
        example = torch.randn(1, 3, 224, 224)  # CPU: the mapped weights are CPU tensors
        exported = torch.export.export(model, (example,))
        trt_model = torch_tensorrt.dynamo.compile(exported, inputs=[example.cuda()])

``cpu_memory_budget`` is the host memory the compiling process may use, counting what it already uses,
and each engine gets up to a quarter of what's left. In :ref:`compile_from_mapped_weights`, a model with
192 MiB of weights built this way as 7 engines had at most 32 MiB of its weights file resident while
compiling, against all 192 MiB when built as one engine.

``low_memory`` also takes the compile settings that lower memory during compilation:
``offload_module_to_cpu``, ``enable_resource_partitioning``, ``cpu_memory_budget`` and
``release_weights_after_build``. They're the defaults for every call to
``torch_tensorrt.dynamo.compile`` inside the block, including in a script you don't own, and so for
``torch_tensorrt.compile(..., ir="dynamo")``. ``torch.compile(backend="tensorrt")``,
``cross_compile_for_windows`` and ``convert_exported_program_to_serialized_trt_engine`` don't use them.
A setting left at ``None`` isn't set, and an argument passed to ``compile`` wins over the block's:

.. code-block:: python

    with torch_tensorrt.dynamo.low_memory(cpu_memory_budget=8 << 30):
        run_export_script()

``low_memory`` uses ``map_weights``, which also works alone on a model you build yourself.

**What to know:**

- **Compile on the CPU.** Mapped weights are CPU tensors, so keep the model and its example inputs on
  the CPU for ``torch.export``. TensorRT takes weights as host pointers and reads each one in place:
  there's no GPU copy and no host copy of the weights.
- **Checkpoints.** Any safetensors file maps if its tensors have the names, dtypes and shapes the model
  uses; a mismatch raises. A checkpoint stored in other dtypes than the model runs in, for example
  float32 for a bfloat16 model, is converted once, as below. The file is checked from its header before
  anything is mapped: a truncated file, bytes after the last tensor, or tensors whose bytes overlap, leave
  gaps or don't hold their dtype and shape raise. Damage inside a tensor's bytes isn't detected, since
  that would mean reading the whole file.
- **Converting without loading.** ``save_weights_for_mapping(model, path, source=...)`` writes a model
  built on ``meta`` straight from a checkpoint. It writes one tensor at a time. For each meta tensor it
  calls ``source(name, tensor)``, which returns the data whole or in chunks along dimension 0, and it
  casts each chunk to the model's dtype. ``checkpoint_source(checkpoint, key=...)`` is a ready source that
  reads a safetensors file, or a sharded checkpoint's ``model.safetensors.index.json``, in 64 MiB chunks,
  so even a float32 checkpoint is never loaded. Change the meta model the way the compiled model will be
  changed first, such as casting it to bfloat16, so the file holds the dtypes and shapes the export maps.
  For pi0.5, converting the 14.5 GB float32 checkpoint held 0.9 GiB of process memory, against 9.3 GiB
  for loading it on the CPU and writing the loaded model. Buffers a checkpoint doesn't store, such as
  non-persistent rotary frequencies, must be real tensors, not on ``meta``: accelerate's
  ``init_empty_weights()`` builds a model that way. ``save_weights_for_mapping(model, path)`` without a
  source writes a model that is already loaded.

  .. code-block:: python

      from torch_tensorrt.dynamo.tools import checkpoint_source, save_weights_for_mapping

      with torch.device("meta"):
          model = MyModel().to(torch.bfloat16)
      save_weights_for_mapping(
          model, "weights.safetensors", source=checkpoint_source("checkpoint.safetensors")
      )

- **Missing names** raise in ``low_memory``: every tensor must be in the file, non-persistent buffers
  such as rotary frequencies included, and ``save_weights_for_mapping`` writes those too.
  ``map_weights`` alone raises for any name in the model's ``state_dict``, but doesn't require
  non-persistent buffers. With ``strict=False``, it lists missing names in its result's ``missing`` and
  leaves them as they were. A missing tensor on ``meta`` stays there, with no values, and a warning names
  it. On any error the model is left unchanged.
- **The file must stay in place and unchanged** while the model or its weights are in use: rewriting it
  changes or invalidates the weights. ``map_weights`` maps it copy-on-write and never writes to it.
  Weights are read when compilation first uses them, so ``compile`` checks that the file of each mapped
  weight the program reads is still the one mapped (same inode, size and modification time) and raises
  otherwise; reading a weight past the end of a file truncated since would kill the process with
  ``SIGBUS``. ``release_weights_after_build=True`` also frees weights that aren't mapped, which the
  source model then can't use.
- **Releasing pages uses** ``madvise``: ``MADV_PAGEOUT`` on Linux 5.4 and later, which keeps writes made
  to a weight, else ``MADV_DONTNEED``, which discards them, so the weight reverts to the file's data. On
  Windows the mapping works, but pages stay resident until the file is unmapped, and a warning says so.

With mapped weights, CPU compilation and the resource partitioner, the host's peak stays near the
largest single engine build instead of the whole model. For pi0.5 (6.7 GiB of bfloat16 weights), host
RSS before compilation went from 8.7 to 2.9 GiB. See :ref:`compile_from_mapped_weights` for a runnable
example.


