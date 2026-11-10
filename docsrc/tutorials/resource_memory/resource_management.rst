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

The ways to lower that memory, each described below:

.. list-table::
   :header-rows: 1

   * - Technique
     - Lowers
     - How
   * - Memory trimming
     - Host memory kept after each build
     - On by default (``TORCHTRT_ENABLE_BUILDER_MALLOC_TRIM``)
   * - Offloading to the CPU
     - GPU memory, for more host memory
     - ``offload_module_to_cpu=True``
   * - Compiling on the CPU
     - GPU memory, with about the same host memory
     - Export with CPU inputs, compile with GPU inputs
   * - Several smaller engines
     - Host memory of each build
     - ``enable_resource_partitioning=True``, ``cpu_memory_budget``
   * - Releasing weights as engines are built
     - Weights held during compilation
     - ``release_weights_after_build``
   * - Mapping the weights from their file
     - Host memory before and during compilation
     - ``torch_tensorrt.dynamo.tools.map_weights``
   * - Weight streaming
     - GPU memory at run time
     - ``enable_weight_streaming=True``

They combine. ``torch_tensorrt.dynamo.low_memory`` maps the weights, which leaves the model
on the CPU, and sets the partitioner and the weight release for every compile in a block;
see `Compiling from Memory-Mapped Weights`_.

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

3. **Build several smaller engines**

   Resource partitioning splits each TensorRT subgraph into pieces whose weights fit a
   per-engine budget, and builds them one after another, so the builder's working memory
   scales with the largest piece instead of the whole model:

   .. code-block:: python

      trt_model = torch_tensorrt.compile(
          model,
          arg_inputs=inputs,
          enable_resource_partitioning=True,
          cpu_memory_budget=16 * 1024**3,  # bytes
      )

   The per-engine budget is ``(cpu_memory_budget - resident memory) / 4``. Resident memory
   is the process's RSS when partitioning runs, so it includes weights held on the host
   (on a CPU compile, every weight already read into memory, see below): set
   ``cpu_memory_budget`` above that. Without ``cpu_memory_budget``, the budget is the
   available system memory divided by 4. See :ref:`low_cpu_memory_compilation` for a
   runnable example.

   Each engine boundary adds a little latency. A split that cuts through a region reusing
   the same weights, such as an unrolled loop, also costs memory: every engine that uses
   those weights carries its own copy. For a 3.6-billion-parameter model with a 10-step unrolled
   denoising loop compiled into 16 engines, the engines totaled 10.9 GB against 5.3 GB as
   one engine (GPU memory at inference: 11.5 GB against 5.5 GB), and inference was 2%
   slower.

4. **Release weights as their engines are built**

   Each TensorRT engine carries its own copy of its weights, so the model's copy of a
   weight isn't needed for the rest of compilation once the last engine that reads it is
   built. ``release_weights_after_build`` frees it then, and frees the weights no engine
   reads before the first build. Weights read by layers left in PyTorch are kept.

   * ``None`` (default): only weights mapped from a file (option 5) are released. Their
     pages are dropped and re-read from the file if used again, so the model keeps working.
   * ``True``: the model's other weights are freed too. The model shares that memory, so
     it can't be used after compiling, even if compiling fails.
   * ``False``: nothing is released.

   With several engines (option 3), compilation then holds only the weights still to be
   built. For the same model compiled on the CPU as 9 engines, the peak of host and GPU
   memory together went from 25.7 to 18.9 GiB with ``True``.

5. **Map the weights from their file**

   Instead of loading the weights before compiling, point the model at its safetensors
   file: a weight's pages are then read only while TensorRT builds the engine that uses
   it. See `Compiling from Memory-Mapped Weights`_.

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

2. **Compile on the CPU**

   Keep the model on the CPU: export it with CPU example inputs and compile the
   ``ExportedProgram`` with inputs on the GPU. TensorRT takes weights from host memory, and
   the converters pass a CPU weight's own memory to it instead of copying it, so the model
   never has a copy on the GPU. Unlike option 1, no model copy is added on the host: host
   memory stays about where a compile on the GPU puts it.

   .. code-block:: python

      model = model.eval()  # on the CPU
      exported = torch.export.export(model, (example_input,))  # example_input on the CPU
      trt_model = torch_tensorrt.dynamo.compile(exported, inputs=[example_input.cuda()])
      trt_model = trt_model.cuda()  # layers left to PyTorch still hold CPU weights
      output = trt_model(example_input.cuda())

   Peak GPU memory drops by about **1x** the model size: the GPU holds only the engine
   being built and the engines already built. This also matters where CPU and GPU share
   memory (Jetson), since the sum of both is what has to fit.

   The engine is the same as a GPU compile's, so latency doesn't change. For ResNet-50, ViT,
   BERT, GPT-2 and a 4-layer Llama, the engines were byte-identical with a shared timing
   cache. Constant folding runs where the constants are, here on the CPU, so a float32 folded
   constant can differ from a GPU compile's in the last bit. bfloat16, float16 and float64
   constants match exactly.

   Use ``torch.export`` and ``torch_tensorrt.dynamo.compile`` as above:
   ``torch_tensorrt.compile(model, ir="dynamo", ...)`` doesn't accept a model on the CPU
   (``FakeTensorDeviceMismatchError``).

   TensorRT engines carry their own copy of their weights; layers that run in PyTorch
   (operators without a converter, listed in ``torch_executed_ops``, or in blocks smaller
   than ``min_block_size``) use the model's tensors, so move the compiled module to the GPU
   before running it. Only those layers' weights move, and since they are the model's own
   tensors, they move in the original model too; moving the original model back to the
   CPU moves them out of the compiled module.

   With a model on the CPU, ``offload_module_to_cpu=True`` has nothing to offload, but it
   makes the compiler move the layers that run in PyTorch to ``device`` for you, which
   replaces the ``.cuda()`` call.

   .. note::

      Code that enables autocast only for CUDA tensors (``torch.autocast("cuda", ...)``, or
      autocast gated on ``tensor.is_cuda``) does not apply while the model runs on the CPU
      during export, which can leave low-precision weights next to FP32 activations and fail
      with a dtype mismatch during export or conversion (for example
      ``IMatrixMultiplyLayer must have same input types``). Wrap those regions in
      ``torch.autocast("cpu", dtype=...)`` for the export.

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
building it allocates no weights, maps them, and drops their pages when the block ends, unless
``release_weights_after_build=False``. Its ``compile_settings`` turn the resource partitioner on, so the
model is built as several engines, each small enough to build within ``cpu_memory_budget`` (or the
available memory without one), so only about one engine's weights are resident at a time. They also give a
part of the model that runs in PyTorch its weights on the GPU after compilation
(``offload_module_to_cpu``). Pass them to ``compile``:

.. code-block:: python

    import torch
    import torch_tensorrt

    # Map the checkpoint's weights, and build engines that fit an 8 GiB host memory budget.
    with torch_tensorrt.dynamo.low_memory(
        lambda: MyModel().eval(), "model.safetensors", cpu_memory_budget=8 << 30
    ) as low_mem:
        print(f"{low_mem.report.mapped_bytes / 2**30:.2f} GiB mapped")
        example = torch.randn(1, 3, 224, 224)  # CPU: the mapped weights are CPU tensors
        exported = torch.export.export(low_mem.model, (example,))
        trt_model = torch_tensorrt.dynamo.compile(
            exported, inputs=[example.cuda()], **low_mem.compile_settings
        )

``cpu_memory_budget`` is the host memory the compiling process may use, counting what it already uses,
and each engine gets up to a quarter of what's left. In :ref:`compile_from_mapped_weights`, a model with
192 MiB of weights built this way as 7 engines had at most 32 MiB of its weights file resident while
compiling, against all 192 MiB when built as one engine.

``low_memory`` takes the compile settings that lower memory during compilation:
``offload_module_to_cpu``, ``enable_resource_partitioning``, ``cpu_memory_budget`` and
``release_weights_after_build``. ``compile_settings`` holds the ones given, with the first two on unless
set. They apply only to the calls they're passed to: another model compiled inside the block keeps its own
settings.

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
  that would mean reading the whole file. Tied weights, and distinct tensors that are one view of a
  storage (``w`` and ``Parameter(w.detach())``), are stored once and mapped to one view, so they stay
  shared; tensors that share a storage otherwise raise.
- **Converting without loading.** ``save_weights_for_mapping(model, path, source=...)`` writes a model
  built on ``meta`` straight from a checkpoint. It writes one tensor at a time. For each tensor it calls
  ``source(name, tensor)``, which returns the data whole or in chunks along dimension 0, and it casts
  each chunk to the model's dtype. ``checkpoint_source(checkpoint, key=...)`` is a ready source that
  reads a safetensors file, or a sharded checkpoint's ``model.safetensors.index.json``, in 64 MiB chunks,
  so even a float32 checkpoint is never loaded. Change the meta model the way the compiled model will be
  changed first, such as casting it to bfloat16, so the file holds the dtypes and shapes the export maps.
  For pi0.5, converting the 14.5 GB float32 checkpoint held 0.9 GiB of process memory, against 9.3 GiB
  for loading it on the CPU and writing the loaded model. A tensor the source doesn't store (it raises
  ``KeyError``) is written from the model, with a warning if it is persistent, so buffers a checkpoint
  doesn't store, such as non-persistent rotary frequencies, must be real tensors, not on ``meta``.
  accelerate's ``init_empty_weights()`` builds a model that way, and its persistent buffers, such as batch
  norm statistics, still come from the checkpoint. ``save_weights_for_mapping(model, path)`` without a
  source writes a model that is already loaded.

  .. code-block:: python

      from torch_tensorrt.dynamo.tools import checkpoint_source, save_weights_for_mapping

      with torch.device("meta"):
          model = MyModel().to(torch.bfloat16)
      save_weights_for_mapping(
          model, "weights.safetensors", source=checkpoint_source("checkpoint.safetensors")
      )

- **Missing names** raise for any name in the model's ``state_dict``, in ``map_weights`` and so in
  ``low_memory``; non-persistent buffers, such as rotary frequencies, aren't required, so recompute those
  or write them with ``save_weights_for_mapping``. With ``strict=False``, ``map_weights`` lists missing
  names in its result's ``missing`` and leaves them as they were. A missing tensor on ``meta`` stays
  there, with no values, and a warning names it: a model that reads it can't be exported or compiled,
  and ``compile`` names it; one that never reads it is fine. On any error the model is left unchanged.
- **The file must stay in place and unchanged** while the model or its weights are in use: rewriting it
  changes or invalidates the weights. ``map_weights`` maps it copy-on-write and never writes to it. It
  reserves no memory for the mapping, so a file larger than RAM and swap maps too, except with
  ``vm.overcommit_memory=2``. Weights are read when compilation first uses them, so ``compile`` checks that the file of each mapped
  weight the program reads is still the one mapped (same inode, size and modification time) and raises
  otherwise; reading a weight past the end of a file truncated since would kill the process with
  ``SIGBUS``. ``release_weights_after_build=True`` also frees weights that aren't mapped, which the
  source model then can't use.
- **Releasing pages uses** ``madvise``. Where the process owns the weights file or may write to it,
  ``MADV_PAGEOUT`` (Linux 5.4 and later) drops a weight's pages and keeps writes made to it; it leaves
  pages another process maps too. Elsewhere, such as a root-owned file in a container image read by a
  normal user, ``MADV_DONTNEED`` drops them and discards such writes, so the weight reverts to the file's
  data. The bytes reported released are the pages that left the process's memory. On Windows the mapping
  works, but pages stay resident until the file is unmapped, and a warning says so.

With mapped weights, CPU compilation and the resource partitioner, the host's peak stays near the
largest single engine build instead of the whole model. For pi0.5 (6.7 GiB of bfloat16 weights), host
RSS before compilation went from 8.7 to 2.9 GiB. See :ref:`compile_from_mapped_weights` for a runnable
example.


