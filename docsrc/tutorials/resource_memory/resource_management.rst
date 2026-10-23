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
   ``cpu_memory_budget`` above that. Without ``cpu_memory_budget``, the budget is the available system memory divided
   by 4. See :ref:`low_cpu_memory_compilation` for a runnable example.

   Each engine boundary adds a little latency. A split that cuts through a region reusing
   the same weights, such as an unrolled loop, also costs memory: every engine that uses
   those weights carries its own copy. For a 3B-parameter model with a 10-step unrolled
   denoising loop compiled into 16 engines, the engines totaled 10.9 GB against 5.3 GB as
   one engine (GPU memory at inference: 11.5 GB against 5.5 GB), and inference was 2%
   slower.

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


