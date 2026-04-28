.. _performance_tuning:

Performance Tuning Guide
========================

Torch-TensorRT compiles PyTorch models to TensorRT engines, but getting the best
performance requires understanding how TRT optimization works and measuring correctly.
This guide covers why compiled models can appear slow and how to extract maximum speedup.

----

Common Benchmarking Issues
---------------------------------

**Not warming up**

TRT engines, like all GPU kernels, need a warm-up pass to load into GPU memory and
trigger JIT kernel selection:

.. code-block:: python

    import torch
    import torch_tensorrt

    trt_model = torch_tensorrt.compile(model, ir="dynamo", arg_inputs=inputs)

    # Warm up — these runs don't count
    for _ in range(5):
        trt_model(*inputs)

    # Now measure
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(100):
        trt_model(*inputs)
    end.record()
    torch.cuda.synchronize()
    print(f"Avg latency: {start.elapsed_time(end) / 100:.3f} ms")

.. warning::

    ``time.time()`` measures wall-clock time including Python overhead and CPU/GPU
    synchronization gaps. Use CUDA events (``torch.cuda.Event``) for accurate GPU
    latency measurements.

**Comparing against an unoptimized baseline**

PyTorch eager mode benefits from the same GPU memory warm-up effect. Run both the
baseline and the TRT model with the same number of warm-up iterations, and time both
with CUDA events.

**The model is too small**

Overhead from the Python–TRT bridge, memory copies, and kernel launch dominates for
very small models or very small batch sizes. TRT typically shows the largest gains on:

- Large matrix multiplications (Transformers, large MLPs)
- Convolutional models with many layers
- Batch sizes > 1 for latency, or large batches for throughput

A 3-layer MLP on batch size 1 is unlikely to be faster in TRT than in eager mode.
Use :ref:`dryrun` to check TRT coverage before committing to a full compile.

----

Using the Right Precision
--------------------------

The single biggest speedup lever is precision. TRT can run in FP32, FP16, BF16,
INT8, or FP8 — but only if you tell it to.

**Explicit typing (strong typing, always enabled)** — cast your model and inputs to the target dtype:

.. code-block:: python

    # FP16: cast model weights and inputs
    model = model.half()
    inputs = [inp.half() for inp in inputs]
    trt_model = torch_tensorrt.compile(
        model, ir="dynamo", arg_inputs=inputs,
    )

**Autocast** — let Torch-TensorRT automatically lower eligible layers to a reduced precision:

.. code-block:: python

    trt_model = torch_tensorrt.compile(
        model, ir="dynamo", arg_inputs=inputs,
        enable_autocast=True,
        autocast_low_precision_type=torch.float16,
    )

.. list-table::
   :widths: 20 40 40
   :header-rows: 1

   * - Precision
     - How to enable (new API)
     - When to use
   * - FP32 only
     - Default (model weights/inputs in FP32)
     - Accuracy-critical, no speedup vs PyTorch
   * - FP16
     - ``model.half()`` + inputs in FP16
     - Standard choice; 2–3× speedup on Volta+
   * - BF16
     - ``model.bfloat16()`` + inputs in BF16
     - Better numerical range than FP16; Ampere+
   * - FP16 mixed (autocast)
     - ``enable_autocast=True, autocast_low_precision_type=torch.float16``
     - Automatically keeps sensitive layers in FP32
   * - INT8 (with calibration)
     - ModelOpt QDQ nodes
     - Highest throughput; requires ModelOpt quantization
   * - FP8 (Hopper+)
     - ModelOpt FP8 QDQ nodes
     - Best accuracy–throughput tradeoff for LLMs on H100

See :ref:`quantization` for the full INT8/FP8 workflow.

**TF32 (default on Ampere+)**

Ampere and newer GPUs automatically use TF32 for FP32 matrix multiplications —
this is a hardware behavior, not a Torch-TensorRT setting. TF32 gives most of the
FP16 speedup with near-FP32 accuracy. If you need strict FP32, add:

.. code-block:: python

    torch_tensorrt.compile(model, ..., disable_tf32=True)

----

Tuning opt_shape
----------------

TensorRT builds separate kernel implementations for each ``(min, opt, max)`` shape
range. The ``opt_shape`` is the shape TRT tunes for most aggressively:

.. code-block:: python

    inputs = [
        torch_tensorrt.Input(
            min_shape=(1,  3, 224, 224),
            opt_shape=(16, 3, 224, 224),  # <-- tune for this shape
            max_shape=(32, 3, 224, 224),
            dtype=torch.float16,
        )
    ]

**Rule of thumb**: set ``opt_shape`` to the batch size / image size you see most often
in production. If you deploy at batch size 8, set ``opt_shape`` accordingly even if the
engine supports 1–32.

----

Optimization Level
------------------

``optimization_level`` (0–5, default 3) controls how long TRT spends searching for
faster kernel implementations. Higher values produce faster engines at the cost of
longer compile time.

.. code-block:: python

    trt_model = torch_tensorrt.compile(
        model, ir="dynamo", arg_inputs=inputs,
        optimization_level=5,          # maximize performance (slow to compile)
    )

For interactive development use ``optimization_level=0`` (fast compile, decent performance).
For production builds use 3–5.

----

TRT Coverage and Graph Breaks
------------------------------

Performance degrades when a large fraction of the model runs in PyTorch instead of TRT.
Use :ref:`dryrun` to see the partition layout:

.. code-block:: python

    trt_model = torch_tensorrt.compile(
        model, ir="dynamo", arg_inputs=inputs,
        dryrun=True,
    )

Look for lines like::

    TRTInterpreter block (100 operators)     <-- good: large TRT block
    PyTorch block      (3 operators)         <-- graph break

If you see many small TRT blocks separated by PyTorch blocks:

* Check :ref:`supported_ops` for the op causing the break.
* Add ``torch_executed_ops={"op_to_skip"}`` to explicitly push a problematic op to
  PyTorch, which may allow the surrounding TRT blocks to merge.
* Lower ``min_block_size`` (default 5) to allow smaller TRT subgraphs; this reduces
  PyTorch fallback at the cost of more kernel launch overhead per block.
* Set ``use_fast_partitioner=False`` for a global partitioning algorithm that often
  produces fewer, larger TRT blocks (slower to compile).

----

CUDA Graphs
-----------

For latency-critical inference (fixed input shapes, no graph breaks), CUDA Graphs
eliminate kernel launch overhead by recording the CUDA op sequence and replaying it:

.. code-block:: python

    import torch_tensorrt

    trt_model = torch_tensorrt.compile(model, ir="dynamo", arg_inputs=inputs)

    with torch.no_grad():
        with torch_tensorrt.runtime.enable_cudagraphs(trt_model) as cg_model:
            output = cg_model(*inputs)   # first call: records the graph
            output = cg_model(*inputs)   # subsequent calls: fast replay

CUDA Graphs require fixed shapes at runtime. They give the largest gains when:

* Inference is called repeatedly in a tight loop.
* The model has many small kernels (attention, layer norm, etc.).
* You are running batch size 1 latency benchmarks.

See :ref:`cuda_graphs` for details.

----

Engine Caching
--------------

TRT engine compilation can take minutes for large models. **Engine caching** saves
the compiled engine to disk so subsequent runs skip the compilation step:

.. code-block:: python

    import torch_tensorrt
    from torch_tensorrt.dynamo._compiler import compile

    trt_model = torch_tensorrt.compile(
        model,
        ir="dynamo",
        arg_inputs=inputs,
        cache_built_engines=True,
        reuse_cached_engines=True,
    )

See :ref:`engine_cache` for the full caching workflow.

----

Memory and Throughput Tradeoffs
--------------------------------

**Weight streaming** (Ampere+)

For models too large to fit in GPU memory at full precision, weight streaming loads
weights on-demand from CPU. This reduces peak GPU memory at the cost of some throughput:

.. code-block:: python

    trt_model = torch_tensorrt.compile(
        model, ir="dynamo", arg_inputs=inputs,
        # enable weight streaming at compile time
        enable_weight_streaming=True,
    )

    with torch_tensorrt.runtime.weight_streaming(trt_model) as ws_module:
        # Control what fraction of weights stay on GPU (0.0–1.0)
        ws_module.device_budget = 0.5 * ws_module.streamable_weights_size
        output = ws_module(*inputs)

**Workspace size**

TRT allocates scratch memory (workspace) for intermediate activations. Larger
workspace lets TRT pick faster algorithms. Reduce it to cut peak memory if OOM:

.. code-block:: python

    trt_model = torch_tensorrt.compile(
        model, ir="dynamo", arg_inputs=inputs,
        workspace_size=1 << 28,  # 256 MB (default is much larger)
    )

----

Profiling with Nsight
---------------------

For kernel-level analysis, wrap inference in a Nsight annotation:

.. code-block:: python

    import torch.cuda.profiler as profiler

    # Warm up first
    for _ in range(5):
        trt_model(*inputs)

    with profiler.profile():
        for _ in range(100):
            trt_model(*inputs)

Run with::

    nsys profile -o report python your_script.py
    ncu --set full python your_script.py

In Nsight Systems, look for long gaps between CUDA kernels (Python overhead) and
compare the kernel timelines for the TRT model vs the baseline.

----

Benchmarking Checklist
-----------------------

.. list-table::
   :widths: 10 90
   :header-rows: 0

   * - ☐
     - Warm up with at least 5–10 forward passes before measuring
   * - ☐
     - Use CUDA events (not ``time.time()``) for GPU timing
   * - ☐
     - Warm up the baseline model the same way
   * - ☐
     - Use FP16 precision (``model.half()`` with FP16 inputs, or ``enable_autocast=True``) unless you need FP32
   * - ☐
     - Run ``dryrun=True`` to confirm TRT coverage is high
   * - ☐
     - Set ``opt_shape`` to match your most common production input shape
   * - ☐
     - For latency workloads: enable CUDA graphs
   * - ☐
     - For large models: try weight streaming or INT8 quantization
