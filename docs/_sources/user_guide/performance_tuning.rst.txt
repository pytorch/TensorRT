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

----

Global Performance Tuning
-------------------------

TensorRT's
`Global Performance Tuning <https://docs.nvidia.com/deeplearning/tensorrt/latest/performance/tuning.html>`_
searches internal builder knobs, collectively called a *build route*, for a faster
engine. Torch-TensorRT exposes this capability for Dynamo TRT partitions.

The feature requires a TensorRT build with Global Performance Tuning support. It is
available starting with TensorRT 11.1 and is currently unavailable with TensorRT-RTX
or on Windows. Check the installed build before configuring a sweep:

.. code-block:: python

    from torch_tensorrt.dynamo import is_global_perf_tuning_available

    if not is_global_perf_tuning_available():
        raise RuntimeError("Global Performance Tuning is unavailable")

Discovering and applying build routes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``get_all_build_routes()`` is the Torch-TensorRT equivalent of
``trtexec --helpBuildRoute``:

.. code-block:: python

    from torch_tensorrt.dynamo import get_all_build_routes

    knobs = get_all_build_routes()
    print("tuner version:", knobs["tuner_version"])
    for knob in knobs["tuner_options"]:
        print(knob["allowed_values"], knob["default_value"])

A route is a space-separated sequence of ``-knob=value`` tokens. Using ``build_route``
to apply a known route without running a search:

.. code-block:: python

    trt_model = torch_tensorrt.compile(
        model,
        ir="dynamo",
        arg_inputs=inputs,
        build_route="-slice_fusion=off -kgen:codegen:cuda_tile=1",
    )

Route expressions use brackets to provide the values to search. They can contain
Boolean values, finite enums, or an explicit subset of an open-ended integer knob:

.. code-block:: text

    -match_ragged_mha=[on|off]
    -kgen:codegen:cuda_tile=[0|1|2|3]
    -cask_fusion:num_tactics=[10|20]
    -peep:max=[1|2]

For an open-ended knob, Torch-TensorRT searches only the values explicitly listed in
the expression; it does not invent a range. A fixed token such as
``-slice_fusion=off`` is included in every route and does not increase the number of
trials.

Running a sweep
^^^^^^^^^^^^^^^

.. code-block:: python

    trt_model = torch_tensorrt.compile(
        model,
        ir="dynamo",
        arg_inputs=inputs,
        tune_build_routes=(
            "-match_ragged_mha=[on|off] "
            "-kgen:codegen:cuda_tile=[0|1|2|3]"
        ),
        tuning_search="mixed",
        accuracy_threshold=0.01,
        accuracy_algorithm="cos",
        tuning_cache_file="/tmp/torch_trt_tune.jsonl",
    )

Tuning is performed independently for each TRT partition. Use
``require_full_compilation=True`` when the model should be tuned as one engine, which
is closest to whole-network ``trtexec`` behavior.

The search algorithms are:

* ``fast``: build one baseline, then change one knob at a time. For knobs with
  ``n_i`` candidate values, this produces ``1 + sum(n_i - 1)`` trials.
* ``full``: build the Cartesian product of all candidate values. This produces
  ``product(n_i)`` trials and can become expensive quickly.
* ``mixed``: run ``fast`` first, then build combinations of only the knobs for which
  at least one one-at-a-time value was faster than the baseline. Duplicate routes
  between the two phases are not rebuilt.

If the TensorRT default for a knob is not present in the expression, ``fast`` and
``mixed`` use the first listed value as that knob's baseline. Timing comparisons are
strict; any finite value below the baseline marks the knob as improved. Small timing
fluctuations can therefore expand a ``mixed`` search. Use ``full`` for exhaustive,
repeatable route coverage, and start with ``fast`` when compile time matters.

``tuning_dry_run=True`` prints the expanded routes without building engines. It
requires ``tune_build_routes`` or ``tune_build_route_file`` and cannot be combined
with ``mixed``.

Process isolation
^^^^^^^^^^^^^^^^^

Each trial is built, checked, and benchmarked in a fresh process created with the
Python ``spawn`` start method. A TensorRT or driver abort terminates only that child.
The parent checks the child exit status and the atomically published result before
continuing to the next route.

Because ``spawn`` imports the main module in each child, executable scripts must use
the standard entry-point guard:

.. code-block:: python

    def main():
        torch_tensorrt.compile(...)


    if __name__ == "__main__":
        main()

Without this guard, a child can execute the compile call again while it is still
starting, causing a multiprocessing bootstrapping error.

Input samples, accuracy, and timing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Torch-TensorRT materializes one deterministic set of example tensors in the parent
and uses it for every route. Generated samples use a local fixed seed without
changing the application's RNG state.

When ``accuracy_threshold`` is set, eager Torch reference outputs are computed once
before the sweep. The outputs are moved to CPU for safe transfer and moved to the
target GPU in each child for comparison with that trial's TRT outputs. Nested
``list``, ``tuple``, and ``dict`` outputs are checked tensor by tensor. If any output
has a non-finite loss or exceeds the threshold, that route cannot win. Setting
``accuracy_threshold=None`` skips both reference execution and accuracy checking.

Supported metrics are ``l0``, ``l1``, ``l2``, ``lInf``, and ``cos``; lower is better.
``accuracy_atol`` and ``accuracy_rtol`` apply only to ``l0``. Each successful route is
warmed up three times and measured ten times with CUDA events. The recorded
``gpu_time`` is the median latency in milliseconds.

Cache and resume
^^^^^^^^^^^^^^^^

``tuning_cache_file`` is a base path. For example, ``/tmp/tune.jsonl`` produces one
``/tmp/tune.<partition_key>.jsonl`` file per TRT partition. The first JSONL row stores
the sweep configuration and later rows store each completed trial.

Resume an interrupted sweep by supplying only the same base path:

.. code-block:: python

    trt_model = torch_tensorrt.compile(
        model,
        ir="dynamo",
        arg_inputs=inputs,
        tuning_continue=True,
        tuning_cache_file="/tmp/torch_trt_tune.jsonl",
    )

Do not combine ``tuning_continue=True`` with a new route expression or
``tuning_dry_run``. Mixed-search phases are reconstructed from the global iteration
indices and the completed fast-phase results.

The cache deliberately uses a lightweight graph fingerprint and a simple,
``trtexec``-like JSONL layout. It is not interchangeable with a ``trtexec`` cache and
does not fully identify weights, input profiles, compilation settings, or hardware.
Use a different base path, or remove the old partition files, after changing any of
those inputs.

Failures and resource usage
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Some knob values are model-dependent and may fail to build. Failed children and
accuracy failures are recorded with ``crash=true`` or a non-empty ``error_message``
and are excluded from winner selection. If all routes fail, tuning raises an error
instead of returning an invalid engine.

Trial payload and result files are temporary and are removed after each candidate.
Only the current winner's engine bytes are retained during a new sweep. When engine
caching is enabled and the engine is eligible for weight stripping/refit, only the
final winner is inserted into the configured Engine Cache.

``tuning_timeout_s`` limits search time. When the remaining budget expires, the
active child is killed and recorded as failed, and no additional search trial is
started. Use ``-1`` to disable the timeout.

Build-route performance is specific to the model, input profile, GPU, and TensorRT
version. Re-tune after changing any of them. Full and mixed sweeps can multiply
compilation time, especially for models with multiple TRT partitions.

See :ref:`global_perf_tuning_attention_example` for a runnable attention example.
