.. _compilation_settings:

CompilationSettings Reference
==============================

``CompilationSettings`` is the single dataclass that controls every aspect of Torch-TensorRT
Dynamo compilation. It is passed (directly or via keyword arguments) to
:func:`torch_tensorrt.dynamo.compile`, :func:`torch_tensorrt.compile`, and
:func:`torch.compile` with ``backend="tensorrt"``.

.. code-block:: python

    import torch
    import torch_tensorrt

    trt_gm = torch_tensorrt.dynamo.compile(
        exported_program,
        arg_inputs=inputs,
        use_explicit_typing=True,  # respects dtypes set in model/inputs
        min_block_size=3,
        optimization_level=4,
    )

All parameters have sensible defaults. Change only what you need.

----

Core Parameters
---------------

.. list-table::
   :widths: 28 12 60
   :header-rows: 1

   * - Parameter
     - Default
     - Description
   * - ``enabled_precisions`` (**DEPRECATED**)
     - ``{dtype.f32}``
     - Set of precisions the TensorRT builder may use. Any combination of
       ``torch.float32``, ``torch.float16``, ``torch.bfloat16``, ``torch.int8``,
       ``torch.float8_e4m3fn``. Adding a lower precision does not force its use — TRT
       selects the best kernel per layer. For INT8 calibration or FP8, use ModelOpt
       quantization first.
   * - ``min_block_size``
     - ``5``
     - Minimum number of consecutive TRT-capable operators required to form a TRT
       engine block. Subgraphs smaller than this are merged back into PyTorch. Lower
       values increase TRT coverage but may add engine-launch overhead for tiny blocks.
       Use :ref:`dryrun mode <dryrun>` to find the sweet spot.
   * - ``torch_executed_ops``
     - ``set()``
     - Force specific ``torch.ops`` targets to run in PyTorch regardless of converter
       support. Example: ``{torch.ops.aten.add.Tensor}`` to keep all adds in PyTorch.
   * - ``require_full_compilation``
     - ``False``
     - Raise an error if any node cannot be placed in TensorRT. Useful for CI
       correctness gates on models that are known to be fully TRT-compatible.
   * - ``device``
     - current CUDA device
     - :class:`torch_tensorrt.Device` specifying the GPU to compile for.
   * - ``use_python_runtime``
     - ``False`` (auto)
     - ``False`` uses the C++ runtime (recommended — serializable, CUDAGraphs,
       multi-device safe). ``True`` forces the Python runtime (simpler to instrument
       for debugging but not serializable to ``ExportedProgram``). ``None`` selects C++
       if available.
   * - ``pass_through_build_failures``
     - ``False``
     - When ``True``, TRT engine build errors raise exceptions rather than fall back to PyTorch.
       Useful during development when some subgraphs are not yet fully convertible.

----

Optimization Tuning
-------------------

.. list-table::
   :widths: 28 12 60
   :header-rows: 1

   * - Parameter
     - Default
     - Description
   * - ``optimization_level``
     - ``None`` (TRT default = 3)
     - Integer 0–5. Higher levels let TRT spend more time searching for faster kernels
       at the cost of longer compile time. 0 = fastest build, 5 = best runtime
       performance. TRT's built-in default (3) is a good balance for most workloads.
   * - ``workspace_size``
     - ``0`` (TRT default)
     - Maximum GPU memory (bytes) TRT may allocate as scratch space during engine build.
       ``0`` means TRT picks a size automatically. Increase if you see workspace-related
       build warnings on large models.
   * - ``num_avg_timing_iters``
     - ``1``
     - Number of iterations used to time and select kernels during the build phase.
       Higher values reduce timing noise and can improve kernel selection on NUMA or
       shared-GPU environments, at the cost of longer compile time.
   * - ``max_aux_streams``
     - ``None`` (TRT default)
     - Maximum number of auxiliary CUDA streams TRT may use per engine for concurrent
       layer execution. ``None`` lets TRT decide. Set to ``0`` to disable auxiliary
       streams and run all layers on the main stream.
   * - ``tiling_optimization_level``
     - ``"none"``
     - Controls how aggressively TRT searches for tiling strategies. Options:
       ``"none"`` (no tiling search), ``"fast"`` (quick scan), ``"moderate"``,
       ``"full"`` (exhaustive search — best performance but slowest compile). Tiling
       can substantially improve throughput for convolution-heavy models.
   * - ``l2_limit_for_tiling``
     - ``-1`` (no limit)
     - Target L2 cache usage limit in bytes for tiling optimization. Use when you want
       tiling kernels to fit within a specific L2 budget (e.g., on multi-tenant GPUs).
       ``-1`` disables the constraint.
   * - ``sparse_weights``
     - ``False``
     - Allow TRT to use sparse-weight kernels for qualified layers. Requires 2:4
       structured sparsity in the model weights. Can provide significant throughput
       improvements on Ampere+ GPUs with sparse weights.
   * - ``disable_tf32``
     - ``False``
     - Disable TensorFloat-32 (TF32) accumulation. TF32 is enabled by default on
       Ampere and newer GPUs and provides FP32-range with FP16-speed for matmul/conv.
       Disable only when you need strict IEEE FP32 semantics.
   * - ``attn_bias_is_causal``
     - ``True``
     - Whether the attn_bias in efficient SDPA is causal. Default is True. This can 
       accelerate models from HF because attn_bias is always a causal mask in HF. 
       If you want to use non-causal attn_bias, you can set this to False.

----

Precision and Typing
---------------------

.. list-table::
   :widths: 28 12 60
   :header-rows: 1

   * - Parameter
     - Default
     - Description
   * - ``truncate_double``
     - ``False``
     - Automatically truncate ``float64`` (double) inputs and weights to ``float32``
       before passing to TRT. TRT does not natively support float64; enable this to
       compile models that use double without changing the model code.
   * - ``use_fp32_acc``
     - ``False``
     - Insert FP32 cast nodes around matmul layers so that accumulation happens in
       FP32 even when the network runs in FP16. Improves numerical accuracy for
       transformer models at a small throughput cost. Requires ``use_explicit_typing=True``.
   * - ``use_explicit_typing``
     - ``True``
     - Respect the dtypes set in the PyTorch model (strong typing). When ``True``,
       TRT will not silently up/downcast layers. This is the recommended approach for
       controlling precision — set dtypes in your model/inputs directly. Required by
       autocast and ``use_fp32_acc``.

----

Autocast (Automatic Mixed Precision)
--------------------------------------

Autocast is an alternative to manually setting layer precisions a model. It analyses the graph and
selectively lowers eligible operations to a reduced precision, skipping nodes that are
numerically sensitive. Enable it with ``enable_autocast=True``.

.. note::

   When ``enable_autocast=True``, ``use_explicit_typing`` is automatically set to
   ``True`` as well.

.. code-block:: python

    trt_gm = torch_tensorrt.dynamo.compile(
        exported_program,
        arg_inputs=inputs,
        enable_autocast=True,
        autocast_low_precision_type=torch.float16,
        autocast_excluded_ops={torch.ops.aten.softmax.int},
        autocast_max_output_threshold=1024.0,
    )

.. list-table::
   :widths: 30 12 58
   :header-rows: 1

   * - Parameter
     - Default
     - Description
   * - ``enable_autocast``
     - ``False``
     - Enable graph-aware automatic mixed precision. Analyses op outputs and reduction
       depths to assign each node to FP32 or low precision.
   * - ``autocast_low_precision_type``
     - ``None``
     - The reduced precision to cast down to. Supported: ``torch.float16``,
       ``torch.bfloat16``. ``None`` disables actual casting (no-op even if
       ``enable_autocast=True``).
   * - ``autocast_excluded_nodes``
     - ``set()``
     - Set of **regex patterns** matched against node names. Nodes whose names match
       any pattern are kept in FP32. Example: ``{".*layer_norm.*"}`` keeps all
       LayerNorm ops in FP32.
   * - ``autocast_excluded_ops``
     - ``set()``
     - Set of ``torch.ops`` targets always kept in FP32 regardless of other autocast
       decisions. Example: ``{torch.ops.aten.softmax.int}``.
   * - ``autocast_max_output_threshold``
     - ``512.0``
     - Nodes whose outputs exceed this absolute value are kept in FP32. Guards against
       overflow in activations with large dynamic range (e.g., unnormalized logits).
   * - ``autocast_max_depth_of_reduction``
     - ``None`` (∞)
     - Maximum reduction depth allowed in low precision. Reduction depth measures how
       many reduction operations (sum, mean, etc.) feed into a node. Nodes with higher
       depth are kept in FP32 to prevent error accumulation. ``None`` means no limit.
   * - ``autocast_calibration_dataloader``
     - ``None``
     - A ``torch.utils.data.DataLoader`` used to calibrate output statistics (e.g.,
       absolute max values) for each node. When provided, the autocast pass runs the
       model on calibration data to make per-node precision decisions based on
       observed ranges rather than static thresholds.

----

Weight Management
-----------------

These settings control how TRT engines store and manage weights. They primarily affect
serialized engine size and whether the engine can be refitted without recompilation.

.. list-table::
   :widths: 28 12 60
   :header-rows: 1

   * - Parameter
     - Default
     - Description
   * - ``immutable_weights``
     - ``True``
     - Build non-refittable engines. When ``True``, TRT can apply more aggressive
       weight-fusing optimizations but the engine cannot be updated via
       :func:`~torch_tensorrt.dynamo.refit_module_weights`. Set to ``False`` if you
       plan to do weight updates (e.g., LoRA, fine-tuning).
   * - ``refit_identical_engine_weights``
     - ``False``
     - When multiple subgraphs share identical weight tensors (e.g., weight-tied
       language models), refit all engines with the same weights in a single pass.
       Requires ``immutable_weights=False``.
   * - ``strip_engine_weights``
     - ``False``
     - Serialize engines without weight data. Produces smaller engine files; weights
       must be refitted before the engine can run. Useful for distributing
       architecture-only engine blueprints. On TRT ≥ 10.14 this is handled
       automatically via TRT's ``INCLUDE_REFIT`` serialization flag.
   * - ``enable_weight_streaming``
     - ``False``
     - Enable TRT weight streaming for engines whose weights exceed GPU memory.
       Weights are streamed from host memory during inference. Requires TRT support
       and is typically used for very large models.

----

Hardware Compatibility
-----------------------

.. list-table::
   :widths: 28 12 60
   :header-rows: 1

   * - Parameter
     - Default
     - Description
   * - ``version_compatible``
     - ``False``
     - Build engines that can run on a newer TRT version than the one used to compile
       them (forward ABI compatibility). Disables some runtime optimizations; use when
       you need to ship engine files that may be loaded by a future TRT version.
   * - ``hardware_compatible``
     - ``False``
     - Build engines compatible with GPU architectures other than the compilation GPU.
       Currently supports NVIDIA Ampere and newer. Useful for compiling on one GPU SKU
       and deploying on another within the same generation.
   * - ``engine_capability``
     - ``STANDARD``
     - Restrict kernel selection to safe GPU kernels (``SAFETY``) or safe DLA kernels
       (``DLA_STANDALONE``). Use ``STANDARD`` for normal inference workloads.
   * - ``enable_cross_compile_for_windows``
     - ``False``
     - Build engines on Linux that are deployable on Windows (x86-64). Disables Python
       runtime, lazy engine init, and engine caching. See
       :func:`torch_tensorrt.dynamo.cross_compile_for_windows`.

----

Memory and Resource Management
--------------------------------

.. list-table::
   :widths: 28 12 60
   :header-rows: 1

   * - Parameter
     - Default
     - Description
   * - ``enable_resource_partitioning``
     - ``False``
     - Split oversized TRT subgraphs so each piece fits within ``cpu_memory_budget``
       during engine build. Prevents OOM errors on large models at the cost of
       slightly more engines. See :ref:`partitioning` for details.
   * - ``cpu_memory_budget``
     - ``None``
     - Byte budget per TRT engine during build, used by resource partitioning. ``None``
       uses available system memory (reported by ``psutil``).
   * - ``offload_module_to_cpu``
     - ``False``
     - Move the model weights to CPU RAM before compilation to reduce GPU memory
       pressure during the build phase. Weights are loaded back to GPU at runtime.
   * - ``dynamically_allocate_resources``
     - ``False``
     - Let TRT dynamically allocate memory for intermediate tensors at runtime rather
       than pre-allocating. Can reduce peak memory footprint at a small runtime cost.

----

Graph Partitioning
-------------------

.. list-table::
   :widths: 28 12 60
   :header-rows: 1

   * - Parameter
     - Default
     - Description
   * - ``use_fast_partitioner``
     - ``True``
     - Use the adjacency (``fast_partition``) partitioner. Set to ``False`` to use the
       globally-optimal ``global_partition`` which minimizes engine count but is slower
       to compile. See :ref:`partitioning`.
   * - ``assume_dynamic_shape_support``
     - ``False``
     - Skip per-converter dynamic shape checks; treat all converters as dynamic-capable.
       Use when you have already validated all your ops with dynamic shapes and want
       to skip the safety gate.
   * - ``enable_experimental_decompositions``
     - ``False``
     - Enable the full set of core ATen decompositions instead of the curated subset.
       May expose more ops to TRT conversion at the cost of potential numerical
       differences.
   * - ``use_distributed_mode_trace``
     - ``False``
     - Use ``aot_autograd`` for tracing instead of the default path. Required when the
       model contains ``DTensor`` or other distributed tensors.
   * - ``decompose_attention``
     - ``False``
     - Decompose attention layers into smaller ops. We have converters for handling attention ops,
       but if you want to decompose them into smaller ops, you can set this to True.

----

Compilation Workflow
---------------------

.. list-table::
   :widths: 28 12 60
   :header-rows: 1

   * - Parameter
     - Default
     - Description
   * - ``dryrun``
     - ``False``
     - Run the full partitioning pipeline without building any TRT engines.
       ``True`` prints a detailed partition report. A string path (e.g.
       ``"/tmp/report.txt"``) also saves the report to a file. See :ref:`dryrun`.
   * - ``lazy_engine_init``
     - ``False``
     - Defer TRT engine deserialization until all engines have been built.
       Works around resource contraints and builder overhad but engines
       may be less well tuned to their deployment resource availability
   * - ``debug``
     - ``False``
     - Enable verbose TRT builder logs at ``DEBUG`` level.

----

Engine Caching
---------------

.. list-table::
   :widths: 28 12 60
   :header-rows: 1

   * - Parameter
     - Default
     - Description
   * - ``cache_built_engines``
     - ``False``
     - Persist compiled TRT engines to disk after building. Combine with
       ``reuse_cached_engines`` for faster subsequent compilations of the same graph.
   * - ``reuse_cached_engines``
     - ``False``
     - Load TRT engines from the disk cache on cache hit, skipping the build step.
       The cache key includes graph structure, input specs, and all engine-invariant
       settings (see :ref:`engine_cache`).
   * - ``timing_cache_path``
     - ``/tmp/torch_tensorrt_engine_cache/timing_cache.bin``
     - Path for TRT's timing cache file. The timing cache records kernel timing data
       across sessions, speeding up subsequent engine builds for similar subgraphs even
       when the engine cache itself is cold. Not used for TensorRT-RTX (no autotuning).
   * - ``runtime_cache_path``
     - ``/tmp/torch_tensorrt_engine_cache/runtime_cache.bin``
     - Path for the TensorRT-RTX runtime cache file. The runtime cache stores JIT
       compilation results at inference time, preventing repeated compilation of
       kernels and graphs across sessions. Uses file locking for concurrent access
       safety. Only used with TensorRT-RTX; ignored for standard TensorRT.

----

DLA Parameters
---------------

Deep Learning Accelerator (DLA) settings are only relevant when compiling for Jetson or
DRIVE platforms. Set ``engine_capability=EngineCapability.DLA_STANDALONE`` to target DLA.

.. list-table::
   :widths: 28 12 60
   :header-rows: 1

   * - Parameter
     - Default
     - Description
   * - ``dla_sram_size``
     - ``1 MB``
     - Fast software-managed SRAM used by DLA for intra-layer communication (bytes).
   * - ``dla_local_dram_size``
     - ``1 GB``
     - Host DRAM used by DLA for intermediate tensor storage across layers (bytes).
   * - ``dla_global_dram_size``
     - ``512 MB``
     - Host DRAM used by DLA for weights and metadata (bytes).

----

Engine-Invariant Settings
--------------------------

Changing any of the following settings invalidates cached engines — the engine must be
rebuilt from scratch:

``enabled_precisions``, ``max_aux_streams``, ``version_compatible``,
``optimization_level``, ``disable_tf32``, ``sparse_weights``,
``engine_capability``, ``hardware_compatible``, ``refit_identical_engine_weights``,
``immutable_weights``, ``enable_weight_streaming``, ``tiling_optimization_level``,
``l2_limit_for_tiling``, ``enable_autocast``, ``autocast_low_precision_type``,
``autocast_excluded_nodes``, ``autocast_excluded_ops``,
``autocast_max_output_threshold``, ``autocast_max_depth_of_reduction``,
``autocast_calibration_dataloader``, ``decompose_attention``, ``attn_bias_is_causal``.

Settings not in this list (e.g., ``debug``, ``dryrun``, ``pass_through_build_failures``)
can be changed without invalidating the cache.
