.. _troubleshooting:

Troubleshooting
===============

This guide lists the most common errors encountered when compiling or running
Torch-TensorRT models, with root-cause explanations and recommended fixes.

For systematic compilation debugging use the :ref:`debugger` context manager.
For pre-build coverage analysis use :ref:`dryrun`.

----

Compilation Errors
------------------

**"Lowering failed for node … No converter found"**

    A TensorRT converter does not exist for the ATen op in the graph. Options:

    * Check :ref:`supported_ops` to confirm the op is listed. If it is, the issue
      may be that the specific overload or dtype combination is not covered.
    * Add ``torch_executed_ops={<op>}`` to run that op in PyTorch instead of TRT.
    * If ``require_full_compilation=True``, remove it or add the op to
      ``torch_executed_ops``.
    * Lower ``min_block_size`` so the surrounding TRT-convertible ops form a block
      even without the failing op.

**"Block size … is smaller than min_block_size"**

    The TRT partition contains only a few convertible ops — fewer than
    ``min_block_size`` — and falls back to PyTorch. This is expected behavior.

    * Use ``dryrun=True`` to see which ops are unsupported and where the partitions
      form; see :ref:`dryrun`.
    * Lower ``min_block_size`` to allow smaller TRT blocks, or
    * Investigate whether the model can be restructured to group more ops together.

**"assert_size_stride ... Expected stride ... but got ..."**

    An input tensor's stride does not match what was used during tracing. Most often
    caused by non-contiguous tensors (e.g. after a ``permute`` with no explicit
    ``contiguous()`` call).

    * Add ``.contiguous()`` before passing the tensor to the model.
    * If you use complex inputs, see :ref:`complex tensor handling note <complex_inputs>`.

**Compilation hangs or takes very long (>30 minutes)**

    TRT engine compilation is CPU/GPU intensive and can legitimately take a long time for
    large models — especially with ``optimization_level=3`` (default) or higher.

    * Check that compilation is actually progressing by enabling debug logging:

      .. code-block:: python

          torch_tensorrt.compile(model, ..., debug=True)

      You should see per-layer TRT build messages. If the last message is stuck at the
      same layer for more than 10 minutes, compilation may be hung.

    * Reduce ``optimization_level`` to 0 or 1 during development. Use higher levels only
      for final production builds.

    * Large models (>1B parameters) may need ``offload_module_to_cpu=True`` to avoid
      OOM during compilation.

    * If using ``torch.compile(backend="torch_tensorrt")`` (JIT), compilation is
      triggered on the first forward pass — the call will block until done. For large
      models, prefer the AOT path (``ir="dynamo"``) to make the compilation step
      explicit.

    * Check whether the model contains ops that trigger very expensive TRT searches
      (e.g. very large convolutions). Use :ref:`dryrun` to identify the problematic ops
      and consider using ``torch_executed_ops`` to push them to PyTorch.

**Export fails with "Cannot export ..." / data-dependent control flow error**

    The model uses Python-level branching on tensor values which
    ``torch.export.export`` (strict mode) cannot trace.

    * Use ``torch_tensorrt.dynamo.trace(model, inputs, strict=False)`` to enable
      non-strict tracing (allows data-dependent control flow).
    * Alternatively, rewrite the dynamic branch as a TRT-compatible conditional.

**"ModuleNotFoundError: No module named 'modelopt'"**

    The model has INT8/FP8 quantization nodes but ModelOpt is not installed.

    .. code-block:: bash

        pip install nvidia-modelopt

    See :ref:`quantization` for the full quantization workflow.

----

Memory Errors During Compilation
---------------------------------

**CUDA out-of-memory during engine build**

    * Set ``offload_module_to_cpu=True`` in compilation settings to free the
      original model from GPU during compilation.
    * Reduce ``workspace_size`` (default is large to allow TRT to find optimal
      kernels).
    * Use ``lazy_engine_init=True`` to defer engine initialization until all
      subgraph compilations are complete.
    * See :ref:`resource_management` for a systematic memory reduction strategy.

**Process killed (OOM) on CPU**

    TRT compilation can use up to 5× the model size in CPU memory.

    * Set ``TORCHTRT_ENABLE_BUILDER_MALLOC_TRIM=1`` to reduce to ~3× model size.
    * Disable ``offload_module_to_cpu`` (``False``) to drop another 1× copy.

----

Runtime Errors
--------------

**"Engine failed to deserialize" / engine load fails**

    * The TRT version on the loading machine is older than the one used to build
      the engine. Upgrade TRT or rebuild with ``version_compatible=True``.
    * The GPU compute capability is lower than on the build machine. Rebuild with
      ``hardware_compatible=True`` (requires Ampere or newer).
    * The ``.ep`` file was generated with ``use_python_runtime=True`` which is not
      serializable. Rebuild with the default C++ runtime.

**Shape mismatch at runtime / "Invalid input shape"**

    * The model was compiled for static shapes but receives a different shape at
      runtime. Recompile using :class:`~torch_tensorrt.Input` with
      ``min/opt/max`` shapes to enable dynamic shapes.
    * If using ``MutableTorchTensorRTModule``, call
      ``set_expected_dynamic_shape_range`` before the first forward pass.

**Wrong numerical results (large error vs PyTorch)**

    * Enable higher precision: cast model and inputs to ``float32``, or remove
      ``model.half()`` / ``model.bfloat16()`` calls. Strong typing is always enabled —
      set dtypes directly in the model and inputs.
    * TF32 is enabled by default on Ampere and newer GPUs. Disable it with
      ``disable_tf32=True`` for bit-exact FP32 comparison.
    * If using cross-compiled Windows engines, floating-point results may differ
      slightly due to driver differences. Use ``optimization_level=0`` to minimize
      kernel specialization.

**"DynamicOutputAllocator required" / nonzero / unique ops fail**

    The model contains data-dependent-shape ops (``nonzero``, ``unique``,
    ``masked_select``, etc.) which require TRT's output allocator.

    * Use ``PythonTorchTensorRTModule`` (``use_python_runtime=True``) — it
      activates the dynamic output allocator automatically via
      ``requires_output_allocator=True``.
    * See :ref:`cuda_graphs` for ``DynamicOutputAllocator`` details.

----

Accuracy / Performance Issues
-------------------------------

**Model is slower than expected after compilation**

    * **Warm up** the model with at least 5 forward passes before measuring — TRT
      engines load kernels lazily on first use.
    * Use CUDA events for timing, not ``time.time()`` (wall-clock includes Python and
      CPU/GPU sync overhead).
    * Run with ``dryrun=True`` to check what fraction of ops are running in TRT
      vs PyTorch (high PyTorch fallback = low coverage = slow).
    * Increase ``optimization_level`` (0–5, default 3) to allow TRT more time to
      find faster kernels.
    * Use FP16 for throughput-critical workloads: ``model.half()`` with FP16 inputs,
      or ``enable_autocast=True`` + ``autocast_low_precision_type=torch.float16``.
    * Try ``use_fast_partitioner=False`` for global partitioning — it is slower to
      compile but may produce better-performing partitions.
    * See :ref:`performance_tuning` for a complete benchmarking guide.

**High latency with variable input sizes**

    * Set ``min/opt/max`` shapes on :class:`~torch_tensorrt.Input` — TRT
      optimizes for the ``opt`` shape. Make ``opt`` match the most frequent
      production input.
    * Use CUDAGraphs (``enable_cudagraphs=True``) for fixed-shape low-latency
      inference; see :ref:`cuda_graphs`.

**"How do I do weight-only quantization (A16W8 / W8A16)?"**

    Weight-only quantization compresses model weights to INT8 while keeping activations
    in FP16/BF16. This is common for large language models where memory bandwidth is the
    bottleneck. To do this with ModelOpt, create a custom quantization config that
    enables ``weight_quantizer`` but disables ``input_quantizer``:

    .. code-block:: python

        import modelopt.torch.quantization as mtq

        # Custom weight-only INT8 config
        quant_cfg = {
            "quant_cfg": {
                "*weight_quantizer": {"num_bits": 8, "axis": 0},
                "*input_quantizer": {"enable": False},  # activations stay FP16
                "default": {"enable": False},
            },
            "algorithm": "max",
        }

        mtq.quantize(model, quant_cfg, forward_loop=calibration_loop)

        trt_model = torch_tensorrt.compile(
            model, ir="dynamo", arg_inputs=inputs,
        )

    See `NVIDIA ModelOpt documentation <https://nvidia.github.io/TensorRT-Model-Optimizer/>`_
    for the full list of built-in quantization configs and customization options.

----

Distributed / Tensor-Parallel Issues
--------------------------------------

**"DTensor inputs detected but use_distributed_mode_trace=False"**

    The model uses ``DTensor`` parameters (from ``parallelize_module``) but the
    default export path does not handle them.

    * Set ``use_distributed_mode_trace=True`` in compilation options.
    * See :ref:`distributed_inference` for the full tensor-parallel workflow.

----

Getting More Information
-------------------------

1. **Enable debug logging** — wrap the compilation call in the
   :class:`~torch_tensorrt.dynamo.Debugger` context with ``log_level="debug"``
   and inspect ``<logging_dir>/torch_tensorrt_logging.log``.

2. **Capture FX graphs** — use ``capture_fx_graph_before`` /
   ``capture_fx_graph_after`` in the :class:`~torch_tensorrt.dynamo.Debugger` to
   see what the graph looks like at each lowering-pass boundary.

3. **Dryrun** — compile with ``dryrun=True`` to see the partition layout and
   coverage percentage without actually building TRT engines.

4. **Layer info** — compile with ``save_layer_info=True`` in the Debugger and
   inspect ``engine_layer_info.json`` to see what TRT kernels were selected.

5. **File a bug** — include the output of the above and the model's
   ``exported_program.print_readable()`` when reporting issues at
   https://github.com/pytorch/TensorRT/issues.
