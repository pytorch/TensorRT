# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass, field
from typing import Any, Collection, FrozenSet, List, Optional, Set, Tuple, Union

import tensorrt as trt
import torch
from torch.fx.node import Target
from torch_tensorrt._Device import Device
from torch_tensorrt._enums import EngineCapability, dtype
from torch_tensorrt.dynamo._defaults import (
    ACCURACY_ALGORITHM,
    ACCURACY_ATOL,
    ACCURACY_RTOL,
    ACCURACY_THRESHOLD,
    ASSUME_DYNAMIC_SHAPE_SUPPORT,
    ATTN_BIAS_IS_CAUSAL,
    AUTOCAST_CALIBRATION_DATALOADER,
    AUTOCAST_EXCLUDED_NODE_NAMES,
    AUTOCAST_EXCLUDED_NODES,
    AUTOCAST_EXCLUDED_OPS,
    AUTOCAST_LOW_PRECISION_TYPE,
    AUTOCAST_MAX_DEPTH_OF_REDUCTION,
    AUTOCAST_MAX_OUTPUT_THRESHOLD,
    AUTOCAST_MODEL_REGIONS,
    BUILD_ROUTE,
    CACHE_BUILT_ENGINES,
    COLLAPSE_HIGH_RANK_PERMUTES,
    CONVERT_ANTIALIASED_RESIZE,
    CPU_MEMORY_BUDGET,
    DECOMPOSE_ATTENTION,
    DEDUPLICATE_WEIGHTS,
    DISABLE_TF32,
    DISABLED_CONSTANT_FOLD_EXCLUSIONS,
    DLA_GLOBAL_DRAM_SIZE,
    DLA_LOCAL_DRAM_SIZE,
    DLA_SRAM_SIZE,
    DRYRUN,
    DYNAMICALLY_ALLOCATE_RESOURCES,
    ENABLE_AUTOCAST,
    ENABLE_CROSS_COMPILE_FOR_WINDOWS,
    ENABLE_EXPERIMENTAL_DECOMPOSITIONS,
    ENABLE_RESOURCE_PARTITIONING,
    ENABLE_WEIGHT_STREAMING,
    ENGINE_CAPABILITY,
    FALLBACK_DATA_DEPENDENT_OPS,
    FALLBACK_PYTORCH_PLUGIN_OPS,
    FOLD_SHARED_KV_ATTENTION,
    GROUPED_MM_DENSE_MAX_MACS,
    HARDWARE_COMPATIBLE,
    IMMUTABLE_WEIGHTS,
    L2_LIMIT_FOR_TILING,
    LAZY_ENGINE_INIT,
    MAX_AUX_STREAMS,
    MAX_PARTITION_SIZE,
    MIN_BLOCK_SIZE,
    NUM_AVG_TIMING_ITERS,
    OFFLOAD_MODULE_TO_CPU,
    OPTIMIZATION_LEVEL,
    PASS_THROUGH_BUILD_FAILURES,
    REFIT_IDENTICAL_ENGINE_WEIGHTS,
    RELEASE_WEIGHTS_AFTER_BUILD,
    REPLACE_INTERLEAVE_STACK_WITH_CAT,
    REQUIRE_FULL_COMPILATION,
    REUSE_CACHED_ENGINES,
    SPARSE_WEIGHTS,
    STRIP_ENGINE_WEIGHTS,
    TARGET_COMPUTE_CAPABILITIES,
    TILING_OPTIMIZATION_LEVEL,
    TIMING_CACHE_PATH,
    TRUNCATE_DOUBLE,
    TUNE_BUILD_ROUTE_FILE,
    TUNE_BUILD_ROUTES,
    TUNING_CACHE_FILE,
    TUNING_CONTINUE,
    TUNING_DRY_RUN,
    TUNING_SEARCH,
    TUNING_TIMEOUT_S,
    USE_COMPLEX_DECOMPOSITION,
    USE_DISTRIBUTED_MODE_TRACE,
    USE_FAST_PARTITIONER,
    USE_FP32_ACC,
    VERSION_COMPATIBLE,
    WORKSPACE_SIZE,
    default_device,
)


def _normalize_disabled_constant_fold_exclusions(
    rule_ids: Collection[str],
) -> Set[str]:
    # Deferred import: torch_tensorrt.dynamo.lowering.passes imports this module,
    # so importing the rule registry at module scope would be circular. Reaching
    # for the core directly is safe from any point of that cycle because it
    # depends on nothing but torch.
    from torch_tensorrt.dynamo.lowering.constant_fold_exclusions._core import (
        validate_disabled_constant_fold_exclusions,
    )

    # Typed local: --strict mypy rejects returning the untyped helper directly.
    normalized: Set[str] = validate_disabled_constant_fold_exclusions(rule_ids)
    return normalized


def _autocast_op_names(ops: Optional[Collection[Target]]) -> Set[str]:
    # Autocast matches ops by name, and op objects cannot be pickled into the
    # engine metadata.
    from torch_tensorrt.dynamo.conversion._ConverterRegistry import (
        ConverterRegistry,
    )

    if ops is None:
        return set()
    if isinstance(ops, str):
        ops = (ops,)
    return {ConverterRegistry.qualified_name_or_str(op) for op in ops}


@dataclass
class CompilationSettings:
    """Compilation settings for Torch-TensorRT Dynamo Paths

    Args:
        workspace_size (int): Workspace TRT is allowed to use for the module (0 is default)
        min_block_size (int): Minimum number of operators per TRT-Engine Block
        torch_executed_ops (Collection[Target]): Collection of operations to run in Torch, regardless of converter coverage
        torch_executed_modules (Collection[str]): Collection of modules to run in Torch
        pass_through_build_failures (bool): Whether to fail on TRT engine build errors (True) or not (False)
        max_aux_streams (Optional[int]): Maximum number of allowed auxiliary TRT streams for each engine
        version_compatible (bool): Provide version forward-compatibility for engine plan files
        optimization_level (Optional[int]): Builder optimization 0-5, higher levels imply longer build time,
            searching for more optimization options. TRT defaults to 3
        truncate_double (bool): Whether to truncate float64 TRT engine inputs or weights to float32
        use_fast_partitioner (bool): Whether to use the fast or global graph partitioning system
        enable_experimental_decompositions (bool): Whether to enable all core aten decompositions
            or only a selected subset of them
        device (Device): GPU to compile the model on
        require_full_compilation (bool): Whether to require the graph is fully compiled in TensorRT.
            Only applicable for `ir="dynamo"`; has no effect for `torch.compile` path
        assume_dynamic_shape_support (bool): Setting this to true enables the converters work for both dynamic and static shapes. Default: False
        disable_tf32 (bool): Whether to disable TF32 computation for TRT layers
        sparse_weights (bool): Whether to allow the builder to use sparse weights
        engine_capability (trt.EngineCapability): Restrict kernel selection to safe gpu kernels or safe dla kernels
        num_avg_timing_iters (int): Number of averaging timing iterations used to select kernels
        dla_sram_size (int): Fast software managed RAM used by DLA to communicate within a layer.
        dla_local_dram_size (int): Host RAM used by DLA to share intermediate tensor data across operations
        dla_global_dram_size (int): Host RAM used by DLA to store weights and metadata for execution
        dryrun (Union[bool, str]): Toggle "Dryrun" mode, which runs everything through partitioning, short of conversion to
            TRT Engines. Prints detailed logs of the graph structure and nature of partitioning. Optionally saves the
            output to a file if a string path is specified
        hardware_compatible (bool): Build the TensorRT engines compatible with GPU architectures other than that of the GPU on which the engine was built (currently works for NVIDIA Ampere and newer)
        target_compute_capabilities (Optional[List[Tuple[int, int]]]): Compute capabilities to build for, e.g. ``[(7, 5)]`` for Turing. Defaults to ``None``. The compilation then targets the current device, from ``torch.cuda.get_device_capability()``. TensorRT-RTX only -- raises on standard TensorRT. Drives both engine targeting and op partitioning: a compiled artifact carries a single partitioning, so an op unsupported on any listed target falls back to PyTorch for all of them.
        timing_cache_path (str): Path to the timing cache if it exists (or) where it will be saved after compilation. Not used for TensorRT-RTX (no autotuning).
        cache_built_engines (bool): Whether to save the compiled TRT engines to storage
        reuse_cached_engines (bool): Whether to load the compiled TRT engines from storage
        use_fp32_acc (bool): Enable FP32 accumulation for FP16 matmul layers while retaining FP16
            inputs and outputs. When combined with ``decompose_attention=True``, the complete
            decomposed FP16 scaled dot product attention calculation runs in FP32 and only its
            final output is cast back to FP16. This option has no effect on FP32 or BF16 inputs.
        refit_identical_engine_weights (bool): Whether to refit the engine with identical weights
        strip_engine_weights (bool): Whether to strip the engine weights
        immutable_weights (bool): Build non-refittable engines. This is useful for some layers that are not refittable. If this argument is set to true, `strip_engine_weights` and `refit_identical_engine_weights` will be ignored
        enable_weight_streaming (bool): Enable weight streaming.
        enable_cross_compile_for_windows (bool): By default this is False means TensorRT engines can only be executed on the same platform where they were built.
            True will enable cross-platform compatibility which allows the engine to be built on Linux and run on Windows
        tiling_optimization_level (str): The optimization level of tiling strategies. A higher level allows TensorRT to spend more time searching for better tiling strategy. We currently support ["none", "fast", "moderate", "full"].
        l2_limit_for_tiling (int): The target L2 cache usage limit (in bytes) for tiling optimization (default is -1 which means no limit).
        use_distributed_mode_trace (bool):  Using aot_autograd to trace the graph. This is enabled when DTensors or distributed tensors are present in distributed model
        enable_autocast (bool): Whether to enable autocast.
        autocast_low_precision_type (Optional[Union[torch.dtype, dtype]]): The precision to reduce to. We currently support torch.float16 and torch.bfloat16. Default is None, which means no low precision is used.
        autocast_excluded_nodes (Collection[str]): The set of regex patterns for the paths of the modules, such as "block.norm", whose nodes should remain in FP32. Default is [].
        autocast_excluded_node_names (Collection[str]): The set of regex patterns for the names of the nodes in the exported graph, such as "layer_norm_1", that should remain in FP32. A pattern matches from the start of a name, so "layer_norm_1$" matches layer_norm_1 and not layer_norm_10. Default is [].
        autocast_excluded_ops (Collection[Target]): The set of targets (ATen ops) that should remain in FP32. Default is [].
        autocast_max_output_threshold (float): Maximum absolute value for node outputs, nodes with outputs greater than this value will remain in FP32. Needs autocast_calibration_dataloader. Default is 512.
        autocast_max_depth_of_reduction (Optional[int]): Maximum depth of reduction allowed in low precision. Nodes with higher reduction depths will remain in FP32. This helps prevent excessive accuracy loss in operations particularly sensitive to reduced precision, as higher-depth reductions may amplify computation errors in low precision formats. If not provided, infinity will be used. Needs autocast_calibration_dataloader. Default is None.
        autocast_calibration_dataloader (Optional[torch.utils.data.DataLoader]): The dataloader to use for autocast calibration. Default is None.
        autocast_model_regions (str): What autocast does with the regions the model itself runs under torch.autocast. "keep" builds them with the precisions that PyTorch autocast gives them. "reclassify" removes their autocast, so that the autocast rules, exclusions and calibration of Torch-TensorRT choose the precision of every op in them too. Regions with enabled=False, which a model uses to keep ops in FP32, keep their autocast. So do regions whose values inlining would change, such as one that writes into a buffer in place, and regions inside a torch.no_grad() block when the model is exported with grad mode on. autocast_excluded_node_names also matches the names that ops have in the regions. Default is "keep".
        max_partition_size (Optional[int]): The weight bytes per TensorRT engine the resource partitioner
            splits to: a finer control than ``cpu_memory_budget``, which it replaces, so the two can't both be set. The
            size is then not derived from the budget and the process's resident memory, so the same model and size always
            give the same split. Setting it turns resource partitioning on. Default is None.
        offload_module_to_cpu (bool): Offload the model to CPU to reduce memory footprint during compilation
        dynamically_allocate_resources (bool): Dynamically allocate resources for TensorRT engines
        release_weights_after_build (Optional[bool]): Release the memory of each model weight once
            the last TensorRT engine that reads it is built, and of weights no partition reads before
            the first build, so compilation holds only the weights still to be built.
            ``None`` (default) releases only weights mapped by ``map_weights``: their pages are dropped
            and re-read from the file if used again, so the source module keeps working, but an in-place
            write to a mapped weight is lost. ``True`` also frees ordinary weights: the model's tensors
            share that memory, so the source module (and an ``ExportedProgram`` exported from it) must
            not be used after compilation, even if compilation fails. ``False`` releases nothing.
            Weights read by subgraphs that run in PyTorch are kept. Applies to
            ``torch_tensorrt.dynamo.compile``; no effect under ``torch.compile`` (parameters arrive as
            graph inputs). ``MutableTorchTensorRTModule`` uses ``False``.
        decompose_attention (bool): Whether to decompose attention layers into smaller operations
            instead of using the attention converters. When combined with ``use_fp32_acc=True``,
            decomposed FP16 attention keeps its intermediate calculation in FP32 and casts only
            the final output back to FP16.
        disabled_constant_fold_exclusions (Collection[str]): IDs of predefined
            Torch-TensorRT rules to disable. Rules are registered by lowering
            implementations and enabled by default; compilation users do not
            need to register them. Disabling a rule makes the nodes that only
            it matches foldable. Supported IDs: ``"attention_mask_arange"``,
            ``"scatter_add_positions"``.
            Default is empty.
        attn_bias_is_causal (Union[bool, str]): Which attn_bias of efficient SDPA to replace with TensorRT's built-in causal mask, which is faster and more accurate. True: a bias proven causal, one that depends on no graph input and holds 0 on and below the diagonal and -inf (or the dtype's lowest value) above, and also a bias that cannot be proven, such as one built from an input, with a dynamic sequence length, or from a module buffer under torch.compile; you assert that it is causal. "proven": only a bias proven causal, for models whose input-built masks may not be causal, such as masks with padding. A bias that is computed and found not causal is always kept. A mask longer than 4096, or one whose check fails, is not proven: True replaces it and "proven" keeps it. False: never replace. A replaced bias is not read again, so refitting the weights it came from does not change the engine. Default is True.
        fold_shared_kv_attention (bool): For attention where all query heads share one key/value head (multi-query attention), compute it against that one head instead of the copies that Hugging Face models make for each query head. TensorRT then runs two plain matrix products instead of its fused attention kernel, which can be faster or slower depending on the GPU, so measure it. In float16 the result can round differently; use_fp32_acc=True reduces that. Only static shapes are rewritten, and scaled_dot_product_attention also needs decompose_attention=True. Both preserved matmuls and the reshape/bmm form produced by AOT tracing under torch.compile are supported for repeat_kv style graphs. In the bmm form, broadcasting a single KV head with enable_gqa=True is not folded. Default is False.
        deduplicate_weights (bool): Add a weight that several layers read to the TensorRT network once instead of once per layer, and give the linear layers that share a weight one transposed copy of it. This cuts engine-build host memory for models that reuse weights, such as a denoising loop unrolled into one graph, most of all in float32 builds with disable_tf32=True and in refittable engines. TensorRT may pick other kernels, so results can round differently, and some models need more memory instead, so measure it. Under torch.compile the weights are engine inputs, not constants, so they are not shared. Default is False.
        fallback_data_dependent_ops (bool): If True, operators whose converters require a TensorRT output allocator (i.e. data-dependent output shapes, such as nonzero) are added to torch_executed_ops and run in PyTorch instead of being lowered into a TensorRT engine. This is useful when targeting runtimes that cannot consume a TensorRT output allocator. Default is False.
        fallback_pytorch_plugin_ops (bool): If True, operators whose converters insert a TensorRT plugin that runs PyTorch (such as aten.index_put with accumulate=True, which uses Torch-TensorRT's ScatterAdd plugin, or a generated plugin without an AOT kernel) run in PyTorch instead of being lowered into a TensorRT engine. An accumulating aten.index_put with one 1-D index then stays in TensorRT through TensorRT's ScatterElements plugin, which needs no PyTorch. This needs float32, float16, bfloat16, int32 or int64 data. The input and the values it receives (the input's shape with the indexed dimension set to the index length) must stay under 2**31 elements, so every dynamic size needs an upper bound in the exported program, such as torch.export.Dim(max=...), that keeps them there. Otherwise the operator runs in PyTorch. Use it for runtimes without PyTorch, such as the ExecuTorch TensorRT runtime. Default is False.
        collapse_high_rank_permutes (bool): Rewrite a reshape, permute, reshape chain over more than 8 dimensions, TensorRT's limit, as reshapes and permutes of at most 8 dimensions that give the same values. Window and patch layouts, such as neighborhood attention and the Qwen2-VL image patches, build such chains. Left in PyTorch, each one splits the graph, and the partitioner can then hand many large tensors across one engine boundary. Only static shapes are rewritten. Fewer, larger engines can need much more time and memory to build and to run, so measure it. Default is False.
        convert_antialiased_resize (bool): Run antialiased bilinear and bicubic resize (F.interpolate with antialias=True) in TensorRT instead of PyTorch, so it does not split the graph. Only static input and output heights and widths and floating-point inputs are converted. Some sizes run slower than in PyTorch, so measure it. Default is False.
        grouped_mm_dense_max_macs (Optional[int]): Leave aten._grouped_mm, the grouped matrix multiply of mixture-of-experts layers, to PyTorch when its TensorRT lowering would do more than this many multiply-adds (groups * rows * K * N). That lowering multiplies every row by every group's weight, while PyTorch computes only each row's own group. None turns the limit off. The limit does not apply to an op with one group or under require_full_compilation=True. torch.compile turns require_full_compilation off, so there only None keeps the op in TensorRT. The lowering handles static shapes only, so a dynamic row count runs in PyTorch whatever this is. Use None where the op cannot run outside TensorRT: an ExecuTorch save without a partitioner that takes aten._grouped_mm (such as the CUDA partitioner) stops with a missing out variant, and where PyTorch has no fast kernel for it (float16 and float32, and GPUs other than sm_90 and sm_100) it copies the group offsets to the host, so a module that runs it cannot be captured whole by CUDA graphs. Default is 2**38.
        replace_interleave_stack_with_cat (bool): Write a stack whose only user is a reshape that merges the new axis into a neighbor axis, as in torch.stack([even, odd], dim=3).flatten(2, 3) for a 2x upsample, as a concatenation of unsqueezed inputs. This can make large, image-sized interleaves faster on GPUs where TensorRT fuses the region around them, but slower where TensorRT runs them as separate layers and adds copies, so measure it. Only programs from torch.export are rewritten, not torch.compile graphs. Default is False.
        build_route (Optional[str]): TensorRT Global Performance Tuning build route string (space-separated "-knob=value" tokens). None uses the default route. This feature requires TensorRT with Global Performance Tuning enabled; currently unavailable on TensorRT-RTX / Windows.
        tune_build_routes (Optional[str]): Build-route expression for an autotuning sweep that runs each candidate in a spawned child process (e.g. "-slice_fusion=[on|off] -kgen:codegen:cuda_tile=[0|1|2|3]"). None disables tuning.
        tune_build_route_file (Optional[str]): Path to a file with one route token per line (same as "tune_build_routes"). Mutually exclusive with "tune_build_routes".
        tuning_search (str): Search algorithm: "fast", "full", or "mixed". Default is "fast". "fast" runs a baseline with all knobs at default, then varies one knob at a time. "full" runs a full grid search. "mixed" runs "fast" first, and then "full" only on knobs that improved performance.
        tuning_timeout_s (int): Time budget for the entire tuning process. The current iteration finishes before the loop stops. Use -1 (default) to disable the timeout. Helpful for capping large "full" sweeps.
        tuning_cache_file (Optional[str]): JSONL path for tuning results / resume.
        tuning_continue (bool): Resume an interrupted sweep from "tuning_cache_file". When it is True, "tuning_cache_file" must be provided; "tune_build_routes", "tune_build_route_file", and "tuning_dry_run" must not be provided.
        tuning_dry_run (bool): Enumerate candidate routes without building engines.
        accuracy_threshold (Optional[float]): Max allowed accuracy loss between the engine outputs and eager Torch reference outputs; routes with accuracy loss above this threshold are excluded from best-engine selection. "None" skips accuracy checks.
        accuracy_algorithm (str): Loss metric: "l0", "l1", "l2", "lInf", or "cos" (lower is better).
        accuracy_atol (float): Absolute tolerance for "accuracy_algorithm="l0"".
        accuracy_rtol (float): Relative tolerance for "accuracy_algorithm="l0"".
    """

    workspace_size: int = WORKSPACE_SIZE
    min_block_size: int = MIN_BLOCK_SIZE
    torch_executed_ops: Collection[Target] = field(default_factory=set)
    torch_executed_modules: Collection[str] = field(default_factory=set)
    pass_through_build_failures: bool = PASS_THROUGH_BUILD_FAILURES
    max_aux_streams: Optional[int] = MAX_AUX_STREAMS
    version_compatible: bool = VERSION_COMPATIBLE
    optimization_level: Optional[int] = OPTIMIZATION_LEVEL
    truncate_double: bool = TRUNCATE_DOUBLE
    use_complex_decomposition: bool = USE_COMPLEX_DECOMPOSITION
    use_fast_partitioner: bool = USE_FAST_PARTITIONER
    enable_experimental_decompositions: bool = ENABLE_EXPERIMENTAL_DECOMPOSITIONS
    device: Device = field(default_factory=default_device)
    require_full_compilation: bool = REQUIRE_FULL_COMPILATION
    disable_tf32: bool = DISABLE_TF32
    assume_dynamic_shape_support: bool = ASSUME_DYNAMIC_SHAPE_SUPPORT
    sparse_weights: bool = SPARSE_WEIGHTS
    engine_capability: EngineCapability = field(
        default_factory=lambda: ENGINE_CAPABILITY
    )
    num_avg_timing_iters: int = NUM_AVG_TIMING_ITERS
    dla_sram_size: int = DLA_SRAM_SIZE
    dla_local_dram_size: int = DLA_LOCAL_DRAM_SIZE
    dla_global_dram_size: int = DLA_GLOBAL_DRAM_SIZE
    dryrun: Union[bool, str] = DRYRUN
    hardware_compatible: bool = HARDWARE_COMPATIBLE
    timing_cache_path: str = TIMING_CACHE_PATH
    lazy_engine_init: bool = LAZY_ENGINE_INIT
    cache_built_engines: bool = CACHE_BUILT_ENGINES
    reuse_cached_engines: bool = REUSE_CACHED_ENGINES
    use_fp32_acc: bool = USE_FP32_ACC
    refit_identical_engine_weights: bool = REFIT_IDENTICAL_ENGINE_WEIGHTS
    strip_engine_weights: bool = STRIP_ENGINE_WEIGHTS
    immutable_weights: bool = IMMUTABLE_WEIGHTS
    enable_weight_streaming: bool = ENABLE_WEIGHT_STREAMING
    enable_cross_compile_for_windows: bool = ENABLE_CROSS_COMPILE_FOR_WINDOWS
    tiling_optimization_level: str = TILING_OPTIMIZATION_LEVEL
    l2_limit_for_tiling: int = L2_LIMIT_FOR_TILING
    target_compute_capabilities: Optional[List[Tuple[int, int]]] = (
        TARGET_COMPUTE_CAPABILITIES
    )
    use_distributed_mode_trace: bool = USE_DISTRIBUTED_MODE_TRACE
    offload_module_to_cpu: bool = OFFLOAD_MODULE_TO_CPU
    enable_autocast: bool = ENABLE_AUTOCAST
    autocast_low_precision_type: Optional[dtype] = AUTOCAST_LOW_PRECISION_TYPE
    autocast_excluded_nodes: Collection[str] = field(
        default_factory=lambda: AUTOCAST_EXCLUDED_NODES
    )
    autocast_excluded_node_names: Collection[str] = field(
        default_factory=lambda: AUTOCAST_EXCLUDED_NODE_NAMES
    )
    autocast_excluded_ops: Collection[Target] = field(
        default_factory=lambda: AUTOCAST_EXCLUDED_OPS
    )
    autocast_max_output_threshold: float = AUTOCAST_MAX_OUTPUT_THRESHOLD
    autocast_max_depth_of_reduction: Optional[int] = AUTOCAST_MAX_DEPTH_OF_REDUCTION
    autocast_calibration_dataloader: Optional[torch.utils.data.DataLoader] = (
        AUTOCAST_CALIBRATION_DATALOADER
    )
    autocast_model_regions: str = AUTOCAST_MODEL_REGIONS
    # Set by autocast, not by users: the nodes it kept in high precision using the
    # calibration data. Saved in place of the data so refit can reuse the choice.
    _autocast_calibrated_high_precision_nodes: Optional[FrozenSet[str]] = field(
        default=None, repr=False
    )
    enable_resource_partitioning: bool = ENABLE_RESOURCE_PARTITIONING
    cpu_memory_budget: Optional[int] = CPU_MEMORY_BUDGET
    max_partition_size: Optional[int] = MAX_PARTITION_SIZE
    dynamically_allocate_resources: bool = DYNAMICALLY_ALLOCATE_RESOURCES
    release_weights_after_build: Optional[bool] = RELEASE_WEIGHTS_AFTER_BUILD
    decompose_attention: bool = DECOMPOSE_ATTENTION
    disabled_constant_fold_exclusions: Collection[str] = (
        DISABLED_CONSTANT_FOLD_EXCLUSIONS
    )
    attn_bias_is_causal: Union[bool, str] = ATTN_BIAS_IS_CAUSAL
    fold_shared_kv_attention: bool = FOLD_SHARED_KV_ATTENTION
    deduplicate_weights: bool = DEDUPLICATE_WEIGHTS
    fallback_data_dependent_ops: bool = FALLBACK_DATA_DEPENDENT_OPS
    fallback_pytorch_plugin_ops: bool = FALLBACK_PYTORCH_PLUGIN_OPS
    collapse_high_rank_permutes: bool = COLLAPSE_HIGH_RANK_PERMUTES
    convert_antialiased_resize: bool = CONVERT_ANTIALIASED_RESIZE
    grouped_mm_dense_max_macs: Optional[int] = GROUPED_MM_DENSE_MAX_MACS
    replace_interleave_stack_with_cat: bool = REPLACE_INTERLEAVE_STACK_WITH_CAT
    build_route: Optional[str] = BUILD_ROUTE
    tune_build_routes: Optional[str] = TUNE_BUILD_ROUTES
    tune_build_route_file: Optional[str] = TUNE_BUILD_ROUTE_FILE
    tuning_search: str = TUNING_SEARCH
    tuning_timeout_s: int = TUNING_TIMEOUT_S
    tuning_cache_file: Optional[str] = TUNING_CACHE_FILE
    tuning_continue: bool = TUNING_CONTINUE
    tuning_dry_run: bool = TUNING_DRY_RUN
    accuracy_threshold: Optional[float] = ACCURACY_THRESHOLD
    accuracy_algorithm: str = ACCURACY_ALGORITHM
    accuracy_atol: float = ACCURACY_ATOL
    accuracy_rtol: float = ACCURACY_RTOL

    def __post_init__(self) -> None:
        self.disabled_constant_fold_exclusions = (
            _normalize_disabled_constant_fold_exclusions(
                self.disabled_constant_fold_exclusions
            )
        )
        if self.grouped_mm_dense_max_macs is not None and (
            isinstance(self.grouped_mm_dense_max_macs, bool)
            or not isinstance(self.grouped_mm_dense_max_macs, int)
            or self.grouped_mm_dense_max_macs <= 0
        ):
            raise ValueError(
                "grouped_mm_dense_max_macs must be a positive integer or None, "
                f"got {self.grouped_mm_dense_max_macs!r}"
            )
        if self.max_partition_size is not None:
            if self.max_partition_size <= 0:
                raise ValueError(
                    "max_partition_size must be a positive number of bytes, "
                    f"got {self.max_partition_size}"
                )
            if self.cpu_memory_budget is not None:
                raise ValueError(
                    "Set max_partition_size or cpu_memory_budget, not both: the size replaces "
                    "the budget as the size the resource partitioner splits to."
                )
            self.enable_resource_partitioning = True
        if self.attn_bias_is_causal not in (True, False, "proven"):
            raise ValueError(
                'attn_bias_is_causal must be True, False or "proven", got '
                f"{self.attn_bias_is_causal!r}"
            )
        self.autocast_excluded_ops = _autocast_op_names(self.autocast_excluded_ops)
        # None (mapped weights only) is a third value, so a falsy 0 must not read as "not False".
        if self.release_weights_after_build is not None:
            self.release_weights_after_build = bool(self.release_weights_after_build)
        if isinstance(self.autocast_excluded_node_names, str):
            self.autocast_excluded_node_names = {self.autocast_excluded_node_names}
        if self.autocast_model_regions not in ("keep", "reclassify"):
            raise ValueError(
                'autocast_model_regions must be "keep" or "reclassify", got '
                f"{self.autocast_model_regions!r}"
            )
        if self.target_compute_capabilities:
            from torch_tensorrt._features import ENABLED_FEATURES

            if not ENABLED_FEATURES.tensorrt_rtx:
                raise ValueError(
                    "target_compute_capabilities is only supported on TensorRT-RTX. "
                    "Standard TensorRT always builds for the current device."
                )

    def __getstate__(self) -> dict[str, Any]:
        from torch_tensorrt.dynamo.conversion._ConverterRegistry import (
            ConverterRegistry,
        )

        state = self.__dict__.copy()
        # Pickles and copies (saved programs, the engine cache, each engine's own
        # settings) leave out the calibration data. Refit then reuses the recorded
        # precisions instead of deciding again from new weights.
        state["autocast_calibration_dataloader"] = None
        state.pop("autocast_calibration_trace", None)
        state["torch_executed_ops"] = {
            op if isinstance(op, str) else ConverterRegistry.qualified_name_or_str(op)
            for op in state["torch_executed_ops"]
        }
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        state.pop("use_python_runtime", None)
        state["disabled_constant_fold_exclusions"] = (
            _normalize_disabled_constant_fold_exclusions(
                state.get("disabled_constant_fold_exclusions", ())
            )
        )
        state["autocast_excluded_ops"] = _autocast_op_names(
            state.get("autocast_excluded_ops", ())
        )
        state.setdefault("fallback_data_dependent_ops", FALLBACK_DATA_DEPENDENT_OPS)
        state.setdefault("fallback_pytorch_plugin_ops", FALLBACK_PYTORCH_PLUGIN_OPS)
        state.setdefault("collapse_high_rank_permutes", COLLAPSE_HIGH_RANK_PERMUTES)
        state.setdefault("convert_antialiased_resize", CONVERT_ANTIALIASED_RESIZE)
        # Settings saved before this field always converted aten._grouped_mm, and refit
        # must rebuild the same engines.
        state.setdefault("grouped_mm_dense_max_macs", None)
        state.setdefault(
            "replace_interleave_stack_with_cat", REPLACE_INTERLEAVE_STACK_WITH_CAT
        )
        state.setdefault("fold_shared_kv_attention", FOLD_SHARED_KV_ATTENTION)
        state.setdefault("release_weights_after_build", RELEASE_WEIGHTS_AFTER_BUILD)
        state.setdefault("deduplicate_weights", DEDUPLICATE_WEIGHTS)
        state.setdefault("build_route", BUILD_ROUTE)
        state.setdefault("tune_build_routes", TUNE_BUILD_ROUTES)
        state.setdefault("tune_build_route_file", TUNE_BUILD_ROUTE_FILE)
        state.setdefault("tuning_search", TUNING_SEARCH)
        state.setdefault("tuning_timeout_s", TUNING_TIMEOUT_S)
        state.setdefault("tuning_cache_file", TUNING_CACHE_FILE)
        state.setdefault("tuning_continue", TUNING_CONTINUE)
        state.setdefault("tuning_dry_run", TUNING_DRY_RUN)
        state.setdefault("accuracy_threshold", ACCURACY_THRESHOLD)
        state.setdefault("accuracy_algorithm", ACCURACY_ALGORITHM)
        state.setdefault("accuracy_atol", ACCURACY_ATOL)
        state.setdefault("accuracy_rtol", ACCURACY_RTOL)
        state.setdefault("max_partition_size", MAX_PARTITION_SIZE)
        state.setdefault("autocast_excluded_node_names", AUTOCAST_EXCLUDED_NODE_NAMES)
        state.setdefault("autocast_model_regions", AUTOCAST_MODEL_REGIONS)
        state.setdefault("_autocast_calibrated_high_precision_nodes", None)
        self.__dict__.update(state)


# If any of the following setting is changed, the engine should be rebuilt.
_SETTINGS_TO_BE_ENGINE_INVARIANT = {
    "max_aux_streams",
    "version_compatible",
    "optimization_level",
    "disable_tf32",
    "sparse_weights",
    "engine_capability",
    "hardware_compatible",
    "target_compute_capabilities",
    "refit_identical_engine_weights",
    "immutable_weights",
    "enable_weight_streaming",
    "tiling_optimization_level",
    "l2_limit_for_tiling",
    "enable_autocast",
    "autocast_low_precision_type",
    "autocast_excluded_nodes",
    "autocast_excluded_node_names",
    "autocast_excluded_ops",
    "autocast_max_output_threshold",
    "autocast_max_depth_of_reduction",
    "autocast_model_regions",
    "decompose_attention",
    "disabled_constant_fold_exclusions",
    "attn_bias_is_causal",
    "fold_shared_kv_attention",
    "deduplicate_weights",
    "build_route",
    "fallback_pytorch_plugin_ops",
    "collapse_high_rank_permutes",
    "convert_antialiased_resize",
    "grouped_mm_dense_max_macs",
    "replace_interleave_stack_with_cat",
}


if not hasattr(trt.SerializationFlag, "INCLUDE_REFIT"):  # for TensorRT < 10.14
    _SETTINGS_TO_BE_ENGINE_INVARIANT.add("strip_engine_weights")


def settings_are_compatible(
    set_a: CompilationSettings, set_b: CompilationSettings
) -> Tuple[bool, Set[str]]:
    incompatible_settings: Set[str] = set()

    for f in _SETTINGS_TO_BE_ENGINE_INVARIANT:
        if getattr(set_a, f) != getattr(set_b, f):
            incompatible_settings.add(f)

    if len(incompatible_settings) == 0:
        return True, set()
    else:
        return False, incompatible_settings
