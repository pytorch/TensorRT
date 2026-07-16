import logging
import os
import warnings
from datetime import datetime
from typing import (
    Any,
    Callable,
    Dict,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Set,
    Tuple,
)

import numpy as np
import tensorrt as trt
import torch
import torch.fx
from torch.fx.experimental.proxy_tensor import unset_fake_temporarily
from torch.fx.node import _get_qualified_name
from torch.fx.passes.shape_prop import TensorMetadata
from torch.utils._python_dispatch import _disable_current_modes

from torch_tensorrt import ENABLED_FEATURES
from torch_tensorrt._enums import dtype
from torch_tensorrt._Input import Input
from torch_tensorrt._utils import is_tensorrt_version_supported
from torch_tensorrt.dynamo._engine_cache import BaseEngineCache
from torch_tensorrt.dynamo._settings import CompilationSettings
from torch_tensorrt.dynamo.conversion._ConversionContext import ConversionContext
from torch_tensorrt.dynamo.conversion._ConverterRegistry import (
    DYNAMO_CONVERTERS as CONVERTERS,
)
from torch_tensorrt.dynamo.conversion._ConverterRegistry import (
    CallingConvention,
)
from torch_tensorrt.dynamo.conversion._TRTBuilderMonitor import TRTBulderMonitor
from torch_tensorrt.dynamo.conversion.converter_utils import (
    get_node_io,
    get_node_name,
    get_trt_tensor,
    to_torch,
)
from torch_tensorrt.dynamo.debug._DebuggerConfig import DebuggerConfig
from torch_tensorrt.dynamo.debug._supports_debugger import cls_supports_debugger
from torch_tensorrt.dynamo.observer import Observer
from torch_tensorrt.dynamo.utils import (
    DYNAMIC_DIM,
    deallocate_module,
    get_cpu_memory_usage,
    validate_optimization_profiles,
)
from torch_tensorrt.logging import TRT_LOGGER

_LOGGER: logging.Logger = logging.getLogger(__name__)

TRT_INTERPRETER_CALL_PRE_OBSERVER: Observer[Callable[[torch.fx.GraphModule], None]] = (
    Observer("TRT_INTERPRETER_CALL_PRE_OBSERVER")
)


class UnsupportedOperatorException(RuntimeError):
    pass


class TRTInterpreterResult(NamedTuple):
    engine: trt.ICudaEngine
    input_names: List[str]
    output_names: List[str]
    requires_output_allocator: bool
    requires_native_multidevice: bool


@cls_supports_debugger
class TRTInterpreter(torch.fx.Interpreter):  # type: ignore[misc]
    def __init__(
        self,
        module: torch.fx.GraphModule,
        input_specs: Sequence[Input],
        output_dtypes: Optional[Sequence[dtype]] = None,
        compilation_settings: CompilationSettings = CompilationSettings(),
        engine_cache: Optional[BaseEngineCache] = None,
        *,
        _debugger_config: Optional[DebuggerConfig] = None,
    ):
        super().__init__(module)

        self.logger = TRT_LOGGER
        self.builder = trt.Builder(self.logger)
        self._debugger_config = _debugger_config
        flag = 0
        # rtx build has strongly typed enabled by default at the network level
        if not ENABLED_FEATURES.tensorrt_rtx:
            STRONGLY_TYPED = 1 << (int)(
                trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED
            )
            flag |= STRONGLY_TYPED

        self.ctx = ConversionContext(
            self.builder.create_network(flag), compilation_settings
        )

        self.compilation_settings = compilation_settings
        # Always update the global converter registry with the current compilation
        # settings.  The registry is a process-global singleton; without an
        # unconditional update, torch_executed_ops set by one compilation leak
        # into subsequent compilations in the same process (e.g. across tests in
        # an xdist worker), making those ops incorrectly appear as disallowed.
        CONVERTERS.set_compilation_settings(compilation_settings)
        self.validate_compile_settings()
        missing_ops = self.validate_conversion()
        if missing_ops:
            warnings.warn(
                "Interpretation will fail due to missing operations \n"
                + "\n".join(f"{i}" for i in missing_ops)
            )

        # Optimization profiles. Profiles are an ordered list on
        # ``Input.profiles``; profile index i is built from each input's
        # ``profiles[i]``. The count is derived from the input specs (submodule
        # inputs inherit the same number of profiles via propagation). It is
        # ``0`` when no input declares ``profiles``, in which case we fall back
        # to the historical behavior: a single profile for dynamic inputs and
        # none for fully static engines.
        self.optimization_profile_count: int = validate_optimization_profiles(
            input_specs
        )
        has_dynamic_input = any(
            input_spec.shape_mode == Input._ShapeMode.DYNAMIC
            for input_spec in input_specs
        )
        num_profiles = (
            self.optimization_profile_count
            if self.optimization_profile_count
            else (1 if has_dynamic_input else 0)
        )
        self.optimization_profiles: Optional[List[trt.IOptimizationProfile]] = (
            [self.builder.create_optimization_profile() for _ in range(num_profiles)]
            if num_profiles > 0
            else None
        )

        self.input_specs = input_specs
        self.input_specs_iter = 0
        self._cur_node_name: Optional[str] = None
        self._cur_node: Optional[torch.fx.Node] = None
        self._input_names: List[str] = []
        self._output_names: List[str] = []
        self._itensor_to_tensor_meta: Dict[trt.tensorrt.ITensor, TensorMetadata] = (
            dict()
        )

        # Data types for TRT Module output Tensors
        self.output_dtypes = (
            [dtype._from(o) for o in output_dtypes] if output_dtypes else None
        )

        # Mapping of constants to shapes and dtypes
        self.const_mapping: Dict[str, Tuple[Sequence[int], str]] = {}

        # Engine cache for storing and reusing TRT engines
        self.engine_cache = engine_cache

    def validate_conversion(self) -> Set[str]:
        missing_converters: Set[str] = set()

        for node in self.module.graph.nodes:
            if node.op == "call_function" and CONVERTERS.get(node) is None:
                missing_converters.add(f"{node.op} {_get_qualified_name(node.target)}")
            elif node.op == "call_method" and CONVERTERS.get(node) is None:
                missing_converters.add(f"{node.op} torch.Tensor.{node.target}")
            elif node.op == "call_module":
                submod = self.fetch_attr(node.target)
                submod_type = getattr(submod, "_base_class_origin", type(submod))
                if CONVERTERS.get(node) is None:
                    missing_converters.add(f"{node.op} {torch.typename(submod_type)}")

        return missing_converters

    @staticmethod
    def _args_str(args: List[Any]) -> str:
        def clean_repr(x: Any, depth: int = 0) -> Any:
            if isinstance(x, trt.ITensor):
                return f"{x.name} <tensorrt.ITensor [shape={x.shape}, dtype={x.dtype}]>"
            elif isinstance(x, torch.Tensor):
                return f"<torch.Tensor [shape={x.shape}, dtype={x.dtype}]>"
            elif isinstance(x, np.ndarray):
                return (
                    f"<torch.Tensor as np.ndarray [shape={x.shape}, dtype={x.dtype}]>"
                )
            elif isinstance(x, Sequence) and not isinstance(x, str):
                if depth < 3:
                    return type(x)([clean_repr(i, depth=depth + 1) for i in x])  # type: ignore[call-arg]
                else:
                    return "(...)"
            else:
                return f"{x} <{type(x).__name__}>"

        str_args = [clean_repr(a) for a in args]
        return repr(tuple(str_args))

    def validate_compile_settings(self) -> None:
        pass

    def _populate_trt_builder_config(
        self,
        strict_type_constraints: bool = False,
        tactic_sources: Optional[int] = None,
    ) -> trt.IBuilderConfig:
        builder_config = self.builder.create_builder_config()

        # Enable TRT's native multi-device runtime preview feature when the
        # Torch-TRT runtime was built with NCCL collectives support. Without
        # this, IBuilder::buildEngineWithConfig() rejects networks that contain
        # IDistCollectiveLayer with "PreviewFeature::kMULTIDEVICE_RUNTIME_10_16
        # is not enabled in the builder config".
        if ENABLED_FEATURES.native_trt_collectives and hasattr(
            trt.PreviewFeature, "MULTIDEVICE_RUNTIME_10_16"
        ):
            builder_config.set_preview_feature(
                trt.PreviewFeature.MULTIDEVICE_RUNTIME_10_16, True
            )

        if self._debugger_config and self._debugger_config.engine_builder_monitor:
            builder_config.progress_monitor = TRTBulderMonitor()

        if self.compilation_settings.workspace_size != 0:
            builder_config.set_memory_pool_limit(
                trt.MemoryPoolType.WORKSPACE, self.compilation_settings.workspace_size
            )

        if is_tensorrt_version_supported("8.2"):
            builder_config.profiling_verbosity = (
                trt.ProfilingVerbosity.DETAILED
                if self._debugger_config
                else trt.ProfilingVerbosity.LAYER_NAMES_ONLY
            )

        if is_tensorrt_version_supported("8.6"):
            if self.compilation_settings.max_aux_streams is not None:
                _LOGGER.info(
                    f"Setting max aux streams to {self.compilation_settings.max_aux_streams}"
                )
                builder_config.max_aux_streams = (
                    self.compilation_settings.max_aux_streams
                )
            if self.compilation_settings.version_compatible:
                _LOGGER.info("Using version compatible")
                builder_config.set_flag(trt.BuilderFlag.VERSION_COMPATIBLE)
                builder_config.set_flag(trt.BuilderFlag.EXCLUDE_LEAN_RUNTIME)
            if self.compilation_settings.hardware_compatible:
                _LOGGER.info("Using hardware compatible")
                builder_config.hardware_compatibility_level = (
                    trt.HardwareCompatibilityLevel.AMPERE_PLUS
                )
            if self.compilation_settings.optimization_level is not None:
                _LOGGER.info(
                    f"Using optimization level {self.compilation_settings.optimization_level}"
                )
                builder_config.builder_optimization_level = (
                    self.compilation_settings.optimization_level
                )

        builder_config.engine_capability = (
            self.compilation_settings.engine_capability.to(trt.EngineCapability)
        )
        builder_config.avg_timing_iterations = (
            self.compilation_settings.num_avg_timing_iters
        )

        if self.compilation_settings.device.device_type == trt.DeviceType.DLA:
            device_info = torch.cuda.get_device_properties(
                self.compilation_settings.device.gpu_id
            )
            assert (device_info.major == 8 and device_info.minor == 7) or (
                device_info.major == 7 and device_info.minor == 2
            ), "DLA is not available on non AGX systems"
            builder_config.DLA_core = self.compilation_settings.device.dla_core
            _LOGGER.info(f"Using DLA core {self.compilation_settings.device.dla_core}")
            builder_config.set_memory_pool_limit(
                trt.MemoryPoolType.DLA_MANAGED_SRAM,
                self.compilation_settings.dla_sram_size,
            )
            builder_config.set_memory_pool_limit(
                trt.MemoryPoolType.DLA_LOCAL_DRAM,
                self.compilation_settings.dla_local_dram_size,
            )
            builder_config.set_memory_pool_limit(
                trt.MemoryPoolType.DLA_GLOBAL_DRAM,
                self.compilation_settings.dla_global_dram_size,
            )

        if self.compilation_settings.sparse_weights:
            builder_config.set_flag(trt.BuilderFlag.SPARSE_WEIGHTS)

        if self.compilation_settings.disable_tf32:
            builder_config.clear_flag(trt.BuilderFlag.TF32)

        if self.compilation_settings.immutable_weights:
            # non-refittable engine
            if self.compilation_settings.strip_engine_weights:
                _LOGGER.warning("strip_engine_weights will be ignored.")
            if self.compilation_settings.refit_identical_engine_weights:
                _LOGGER.warning("refit_identical_engine_weights will be ignored.")
        else:
            # refittable engine
            if self.compilation_settings.refit_identical_engine_weights:
                builder_config.set_flag(trt.BuilderFlag.REFIT_IDENTICAL)
            else:
                builder_config.set_flag(trt.BuilderFlag.REFIT)

            if self.compilation_settings.strip_engine_weights:
                builder_config.set_flag(trt.BuilderFlag.STRIP_PLAN)

        if strict_type_constraints:
            builder_config.set_flag(trt.BuilderFlag.STRICT_TYPES)

        if self.optimization_profiles is not None:
            if len(self.optimization_profiles) > 0:
                for optimization_profile in self.optimization_profiles:
                    builder_config.add_optimization_profile(optimization_profile)

        if tactic_sources is not None:
            builder_config.set_tactic_sources(tactic_sources=tactic_sources)

        if self.compilation_settings.enable_cross_compile_for_windows:
            builder_config.runtime_platform = trt.RuntimePlatform.WINDOWS_AMD64
            _LOGGER.info(
                "Setting runtime_platform as trt.RuntimePlatform.WINDOWS_AMD64"
            )

        if self.compilation_settings.enable_weight_streaming:
            builder_config.set_flag(trt.BuilderFlag.WEIGHT_STREAMING)

        if self.ctx.requires_aliased_plugin_io:
            aliased_io_feature = getattr(
                trt.PreviewFeature, "ALIASED_PLUGIN_IO_10_03", None
            )
            if aliased_io_feature is None:
                raise RuntimeError(
                    "An in-place QDP plugin declared aliased I/O, but this TensorRT"
                    " version does not expose PreviewFeature.ALIASED_PLUGIN_IO_10_03."
                    " TensorRT 10.3+ is required for aliased plugin I/O."
                )
            builder_config.set_preview_feature(aliased_io_feature, True)
            _LOGGER.info("Enabling preview feature ALIASED_PLUGIN_IO_10_03")

        if is_tensorrt_version_supported("10.8"):
            TilingOptimizationLevel = {
                "none": trt.TilingOptimizationLevel.NONE,
                "fast": trt.TilingOptimizationLevel.FAST,
                "moderate": trt.TilingOptimizationLevel.MODERATE,
                "full": trt.TilingOptimizationLevel.FULL,
            }
            assert (
                self.compilation_settings.tiling_optimization_level
                in TilingOptimizationLevel
            ), f"Invalid tiling optimization level: {self.compilation_settings.tiling_optimization_level}. We currently support {TilingOptimizationLevel.keys()}."
            builder_config.tiling_optimization_level = TilingOptimizationLevel[
                self.compilation_settings.tiling_optimization_level
            ]

            if self.compilation_settings.l2_limit_for_tiling != -1:
                builder_config.l2_limit_for_tiling = (
                    self.compilation_settings.l2_limit_for_tiling
                )

        return builder_config

    def _create_timing_cache(
        self,
        builder_config: trt.IBuilderConfig,
        timing_cache_path: str = "",
    ) -> None:
        """
        Create a timing cache to enable faster build time for TRT engines.
        By default the timing_cache_path="/tmp/timing_cache.bin"
        Skipped for TensorRT-RTX since it does not use autotuning.
        """
        if ENABLED_FEATURES.tensorrt_rtx:
            _LOGGER.info(
                "Skipping timing cache creation for TensorRT-RTX (no autotuning)"
            )
            return

        buffer = b""
        if os.path.isfile(timing_cache_path):
            # Load from existing cache
            with open(timing_cache_path, mode="rb") as timing_cache_file:
                buffer = timing_cache_file.read()
        cache = builder_config.create_timing_cache(buffer)
        builder_config.set_timing_cache(cache, False)

    def _save_timing_cache(
        self,
        builder_config: trt.IBuilderConfig,
        timing_cache_path: str,
    ) -> None:
        """
        This is called after a TensorRT engine is built. Save the timing cache.
        Skipped for TensorRT-RTX since it does not use autotuning.
        """
        if ENABLED_FEATURES.tensorrt_rtx:
            return

        timing_cache = builder_config.get_timing_cache()
        os.makedirs(os.path.dirname(timing_cache_path), exist_ok=True)
        with open(timing_cache_path, "wb") as timing_cache_file:
            timing_cache_file.write(memoryview(timing_cache.serialize()))

    def _construct_trt_network_def(self) -> None:
        """
        Run the interpreter on each node to get TRT INetwork
        """
        TRT_INTERPRETER_CALL_PRE_OBSERVER.observe(self.module)

        self.input_specs_iter = 0
        run_module_start_time = datetime.now()
        super().run()
        _LOGGER.info(
            f"TRT INetwork construction elapsed time: {datetime.now() - run_module_start_time}"
        )

    def run(
        self,
        strict_type_constraints: bool = False,
        tactic_sources: Optional[int] = None,
    ) -> TRTInterpreterResult:
        """
        Build TensorRT engine with some configs.
        Args:
            strict_type_constraints: Usually we should set it to False unless we want to control the precision of certain layer for numeric reasons.
            tactic_sources: set up tactic sources for certain layer
        Return:
            TRTInterpreterResult
        """
        self._construct_trt_network_def()
        _LOGGER.debug(
            f"CPU memory usage after network construction: {get_cpu_memory_usage()} MB"
        )

        if self.compilation_settings.offload_module_to_cpu:
            deallocate_module(self.module)

        build_engine_start_time = datetime.now()
        _LOGGER.info("Not found cached TRT engines. Start building engine.")

        builder_config = self._populate_trt_builder_config(
            strict_type_constraints, tactic_sources
        )

        self._create_timing_cache(
            builder_config, self.compilation_settings.timing_cache_path
        )

        if (
            ENABLED_FEATURES.tensorrt_rtx
            or self.compilation_settings.version_compatible
        ):
            # TODO: When TRT-RTX matures, change it to build_engine_with_config
            serialized_engine = self.builder.build_serialized_network(
                self.ctx.net, builder_config
            )
            if serialized_engine is None:
                raise RuntimeError(
                    "TensorRT build_serialized_network returned None; engine build failed."
                )
            runtime = trt.Runtime(TRT_LOGGER)
            cuda_engine = runtime.deserialize_cuda_engine(serialized_engine)
        else:

            cuda_engine = self.builder.build_engine_with_config(
                self.ctx.net, builder_config
            )
        assert cuda_engine

        _LOGGER.debug(
            f"CPU memory usage after engine building: {get_cpu_memory_usage()} MB"
        )

        _LOGGER.info(
            f"Build TRT engine elapsed time: {datetime.now() - build_engine_start_time}"
        )
        self.ctx.clear_cpu_weights_reference_holder()

        self._save_timing_cache(
            builder_config, self.compilation_settings.timing_cache_path
        )

        return TRTInterpreterResult(
            cuda_engine,
            self._input_names,
            self._output_names,
            self.ctx.requires_output_allocator,
            self.ctx.requires_native_multidevice,
        )

    def run_node(self, n: torch.fx.Node) -> torch.fx.Node:
        self._cur_node_name = get_node_name(n)
        self._cur_node = n

        if _LOGGER.isEnabledFor(logging.DEBUG):
            _LOGGER.debug(
                f"Converting node {self._cur_node_name} (kind: {n.target}, args: {TRTInterpreter._args_str(n.args)})"
            )

        trt_node: torch.fx.Node = super().run_node(n)

        if n.op == "get_attr":
            self.const_mapping[str(n)] = (tuple(trt_node.shape), str(trt_node.dtype))

        _LOGGER.info(
            f"Converted node {self._cur_node_name} [{n.target}] ({get_node_io(n, self.const_mapping)})"
        )

        if isinstance(trt_node, trt.ITensor):
            self._itensor_to_tensor_meta[trt_node] = n.meta.get("tensor_meta")

        return trt_node

    def _per_profile_shapes(
        self, current_input: Input
    ) -> List[Tuple[Sequence[int], Sequence[int], Sequence[int]]]:
        """Return ``(min, opt, max)`` shapes for each optimization profile index.

        - If multiple profiles are active and the input declares ``profiles``,
          each entry uses the input's range at that profile index.
        - If multiple profiles are active but the input has none (static-like
          input reused across regimes), every entry repeats the union range.
        - If no profiles are active, there is a single entry (the union range) —
          the historical single-profile behavior.
        """
        assert isinstance(current_input.shape, dict)
        union = (
            current_input.shape["min_shape"],
            current_input.shape["opt_shape"],
            current_input.shape["max_shape"],
        )
        if not self.optimization_profile_count:
            return [union]

        result: List[Tuple[Sequence[int], Sequence[int], Sequence[int]]] = []
        for i in range(self.optimization_profile_count):
            if current_input.profiles and i < len(current_input.profiles):
                prof = current_input.profiles[i]
                result.append((prof["min_shape"], prof["opt_shape"], prof["max_shape"]))
            else:
                result.append(union)
        return result

    def placeholder(self, target: str, args: Any, kwargs: Any) -> trt.ITensor:
        self._input_names.append(target)
        current_input = self.input_specs[self.input_specs_iter]
        self.input_specs_iter += 1
        # Set optimization profile for dynamic input shape
        shape = None
        if current_input.shape_mode == Input._ShapeMode.DYNAMIC:
            assert isinstance(current_input.shape, dict)
            shape = []
            assert self.optimization_profiles is not None

            # Build the per-profile (min, opt, max) shapes for this input. Each
            # TRT optimization profile index gets the input's corresponding
            # profile range; inputs without profiles (e.g. static tensors reused
            # across regimes) repeat their single union range in every profile.
            per_profile_shapes = self._per_profile_shapes(current_input)

            for profile_idx, opt_profile in enumerate(self.optimization_profiles):
                min_shape, opt_shape, max_shape = per_profile_shapes[profile_idx]
                assert len(min_shape) == len(opt_shape) == len(max_shape)
                if current_input.is_shape_tensor:
                    # For shape_tensors, min/opt/max_shapes correspond to actual
                    # values of the shapes provided during runtime.
                    opt_profile.set_shape_input(target, min_shape, opt_shape, max_shape)
                else:
                    opt_profile.set_shape(target, min_shape, opt_shape, max_shape)

            # The INetwork input shape uses the union envelope to mark which
            # dims are dynamic (-1). A dim is static only if it is identical
            # across every profile's min/opt/max.
            union_min = current_input.shape["min_shape"]
            union_opt = current_input.shape["opt_shape"]
            union_max = current_input.shape["max_shape"]
            if current_input.is_shape_tensor:
                shape.append(len(union_opt))
            else:
                for i in range(len(union_min)):
                    dim_is_static = all(
                        per_profile_shapes[p][0][i]
                        == per_profile_shapes[p][1][i]
                        == per_profile_shapes[p][2][i]
                        == per_profile_shapes[0][0][i]
                        for p in range(len(per_profile_shapes))
                    )
                    if dim_is_static:
                        shape.append(union_min[i])
                    else:
                        # -1 to represent the dynamic dimension
                        shape.append(DYNAMIC_DIM)
        elif (
            not current_input.is_shape_tensor
            and current_input.shape_mode == Input._ShapeMode.STATIC
        ):
            assert isinstance(current_input.shape, tuple)
            shape = list(current_input.shape)
        else:
            raise RuntimeError(
                f"Unable to access shape spec for input: {target} (got: {current_input})"
            )

        trt_input_dtype = current_input.dtype.to(trt.DataType, use_default=True)
        _LOGGER.debug(
            f"Adding input to in-progress INetwork: {target} [shape={shape}, dtype={trt_input_dtype}]"
        )

        return self.ctx.net.add_input(
            name=target,
            shape=tuple(shape),
            dtype=trt_input_dtype,
        )

    def call_module(
        self, target: str, args: Any, kwargs: Any
    ) -> Any:  # Probably should be Tuple[trt.ITensor]? Case for Any?
        assert isinstance(target, str)
        submod = self.fetch_attr(target)
        submod_type = getattr(submod, "_base_class_origin", type(submod))
        converter_packet = CONVERTERS.get(self._cur_node)

        if converter_packet is None:
            raise UnsupportedOperatorException(
                f"Conversion of module of type {submod_type} not currently supported!"
            )

        converter, calling_convention, _ = converter_packet

        assert self._cur_node_name is not None

        if calling_convention is CallingConvention.LEGACY:
            return converter(self.ctx.net, submod, args, kwargs, self._cur_node_name)
        else:
            return converter(self.ctx, submod, args, kwargs, self._cur_node_name)

    def call_function(self, target: str, args: Any, kwargs: Any) -> Any:
        # TODO: Why is this stateful? We should be able to take in the inputs
        converter_packet = CONVERTERS.get(self._cur_node)
        if converter_packet is None:
            raise UnsupportedOperatorException(
                f"Conversion of function {torch.typename(target)} not currently supported!"
            )

        converter, calling_convention, converter_info = converter_packet
        if converter_info.get("requires_output_allocator", False):
            self.ctx.requires_output_allocator = True
            _LOGGER.debug(f"{target} requires output allocator")

        if converter_info.get("requires_native_multidevice", False):
            self.ctx.requires_native_multidevice = True
            _LOGGER.debug(f"{target} requires native multi-device support")

        self.ctx.current_node = self._cur_node
        if calling_convention is CallingConvention.LEGACY:
            return converter(self.ctx.net, target, args, kwargs, self._cur_node_name)
        else:
            return converter(self.ctx, target, args, kwargs, self._cur_node_name)

    def get_attr(self, target: str, args: Any, kwargs: Any) -> torch.Tensor:
        with _disable_current_modes(), unset_fake_temporarily():
            frozen_attr = self.fetch_attr(target)
            if isinstance(frozen_attr, torch.nn.Parameter):
                constant_tensor = frozen_attr.data
            else:
                constant_tensor = frozen_attr

            return to_torch(constant_tensor)

    def call_method(self, target: str, args: Any, kwargs: Any) -> Any:
        assert isinstance(target, str)
        converter_packet = CONVERTERS.get(self._cur_node)

        if converter_packet is None:
            raise UnsupportedOperatorException(
                f"Conversion of method {target} not currently supported!"
            )
        converter, calling_convention, _ = converter_packet

        if calling_convention is CallingConvention.LEGACY:
            return converter(self.ctx.net, target, args, kwargs, self._cur_node_name)
        else:
            return converter(self.ctx, target, args, kwargs, self._cur_node_name)

    def _cast_output_dtype(
        self,
        output: trt.ITensor,
        output_dtype: trt.DataType,
        output_name: str,
    ) -> trt.ITensor:
        if output.dtype == output_dtype:
            return output

        layer = self.ctx.net.add_cast(output, output_dtype)
        layer.name = f"Cast output {output_name} from {output.dtype} to {output_dtype}"
        return layer.get_output(0)

    def output(self, target: str, args: Any, kwargs: Any) -> List[Any]:
        assert len(args) == 1
        if isinstance(args[0], tuple):
            outputs = args[0]
        elif isinstance(args[0], list):
            outputs = tuple(args[0])
        else:
            outputs = (args[0],)

        for output_idx in range(len(outputs)):
            output = outputs[output_idx]

            if not isinstance(output, trt.ITensor):
                new_output = get_trt_tensor(self.ctx, output, target)
                outputs = (
                    outputs[:output_idx] + (new_output,) + outputs[output_idx + 1 :]
                )

        if not all(isinstance(output, trt.ITensor) for output in outputs):
            raise RuntimeError("TensorRT requires all outputs to be Tensor!")

        if self.output_dtypes is not None and len(self.output_dtypes) != len(outputs):
            raise RuntimeError(
                f"Specified output dtypes ({len(self.output_dtypes)}) differ from number of outputs ({len(outputs)})"
            )

        marked_outputs_ids = []
        for i, output in enumerate(outputs):
            # In some cases, the same output tensor may be marked multiple times, such as _to_copy,
            # so we skip marking if the output is already marked
            if id(output) in marked_outputs_ids:
                continue
            marked_outputs_ids.append(id(output))

            name = f"output{i}"

            if self.output_dtypes is not None:
                output_dtype = self.output_dtypes[i]
            elif any(
                op_name in output.name.split("_")
                for op_name in (
                    "eq",
                    "gt",
                    "lt",
                    "or",
                    "xor",
                    "and",
                    "not",
                    "ne",
                    "isinf",
                    "isnan",
                    "any",
                )
            ):
                output_dtype = dtype.b
            else:
                output_dtype = dtype.unknown

            if output_dtype is not dtype.unknown:
                output = self._cast_output_dtype(
                    output,
                    output_dtype.to(trt.DataType, use_default=True),
                    name,
                )

            output.name = name
            outputs = outputs[:i] + (output,) + outputs[i + 1 :]
            self.ctx.net.mark_output(output)

            self._output_names.append(name)
            _LOGGER.debug(
                f"Marking output {name} [shape={output.shape}, dtype={output.dtype}]"
            )

        return list(outputs)
