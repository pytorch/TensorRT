import logging
import warnings
from datetime import datetime
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Sequence, Set, Tuple

import numpy as np
import tensorrt as trt
import torch
import torch.fx
from torch.fx.node import _get_qualified_name
from torch.fx.passes.shape_prop import TensorMetadata
from torch.utils._python_dispatch import _disable_current_modes
from torch_tensorrt._Input import Input
from torch_tensorrt.dynamo._settings import CompilationSettings
from torch_tensorrt.dynamo.conversion._ConversionContext import ConversionContext
from torch_tensorrt.dynamo.conversion._ConverterRegistry import (
    DYNAMO_CONVERTERS as CONVERTERS,
)
from torch_tensorrt.dynamo.conversion._ConverterRegistry import CallingConvention
from torch_tensorrt.dynamo.conversion.converter_utils import (
    get_node_io,
    get_node_name,
    get_trt_tensor,
)
from torch_tensorrt.fx.observer import Observer
from torch_tensorrt.fx.utils import Frameworks, unified_dtype_converter

from packaging import version

_LOGGER: logging.Logger = logging.getLogger(__name__)

TRT_INTERPRETER_CALL_PRE_OBSERVER: Observer[Callable[[torch.fx.GraphModule], None]] = (
    Observer("TRT_INTERPRETER_CALL_PRE_OBSERVER")
)


class UnsupportedOperatorException(RuntimeError):
    pass


class TRTInterpreterResult(NamedTuple):
    engine: Any
    input_names: Sequence[str]
    output_names: Sequence[str]
    serialized_cache: bytearray


class TRTInterpreter(torch.fx.Interpreter):  # type: ignore[misc]
    def __init__(
        self,
        module: torch.fx.GraphModule,
        input_specs: Sequence[Input],
        logger_level: trt.ILogger.Severity = trt.ILogger.Severity.WARNING,
        output_dtypes: Optional[Sequence[torch.dtype]] = None,
        compilation_settings: CompilationSettings = CompilationSettings(),
    ):
        super().__init__(module)

        # TODO: @narendasan replace with Torch-TensorRT Logger
        self.logger = trt.Logger(logger_level)
        self.builder = trt.Builder(self.logger)

        flag = 0

        # It is deprecated to not use this flag
        EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        flag |= EXPLICIT_BATCH

        self.ctx = ConversionContext(
            self.builder.create_network(flag), compilation_settings
        )

        missing_ops = self.validate_conversion()
        if missing_ops:
            # TODO: @narendasan make sure to set logging.captureWarnings(True)
            warnings.warn(
                "Interpretation will fail due to missing operations \n"
                + "\n".join(f"{i}" for i in missing_ops)
            )

        self.optimization_profiles: Optional[List[trt.IOptimizationProfile]] = (
            [self.builder.create_optimization_profile()]
            if any(
                input_spec.shape_mode == Input._ShapeMode.DYNAMIC
                for input_spec in input_specs
            )
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
        self.compilation_settings = compilation_settings

        # Data types for TRT Module output Tensors
        self.output_dtypes = output_dtypes

        # Mapping of constants to shapes and dtypes
        self.const_mapping: Dict[str, Tuple[Sequence[int], str]] = {}

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

    def run(
        self,
        force_fp32_output: bool = False,
        strict_type_constraints: bool = False,
        algorithm_selector: Optional[trt.IAlgorithmSelector] = None,
        timing_cache: Optional[trt.ITimingCache] = None,
        tactic_sources: Optional[int] = None,
    ) -> TRTInterpreterResult:
        """
        Build TensorRT engine with some configs.
        Args:
            force_fp32_output: force output to be fp32
            strict_type_constraints: Usually we should set it to False unless we want to control the precision of certain layer for numeric reasons.
            algorithm_selector: set up algorithm selection for certain layer
            timing_cache: enable timing cache for TensorRT
        Return:
            TRTInterpreterResult
        """
        TRT_INTERPRETER_CALL_PRE_OBSERVER.observe(self.module)

        precision = self.compilation_settings.precision
        # For float outputs, we set their dtype to fp16 only if precision == torch.float16 and
        # force_fp32_output=False. Overriden by specifying output_dtypes
        self.output_fp16 = not force_fp32_output and precision == torch.float16

        if precision == torch.int8 and not self.builder.platform_has_fast_int8:
            raise RuntimeError("Current platform doesn't support fast native int8!")

        if precision == torch.float16 and not self.builder.platform_has_fast_fp16:
            warnings.warn("Current platform doesn't support fast native fp16!")

        self.input_specs_iter = 0
        run_module_start_time = datetime.now()
        super().run()
        _LOGGER.info(
            f"TRT INetwork construction elapsed time: {datetime.now() - run_module_start_time}"
        )
        build_engine_start_time = datetime.now()

        builder_config = self.builder.create_builder_config()

        if self.compilation_settings.workspace_size != 0:
            builder_config.set_memory_pool_limit(
                trt.MemoryPoolType.WORKSPACE, self.compilation_settings.workspace_size
            )

        cache = None
        if timing_cache:
            cache_file = np.array(timing_cache)
            cache = builder_config.create_timing_cache(cache_file.tobytes())
        else:
            cache = builder_config.create_timing_cache(b"")
        builder_config.set_timing_cache(cache, False)

        if version.parse(trt.__version__) >= version.parse("8.2"):
            builder_config.profiling_verbosity = (
                trt.ProfilingVerbosity.VERBOSE
                if self.compilation_settings.debug
                else trt.ProfilingVerbosity.LAYER_NAMES_ONLY
            )

        if version.parse(trt.__version__) >= version.parse("8.6"):
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

        builder_config.engine_capability = self.compilation_settings.engine_capability
        builder_config.avg_timing_iterations = (
            self.compilation_settings.num_avg_timing_iters
        )

        if self.compilation_settings.device.device_type == trt.DeviceType.DLA:
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

        if precision == torch.float16:
            builder_config.set_flag(trt.BuilderFlag.FP16)

        if precision == torch.int8:
            builder_config.set_flag(trt.BuilderFlag.INT8)

        if self.compilation_settings.sparse_weights:
            builder_config.set_flag(trt.BuilderFlag.SPARSE_WEIGHTS)

        if self.compilation_settings.disable_tf32:
            builder_config.clear_flag(trt.BuilderFlag.TF32)

        if self.compilation_settings.refit:
            builder_config.set_flag(trt.BuilderFlag.REFIT)

        if strict_type_constraints:
            builder_config.set_flag(trt.BuilderFlag.STRICT_TYPES)

        if self.optimization_profiles is not None:
            if len(self.optimization_profiles) > 0:
                for optimization_profile in self.optimization_profiles:
                    builder_config.add_optimization_profile(optimization_profile)

        if algorithm_selector:
            builder_config.set_flag(trt.BuilderFlag.DISABLE_TIMING_CACHE)
            builder_config.algorithm_selector = algorithm_selector

        if tactic_sources is not None:
            builder_config.set_tactic_sources(tactic_sources=tactic_sources)

        engine = self.builder.build_engine(self.ctx.net, builder_config)
        assert engine

        serialized_cache = (
            bytearray(cache.serialize())
            if builder_config.get_timing_cache()
            else bytearray()
        )
        _LOGGER.info(
            f"Build TRT engine elapsed time: {datetime.now() - build_engine_start_time}"
        )
        _LOGGER.info(f"TRT Engine uses: {engine.device_memory_size} bytes of Memory")

        return TRTInterpreterResult(
            engine, self._input_names, self._output_names, serialized_cache
        )

    def run_node(self, n: torch.fx.Node) -> torch.fx.Node:
        self._cur_node_name = get_node_name(n)
        self._cur_node = n
        # add "_itensor_to_tensor_meta"
        kwargs = dict(n.kwargs)
        kwargs["_itensor_to_tensor_meta"] = self._itensor_to_tensor_meta
        n.kwargs = kwargs

        # run the node
        _LOGGER.debug(
            f"Running node {self._cur_node_name}, a {self._cur_node.op} node "
            f"with target {self._cur_node.target} in the TensorRT Interpreter"
        )
        trt_node: torch.fx.Node = super().run_node(n)

        if n.op == "get_attr":
            self.const_mapping[str(n)] = (tuple(trt_node.shape), str(trt_node.dtype))

        _LOGGER.debug(
            f"Ran node {self._cur_node_name} with properties: {get_node_io(n, self.const_mapping)}"
        )

        # remove "_itensor_to_tensor_meta"
        kwargs = dict(n.kwargs)
        del kwargs["_itensor_to_tensor_meta"]
        n.kwargs = kwargs

        if isinstance(trt_node, trt.tensorrt.ITensor):
            self._itensor_to_tensor_meta[trt_node] = n.meta.get("tensor_meta")

        return trt_node

    def placeholder(self, target: str, args: Any, kwargs: Any) -> trt.ITensor:
        self._input_names.append(target)
        current_input = self.input_specs[self.input_specs_iter]
        self.input_specs_iter += 1
        # Set optimization profile for dynamic input shape
        shape = None
        if current_input.shape_mode == Input._ShapeMode.DYNAMIC:
            assert isinstance(current_input.shape, dict)
            shape = []
            min_shape = current_input.shape["min_shape"]
            opt_shape = current_input.shape["opt_shape"]
            max_shape = current_input.shape["max_shape"]
            # TODO: Does not support disjoint optimization profiles?
            assert self.optimization_profiles is not None
            self.optimization_profiles[0].set_shape(
                target, min_shape, opt_shape, max_shape
            )

            assert len(min_shape) == len(opt_shape) == len(max_shape)
            for i in range(len(min_shape)):
                if min_shape[i] == opt_shape[i] == max_shape[i]:
                    shape.append(min_shape[i])
                else:
                    # -1 to represent the dynamic dimension
                    shape.append(-1)
        elif current_input.shape_mode == Input._ShapeMode.STATIC:
            assert isinstance(current_input.shape, tuple)
            shape = list(current_input.shape)
        else:
            raise RuntimeError(
                f"Unable to access shape spec for input: {target} (got: {current_input})"
            )

        return self.ctx.net.add_input(
            name=target,
            shape=tuple(shape),
            dtype=unified_dtype_converter(current_input.torch_dtype, Frameworks.TRT),
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

        converter, calling_convention = converter_packet

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

        converter, calling_convention = converter_packet

        assert self._cur_node_name is not None
        if calling_convention is CallingConvention.LEGACY:
            return converter(self.ctx.net, target, args, kwargs, self._cur_node_name)
        else:
            return converter(self.ctx, target, args, kwargs, self._cur_node_name)

    def get_attr(self, target: str, args: Any, kwargs: Any) -> np.ndarray:
        with _disable_current_modes():
            from torch_tensorrt.dynamo.conversion.converter_utils import to_numpy

            frozen_attr = self.fetch_attr(target)

            if isinstance(frozen_attr, torch.nn.Parameter):
                constant_tensor = frozen_attr.data
            else:
                constant_tensor = frozen_attr

            network_constant = to_numpy(constant_tensor)

        return network_constant

    def call_method(self, target: str, args: Any, kwargs: Any) -> Any:
        assert isinstance(target, str)
        converter_packet = CONVERTERS.get(self._cur_node)

        if converter_packet is None:
            raise UnsupportedOperatorException(
                f"Conversion of method {target} not currently supported!"
            )
        converter, calling_convention = converter_packet

        assert self._cur_node_name is not None
        if calling_convention is CallingConvention.LEGACY:
            return converter(self.ctx.net, target, args, kwargs, self._cur_node_name)
        else:
            return converter(self.ctx, target, args, kwargs, self._cur_node_name)

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

            if not isinstance(output, trt.tensorrt.ITensor):
                new_output = get_trt_tensor(self.ctx, output, target)
                outputs = (
                    outputs[:output_idx] + (new_output,) + outputs[output_idx + 1 :]
                )

        if not all(isinstance(output, trt.tensorrt.ITensor) for output in outputs):
            raise RuntimeError("TensorRT requires all outputs to be Tensor!")

        if self.output_dtypes is not None and len(self.output_dtypes) != len(outputs):
            raise RuntimeError(
                f"Specified output dtypes ({len(self.output_dtypes)}) differ from number of outputs ({len(outputs)})"
            )

        for i, output in enumerate(outputs):
            if any(
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
                    "any",
                )
            ):
                output_bool = True
            else:
                output_bool = False
            name = f"output{i}"
            output.name = name
            self.ctx.net.mark_output(output)
            if output_bool:
                output.dtype = trt.bool
            elif self.output_dtypes is not None:
                output.dtype = unified_dtype_converter(
                    self.output_dtypes[i], Frameworks.TRT
                )
            elif self.output_fp16 and output.dtype == trt.float32:
                output.dtype = trt.float16
            self._output_names.append(name)

        return list(outputs)
