import logging
from copy import deepcopy
from enum import Enum, auto
from typing import Any, Collection, Dict, Iterator, List, Optional, Set, Union

import numpy as np
import torch
from torch.fx.node import Target
from torch_tensorrt._Device import Device
from torch_tensorrt._enums import EngineCapability, dtype
from torch_tensorrt.dynamo import _defaults
from torch_tensorrt.dynamo._compiler import compile as dynamo_compile
from torch_tensorrt.dynamo._refit import refit_module_weights
from torch_tensorrt.dynamo._settings import CompilationSettings
from torch_tensorrt.dynamo.utils import (
    check_output_equal,
    to_torch_device,
    to_torch_tensorrt_device,
)

logger = logging.getLogger(__name__)


class RefitFlag(Enum):
    UNKNOWN = auto()
    NEEDS_REFIT = auto()
    NEEDS_RECOMPILE = auto()
    LIVE = auto()


class RefitState:
    _state: RefitFlag = RefitFlag.NEEDS_RECOMPILE

    def set_state(self, state: RefitFlag) -> None:
        if isinstance(state, RefitFlag):
            self._state = state
        else:
            raise ValueError(f"Invalid state: {state}")

    def get_state(self) -> RefitFlag:
        return self._state


class MutableTorchTensorRTModule(object):
    """
    Initialize a MutableTorchTensorRTModule to seamlessly manipulate it like a regular PyTorch module.
    All TensorRT compilation and refitting processes are handled automatically as you work with the module.
    Any changes to its attributes or loading a different state_dict will trigger refitting or recompilation,
    which will be managed during the next forward pass.

    The MutableTorchTensorRTModule takes a PyTorch module and a set of configuration settings for the compiler.
    Once compilation is complete, the module maintains the connection between the TensorRT graph module and the original PyTorch module.
    Any modifications made to the MutableTorchTensorRTModule will be reflected in both the TensorRT graph module and the original PyTorch module.
    """

    def __init__(
        self,
        pytorch_model: torch.nn.Module,
        *,
        device: Optional[Union[Device, torch.device, str]] = _defaults.DEVICE,
        disable_tf32: bool = _defaults.DISABLE_TF32,
        assume_dynamic_shape_support: bool = _defaults.ASSUME_DYNAMIC_SHAPE_SUPPORT,
        sparse_weights: bool = _defaults.SPARSE_WEIGHTS,
        enabled_precisions: Set[
            Union[torch.dtype, dtype]
        ] = _defaults.ENABLED_PRECISIONS,
        engine_capability: EngineCapability = _defaults.ENGINE_CAPABILITY,
        make_refittable: bool = _defaults.MAKE_REFITTABLE,
        debug: bool = _defaults.DEBUG,
        num_avg_timing_iters: int = _defaults.NUM_AVG_TIMING_ITERS,
        workspace_size: int = _defaults.WORKSPACE_SIZE,
        dla_sram_size: int = _defaults.DLA_SRAM_SIZE,
        dla_local_dram_size: int = _defaults.DLA_LOCAL_DRAM_SIZE,
        dla_global_dram_size: int = _defaults.DLA_GLOBAL_DRAM_SIZE,
        truncate_double: bool = _defaults.TRUNCATE_DOUBLE,
        require_full_compilation: bool = _defaults.REQUIRE_FULL_COMPILATION,
        min_block_size: int = _defaults.MIN_BLOCK_SIZE,
        torch_executed_ops: Optional[Collection[Target]] = None,
        torch_executed_modules: Optional[List[str]] = None,
        pass_through_build_failures: bool = _defaults.PASS_THROUGH_BUILD_FAILURES,
        max_aux_streams: Optional[int] = _defaults.MAX_AUX_STREAMS,
        version_compatible: bool = _defaults.VERSION_COMPATIBLE,
        optimization_level: Optional[int] = _defaults.OPTIMIZATION_LEVEL,
        use_python_runtime: bool = _defaults.USE_PYTHON_RUNTIME,
        use_fast_partitioner: bool = _defaults.USE_FAST_PARTITIONER,
        enable_experimental_decompositions: bool = _defaults.ENABLE_EXPERIMENTAL_DECOMPOSITIONS,
        dryrun: bool = _defaults.DRYRUN,
        hardware_compatible: bool = _defaults.HARDWARE_COMPATIBLE,
        timing_cache_path: str = _defaults.TIMING_CACHE_PATH,
        **kwargs: Any,
    ) -> None:
        """

        Arguments:
            pytorch_model (torch.nn.module): Source module that needs to be accelerated

        Keyword Arguments:
            device (Union(torch_tensorrt.Device, torch.device, dict)): Target device for TensorRT engines to run on ::

                device=torch_tensorrt.Device("dla:1", allow_gpu_fallback=True)

            disable_tf32 (bool): Force FP32 layers to use traditional as FP32 format vs the default behavior of rounding the inputs to 10-bit mantissas before multiplying, but accumulates the sum using 23-bit mantissas
            assume_dynamic_shape_support (bool): Setting this to true enables the converters work for both dynamic and static shapes. Default: False
            sparse_weights (bool): Enable sparsity for convolution and fully connected layers.
            enabled_precision (Set(Union(torch.dtype, torch_tensorrt.dtype))): The set of datatypes that TensorRT can use when selecting kernels
            refit (bool): Enable refitting
            debug (bool): Enable debuggable engine
            capability (torch_tensorrt.EngineCapability): Restrict kernel selection to safe gpu kernels or safe dla kernels
            num_avg_timing_iters (int): Number of averaging timing iterations used to select kernels
            workspace_size (int): Maximum size of workspace given to TensorRT
            dla_sram_size (int): Fast software managed RAM used by DLA to communicate within a layer.
            dla_local_dram_size (int): Host RAM used by DLA to share intermediate tensor data across operations
            dla_global_dram_size (int): Host RAM used by DLA to store weights and metadata for execution
            truncate_double (bool): Truncate weights provided in double (float64) to float32
            calibrator (Union(torch_tensorrt._C.IInt8Calibrator, tensorrt.IInt8Calibrator)): Calibrator object which will provide data to the PTQ system for INT8 Calibration
            require_full_compilation (bool): Require modules to be compiled end to end or return an error as opposed to returning a hybrid graph where operations that cannot be run in TensorRT are run in PyTorch
            min_block_size (int): The minimum number of contiguous TensorRT convertible operations in order to run a set of operations in TensorRT
            torch_executed_ops (Collection[Target]): Set of aten operators that must be run in PyTorch. An error will be thrown if this set is not empty but ``require_full_compilation`` is True
            torch_executed_modules (List[str]): List of modules that must be run in PyTorch. An error will be thrown if this list is not empty but ``require_full_compilation`` is True
            pass_through_build_failures (bool): Error out if there are issues during compilation (only applicable to torch.compile workflows)
            max_aux_stream (Optional[int]): Maximum streams in the engine
            version_compatible (bool): Build the TensorRT engines compatible with future versions of TensorRT (Restrict to lean runtime operators to provide version forward compatibility for the engines)
            optimization_level: (Optional[int]): Setting a higher optimization level allows TensorRT to spend longer engine building time searching for more optimization options. The resulting engine may have better performance compared to an engine built with a lower optimization level. The default optimization level is 3. Valid values include integers from 0 to the maximum optimization level, which is currently 5. Setting it to be greater than the maximum level results in identical behavior to the maximum level.
            use_python_runtime: (bool): Return a graph using a pure Python runtime, reduces options for serialization
            use_fast_partitioner: (bool): Use the adjacency based partitioning scheme instead of the global partitioner. Adjacency partitioning is faster but may not be optimal. Use the global paritioner (``False``) if looking for best performance
            enable_experimental_decompositions (bool): Use the full set of operator decompositions. These decompositions may not be tested but serve to make the graph easier to convert to TensorRT, potentially increasing the amount of graphs run in TensorRT.
            dryrun (bool): Toggle for "Dryrun" mode, running everything except conversion to TRT and logging outputs
            hardware_compatible (bool): Build the TensorRT engines compatible with GPU architectures other than that of the GPU on which the engine was built (currently works for NVIDIA Ampere and newer)
            timing_cache_path (str): Path to the timing cache if it exists (or) where it will be saved after compilation
            lazy_engine_init (bool): Defer setting up engines until the compilation of all engines is complete. Can allow larger models with multiple graph breaks to compile but can lead to oversubscription of GPU memory at runtime.
            **kwargs: Any,
        Returns:
            MutableTorchTensorRTModule
        """
        # The order to initialize this module is
        # 1. Set init_finished to False
        # 2. Initialize all attributes
        # 3. Add the module base class
        # 4. Set the init_finished to True
        # After initialization, no new attribute should be added to the module __dict__
        # Otherwise, it will cause undefined behavior

        object.__setattr__(self, "init_finished", False)
        self.refit_state = RefitState()
        self.pytorch_model = _make_refit_change_trigger(pytorch_model, self.refit_state)
        self.original_model = pytorch_model
        # Process settings
        self.gm: Any = None
        self.exp_program: Any = None
        self.arg_inputs: tuple[Any, ...] = tuple()
        self.kwarg_inputs: dict[str, Any] = {}
        device = to_torch_tensorrt_device(device)
        enabled_precisions = {dtype._from(p) for p in enabled_precisions}
        assert (
            make_refittable
        ), "'make_refittable' has to be True for a MutableTorchTensorRTModule."
        compilation_options = {
            "enabled_precisions": (
                enabled_precisions
                if enabled_precisions
                else _defaults.ENABLED_PRECISIONS
            ),
            "debug": debug,
            "device": device,
            "assume_dynamic_shape_support": assume_dynamic_shape_support,
            "workspace_size": workspace_size,
            "min_block_size": min_block_size,
            "torch_executed_ops": (
                torch_executed_ops if torch_executed_ops is not None else set()
            ),
            "pass_through_build_failures": pass_through_build_failures,
            "max_aux_streams": max_aux_streams,
            "version_compatible": version_compatible,
            "optimization_level": optimization_level,
            "use_python_runtime": use_python_runtime,
            "truncate_double": truncate_double,
            "use_fast_partitioner": use_fast_partitioner,
            "num_avg_timing_iters": num_avg_timing_iters,
            "enable_experimental_decompositions": enable_experimental_decompositions,
            "require_full_compilation": require_full_compilation,
            "disable_tf32": disable_tf32,
            "sparse_weights": sparse_weights,
            "make_refittable": make_refittable,
            "engine_capability": engine_capability,
            "dla_sram_size": dla_sram_size,
            "dla_local_dram_size": dla_local_dram_size,
            "dla_global_dram_size": dla_global_dram_size,
            "dryrun": dryrun,
            "hardware_compatible": hardware_compatible,
            "timing_cache_path": timing_cache_path,
        }

        self.settings = CompilationSettings(**compilation_options)
        self.run_info: Optional[tuple[Any, ...]] = None
        self.state_dict_metadata: dict[str, torch.Size] = {}
        self.store_state_dict_metadata()

        cls = self.__class__
        self.__class__ = type(
            self.original_model.__class__.__name__,
            (cls, pytorch_model.__class__),
            {},
        )
        self.init_finished = True

    def store_state_dict_metadata(self) -> None:

        for k, v in self.original_model.state_dict().items():
            self.state_dict_metadata[k] = v.shape

    def load_state_dict(
        self, state_dict: Dict[str, Any], strict: bool = True, assign: bool = False
    ) -> None:
        self.refit_state.set_state(RefitFlag.NEEDS_REFIT)
        self.original_model.load_state_dict(state_dict, strict=strict, assign=assign)

    @staticmethod
    def _transform_state_dict(sd: Dict[str, Any]) -> Dict[str, torch.nn.Parameter]:
        return {k: torch.nn.Parameter(v, requires_grad=False) for k, v in sd.items()}

    def update_refit_condition(self) -> None:
        # 2-stage check to determine whether the module should be intact, refitted, or recompiled.

        # Default refit
        self.refit_state.set_state(RefitFlag.NEEDS_REFIT)

        # Run the same inputs through pytorch model and compare the result to previous run of graph module
        # to determine whether refit/recompilation is needed. If the output is the same, no further process needed.
        if self.run_info:
            args, kwargs, result = self.run_info
            self.original_model.to(to_torch_device(self.settings.device))
            new_result = self.original_model(*args, **kwargs)
            self.original_model.cpu()
            torch.cuda.empty_cache()
            if check_output_equal(result, new_result):
                self.refit_state.set_state(RefitFlag.LIVE)
                return

        # Since we do not have access to the previous state_dict, we can only use state_dict_metadata
        # to determine whether the keys or weight shape is changed.
        sd, sd_meta = self.original_model.state_dict(), self.state_dict_metadata
        if sd.keys() != sd_meta.keys():
            # If keys are not identical, recompile.
            self.refit_state.set_state(RefitFlag.NEEDS_RECOMPILE)
            return

        for k in sd.keys():
            if sd[k].shape != sd_meta[k]:
                # If weight shapes are not identical, recompile.
                self.refit_state.set_state(RefitFlag.NEEDS_RECOMPILE)
                return

        return

    def refit_gm(self) -> None:
        """
        Refit the TRT graph module with any updates.
        This function should be called whenever the weight values get changed but the weight structure remains
        the same.
        MutableTorchTensorRTModule automatically catches weight value updates and call this function to refit the module.
        If it fails to catch the changes, please call this function manually to update the TRT graph module.
        """
        self.original_model.to(to_torch_device(self.settings.device))
        if self.exp_program is None:
            self.exp_program = torch.export.export(
                self.original_model, self.arg_inputs, kwargs=self.kwarg_inputs
            )
        else:
            self.exp_program._state_dict = (
                MutableTorchTensorRTModule._transform_state_dict(
                    self.original_model.state_dict()
                )
            )
        self.gm = refit_module_weights(
            self.gm,
            self.exp_program,
            self.arg_inputs,
            self.kwarg_inputs,
            use_weight_map_cache=True,
            in_place=True,
        )

        self.original_model.cpu()
        torch.cuda.empty_cache()

    def compile(self) -> None:
        """
        (Re)compile the TRT graph module using the PyTorch module.
        This function should be called whenever the weight structure get changed (shape, more layers...)
        MutableTorchTensorRTModule automatically catches weight value updates and call this function to recompile.
        If it fails to catch the changes, please call this function manually to recompile the TRT graph module.
        """
        # Export the module
        self.original_model.to(to_torch_device(self.settings.device))
        self.exp_program = torch.export.export(
            self.original_model,
            self.arg_inputs,
            kwargs=self.kwarg_inputs,
        )
        self.gm = dynamo_compile(
            self.exp_program,
            arg_inputs=self.arg_inputs,
            kwarg_inputs=self.kwarg_inputs,
            **self.settings.__dict__,
        )
        self.original_model.cpu()
        torch.cuda.empty_cache()

    def _validate_inputs(self, *args: Any, **kwargs: Any) -> None:
        if (
            not self.arg_inputs
            or not MutableTorchTensorRTModule.check_inputs_equal(self.arg_inputs, args)
            or not MutableTorchTensorRTModule.check_inputs_equal(
                self.kwarg_inputs, kwargs
            )
        ):
            logger.info("Input change detected.")
            self.refit_state.set_state(RefitFlag.NEEDS_RECOMPILE)
            self.store_inputs(args, kwargs)

    def store_inputs(self, arg_inputs: Any, kwarg_inputs: Any) -> None:
        self.arg_inputs = arg_inputs
        self.kwarg_inputs = kwarg_inputs

    @staticmethod
    def process_kwarg_inputs(inputs: Any) -> Any:
        # Process kwarg inputs to be acceptable for Torch-TensorRT
        if isinstance(inputs, dict):
            # None should be excluded. AOT compile also does not allow dynamic control flow, bool is also excluded.
            return {
                k: MutableTorchTensorRTModule.process_kwarg_inputs(v)
                for k, v in inputs.items()
                if (v is not None and not isinstance(v, bool))
            }
        elif isinstance(inputs, torch.Tensor):
            return inputs
        elif isinstance(inputs, (int, float, np.ndarray)):
            return torch.tensor(inputs)
        elif isinstance(inputs, (list, tuple)):
            if None not in inputs:
                return type(inputs)(
                    [MutableTorchTensorRTModule.process_kwarg_inputs(v) for v in inputs]
                )

        raise ValueError(
            f"Invalid input type {type(inputs)} encountered in the input. "
            + "Allowed input types: {torch_tensorrt.Input, torch.Tensor, list, tuple, dict}"
        )

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        # Step 1: Check whether the input shape has changed
        kwargs = MutableTorchTensorRTModule.process_kwarg_inputs(kwargs)
        self._validate_inputs(*args, **kwargs)

        # Step 2: If the flag is unknown, it could be a recompile or refit.
        if self.refit_state.get_state() == RefitFlag.UNKNOWN:
            # Update the flag
            self.update_refit_condition()

        # Step 3: Refit/recompile accordingly
        if self.refit_state.get_state() == RefitFlag.NEEDS_RECOMPILE:
            logger.info("(Re)Compiling the engine...")
            self.compile()
            self.store_state_dict_metadata()
            self.refit_state.set_state(RefitFlag.LIVE)

        elif self.refit_state.get_state() == RefitFlag.NEEDS_REFIT:
            logger.info("Model weight change detected. Refitting the module...")
            try:
                self.refit_gm()
            except Exception as e:
                logger.error(e)
                logger.error("Model refit failed. Recompiling the graph module.")
                self.compile()
                self.store_state_dict_metadata()
            self.refit_state.set_state(RefitFlag.LIVE)

        result = self.gm(*args, **kwargs)
        # Storing inputs and outputs for verification when the state is unknown
        self.run_info = (args, kwargs, result)
        return result

    def to(self, device: str) -> None:
        logger.warning("Original PyTorch model is moved. CPU offload may failed.")
        self.orignial_model.to(device)

    def __deepcopy__(self, memo: Any) -> Any:
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k != "pytorch_model":
                setattr(result, k, deepcopy(v, memo))
        result.pytorch_model = _make_refit_change_trigger(
            result.original_model, result.refit_state
        )
        return result

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:

        if name in self.__dict__:
            # this object has it
            return getattr(self, name)

        if "pytorch_model" not in self.__dict__:
            raise AttributeError(
                "Module is not properly initiated. Pytorch model is not found in the module."
            )

        return getattr(self.pytorch_model, name)

    def __delattr__(self, name: str) -> Any:

        if name in self.__dict__:
            # this object has it
            super().__delattr__(name)

        return self.pytorch_model.__delattr__(name)

    def __setattr__(self, name: str, value: Any) -> None:
        # When the module finished initialization, any modification to attributes that does not exist
        # in __dict__ will be handled in pytorch module.
        if self.init_finished:
            if name in self.__dict__:
                object.__setattr__(self, name, value)
            else:
                # Capture attribute change
                self.refit_state.set_state(RefitFlag.UNKNOWN)
                # We want to make sure the original PyTorch model does not have a trigger wrapper
                value = recursively_remove_trigger(value)
                setattr(self.original_model, name, value)
        else:
            object.__setattr__(self, name, value)

    @staticmethod
    def check_inputs_equal(
        input1: Any,
        input2: Any,
    ) -> bool:
        # TODO: Add support for dynamic shape
        if isinstance(input1, (tuple, list)):
            if len(input1) != len(input2):
                return False
            for a, b in zip(input1, input2):
                if type(a) != type(b):
                    return False
                if isinstance(a, torch.Tensor) and a.shape != b.shape:
                    return False
                elif isinstance(a, bool) and a != b:
                    return False

        elif isinstance(input1, dict):
            if input1.keys() != input2.keys():
                return False
            for a, b in zip(input1.values(), input2.values()):
                if type(a) != type(b):
                    return False
                if isinstance(a, torch.Tensor) and a.shape != b.shape:
                    return False
                elif isinstance(a, bool) and a != b:
                    return False
                elif isinstance(
                    a, (list, tuple, dict)
                ) and not MutableTorchTensorRTModule.check_inputs_equal(a, b):
                    return False
        return True

    @staticmethod
    def save(module: Any, path: str) -> None:
        # Cast the object back to MutableTorchTensorRTModule to save
        assert (
            not module.settings.use_python_runtime
        ), "Python runtime does not support serialization. Save failed."
        module.init_finished = False
        module.__class__ = MutableTorchTensorRTModule
        exp_program = module.exp_program
        module.pytorch_model = None
        module.exp_program = None
        torch.save(module, path)
        # Restore deleted attributes
        module.exp_program = exp_program
        module.pytorch_model = _make_refit_change_trigger(
            module.original_model, module.refit_state
        )
        cls = module.__class__
        module.__class__ = type(
            module.original_model.__class__.__name__,
            (cls, module.original_model.__class__),
            {},
        )

        module.init_finished = True

    @staticmethod
    def load(path: str) -> Any:
        # When the model get saved, init_finished is set to False.
        # Class is restored to MutableTorchTensorRTModule, and some attribute is deleted
        module = torch.load(path)
        module.pytorch_model = _make_refit_change_trigger(
            module.original_model, module.refit_state
        )
        module.original_model.to(to_torch_device(module.settings.device))
        module.exp_program = torch.export.export(
            module.original_model, module.arg_inputs, kwargs=module.kwarg_inputs
        )
        module.original_model.to("cpu")
        cls = module.__class__
        module.__class__ = type(
            module.original_model.__class__.__name__,
            (cls, module.original_model.__class__),
            {},
        )
        module.init_finished = True
        return module


def recursively_remove_trigger(obj: Any) -> Any:
    # Not safe: If the object has a circular reference (such as a doubly linkded list), this will cause infinite recursion
    if obj.__class__.__name__ == "ChangeTriggerWrapper":
        obj = obj.instance

    if isinstance(obj, dict):
        for k, v in obj.items():
            obj[k] = recursively_remove_trigger(v)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            obj[i] = recursively_remove_trigger(v)
    else:
        if not hasattr(obj, "__dict__") or isinstance(obj, (type,)):
            return obj
        for k, v in obj.__dict__.items():
            if k[:2] != "__" or k[-2:] != "__":
                # We don't want to touch some built in attribute such as __dict__
                setattr(obj, k, recursively_remove_trigger(v))

    return obj


def _make_refit_change_trigger(obj: object, refit_state: RefitState) -> Any:
    subclass: type = obj.__class__

    class ChangeTriggerWrapper(subclass):  # type: ignore
        # The reason why we want to inherent to the subclass is that we want the ChangeTriggerWrapper shares all functions
        # that an ordinary object has. In this way attributes accessed inside a function will be from the __getattr__function
        # of ChangeTriggerWrapper, instead of the object itself, thus be recursively wrapped by ChangeTriggerWrapper.

        def __init__(self, obj: Any):
            object.__setattr__(self, "instance", obj)

        def __getattr__(
            self, name: str
        ) -> Any:  # Called when the attribute does not exist
            obj = getattr(self.instance, name)
            if isinstance(obj, torch.nn.Parameter):
                # Whenever the user retrieve an attribute that could be related to weights, we set the state to UNKNOWN
                self._on_change()
            if (
                hasattr(obj, "__dict__") or isinstance(obj, (torch.nn.ModuleList, list))
            ) and not isinstance(
                obj, ChangeTriggerWrapper
            ):  # prevent nesting wrapper
                return _make_refit_change_trigger(obj, refit_state)
            return obj

        def __setattr__(self, name: str, value: Any) -> None:
            # If we need to set __dict__ or instance, we directly set it to the trigger wrapper.
            # Enable setting __dict__ is because PyTorch proxy uses __new__ to initialize a shallow copy
            # of a module and explicit set the __dict__. If we don't set __dict__ it will get infinite recursion.
            if name in ["__dict__", "instance"]:
                object.__setattr__(self, name, value)
                return
            self._on_change()
            # We want to make sure the original PyTorch model does not have a trigger wrapper
            value = recursively_remove_trigger(value)
            setattr(self.instance, name, value)

        def __delattr__(self, name: str) -> None:
            self._on_change()
            delattr(
                self.instance,
                name,
            )

        def _on_change(self) -> None:
            refit_state.set_state(RefitFlag.UNKNOWN)
            logger.info(
                "Attribute modification detected. The module will be refitted later."
            )

        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            return self.instance(*args, **kwargs)

        def _call_impl(self, *args: Any, **kwargs: Any) -> Any:
            return self.instance._call_impl(*args, **kwargs)

        def forward(self, *args: Any, **kwargs: Any) -> Any:
            return self.instance.forward(*args, **kwargs)

        def __setitem__(self, item: str, value: Any) -> None:
            self._on_change()
            # We want to make sure the original PyTorch model does not have a trigger wrapper
            value = recursively_remove_trigger(value)
            self.instance.__setitem__(item, value)

        def __getitem__(self, items: str) -> Any:
            obj = self.instance.__getitem__(items)
            if isinstance(obj, ChangeTriggerWrapper):
                return obj
            return _make_refit_change_trigger(obj, refit_state)

        def __len__(self) -> int:
            return len(self.instance)

        def __iter__(self) -> Iterator[Any]:
            return iter(self.instance)

    return ChangeTriggerWrapper(obj)
