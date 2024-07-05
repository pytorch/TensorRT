import logging
from typing import Any, Collection, List, Optional, Set, Tuple, Union

import torch
from torch.fx.node import Target
from torch_tensorrt._Device import Device
from torch_tensorrt._enums import EngineCapability, dtype
from torch_tensorrt.dynamo import _defaults
from torch_tensorrt.dynamo._compiler import compile as dynamo_compile
from torch_tensorrt.dynamo._refit import refit_module_weights
from torch_tensorrt.dynamo._settings import CompilationSettings
from torch_tensorrt.dynamo._tracer import trace as dynamo_trace
from torch_tensorrt.dynamo.utils import prepare_inputs, to_torch_tensorrt_device

logger = logging.getLogger(__name__)


class RefitFlag:
    def __init__(self) -> None:
        self.flag = False

    def set_on(self) -> None:
        self.flag = True
        print("RefitFlag is set to ON.")

    def set_off(self) -> None:
        self.flag = False
        print("RefitFlag is set to OFF.")


class MutableTorchTensorRTModule(object):
    def __init__(
        self,
        pytorch_model: torch.nn.Module,
        *,
        device: Optional[Union[Device, torch.device, str]] = _defaults.DEVICE,
        disable_tf32: bool = _defaults.DISABLE_TF32,
        assume_dynamic_shape_support: bool = _defaults.ASSUME_DYNAMIC_SHAPE_SUPPORT,
        sparse_weights: bool = _defaults.SPARSE_WEIGHTS,
        enabled_precisions: (
            Set[torch.dtype | dtype] | Tuple[torch.dtype | dtype]
        ) = _defaults.ENABLED_PRECISIONS,
        engine_capability: EngineCapability = _defaults.ENGINE_CAPABILITY,
        make_refitable: bool = _defaults.MAKE_REFITABLE,
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

        self.refit_flag = RefitFlag()
        self.pytorch_model = _make_refit_change_trigger(pytorch_model, self.refit_flag)
        self.original_model = pytorch_model
        # Process settings
        self.gm: Any = None
        self.exp_program: Any = None
        self.inputs: Optional[tuple[Any, ...]] = None
        device = to_torch_tensorrt_device(device)
        enabled_precisions = {dtype._from(p) for p in enabled_precisions}
        if not make_refitable:
            logger.warning(
                "'make_refitable' has to be True for a MutableTorchTensorRTModule."
            )
            make_refitable = True
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
            "make_refitable": make_refitable,
            "engine_capability": engine_capability,
            "dla_sram_size": dla_sram_size,
            "dla_local_dram_size": dla_local_dram_size,
            "dla_global_dram_size": dla_global_dram_size,
            "dryrun": dryrun,
            "hardware_compatible": hardware_compatible,
            "timing_cache_path": timing_cache_path,
        }

        self.settings = CompilationSettings(**compilation_options)

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.refit_flag.set_on()
        self.pytorch_model.load_state_dict(state_dict)

    def _refit_gm(self) -> None:
        if self.exp_program is None:
            self.exp_program = torch.export.export(
                self.pytorch_model, self.inputs, **self.settings.__dict__
            )
        # TODO: Check refit condition and fallback to recompile
        self.exp_program._state_dict = MutableTorchTensorRTModule._transform_state_dict(
            self.pytorch_model.state_dict()
        )
        self.gm = refit_module_weights(self.gm, self.exp_program, self.inputs)

    def _compile(self) -> None:

        # Export the module
        self.exp_program = dynamo_trace(
            self.original_model, self.torchtrt_inputs, **self.settings.__dict__
        )
        self.gm = dynamo_compile(
            self.exp_program,
            inputs=self.torchtrt_inputs,
            # make_refitable=True,
            **self.settings.__dict__,
        )

    @staticmethod
    def _transform_state_dict(sd: dict[str, Any]) -> dict[str, torch.nn.Parameter]:
        return {k: torch.nn.Parameter(v, requires_grad=False) for k, v in sd.items()}

    def __getattr__(self, name: str) -> Any:

        if name in self.__dict__:
            # this object has it
            return getattr(self, name)

        return getattr(self.pytorch_model, name)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        # We can update this once the kwarg pull request got merged
        return self.forward(*args, **kwargs)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        # TODO: Add support for kwargs
        if not self.inputs or not MutableTorchTensorRTModule.check_inputs(
            self.inputs, args
        ):
            logger.info("Input change detected. (Re)Compiling the engine.")
            self.inputs = args
            self.torchtrt_inputs = prepare_inputs(self.inputs)
            self.refit_flag.set_off()
            self._compile()

        elif self.refit_flag.flag:
            print("Model weight change detected. Refitting the module...")
            self.refit_flag.set_off()
            self._refit_gm()

        return self.gm(*args, **kwargs)

    @staticmethod
    def check_inputs(
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
                if isinstance(a, bool) and a != b:
                    return False

        if isinstance(input1, dict):
            if input1.keys() != input2.keys():
                return False
            for a, b in zip(input1.items(), input2.items()):
                if type(a) != type(b):
                    return False
                if isinstance(a, torch.tensor) and a.shape != b.shape:
                    return False
                if isinstance(a, bool) and a != b:
                    return False
                if isinstance(
                    a, (list, tuple, dict)
                ) and not MutableTorchTensorRTModule.check_inputs(input1, input2):
                    return False
        return True


def _make_refit_change_trigger(obj: object, refit_flag: RefitFlag) -> Any:
    subclass: type = obj.__class__

    class ChangeTriggerWrapper(subclass):  # type: ignore
        def __init__(self, obj: Any):
            object.__setattr__(self, "instance", obj)

        def __getattr__(self, name: str) -> Any:
            # This will cause infinte loop if there is a cycle
            obj = getattr(self.instance, name)
            if not hasattr(obj, "__dict__"):
                return obj
            else:
                return _make_refit_change_trigger(obj, refit_flag)

        def __setattr__(self, name: str, value: Any) -> None:
            self._on_change()
            setattr(self.instance, name, value)

        def __delattr__(self, name: str) -> None:
            self._on_change()
            delattr(
                self.instance,
                name,
            )

        def _on_change(self) -> None:
            refit_flag.set_on()
            print("Change!")

        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            print("Warning: uncatched change in function!")
            return self.instance(*args, **kwargs)

    return ChangeTriggerWrapper(obj)
