from typing import List, Dict, Any, Set
import torch
from torch_tensorrt import _C
import torch_tensorrt._C.ts as _ts_C
from torch_tensorrt import _enums
from torch_tensorrt._Input import Input
from torch_tensorrt._Device import Device
from torch_tensorrt.logging import Level, log
from typing import Tuple, List, Dict
import warnings
from copy import deepcopy


def _internal_input_to_torch_class_input(i: _C.Input) -> torch.classes.tensorrt._Input:
    clone = torch.classes.tensorrt._Input()
    clone._set_min(i.min)
    clone._set_opt(i.opt)
    clone._set_max(i.max)
    clone._set_dtype(i.dtype)
    clone._set_format(i.format)
    clone._set_input_is_dynamic(i.input_is_dynamic)
    clone._set_explicit_set_dtype(i._explicit_set_dtype)
    return clone


def _supported_input_size_type(input_size: Any) -> bool:
    if isinstance(input_size, torch.Size):
        return True
    elif isinstance(input_size, tuple):
        return True
    elif isinstance(input_size, list):
        return True
    else:
        raise TypeError(
            "Input sizes for inputs are required to be a List, tuple or torch.Size or a Dict of three sizes (min, opt, max), found type: "
            + str(type(input_size))
        )


def _parse_input_ranges(input_sizes: List) -> List:

    if any(
        not isinstance(i, dict) and not _supported_input_size_type(i)
        for i in input_sizes
    ):
        raise KeyError(
            "An input size must either be a static size or a range of three sizes (min, opt, max) as Dict"
        )

    parsed_input_sizes = []
    for i in input_sizes:
        if isinstance(i, dict):
            if all(k in i for k in ["min", "opt", "min"]):
                parsed_input_sizes.append(
                    Input(
                        min_shape=i["min"], opt_shape=i["opt"], max_shape=i["max"]
                    )._to_internal()
                )

            elif "opt" in i:
                parsed_input_sizes.append(Input(shape=i["opt"])._to_internal())

            else:
                raise KeyError(
                    "An input size must either be a static size or a range of three sizes (min, opt, max) as Dict"
                )

        elif isinstance(i, list):
            parsed_input_sizes.append(Input(shape=i)._to_internal())

        elif isinstance(i, tuple):
            parsed_input_sizes.append(Input(shape=i)._to_internal())

        elif isinstance(i, torch.Size):
            parsed_input_sizes.append(Input(shape=i)._to_internal())

    return parsed_input_sizes


def _parse_op_precision(precision: Any) -> _enums.dtype:
    if isinstance(precision, torch.dtype):
        if precision == torch.int8:
            return _enums.dtype.int8
        elif precision == torch.half:
            return _enums.dtype.half
        elif precision == torch.float:
            return _enums.dtype.float
        else:
            raise TypeError(
                "Provided an unsupported dtype as operating precision (support: int8, half, float), got: "
                + str(precision)
            )

    elif isinstance(precision, _enums.dtype):
        return precision

    else:
        raise TypeError(
            "Op precision type needs to be specified with a torch.dtype or a torch_tensorrt.dtype, got: "
            + str(type(precision))
        )


def _parse_enabled_precisions(precisions: Any) -> Set:
    parsed_precisions = set()
    if any([isinstance(precisions, type) for type in [list, tuple, set]]):
        for p in precisions:
            parsed_precisions.add(_parse_op_precision(p))
    else:
        parsed_precisions.add(_parse_op_precision(precisions))
    return parsed_precisions


def _parse_device_type(device: Any) -> _enums.DeviceType:
    if isinstance(device, torch.device):
        if device.type == "cuda":
            return _enums.DeviceType.gpu
        else:
            ValueError(
                "Got a device type other than GPU or DLA (type: "
                + str(device.type)
                + ")"
            )
    elif isinstance(device, _enums.DeviceType):
        return device
    elif isinstance(device, str):
        if device == "gpu" or device == "GPU":
            return _enums.DeviceType.gpu
        elif device == "dla" or device == "DLA":
            return _enums.DeviceType.dla
        else:
            ValueError(
                "Got a device type other than GPU or DLA (type: " + str(device) + ")"
            )
    else:
        raise TypeError(
            "Device specification must be of type torch.device, string or torch_tensorrt.DeviceType, but got: "
            + str(type(device))
        )


def _parse_device(device_info: Any) -> _C.Device:
    if isinstance(device_info, dict):
        info = _C.Device()
        if "device_type" not in device_info:
            raise KeyError("Device type is required parameter")
        else:
            assert isinstance(device_info["device_type"], _enums.DeviceType)
            info.device_type = _parse_device_type(device_info["device_type"])

        if "gpu_id" in device_info:
            assert isinstance(device_info["gpu_id"], int)
            info.gpu_id = device_info["gpu_id"]

        if "dla_core" in device_info:
            assert isinstance(device_info["dla_core"], int)
            info.dla_core = device_info["dla_core"]

        if "allow_gpu_fallback" in device_info:
            assert isinstance(device_info["allow_gpu_fallback"], bool)
            info.allow_gpu_fallback = device_info["allow_gpu_fallback"]

        return info
    elif isinstance(device_info, Device):
        return device_info._to_internal()
    elif isinstance(device_info, torch.device):
        return (Device._from_torch_device(device_info))._to_internal()
    else:
        raise ValueError(
            "Unsupported data for device specification. Expected either a dict, torch_tensorrt.Device or torch.Device"
        )


def _parse_torch_fallback(fallback_info: Dict[str, Any]) -> _ts_C.TorchFallback:
    info = _ts_C.TorchFallback()
    if "enabled" not in fallback_info:
        raise KeyError("Enabled is required parameter")
    else:
        assert isinstance(fallback_info["enabled"], bool)
        info.enabled = fallback_info["enabled"]
    if "min_block_size" in fallback_info:
        assert isinstance(fallback_info["min_block_size"], int)
        info.min_block_size = fallback_info["min_block_size"]

    if "forced_fallback_ops" in fallback_info:
        assert isinstance(fallback_info["forced_fallback_ops"], list)
        info.forced_fallback_operators = fallback_info["forced_fallback_ops"]

    if "forced_fallback_modules" in fallback_info:
        assert isinstance(fallback_info["forced_fallback_modules"], list)
        info.forced_fallback_modules = fallback_info["forced_fallback_modules"]

    return info


def _parse_input_signature(input_signature: Any):
    if isinstance(input_signature, tuple):
        input_list = []
        for item in input_signature:
            input = _parse_input_signature(item)
            input_list.append(input)
        return tuple(input_list)
    elif isinstance(input_signature, list):
        input_list = []
        for item in input_signature:
            input = _parse_input_signature(item)
            input_list.append(input)
        return input_list
    elif isinstance(input_signature, Input) or isinstance(
        input_signature, torch.Tensor
    ):
        i = (
            Input._from_tensor(input_signature)
            if isinstance(input_signature, torch.Tensor)
            else input_signature
        )
        clone = _internal_input_to_torch_class_input(i._to_internal())
        return clone
    else:
        raise KeyError(
            "Input signature contains an unsupported type {}".format(
                type(input_signature)
            )
        )


def _parse_compile_spec(compile_spec_: Dict[str, Any]) -> _ts_C.CompileSpec:
    # TODO: Remove deep copy once collections does not need partial compilation
    compile_spec = deepcopy(compile_spec_)
    info = _ts_C.CompileSpec()

    if len(compile_spec["inputs"]) > 0:
        if not all(
            [
                isinstance(i, torch.Tensor) or isinstance(i, Input)
                for i in compile_spec["inputs"]
            ]
        ):
            raise KeyError(
                "Input specs should be either torch_tensorrt.Input or torch.Tensor, found types: {}".format(
                    [type(i) for i in compile_spec["inputs"]]
                )
            )

        inputs = [
            Input._from_tensor(i) if isinstance(i, torch.Tensor) else i
            for i in compile_spec["inputs"]
        ]
        info.inputs = [i._to_internal() for i in inputs]

    elif compile_spec["input_signature"] is not None:
        log(
            Level.Warning,
            "Input signature parsing is an experimental feature, behavior and APIs may change",
        )
        signature = _parse_input_signature(compile_spec["input_signature"])
        info.input_signature = _C.InputSignature(signature)  # py_object

        if not compile_spec["torch_fallback"]["enabled"]:
            raise ValueError(
                "Grouped inputs currently requires partial compilation to be enabled, this restriction will be relaxed in a future release"
            )

        log(
            Level.Debug,
            "Grouped inputs currently requires additional settings to enable the feature",
        )
        log(
            Level.Debug,
            """Adding the following ops to torch_executed_ops:
    - aten::__getitem__
    - prim::ListConstruct
    - prim::ListUnpack
    - prim::TupleIndex
    - prim::TupleConstruct
    - prim::TupleUnpack
""",
        )
        compile_spec["torch_fallback"]["forced_fallback_ops"].append(
            "aten::__getitem__"
        )
        compile_spec["torch_fallback"]["forced_fallback_ops"].append(
            "prim::ListConstruct"
        )
        compile_spec["torch_fallback"]["forced_fallback_ops"].append("prim::ListUnpack")
        compile_spec["torch_fallback"]["forced_fallback_ops"].append("prim::TupleIndex")
        compile_spec["torch_fallback"]["forced_fallback_ops"].append(
            "prim::TupleConstruct"
        )
        compile_spec["torch_fallback"]["forced_fallback_ops"].append(
            "prim::TupleUnpack"
        )

    else:
        raise KeyError(
            'Module input definitions are requried to compile module. Provide a list of torch_tensorrt.Input keyed to "inputs" in the compile spec'
        )

    if "enabled_precisions" in compile_spec:
        info.enabled_precisions = _parse_enabled_precisions(
            compile_spec["enabled_precisions"]
        )

    if "calibrator" in compile_spec:
        info.ptq_calibrator = compile_spec["calibrator"]

    if "sparse_weights" in compile_spec:
        assert isinstance(compile_spec["sparse_weights"], bool)
        info.sparse_weights = compile_spec["sparse_weights"]

    if "disable_tf32" in compile_spec:
        assert isinstance(compile_spec["disable_tf32"], bool)
        info.disable_tf32 = compile_spec["disable_tf32"]

    if "refit" in compile_spec:
        assert isinstance(compile_spec["refit"], bool)
        info.refit = compile_spec["refit"]

    if "debug" in compile_spec:
        assert isinstance(compile_spec["debug"], bool)
        info.debug = compile_spec["debug"]

    if "device" in compile_spec:
        info.device = _parse_device(compile_spec["device"])

    if "capability" in compile_spec:
        assert isinstance(compile_spec["capability"], _enums.EngineCapability)
        info.capability = compile_spec["capability"]

    if "num_avg_timing_iters" in compile_spec:
        assert type(compile_spec["num_avg_timing_iters"]) is int
        info.num_avg_timing_iters = compile_spec["num_avg_timing_iters"]

    if "workspace_size" in compile_spec:
        assert type(compile_spec["workspace_size"]) is int
        info.workspace_size = compile_spec["workspace_size"]

    if "dla_sram_size" in compile_spec:
        assert type(compile_spec["dla_sram_size"]) is int
        info.dla_sram_size = compile_spec["dla_sram_size"]

    if "dla_local_dram_size" in compile_spec:
        assert type(compile_spec["dla_local_dram_size"]) is int
        info.dla_local_dram_size = compile_spec["dla_local_dram_size"]

    if "dla_global_dram_size" in compile_spec:
        assert type(compile_spec["dla_global_dram_size"]) is int
        info.dla_global_dram_size = compile_spec["dla_global_dram_size"]

    if "truncate_long_and_double" in compile_spec:
        assert type(compile_spec["truncate_long_and_double"]) is bool
        info.truncate_long_and_double = compile_spec["truncate_long_and_double"]

    if "torch_fallback" in compile_spec:
        info.torch_fallback = _parse_torch_fallback(compile_spec["torch_fallback"])

    log(Level.Debug, str(info))

    return info


def TensorRTCompileSpec(
    inputs=[],
    input_signature=None,
    device=Device._current_device(),
    disable_tf32=False,
    sparse_weights=False,
    enabled_precisions=set(),
    refit=False,
    debug=False,
    capability=_enums.EngineCapability.default,
    num_avg_timing_iters=1,
    workspace_size=0,
    dla_sram_size=1048576,
    dla_local_dram_size=1073741824,
    dla_global_dram_size=536870912,
    truncate_long_and_double=False,
    calibrator=None,
) -> torch.classes.tensorrt.CompileSpec:
    """Utility to create a formated spec dictionary for using the PyTorch TensorRT backend

    Keyword Args:
        inputs (List[Union(torch_tensorrt.Input, torch.Tensor)]): **Required** List of specifications of input shape, dtype and memory layout for inputs to the module. This argument is required. Input Sizes can be specified as torch sizes, tuples or lists. dtypes can be specified using
            torch datatypes or torch_tensorrt datatypes and you can use either torch devices or the torch_tensorrt device type enum
            to select device type. ::

                input=[
                    torch_tensorrt.Input((1, 3, 224, 224)), # Static NCHW input shape for input #1
                    torch_tensorrt.Input(
                        min_shape=(1, 224, 224, 3),
                        opt_shape=(1, 512, 512, 3),
                        max_shape=(1, 1024, 1024, 3),
                        dtype=torch.int32
                        format=torch.channel_last
                    ), # Dynamic input shape for input #2
                    torch.randn((1, 3, 224, 244)) # Use an example tensor and let torch_tensorrt infer settings
                ]

        device (Union(torch_tensorrt.Device, torch.device, dict)): Target device for TensorRT engines to run on ::

            device=torch_tensorrt.Device("dla:1", allow_gpu_fallback=True)

        disable_tf32 (bool): Force FP32 layers to use traditional as FP32 format vs the default behavior of rounding the inputs to 10-bit mantissas before multiplying, but accumulates the sum using 23-bit mantissas
        sparse_weights (bool): Enable sparsity for convolution and fully connected layers.
        enabled_precision (Set(Union(torch.dtype, torch_tensorrt.dtype))): The set of datatypes that TensorRT can use when selecting kernels
        refit (bool): Enable refitting
        debug (bool): Enable debuggable engine
        capability (torch_tensorrt.EngineCapability): Restrict kernel selection to safe gpu kernels or safe dla kernels
        num_avg_timing_iters (int): Number of averaging timing iterations used to select kernels
        workspace_size (int): Maximum size of workspace given to TensorRT
        truncate_long_and_double (bool): Truncate weights provided in int64 or double (float64) to int32 and float32
        calibrator (Union(torch_tensorrt._C.IInt8Calibrator, tensorrt.IInt8Calibrator)): Calibrator object which will provide data to the PTQ system for INT8 Calibration

      Returns:
        torch.classes.tensorrt.CompileSpec: List of methods and formated spec objects to be provided to ``torch._C._jit_to_tensorrt``
    """

    compile_spec = {
        "inputs": inputs,
        # "input_signature": input_signature,
        "device": device,
        "disable_tf32": disable_tf32,  # Force FP32 layers to use traditional as FP32 format vs the default behavior of rounding the inputs to 10-bit mantissas before multiplying, but accumulates the sum using 23-bit mantissas
        "sparse_weights": sparse_weights,  # Enable sparsity for convolution and fully connected layers.
        "enabled_precisions": enabled_precisions,  # Enabling FP16 kernels
        "refit": refit,  # enable refit
        "debug": debug,  # enable debuggable engine
        "capability": capability,  # Restrict kernel selection to safe gpu kernels or safe dla kernels
        "num_avg_timing_iters": num_avg_timing_iters,  # Number of averaging timing iterations used to select kernels
        "workspace_size": workspace_size,  # Maximum size of workspace given to TensorRT
        "dla_sram_size": dla_sram_size,  # Fast software managed RAM used by DLA to communicate within a layer.
        "dla_local_dram_size": dla_local_dram_size,  # Host RAM used by DLA to share intermediate tensor data across operations
        "dla_global_dram_size": dla_global_dram_size,  # Host RAM used by DLA to store weights and metadata for execution
        "calibrator": calibrator,
        "truncate_long_and_double": truncate_long_and_double,
    }

    parsed_spec = _parse_compile_spec(compile_spec)

    backend_spec = torch.classes.tensorrt.CompileSpec()

    if input_signature is not None:
        raise ValueError(
            "Input signature parsing is not currently supported in the TorchScript backend integration"
        )

    for i in parsed_spec.inputs:
        clone = _internal_input_to_torch_class_input(i)
        backend_spec._append_input(clone)

    d = torch.classes.tensorrt._Device()
    d._set_device_type(int(parsed_spec.device.device_type))
    d._set_gpu_id(parsed_spec.device.gpu_id)
    d._set_dla_core(parsed_spec.device.dla_core)
    d._set_allow_gpu_fallback(parsed_spec.device.allow_gpu_fallback)

    if parsed_spec.torch_fallback.enabled:
        raise RuntimeError(
            "Partial module compilation is not currently supported via the PyTorch TensorRT backend. If you need partial compilation, use torch_tensorrt.compile"
        )

    torch_fallback = torch.classes.tensorrt._TorchFallback()
    torch_fallback._set_enabled(parsed_spec.torch_fallback.enabled)
    torch_fallback._set_min_block_size(parsed_spec.torch_fallback.min_block_size)
    torch_fallback._set_forced_fallback_operators(
        parsed_spec.torch_fallback.forced_fallback_operators
    )
    torch_fallback._set_forced_fallback_modules(
        parsed_spec.torch_fallback.forced_fallback_modules
    )

    backend_spec._set_device(d)
    backend_spec._set_torch_fallback(torch_fallback)
    backend_spec._set_precisions([int(i) for i in parsed_spec.enabled_precisions])

    backend_spec._set_disable_tf32(parsed_spec.disable_tf32)
    backend_spec._set_refit(parsed_spec.refit)
    backend_spec._set_debug(parsed_spec.debug)
    backend_spec._set_refit(parsed_spec.refit)
    backend_spec._set_capability(int(parsed_spec.capability))
    backend_spec._set_num_avg_timing_iters(parsed_spec.num_avg_timing_iters)
    backend_spec._set_workspace_size(parsed_spec.workspace_size)
    backend_spec._set_dla_sram_size(parsed_spec.dla_sram_size)
    backend_spec._set_dla_local_dram_size(parsed_spec.dla_local_dram_size)
    backend_spec._set_dla_global_dram_size(parsed_spec.dla_global_dram_size)
    backend_spec._set_truncate_long_and_double(parsed_spec.truncate_long_and_double)
    backend_spec._set_ptq_calibrator(parsed_spec._get_calibrator_handle())

    return backend_spec
