from typing import List, Dict, Any
import torch
import trtorch._C
from trtorch import _types

def _supported_input_size_type(input_size: Any) -> bool:
    if isinstance(input_size, torch.Size):
        return True
    elif isinstance(input_size, tuple):
        return True
    elif isinstance(input_size, list):
        return True
    else:
        raise TypeError("Input sizes for inputs are required to be a List, tuple or torch.Size or a Dict of three sizes (min, opt, max), found type: " + str(type(input_size)))

def _parse_input_ranges(input_sizes: List) -> List:

    if any (not isinstance(i, dict) and not _supported_input_size_type(i) for i in input_sizes):
        raise KeyError("An input size must either be a static size or a range of three sizes (min, opt, max) as Dict")

    parsed_input_sizes = []
    for i in input_sizes:
        if isinstance(i, dict):
            if all (k in i for k in ["min", "opt", "min"]):
                in_range = trtorch._C.InputRange()
                in_range.min = i["min"]
                in_range.opt = i["opt"]
                in_range.max = i["max"]
                parsed_input_sizes.append(in_range)

            elif "opt" in i:
                in_range = trtorch._C.InputRange()
                in_range.min = i["opt"]
                in_range.opt = i["opt"]
                in_range.max = i["opt"]
                parsed_input_sizes.append(in_range)

            else:
                raise KeyError("An input size must either be a static size or a range of three sizes (min, opt, max) as Dict")

        elif isinstance(i, list):
            in_range = trtorch._C.InputRange()
            in_range.min = i
            in_range.opt = i
            in_range.max = i
            parsed_input_sizes.append(in_range)

        elif isinstance(i, tuple):
            in_range = trtorch._C.InputRange()
            in_range.min = list(i)
            in_range.opt = list(i)
            in_range.max = list(i)
            parsed_input_sizes.append(in_range)

    return parsed_input_sizes

def _parse_op_precision(precision: Any) -> _types.dtype:
    if isinstance(precision, torch.dtype):
        if precision == torch.int8:
            return _types.dtype.int8
        elif precision == torch.half:
            return _types.dtype.half
        elif precision == torch.float:
            return _types.dtype.float
        else:
            raise TypeError("Provided an unsupported dtype as operating precision (support: int8, half, float), got: " + str(precision))

    elif isinstance(precision, _types.DataTypes):
        return precision

    else:
        raise TypeError("Op precision type needs to be specified with a torch.dtype or a trtorch.dtype, got: " + str(type(precision)))

def _parse_device_type(device: Any) -> _types.DeviceType:
    if isinstance(device, torch.device):
        if torch.device.type == 'cuda':
            return _types.DeviceType.gpu
        else:
            raise TypeError("Valid device choices are GPU (and DLA if on Jetson platforms) however got device type" + str(device.type))

    elif isinstance(device, _types.DeviceType):
        return device

    else:
        raise TypeError("Device specification must be of type torch.device or trtorch.DeviceType, but got: " + str(type(device)))

def _parse_compile_spec(compile_spec: Dict[str, Any]) -> trtorch._C.CompileSpec:
    info = trtorch._C.CompileSpec()
    if "input_shapes" not in compile_spec:
        raise KeyError("Input shapes for inputs are required as a List, provided as either a static sizes or a range of three sizes (min, opt, max) as Dict")

    info.input_ranges = _parse_input_ranges(compile_spec["input_shapes"])

    if "op_precision" in compile_spec:
        info.op_precision = _parse_op_precision(compile_spec["op_precision"])

    if "refit" in compile_spec:
        assert isinstance(compile_spec["refit"], bool)
        info.refit = compile_spec["refit"]

    if "debug" in compile_spec:
        assert isinstance(compile_spec["debug"], bool)
        info.debug = compile_spec["debug"]

    if "strict_types" in compile_spec:
        assert isinstance(compile_spec["strict_types"], bool)
        info.strict_types = compile_spec["strict_types"]

    if "allow_gpu_fallback" in compile_spec:
        assert isinstance(compile_spec["allow_gpu_fallback"], bool)
        info.allow_gpu_fallback = compile_spec["allow_gpu_fallback"]

    if "device" in compile_spec:
        info.device = _parse_device_type(compile_spec["device"])

    if "capability" in compile_spec:
        assert isinstance(compile_spec["capability"], type.EngineCapability)
        info.capability = compile_spec["capability"]

    if "num_min_timing_iters" in compile_spec:
        assert type(compile_spec["num_min_timing_iters"]) is int
        info.num_min_timing_iters = compile_spec["num_min_timing_iters"]

    if "num_avg_timing_iters" in compile_spec:
        assert type(compile_spec["num_avg_timing_iters"]) is int
        info.num_avg_timing_iters = compile_spec["num_avg_timing_iters"]

    if "workspace_size" in compile_spec:
        assert type(compile_spec["workspace_size"]) is int
        info.workspace_size = compile_spec["workspace_size"]

    if "max_batch_size" in compile_spec:
        assert type(compile_spec["max_batch_size"]) is int
        info.max_batch_size = compile_spec["max_batch_size"]

    return info