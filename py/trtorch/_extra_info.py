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

def _parse_extra_info(extra_info: Dict[str, Any]) -> trtorch._C._ExtraInfo:
    info = trtorch._C._ExtraInfo()
    if "input_shapes" not in extra_info and not isinstance(extra_info["input_shapes"], list):
        raise KeyError("Input shapes for inputs are required as a List, provided as either a static sizes or a range of three sizes (min, opt, max) as Dict")

    info.input_ranges = _parse_input_ranges(extra_info["input_shapes"])
    print(info.input_ranges)

    if "op_precision" in extra_info:
        info.op_precision = _parse_op_precision(extra_info["op_precision"])

    if "refit" in extra_info:
        assert isinstance(extra_info["refit"], bool)
        info.refit = extra_info["refit"]

    if "debug" in extra_info:
        assert isinstance(extra_info["debug"], bool)
        info.debug = extra_info["debug"]

    if "strict_types" in extra_info:
        assert isinstance(extra_info["strict_types"], bool)
        info.strict_types = extra_info["strict_types"]

    if "allow_gpu_fallback" in extra_info:
        assert isinstance(extra_info["allow_gpu_fallback"], bool)
        info.allow_gpu_fallback = extra_info["allow_gpu_fallback"]

    if "device" in extra_info:
        info.device = _parse_device_type(extra_info["device"])

    if "capability" in extra_info:
        assert isinstance(extra_info["capability"], type.EngineCapability)
        info.capability = extra_info["capability"]

    if "num_min_timing_iters" in extra_info:
        assert type(extra_info["num_min_timing_iters"]) is int
        info.num_min_timing_iters = extra_info["num_min_timing_iters"]

    if "num_avg_timing_iters" in extra_info:
        assert type(extra_info["num_avg_timing_iters"]) is int
        info.num_avg_timing_iters = extra_info["num_avg_timing_iters"]

    if "workspace_size" in extra_info:
        assert type(extra_info["workspace_size"]) is int
        info.workspace_size = extra_info["workspace_size"]

    if "max_batch_size" in extra_info:
        assert type(extra_info["max_batch_size"]) is int
        info.max_batch_size = extra_info["max_batch_size"]

    return info