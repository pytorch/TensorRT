"""Serialized TensorRT engine blob layout shared by C++ and Python runtimes.

Field order and indices must stay aligned with the Torch-TensorRT C++ engine
packing (e.g. ``TRTEngine`` / ``register_jit_hooks``). Python-only builds use
this module instead of ``torch.ops.tensorrt.*_IDX()`` helpers.
"""

from __future__ import annotations

from typing import Any, Dict, List, Union

import tensorrt as trt
import torch
import torch_tensorrt

ABI_VERSION = "8"
ABI_TARGET_IDX = 0
NAME_IDX = 1
DEVICE_IDX = 2
ENGINE_IDX = 3
INPUT_BINDING_NAMES_IDX = 4
OUTPUT_BINDING_NAMES_IDX = 5
HW_COMPATIBLE_IDX = 6
SERIALIZED_METADATA_IDX = 7
TARGET_PLATFORM_IDX = 8
REQUIRES_OUTPUT_ALLOCATOR_IDX = 9
RESOURCE_ALLOCATION_STRATEGY_IDX = 10
SERIALIZATION_LEN = 11

SERIALIZED_ENGINE_BINDING_DELIM = "%"
SERIALIZED_RT_DEVICE_DELIM = "%"

SerializedTensorRTEngineFmt = List[Union[str, bytes]]


def serialize_binding_names(binding_names: List[str]) -> str:
    return SERIALIZED_ENGINE_BINDING_DELIM.join(binding_names)


def deserialize_binding_names(binding_names: str) -> List[str]:
    return binding_names.split(SERIALIZED_ENGINE_BINDING_DELIM) if binding_names else []


def serialize_device_info(device: torch_tensorrt.Device) -> str:
    dev_info = torch.cuda.get_device_properties(device.gpu_id)
    rt_info = [
        device.gpu_id,
        dev_info.major,
        dev_info.minor,
        int(device.device_type.to(trt.DeviceType)),
        dev_info.name,
    ]
    return SERIALIZED_RT_DEVICE_DELIM.join(str(value) for value in rt_info)


def parse_device_info(serialized_device_info: str) -> Dict[str, Any]:
    tokens = serialized_device_info.split(SERIALIZED_RT_DEVICE_DELIM)
    if len(tokens) != 5:
        raise RuntimeError(
            f"Unable to deserialize program target device information: {serialized_device_info}"
        )

    target_device_id = int(tokens[0])
    return {
        "id": target_device_id,
        "major": int(tokens[1]),
        "minor": int(tokens[2]),
        "device_type": int(tokens[3]),
        "name": tokens[4],
    }
