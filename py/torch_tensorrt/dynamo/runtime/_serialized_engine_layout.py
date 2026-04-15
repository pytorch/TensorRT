"""Serialized TensorRT engine blob layout shared by C++ and Python runtimes.

Field order and indices must stay aligned with ``core/runtime/runtime.h`` and
``register_jit_hooks.cpp`` (``torch.ops.tensorrt.*``). When the C++ runtime is
loaded, :func:`_assert_serialized_layout_matches_cpp` checks that these literals
match the library; fix either side if the assertion fails.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Tuple, Union

import tensorrt as trt
import torch
import torch_tensorrt
from torch_tensorrt._features import ENABLED_FEATURES

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

# (torch.ops.tensorrt name, module global holding the expected value, normalizer)
_LayoutCheck = Tuple[str, str, Callable[[Any], Any]]
_LAYOUT_CPP_CHECKS: tuple[_LayoutCheck, ...] = (
    ("ABI_VERSION", "ABI_VERSION", str),
    ("ABI_TARGET_IDX", "ABI_TARGET_IDX", int),
    ("NAME_IDX", "NAME_IDX", int),
    ("DEVICE_IDX", "DEVICE_IDX", int),
    ("ENGINE_IDX", "ENGINE_IDX", int),
    ("INPUT_BINDING_NAMES_IDX", "INPUT_BINDING_NAMES_IDX", int),
    ("OUTPUT_BINDING_NAMES_IDX", "OUTPUT_BINDING_NAMES_IDX", int),
    ("HW_COMPATIBLE_IDX", "HW_COMPATIBLE_IDX", int),
    ("SERIALIZED_METADATA_IDX", "SERIALIZED_METADATA_IDX", int),
    ("TARGET_PLATFORM_IDX", "TARGET_PLATFORM_IDX", int),
    ("REQUIRES_OUTPUT_ALLOCATOR_IDX", "REQUIRES_OUTPUT_ALLOCATOR_IDX", int),
    ("RESOURCE_ALLOCATION_STRATEGY_IDX", "RESOURCE_ALLOCATION_STRATEGY_IDX", int),
    ("SERIALIZATION_LEN", "SERIALIZATION_LEN", int),
    ("SERIALIZED_ENGINE_BINDING_DELIM", "SERIALIZED_ENGINE_BINDING_DELIM", str),
    ("SERIALIZED_RT_DEVICE_DELIM", "SERIALIZED_RT_DEVICE_DELIM", str),
)


def _assert_serialized_layout_matches_cpp() -> None:
    """Fail fast if Python layout literals diverge from ``register_jit_hooks.cpp``."""
    if not ENABLED_FEATURES.torch_tensorrt_runtime:
        return
    for op_name, global_name, normalizer in _LAYOUT_CPP_CHECKS:
        expected = globals()[global_name]
        try:
            op = getattr(torch.ops.tensorrt, op_name)
            raw = op()
        except (AttributeError, RuntimeError, TypeError) as e:
            raise RuntimeError(
                f"Could not call torch.ops.tensorrt.{op_name}() to verify serialized layout: {e}"
            ) from e
        got = normalizer(raw)
        if got != expected:
            raise RuntimeError(
                f"Serialized engine layout mismatch: torch.ops.tensorrt.{op_name}() "
                f"returned {got!r} but Python _serialized_engine_layout.{global_name} "
                f"is {expected!r}. Align ``runtime.h`` / ``register_jit_hooks.cpp`` with "
                f"``_serialized_engine_layout.py``."
            )


_assert_serialized_layout_matches_cpp()

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
