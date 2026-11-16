# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import logging
from typing import Protocol

import torch
import torch_tensorrt


class _ComparableDeviceProps(Protocol):
    """Enough for multi-device checks; may be ``_CudaDeviceProperties`` or a simple namespace."""

    major: int
    minor: int
    name: object


logger = logging.getLogger(__name__)


def multi_gpu_device_check() -> None:
    # If multi-device safe mode is disabled and more than 1 device is registered on the machine, warn user
    if (
        not torch_tensorrt.runtime._multi_device_safe_mode._PY_RT_MULTI_DEVICE_SAFE_MODE
        and torch.cuda.device_count() > 1
    ):
        logger.warning(
            "Detected this engine is being instantitated in a multi-GPU system with "
            "multi-device safe mode disabled. For more on the implications of this "
            "as well as workarounds, see the linked documentation "
            "(https://pytorch.org/TensorRT/user_guide/runtime.html#multi-device-safe-mode). "
            f"The engine is set to be instantiated on the current default cuda device, cuda:{torch.cuda.current_device()}. "
            "If this is incorrect, please set the desired cuda device via torch.cuda.set_device(...) and retry."
        )


def _is_switch_required(
    curr_device_id: int,
    engine_device_id: int,
    curr_device_properties: torch._C._CudaDeviceProperties,
    engine_device_properties: _ComparableDeviceProps,
) -> bool:
    """Determines whether a device switch is required based on input device parameters"""
    # Device Capabilities disagree
    if (curr_device_properties.major, curr_device_properties.minor) != (
        engine_device_properties.major,
        engine_device_properties.minor,
    ):
        logger.warning(
            f"Configured SM capability {(engine_device_properties.major, engine_device_properties.minor)} does not match with "
            f"current device SM capability {(curr_device_properties.major, curr_device_properties.minor)}. Switching device context."
        )

        return True

    # Names disagree
    if curr_device_properties.name != engine_device_properties.name:
        logger.warning(
            f"Program compiled for {engine_device_properties.name} but current CUDA device is "
            f"current device SM capability {curr_device_properties.name}. Attempting to switch device context for better compatibility."
        )

        return True

    # Device IDs disagree
    if curr_device_id != engine_device_id:
        logger.warning(
            f"Configured Device ID: {engine_device_id} is different than current device ID: "
            f"{curr_device_id}. Attempting to switch device context for better compatibility."
        )

        return True

    return False
