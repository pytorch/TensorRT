from __future__ import annotations

import logging
from dataclasses import fields, replace
from typing import Any, Callable, Dict, Optional, Sequence

import torch
from torch_tensorrt._Device import Device
from torch_tensorrt._Input import Input
from torch_tensorrt.dynamo import CompilationSettings

from packaging import version

logger = logging.getLogger(__name__)

COSINE_THRESHOLD = 0.99


def use_python_runtime_parser(use_python_runtime: Optional[bool] = None) -> bool:
    """Parses a user-provided input argument regarding Python runtime

    Automatically handles cases where the user has not specified a runtime (None)

    Returns True if the Python runtime should be used, False if the C++ runtime should be used
    """
    using_python_runtime = use_python_runtime
    reason = ""

    # Runtime was manually specified by the user
    if using_python_runtime is not None:
        reason = "as requested by user"
    # Runtime was not manually specified by the user, automatically detect runtime
    else:
        try:
            from torch_tensorrt.dynamo.runtime import TorchTensorRTModule  # noqa: F401

            using_python_runtime = False
            reason = "since C++ dependency was detected as present"
        except ImportError:
            using_python_runtime = True
            reason = "since import failed, C++ dependency not installed"

    logger.info(
        f"Using {'Python-only' if using_python_runtime else 'Default'} Torch-TRT Runtime ({reason})"
    )

    return using_python_runtime


def cosine_similarity(gt_tensor: torch.Tensor, pred_tensor: torch.Tensor) -> float:
    gt_tensor = gt_tensor.flatten().to(torch.float32)
    pred_tensor = pred_tensor.flatten().to(torch.float32)
    if torch.sum(gt_tensor) == 0.0 or torch.sum(pred_tensor) == 0.0:
        if torch.allclose(gt_tensor, pred_tensor, atol=1e-4, rtol=1e-4, equal_nan=True):
            return 1.0
    res_t = torch.nn.functional.cosine_similarity(
        gt_tensor, pred_tensor, dim=0, eps=1e-6
    )
    res: float = res_t.cpu().detach().item()

    return res


def prepare_inputs(
    inputs: Input | torch.Tensor | Sequence[Any] | Dict[Any, Any],
    device: torch.device = torch.device("cuda"),
) -> Any:
    if isinstance(inputs, Input):
        if isinstance(inputs.shape, dict):
            return inputs, inputs.example_tensor(
                optimization_profile_field="opt_shape"
            ).to(device)
        else:
            return inputs, inputs.example_tensor().to(device)

    elif isinstance(inputs, torch.Tensor):
        return Input.from_tensor(inputs), inputs

    elif isinstance(inputs, list):
        torchtrt_input_list = []
        torch_input_list = []
        for input_obj in inputs:
            torchtrt_input, torch_input = prepare_inputs(input_obj)
            torchtrt_input_list.append(torchtrt_input)
            torch_input_list.append(torch_input)

        return torchtrt_input_list, torch_input_list

    elif isinstance(inputs, tuple):
        torchtrt_inputs_tup = []
        torch_inputs_tup = []
        for input_obj in inputs:
            torchtrt_input, torch_input = prepare_inputs(input_obj)
            torchtrt_inputs_tup.append(torchtrt_input)
            torch_inputs_tup.append(torch_input)

        return tuple(torchtrt_inputs_tup), tuple(torch_inputs_tup)

    elif isinstance(inputs, dict):
        torchtrt_inputs_dict: Dict[Any, Any] = dict()
        torch_inputs_dict: Dict[Any, Any] = dict()

        for key, input_obj in inputs.items():
            torchtrt_input, torch_input = prepare_inputs(input_obj)
            torchtrt_inputs_dict[key] = torchtrt_input
            torch_inputs_dict[key] = torch_input

        return torchtrt_inputs_dict, torch_inputs_dict

    else:
        raise ValueError(
            f"Invalid input type {type(inputs)} encountered in the dynamo_compile input parsing. "
            + "Allowed input types: {torch_tensorrt.Input, torch.Tensor, list, tuple, dict}"
        )


def prepare_device(device: Device | torch.device) -> torch.device:
    _device: torch.device
    if isinstance(device, Device):
        if device.gpu_id != -1:
            _device = torch.device(device.gpu_id)
        else:
            raise ValueError("Invalid GPU ID provided for the CUDA device provided")

    elif isinstance(device, torch.device):
        _device = device

    else:
        raise ValueError(
            "Invalid device provided. Supported options: torch.device | torch_tensorrt.Device"
        )

    return _device


def parse_dynamo_kwargs(kwargs: Any) -> CompilationSettings:
    """Parses the kwargs field of a Dynamo backend

    Args:
        kwargs: Keyword arguments dictionary provided to the backend
    Returns:
        CompilationSettings object with relevant kwargs
    """

    # Initialize an empty CompilationSettings object
    settings = CompilationSettings()

    # If the user specifies keyword args, overwrite those fields in settings
    # Validate all specified kwargs to ensure they are true fields of the dataclass
    #
    # Note: kwargs provided by torch.compile are wrapped in the "options" key
    if kwargs:
        if "options" in kwargs and len(kwargs) == 1:
            kwargs = kwargs["options"]

        valid_attrs = {attr.name for attr in fields(settings)}
        valid_kwargs = {k: v for k, v in kwargs.items() if k in valid_attrs}
        settings = replace(settings, **valid_kwargs)

    # Enable debug/verbose mode if requested
    if settings.debug:
        logger.setLevel(logging.DEBUG)

    # Parse input runtime specification
    settings.use_python_runtime = use_python_runtime_parser(settings.use_python_runtime)

    logger.debug(f"Compiling with Settings:\n{settings}")

    return settings


def req_torch_version(min_torch_version: str = "2.dev") -> Callable[..., Any]:
    """
    Create a decorator which verifies the Torch version installed
    against a specified version range

    Args:
        min_torch_version (str): The minimum required Torch version
        for the decorated function to work properly

    Returns:
        A decorator which raises a descriptive error message if
        an unsupported Torch version is used
    """

    def nested_decorator(f: Callable[..., Any]) -> Callable[..., Any]:
        def function_wrapper(*args: Any, **kwargs: Any) -> Any:
            # Parse minimum and current Torch versions
            min_version = version.parse(min_torch_version)
            current_version = version.parse(torch.__version__)

            if current_version < min_version:
                raise AssertionError(
                    f"Expected Torch version {min_torch_version} or greater, "
                    + f"when calling {f}. Detected version {torch.__version__}"
                )
            else:
                return f(*args, **kwargs)

        return function_wrapper

    return nested_decorator
