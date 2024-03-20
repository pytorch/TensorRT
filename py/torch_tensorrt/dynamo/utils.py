from __future__ import annotations

import logging
from dataclasses import fields, replace
from typing import Any, Callable, Dict, Optional, Sequence, Union

import torch
import torch_tensorrt
from torch_tensorrt._Device import Device
from torch_tensorrt._Input import Input
from torch_tensorrt.dynamo._defaults import PRECISION
from torch_tensorrt.dynamo._settings import CompilationSettings

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


def input_is_dynamic(inputs: Sequence[Union[Input, torch.Tensor]]) -> bool:
    """
    Return true if the provided inputs are `torch_tensorrt.Input` objects and have dynamic shapes.
    """
    return not any(isinstance(input, torch.Tensor) for input in inputs) and any(
        input.shape_mode == Input._ShapeMode.DYNAMIC for input in inputs
    )


def get_torch_inputs(
    inputs: Sequence[Input], device: Union[Device, torch.device, str], mode: str = ""
) -> Sequence[torch.tensor]:
    """
    Return the torch_tensor from the Input object. If mode is set, this implies
    user is using dynamic shaped inputs and return the corresponding input based
    on the mode requested.
    """
    device = to_torch_device(device)
    if mode:
        return [
            input.example_tensor(mode).to(device)
            for input in inputs
            if isinstance(input, Input)
        ]
    return [
        input.torch_tensor.to(device) for input in inputs if isinstance(input, Input)
    ]


def set_log_level(parent_logger: Any, level: Any) -> None:
    """
    Sets the log level to the user provided level.
    This is used to set debug logging at a global level
    at entry points of tracing, dynamo and torch_compile compilation.
    """
    if parent_logger:
        parent_logger.setLevel(level)


def prepare_inputs(
    inputs: Input | torch.Tensor | Sequence[Any] | Dict[Any, Any],
    disable_memory_format_check: bool = False,
) -> Any:
    if isinstance(inputs, Input):
        return inputs

    elif isinstance(inputs, torch.Tensor):
        return Input.from_tensor(
            inputs, disable_memory_format_check=disable_memory_format_check
        )

    elif isinstance(inputs, (list, tuple)):
        torchtrt_input_list = []
        for input_obj in inputs:
            torchtrt_input = prepare_inputs(
                input_obj, disable_memory_format_check=disable_memory_format_check
            )
            torchtrt_input_list.append(torchtrt_input)

        return (
            torchtrt_input_list
            if isinstance(inputs, list)
            else tuple(torchtrt_input_list)
        )

    elif isinstance(inputs, dict):
        torchtrt_inputs_dict: Dict[Any, Any] = dict()

        for key, input_obj in inputs.items():
            torchtrt_input = prepare_inputs(
                input_obj, disable_memory_format_check=disable_memory_format_check
            )
            torchtrt_inputs_dict[key] = torchtrt_input

        return torchtrt_inputs_dict

    else:
        raise ValueError(
            f"Invalid input type {type(inputs)} encountered in the dynamo_compile input parsing. "
            + "Allowed input types: {torch_tensorrt.Input, torch.Tensor, list, tuple, dict}"
        )


def parse_complex_tensor_structs(
    inputs: Input | torch.Tensor | Sequence[Any] | Dict[Any, Any],
    attribute_to_extract: str,
    apply_fn: Callable[[Any], Any] = lambda x: x,
) -> Any:
    """Parses complex structures of Tensors and returns a mirrored structure
    Extracts key attributes of each singular element, while reconstructing the struct
    Optionally applies a function to each attribute before returning
    """
    if isinstance(inputs, (torch.Tensor, Input)):
        return apply_fn(getattr(inputs, attribute_to_extract, None))

    elif isinstance(inputs, (list, tuple)):
        torchtrt_input_list = []
        for input_obj in inputs:
            torchtrt_input = parse_complex_tensor_structs(
                input_obj, attribute_to_extract, apply_fn
            )
            torchtrt_input_list.append(torchtrt_input)

        return (
            torchtrt_input_list
            if isinstance(inputs, list)
            else tuple(torchtrt_input_list)
        )

    elif isinstance(inputs, dict):
        torchtrt_inputs_dict: Dict[Any, Any] = dict()

        for key, input_obj in inputs.items():
            torchtrt_input = parse_complex_tensor_structs(
                input_obj, attribute_to_extract, apply_fn
            )
            torchtrt_inputs_dict[key] = torchtrt_input

        return torchtrt_inputs_dict

    elif isinstance(inputs, (torch.SymBool, torch.SymFloat, torch.SymInt)):
        raise ValueError(
            f"Detected Torch symbolic input type {type(inputs)} during input parsing. "
            "Symbolic inputs are not currently allowed; please specify dynamic=False "
            "if using torch.compile with the Torch-TensorRT backend."
        )

    else:
        raise ValueError(
            f"Invalid input type {type(inputs)} encountered during Dynamo input parsing. "
            + "Allowed input types: {torch_tensorrt.Input, torch.Tensor, list, tuple, dict}"
        )


def to_torch_device(device: Optional[Union[Device, torch.device, str]]) -> torch.device:
    """Cast a device-type to torch.device

    Returns the corresponding torch.device
    """
    if isinstance(device, Device):
        if device.gpu_id != -1:
            return torch.device(device.gpu_id)
        else:
            raise ValueError("Invalid GPU ID provided for the CUDA device provided")

    elif isinstance(device, torch.device):
        return device

    elif device is None:
        return torch.device(torch.cuda.current_device())

    else:
        return torch.device(device)


def to_torch_tensorrt_device(
    device: Optional[Union[Device, torch.device, str]]
) -> Device:
    """Cast a device-type to torch_tensorrt.Device

    Returns the corresponding torch_tensorrt.Device
    """
    if isinstance(device, Device):
        return device

    elif isinstance(device, torch.device):
        return Device(gpu_id=device.index)

    elif device is None:
        return Device(gpu_id=torch.cuda.current_device())

    else:
        return Device(device)


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

    # TODO: Remove once Dynamo precisions refactoring is complete
    if "enabled_precisions" in kwargs:
        enabled_precisions = kwargs["enabled_precisions"]

        if (
            torch.float16 in enabled_precisions
            or torch_tensorrt.dtype.half in enabled_precisions
        ):
            settings.precision = torch.float16
        elif (
            torch.float32 in enabled_precisions
            or torch_tensorrt.dtype.float in enabled_precisions
        ):
            settings.precision = torch.float32
        elif len(enabled_precisions) == 0:
            logger.info(f"No precision specified, defaulting to {PRECISION}")
            settings.precision = PRECISION
        else:
            raise ValueError(
                f"Precision {enabled_precisions} not supported in the Dynamo Path"
            )

    # Parse input runtime specification
    settings.use_python_runtime = use_python_runtime_parser(settings.use_python_runtime)

    # Ensure device is a torch_tensorrt Device
    settings.device = to_torch_tensorrt_device(settings.device)

    # Check and update device settings
    if "device" not in kwargs:
        logger.info(
            f"Device not specified, using Torch default current device - cuda:{settings.device.gpu_id}. "
            "If this is incorrect, please specify an input device, via the device keyword."
        )

    # Ignore and warn about require_full_compilation flag
    if settings.require_full_compilation:
        logger.warning(
            "Detected require_full_compilation=True for a torch.compile run. "
            "This option has no effect in torch.compile."
        )
        settings.require_full_compilation = False

    logger.info("Compilation Settings: %s\n", settings)

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
