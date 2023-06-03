import torch
import logging
from dataclasses import replace, fields

from torch_tensorrt.dynamo.common import CompilationSettings, use_python_runtime_parser
from typing import Any, Union, Sequence, Dict
from torch_tensorrt import _Input, Device


logger = logging.getLogger(__name__)


def prepare_inputs(
    inputs: Union[_Input.Input, torch.Tensor, Sequence, Dict],
    device: torch.device = torch.device("cuda"),
) -> Any:
    if isinstance(inputs, _Input.Input):
        if isinstance(inputs.shape, dict):
            return inputs.example_tensor(optimization_profile_field="opt_shape").to(
                device
            )
        else:
            return inputs.example_tensor().to(device)

    elif isinstance(inputs, torch.Tensor):
        return inputs

    elif isinstance(inputs, list):
        prepared_input = list()

        for input_obj in inputs:
            prepared_input.append(prepare_inputs(input_obj))

        return prepared_input

    elif isinstance(inputs, tuple):
        prepared_input = list()

        for input_obj in inputs:
            prepared_input.append(prepare_inputs(input_obj))

        return tuple(prepared_input)

    elif isinstance(inputs, dict):
        prepared_input = dict()

        for key, input_obj in inputs.items():
            prepared_input[key] = prepare_inputs(input_obj)

        return prepared_input

    else:
        raise ValueError(
            f"Invalid input type {type(inputs)} encountered in the dynamo_compile input parsing. "
            + "Allowed input types: {torch_tensorrt.Input, torch.Tensor, list, tuple, dict}"
        )


def prepare_device(device: Union[Device, torch.device]) -> torch.device:
    if isinstance(device, Device):
        if device.gpu_id != -1:
            device = torch.device(device.gpu_id)
        else:
            raise ValueError("Invalid GPU ID provided for the CUDA device provided")

    elif isinstance(device, torch.device):
        device = device

    else:
        raise ValueError(
            "Invalid device provided. Supported options: torch.device | torch_tensorrt.Device"
        )

    return device


def parse_dynamo_kwargs(kwargs: Dict) -> CompilationSettings:
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
