import torch
import logging
from dataclasses import replace, fields
from torch_tensorrt.dynamo import CompilationSettings
from typing import Any, Union, Sequence, Dict
from torch_tensorrt import _Input, Device


logger = logging.getLogger(__name__)

COSINE_THRESHOLD = 0.99


def cosine_similarity(gt_tensor, pred_tensor):
    gt_tensor = gt_tensor.flatten().to(torch.float32)
    pred_tensor = pred_tensor.flatten().to(torch.float32)
    if torch.sum(gt_tensor) == 0.0 or torch.sum(pred_tensor) == 0.0:
        if torch.allclose(gt_tensor, pred_tensor, atol=1e-4, rtol=1e-4, equal_nan=True):
            return 1.0
    res = torch.nn.functional.cosine_similarity(gt_tensor, pred_tensor, dim=0, eps=1e-6)
    res = res.cpu().detach().item()

    return res


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

    logger.debug(f"Compiling with Settings:\n{settings}")

    return settings
