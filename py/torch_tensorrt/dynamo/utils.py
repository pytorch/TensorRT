import torch
import logging
from dataclasses import replace, fields
from torch_tensorrt.dynamo import CompilationSettings
from typing import Any, Union, Sequence, Dict
from torch_tensorrt import Input, Device
from typing import Optional

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
            from torch_tensorrt.dynamo.runtime import TorchTensorRTModule

            using_python_runtime = False
            reason = "since C++ dependency was detected as present"
        except ImportError:
            using_python_runtime = True
            reason = "since import failed, C++ dependency not installed"

    logger.info(
        f"Using {'Python-only' if using_python_runtime else 'Default'} Torch-TRT Runtime ({reason})"
    )

    return using_python_runtime


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
    inputs: Union[Input, torch.Tensor, Sequence, Dict],
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
        prepared_input = list()
        torchtrt_inputs = []
        torch_inputs = []
        for input_obj in inputs:
            torchtrt_input, torch_input = prepare_inputs(input_obj)
            torchtrt_inputs.append(torchtrt_input)
            torch_inputs.append(torch_input)

        return torchtrt_inputs, torch_inputs

    elif isinstance(inputs, tuple):
        torchtrt_inputs = []
        torch_inputs = []
        for input_obj in inputs:
            torchtrt_input, torch_input = prepare_inputs(input_obj)
            torchtrt_inputs.append(torchtrt_input)
            torch_inputs.append(torch_input)

        return tuple(torchtrt_inputs), tuple(torch_inputs)

    elif isinstance(inputs, dict):
        torchtrt_inputs = dict()
        torch_inputs = dict()

        for key, input_obj in inputs.items():
            torchtrt_input, torch_input = prepare_inputs(input_obj)
            torchtrt_inputs[key] = torchtrt_input
            torch_inputs[key] = torch_input

        return torchtrt_inputs, torch_inputs

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
