import logging
from typing import Any, List

import torch

logger = logging.getLogger(__name__)


def sanitized_torch_version() -> Any:
    return (
        torch.__version__
        if ".nv" not in torch.__version__
        else torch.__version__.split(".nv")[0]
    )


@torch.library.custom_op(
    "tensorrt::no_op_placeholder_for_execute_engine", mutates_args=()
)
def no_op_placeholder_for_execute_engine(
    inputs: List[torch.Tensor],
    abi_version: str,
    name: str,
    serialized_device_info: str,
    serialized_engine: str,
    serialized_in_binding_names: str,
    serialized_out_binding_names: str,
    serialized_hardware_compatible: str,
    serialized_metadata: str,
    serialized_target_platform: str,
) -> List[torch.Tensor]:
    logger.warning(
        "The saved model is cross compiled for windows in Linux, should only be loadded in Windows via torch_tensorrt.cross_load_in_windows() api."
    )
    raise RuntimeError(
        "The saved model is cross compiled for windows in Linux, should only be loadded in Windows via torch_tensorrt.cross_load_in_windows() api."
    )
