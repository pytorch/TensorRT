from typing import Any

import torch
from torch_tensorrt._version import __version__

def sanitized_torch_version() -> Any:
    return (
        torch.__version__
        if ".nv" not in torch.__version__
        else torch.__version__.split(".nv")[0]
    )
