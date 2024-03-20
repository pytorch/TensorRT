from typing import Any

import torch


def sanitized_torch_version() -> Any:
    return (
        torch.__version__
        if ".nv" not in torch.__version__
        else torch.__version__.split(".nv")[0]
    )
