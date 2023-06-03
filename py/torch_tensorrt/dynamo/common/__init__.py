import logging
from typing import Optional

from ._settings import CompilationSettings
from .input_tensor_spec import InputTensorSpec
from .fx2trt import TRTInterpreter, TRTInterpreterResult


logger = logging.getLogger(__name__)


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
            from torch_tensorrt.dynamo._TorchTensorRTModule import TorchTensorRTModule

            using_python_runtime = False
            reason = "since C++ dependency was detected as present"
        except ImportError:
            using_python_runtime = True
            reason = "since import failed, C++ dependency not installed"

    logger.info(
        f"Using {'Python' if using_python_runtime else 'C++'} {reason} TRT Runtime"
    )

    return using_python_runtime
