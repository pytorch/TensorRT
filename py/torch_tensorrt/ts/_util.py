from torch_tensorrt import __version__
from torch_tensorrt import _C

import torch


def dump_build_info():
    """Prints build information about the torch_tensorrt distribution to stdout"""
    print(get_build_info())


def get_build_info() -> str:
    """Returns a string containing the build information of torch_tensorrt distribution

    Returns:
        str: String containing the build information for torch_tensorrt distribution
    """
    build_info = _C.get_build_info()
    build_info = (
        "Torch-TensorRT Version: "
        + str(__version__)
        + "\n"
        + "Using PyTorch Version: "
        + str(torch.__version__)
        + "\n"
        + build_info
    )
    return build_info


def set_device(gpu_id):
    _C.set_device(gpu_id)
