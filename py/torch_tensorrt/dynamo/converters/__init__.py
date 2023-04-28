# @manual=//deeplearning/trt/python:py_tensorrt
import tensorrt as trt

if hasattr(trt, "__version__"):
    from .activation import *  # noqa: F401 F403
    from .aten_ops_converters import *  # noqa: F401 F403
    from .operator import *  # noqa: F401 F403


    TRT_LOGGER = trt.Logger()
    trt.init_libnvinfer_plugins(TRT_LOGGER, "")
