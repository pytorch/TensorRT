import logging

from torch_tensorrt.fx.converter_registry import (  # noqa
    CONVERTERS,
    NO_EXPLICIT_BATCH_DIM_SUPPORT,
    NO_IMPLICIT_BATCH_DIM_SUPPORT,
    tensorrt_converter,
)
from .lower_setting import LowerSetting  # noqa
from .lower import compile  # usort: skip  #noqa

logging.basicConfig(level=logging.INFO)
