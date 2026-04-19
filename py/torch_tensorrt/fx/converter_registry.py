import logging
from typing import Any, Callable, Dict

from torch.fx.node import Target

CONVERTERS: Dict[Target, Any] = {}


logger = logging.getLogger(__name__)


def tensorrt_converter(
    key: Target,
    enabled: bool = True,
) -> Callable[[Any], Any]:
    def register_converter(converter):
        CONVERTERS[key] = converter
        logger.debug(f"Converter for {key} added to FX Converter Registry")
        return converter

    def disable_converter(converter):
        return converter

    if enabled:
        return register_converter
    else:
        return disable_converter
