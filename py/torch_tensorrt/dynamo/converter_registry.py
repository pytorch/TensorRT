from typing import Any, Callable, Dict

from torch.fx.node import Target
from torch_tensorrt.fx.converter_registry import CONVERTERS

DYNAMO_CONVERTERS: Dict[Target, Any] = dict(CONVERTERS)


def dynamo_tensorrt_converter(
    key: Target,
    enabled: bool = True,
) -> Callable[[Any], Any]:
    def register_converter(converter):
        DYNAMO_CONVERTERS[key] = converter
        return converter

    def disable_converter(converter):
        return converter

    if enabled:
        return register_converter
    else:
        return disable_converter
