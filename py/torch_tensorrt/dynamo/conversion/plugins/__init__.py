from torch_tensorrt.dynamo.conversion.plugins import (  # noqa: F401  (registers wrapper converters on import)
    _auto_functionalized_converter,
)
from torch_tensorrt.dynamo.conversion.plugins._aot_utils import make_aot_extra_args
from torch_tensorrt.dynamo.conversion.plugins._custom_op import custom_op
from torch_tensorrt.dynamo.conversion.plugins._generate_plugin import generate_plugin
from torch_tensorrt.dynamo.conversion.plugins._generate_plugin_converter import (
    generate_plugin_converter,
)
