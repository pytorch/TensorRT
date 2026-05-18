from typing import Callable, Optional

from torch.fx.node import Node

from torch_tensorrt.dynamo._settings import CompilationSettings
from torch_tensorrt.dynamo.conversion._ConverterRegistry import ConverterPriority
from torch_tensorrt.dynamo.conversion.plugins._generate_plugin import generate_plugin
from torch_tensorrt.dynamo.conversion.plugins._generate_plugin_converter import (
    generate_plugin_converter,
)


def custom_op(
    op_name: str,
    capability_validator: Optional[Callable[[Node, CompilationSettings], bool]] = None,
    priority: ConverterPriority = ConverterPriority.STANDARD,
    supports_dynamic_shapes: bool = False,
    requires_output_allocator: bool = False,
    *,
    use_aot_if_available: bool = True,
    _aot_register: Optional[Callable[[], None]] = None,
) -> None:
    """
    Generate the Plugin and corresponding Plugin Converter using external kernels and TensorRT Quick Deployable Plugin APIs.

    Args:
        plugin_name: the plugin name that is used to generate the plugin automatically.
            There should be existing kernels and pytorch custom operation for this plugin name.
        capability_validator:  A lambda that can take a ``torch.fx.Node`` and determine if the
            converter can properly handle this Node. If the validator returns ``False``, the subgraph
            partitioner will make sure this Node is run in PyTorch in the compiled graph.
        priority: Allows developers to override existing converters in the converter registry
        supports_dynamic_shapes: if dynamic shape is supported
        requires_output_allocator: if the converter creates operators which require an Output Allocator to run (e.g. data dependent operators)
        use_aot_if_available: forwarded to ``generate_plugin_converter``; when ``True``
            (the default), the converter prefers the AOT impl if the op has one registered.
        _aot_register: internal hook used by ``torch_tensorrt.kernels`` to register a
            cuda-python AOT impl between the plugin descriptor and the converter. Not
            part of the public API; pass ``None`` (the default) for ordinary use.
    """
    generate_plugin(op_name)
    if _aot_register is not None:
        _aot_register()
    generate_plugin_converter(
        op_name,
        capability_validator,
        priority,
        supports_dynamic_shapes,
        requires_output_allocator,
        use_aot_if_available=use_aot_if_available,
    )
