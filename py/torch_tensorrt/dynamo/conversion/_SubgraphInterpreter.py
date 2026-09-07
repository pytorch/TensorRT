import logging
from typing import Any, Optional, Sequence, Tuple

import torch
from torch.fx.experimental.proxy_tensor import unset_fake_temporarily
from torch.utils._python_dispatch import _disable_current_modes
from torch_tensorrt.dynamo.conversion._ConversionContext import ConversionContext
from torch_tensorrt.dynamo.conversion._ConverterRegistry import (
    DYNAMO_CONVERTERS as CONVERTERS,
)
from torch_tensorrt.dynamo.conversion._ConverterRegistry import (
    CallingConvention,
)
from torch_tensorrt.dynamo.conversion._TRTInterpreter import (
    UnsupportedOperatorException,
)
from torch_tensorrt.dynamo.conversion.converter_utils import get_node_name, to_torch

_LOGGER = logging.getLogger(__name__)


class TRTSubgraphInterpreter(torch.fx.Interpreter):  # type: ignore[misc]
    """Convert an FX GraphModule into an existing TensorRT network.

    Unlike ``TRTInterpreter``, this does not create a builder, network, or
    engine I/O bindings. Placeholders are bound to caller-provided values
    (typically ``IIfConditionalInputLayer`` outputs) via ``Interpreter.run``.
    """

    def __init__(
        self,
        module: torch.fx.GraphModule,
        ctx: ConversionContext,
        name_prefix: str,
    ) -> None:
        super().__init__(module)
        self.ctx = ctx
        self.name_prefix = name_prefix
        self._cur_node: Optional[torch.fx.Node] = None
        self._cur_node_name: Optional[str] = None

    def run_node(self, n: torch.fx.Node) -> Any:
        prev = self.ctx.current_node
        self._cur_node = n
        self._cur_node_name = f"{self.name_prefix}/{get_node_name(n)}"
        self.ctx.current_node = n
        try:
            if _LOGGER.isEnabledFor(logging.DEBUG):
                _LOGGER.debug(
                    "Converting cond-subgraph node %s (kind: %s)",
                    self._cur_node_name,
                    n.target,
                )
            return super().run_node(n)
        finally:
            self.ctx.current_node = prev

    def get_attr(self, target: str, args: Any, kwargs: Any) -> Any:
        del args, kwargs
        with _disable_current_modes(), unset_fake_temporarily():
            attr = self.fetch_attr(target)
            if isinstance(attr, torch.nn.Module):
                return attr
            if isinstance(attr, torch.nn.Parameter):
                attr = attr.data
            return to_torch(attr)

    def call_function(self, target: Any, args: Any, kwargs: Any) -> Any:
        converter_packet = CONVERTERS.get(self._cur_node)
        if converter_packet is None:
            raise UnsupportedOperatorException(
                f"Conversion of function {torch.typename(target)} not currently supported "
                f"inside torch.cond subgraph '{self.name_prefix}'"
            )

        converter, calling_convention, converter_info = converter_packet
        if converter_info.get("requires_output_allocator", False):
            self.ctx.requires_output_allocator = True
            _LOGGER.debug("%s requires output allocator", target)
        if converter_info.get("requires_native_multidevice", False):
            self.ctx.requires_native_multidevice = True
            _LOGGER.debug("%s requires native multi-device support", target)

        if calling_convention is CallingConvention.LEGACY:
            return converter(self.ctx.net, target, args, kwargs, self._cur_node_name)
        return converter(self.ctx, target, args, kwargs, self._cur_node_name)

    def call_method(self, target: str, args: Any, kwargs: Any) -> Any:
        converter_packet = CONVERTERS.get(self._cur_node)
        if converter_packet is None:
            raise UnsupportedOperatorException(
                f"Conversion of method {target} not currently supported "
                f"inside torch.cond subgraph '{self.name_prefix}'"
            )
        converter, calling_convention, _ = converter_packet
        if calling_convention is CallingConvention.LEGACY:
            return converter(self.ctx.net, target, args, kwargs, self._cur_node_name)
        return converter(self.ctx, target, args, kwargs, self._cur_node_name)

    def call_module(self, target: str, args: Any, kwargs: Any) -> Any:
        del args, kwargs
        raise UnsupportedOperatorException(
            f"call_module '{target}' is not supported inside torch.cond subgraphs"
        )


def convert_subgraph(
    ctx: ConversionContext,
    gm: torch.fx.GraphModule,
    operands: Sequence[Any],
    name_prefix: str,
) -> Tuple[Any, ...]:
    """Convert ``gm`` with ``operands`` bound to its placeholders.

    Returns the subgraph outputs as a tuple, matching torch.cond's convention
    that branch graphs return a tuple even for a single tensor.
    """
    interp = TRTSubgraphInterpreter(gm, ctx, name_prefix)
    outputs = interp.run(*operands)
    if not isinstance(outputs, (list, tuple)):
        return (outputs,)
    return tuple(outputs)
