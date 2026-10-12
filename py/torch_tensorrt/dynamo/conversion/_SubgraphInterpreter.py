import logging
from typing import Any, Optional, Sequence, Tuple

import torch
from torch_tensorrt.dynamo.conversion._ConversionContext import ConversionContext
from torch_tensorrt.dynamo.conversion._TRTInterpreter import (
    TRTInterpreter,
    UnsupportedOperatorException,
)
from torch_tensorrt.dynamo.conversion.converter_utils import get_node_name

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

    get_attr = TRTInterpreter.get_attr
    call_function = TRTInterpreter.call_function
    call_method = TRTInterpreter.call_method

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
