"""
Capture symbolic shape expressions from FX graphs for TRT meta kernel.

This module extracts the symbolic relationship between input and output shapes
at compile time, which can then be used by the meta kernel to correctly infer
output shapes without pattern matching.
"""

import logging
from typing import Any, Dict, List, Optional

import torch

logger = logging.getLogger(__name__)


def extract_symbolic_shape_expressions(
    module: torch.fx.GraphModule,
) -> Optional[Dict[str, List[Dict[str, Any]]]]:
    """
    Extract symbolic shape expressions from an FX graph.

    This captures the symbolic expressions (as sympy expressions) for input and output shapes
    that can be applied to input fake tensors at runtime.

    Args:
        module: FX GraphModule with symbolic shapes in node metadata

    Returns:
        Dict with 'inputs' and 'outputs' keys, each containing a list of dicts with shape_exprs and dtype,
        or None if extraction fails
    """
    # Find input nodes (placeholders)
    input_nodes = [node for node in module.graph.nodes if node.op == "placeholder"]

    # Find output node
    output_nodes = [node for node in module.graph.nodes if node.op == "output"]
    if not output_nodes:
        return None

    output_node = output_nodes[0]

    # Collect shape expressions and dtypes for each input
    input_info = []
    for input_node in input_nodes:
        if not hasattr(input_node, "meta") or "val" not in input_node.meta:
            logger.warning(
                "When processing symbolic shapes for TensorRT engine, found no metadata in input node"
            )
            return None

        input_val = input_node.meta["val"]
        if not isinstance(input_val, torch.Tensor):
            logger.warning(
                "When processing symbolic shapes for TensorRT engine, input is not a tensor"
            )
            return None

        # Extract shape as sympy expressions (can be pickled)
        shape_exprs = []
        for dim_size in input_val.shape:
            if isinstance(dim_size, torch.SymInt):
                # Store the sympy expression, which can be pickled
                shape_exprs.append(dim_size.node.expr)
            else:
                # Store concrete integer
                shape_exprs.append(int(dim_size))

        input_info.append(
            {
                "shape_exprs": shape_exprs,
                "dtype": input_val.dtype,
                "name": input_node.name,
            }
        )

    # Extract output values from output node
    output_args = output_node.args[0]
    if not isinstance(output_args, (tuple, list)):
        output_args = (output_args,)

    # Collect shape expressions and dtypes for each output
    output_info = []
    for out_arg in output_args:
        if not hasattr(out_arg, "meta") or "val" not in out_arg.meta:
            logger.warning(
                "When processing symbolic shapes for TensorRT engine, found no metadata in FX Graph"
            )
            return None

        out_val = out_arg.meta["val"]
        if not isinstance(out_val, torch.Tensor):
            logger.warning(
                "When processing symbolic shapes for TensorRT engine, output is not a tensor"
            )
            return None

        # Extract shape as sympy expressions (can be pickled)
        shape_exprs = []
        for dim_size in out_val.shape:
            if isinstance(dim_size, torch.SymInt):
                # Store the sympy expression, which can be pickled
                shape_exprs.append(dim_size.node.expr)
            else:
                # Store concrete integer
                shape_exprs.append(int(dim_size))

        output_info.append(
            {
                "shape_exprs": shape_exprs,
                "dtype": out_val.dtype,
            }
        )

    if not output_info:
        return None

    return {
        "inputs": input_info,
        "outputs": output_info,
    }
