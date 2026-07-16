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


def _resolve_meta_val(node: Any) -> Any:
    """Return ``node.meta['val']``, resolving through in-place ``copy_`` write-backs.

    When a model returns a tensor mutated in place, functionalization can leave the
    write-back ``aten.copy_(dest, src)`` as a graph output with no ``meta['val']``.
    A ``copy_`` takes the shape/dtype of its destination, so fall back to the
    destination's value. Returns ``None`` if no value can be resolved.
    """
    meta = getattr(node, "meta", None)
    if meta is not None and "val" in meta:
        return meta["val"]
    if (
        getattr(node, "op", None) == "call_function"
        and node.target is torch.ops.aten.copy_.default
        and node.args
    ):
        return _resolve_meta_val(node.args[0])
    return None


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
        logger.debug(
            f"Input node '{input_node.name}': type={type(input_val)}, val={input_val}"
        )
        if isinstance(input_val, torch.Tensor):
            shape_exprs = []
            for dim_size in input_val.shape:
                if isinstance(dim_size, torch.SymInt):
                    shape_exprs.append(dim_size.node.expr)
                else:
                    shape_exprs.append(int(dim_size))

            input_info.append(
                {
                    "shape_exprs": shape_exprs,
                    "dtype": input_val.dtype,
                    "name": input_node.name,
                }
            )
        elif isinstance(input_val, (torch.SymInt, torch.SymFloat, int, float, bool)):
            if isinstance(input_val, (torch.SymInt, int)):
                scalar_dtype = torch.int64
            elif isinstance(input_val, (torch.SymFloat, float)):
                scalar_dtype = torch.float64
            else:
                scalar_dtype = torch.bool
            input_info.append(
                {
                    "shape_exprs": [],
                    "dtype": scalar_dtype,
                    "name": input_node.name,
                    "is_scalar": True,
                }
            )
        else:
            logger.warning(
                f"When processing symbolic shapes for TensorRT engine, unsupported input type: {type(input_val)}"
            )
            return None

    # Extract output values from output node
    output_args = output_node.args[0]
    if not isinstance(output_args, (tuple, list)):
        output_args = (output_args,)

    # Collect shape expressions and dtypes for each output
    output_info = []
    for out_arg in output_args:
        out_val = _resolve_meta_val(out_arg)
        if out_val is None:
            logger.warning(
                "When processing symbolic shapes for TensorRT engine, found no metadata in FX Graph"
            )
            return None

        if isinstance(out_val, torch.Tensor):
            shape_exprs = []
            for dim_size in out_val.shape:
                if isinstance(dim_size, torch.SymInt):
                    shape_exprs.append(dim_size.node.expr)
                else:
                    shape_exprs.append(int(dim_size))

            output_info.append(
                {
                    "shape_exprs": shape_exprs,
                    "dtype": out_val.dtype,
                }
            )
        elif isinstance(out_val, (torch.SymInt, torch.SymFloat, int, float, bool)):
            if isinstance(out_val, (torch.SymInt, int)):
                scalar_dtype = torch.int64
            elif isinstance(out_val, (torch.SymFloat, float)):
                scalar_dtype = torch.float64
            else:
                scalar_dtype = torch.bool
            output_info.append(
                {
                    "shape_exprs": [],
                    "dtype": scalar_dtype,
                    "is_scalar": True,
                }
            )
        else:
            logger.warning(
                f"When processing symbolic shapes for TensorRT engine, unsupported output type: {type(out_val)}"
            )
            return None

    if not output_info:
        return None

    return {
        "inputs": input_info,
        "outputs": output_info,
    }
