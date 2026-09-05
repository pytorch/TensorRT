"""
Capture symbolic shape expressions from FX graphs for TRT meta kernel.

This module extracts the symbolic relationship between input and output shapes
at compile time, which can then be used by the meta kernel to correctly infer
output shapes without pattern matching.
"""

import logging
from typing import Any, Dict, List, Optional, Sequence

import torch
from torch_tensorrt._enums import dtype as _dtype
from torch_tensorrt._Input import Input

logger = logging.getLogger(__name__)


def extract_symbolic_shape_expressions(
    module: torch.fx.GraphModule,
    inputs: Optional[Sequence[Input]] = None,
    truncate_double: bool = False,
) -> Optional[Dict[str, List[Dict[str, Any]]]]:
    """
    Extract symbolic shape expressions from an FX graph.

    This captures the symbolic expressions (as sympy expressions) for input and output shapes
    that can be applied to input fake tensors at runtime.

    Args:
        module: FX GraphModule with symbolic shapes in node metadata
        inputs: Engine Input specs from construct_submodule_inputs, aligned by
            name. Used as the dtype source of truth for scalar inputs, which
            have no dtype of their own in FX metadata. Falls back to a
            best-effort default when not provided.
        truncate_double: Record float64 tensor bindings as float32, matching
            the precision TensorRT builds when double truncation is enabled

    Returns:
        Dict with 'inputs' and 'outputs' keys, each containing a list of dicts with shape_exprs and dtype,
        or None if extraction fails
    """
    # dtype.unknown has no torch.dtype equivalent; skip it rather than raise
    # (unset-dtype inputs are never looked up below anyway).
    input_dtypes_by_name = {
        inp.name: inp.dtype.to(torch.dtype)
        for inp in inputs or ()
        if inp.dtype != _dtype.unknown
    }

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
                    "dtype": (
                        torch.float32
                        if truncate_double and input_val.dtype == torch.float64
                        else input_val.dtype
                    ),
                    "name": input_node.name,
                }
            )
        elif isinstance(input_val, (torch.SymInt, torch.SymFloat, int, float, bool)):
            if isinstance(input_val, (torch.SymInt, int)):
                default_scalar_dtype = torch.int64
            elif isinstance(input_val, (torch.SymFloat, float)):
                default_scalar_dtype = torch.float32
            else:
                default_scalar_dtype = torch.bool
            # Prefer the engine's actual binding dtype over the guess above.
            scalar_dtype = input_dtypes_by_name.get(
                input_node.name, default_scalar_dtype
            )
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
        if not hasattr(out_arg, "meta") or "val" not in out_arg.meta:
            logger.warning(
                "When processing symbolic shapes for TensorRT engine, found no metadata in FX Graph"
            )
            return None

        out_val = out_arg.meta["val"]
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
                    "dtype": (
                        torch.float32
                        if truncate_double and out_val.dtype == torch.float64
                        else out_val.dtype
                    ),
                }
            )
        elif isinstance(out_val, (torch.SymInt, torch.SymFloat, int, float, bool)):
            if isinstance(out_val, (torch.SymInt, int)):
                scalar_dtype = torch.int64
            elif isinstance(out_val, (torch.SymFloat, float)):
                # No float64 output binding exists in TensorRT.
                scalar_dtype = torch.float32
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
