"""Schema + fake-run alias detection shared by plugin generation
(``_generate_plugin``), the plugin converter (``_generate_plugin_converter``),
and the ``auto_functionalized`` wrapper converter."""

from typing import Any, Dict, List, Sequence, Set

import torch


def is_tensor_arg(arg: Any) -> bool:
    return bool(arg.type.isSubtypeOf(torch._C.TensorType.get()))


def tensor_positions(schema: Any) -> List[int]:
    """Schema positions of tensor-typed args (== plugin input order)."""
    return [i for i, a in enumerate(schema.arguments) if is_tensor_arg(a)]


def mutated_tensor_indices(schema: Any) -> Set[int]:
    """Indices among tensor args that the schema marks mutated (``Tensor(a!)``)."""
    return {
        t
        for t, i in enumerate(tensor_positions(schema))
        if (a := schema.arguments[i]).alias_info is not None and a.alias_info.is_write
    }


def detect_output_aliases(
    outputs: Sequence[Any], tensor_inputs: Sequence[Any], mutated: Set[int]
) -> Dict[int, int]:
    """``{out_idx: tensor_arg_idx}`` from a fake/meta run: an output aliases a
    mutated input iff the schema marks the input ``is_write`` AND the run
    returns it by identity (the schema gate avoids false positives from
    incidental identity returns of non-mutating ops)."""
    alias_map: Dict[int, int] = {}
    for out_idx, out in enumerate(outputs):
        for t in mutated:
            if t < len(tensor_inputs) and out is tensor_inputs[t]:
                alias_map[out_idx] = t
                break
    return alias_map
