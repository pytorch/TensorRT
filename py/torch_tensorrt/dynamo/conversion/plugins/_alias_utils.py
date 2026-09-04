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


def detect_and_validate_output_aliases(
    op: Any, outputs: Sequence[Any], tensor_inputs: Sequence[Any]
) -> Dict[int, int]:
    """Detect aliases and require every mutated tensor input to be represented.

    PyTorch custom-op schemas cannot declare the real kernel's output as aliasing a
    mutated input. Generated QDP plugins therefore use the fake kernel's object
    identity as the explicit alias signal: the fake kernel must return the exact
    mutated fake tensor, while the real kernel returns a non-aliasing tensor.
    """
    schema = op._schema
    mutated = mutated_tensor_indices(schema)
    alias_map: Dict[int, int] = {}
    for output_index, output in enumerate(outputs):
        for tensor_index in mutated:
            if (
                tensor_index < len(tensor_inputs)
                and output is tensor_inputs[tensor_index]
            ):
                alias_map[output_index] = tensor_index
                break

    missing = mutated.difference(alias_map.values())
    if missing:
        positions = tensor_positions(schema)
        names = [schema.arguments[positions[index]].name for index in sorted(missing)]
        raise RuntimeError(
            f"In-place QDP plugin {op} mutates tensor argument(s) "
            f"{', '.join(names)}, but its fake implementation did not return the "
            "same FakeTensor object for each mutated argument. Return each mutated "
            "argument by identity from the fake kernel (the real kernel should "
            "continue returning non-aliasing tensors)."
        )
    return alias_map
