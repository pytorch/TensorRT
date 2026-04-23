from types import SimpleNamespace

import pytest
import torch
from torch_tensorrt._compile import (
    _count_executorch_engine_nodes,
    _validate_executorch_engine_info,
)
from torch_tensorrt.dynamo.runtime._TorchTensorRTModule import (
    REQUIRES_OUTPUT_ALLOCATOR_IDX,
    SERIALIZATION_LEN,
)


class _SchemaTarget:
    def __init__(self, name):
        self._schema = SimpleNamespace(name=name)


@pytest.mark.unit
def test_validate_executorch_engine_info_rejects_output_allocator():
    engine_info = [""] * SERIALIZATION_LEN
    engine_info[REQUIRES_OUTPUT_ALLOCATOR_IDX] = "1"

    with pytest.raises(RuntimeError, match="output allocator"):
        _validate_executorch_engine_info(engine_info, node_name="trt")


@pytest.mark.unit
def test_count_executorch_engine_nodes_handles_execute_and_placeholder():
    execute_node = SimpleNamespace(
        op="call_function",
        target=torch.ops.tensorrt.execute_engine.default,
    )
    placeholder_node = SimpleNamespace(
        op="call_function",
        target=_SchemaTarget("tensorrt::no_op_placeholder_for_execute_engine"),
    )
    other_node = SimpleNamespace(
        op="call_function",
        target=_SchemaTarget("aten::add"),
    )
    exp_program = SimpleNamespace(
        graph_module=SimpleNamespace(
            graph=SimpleNamespace(nodes=[execute_node, placeholder_node, other_node])
        )
    )

    assert _count_executorch_engine_nodes(exp_program) == 2
