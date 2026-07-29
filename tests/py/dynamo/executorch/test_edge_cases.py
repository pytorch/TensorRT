import os
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from torch_tensorrt._compile import (
    _count_executorch_engine_nodes,
    _validate_executorch_engine_info,
    _write_external_tensor_data,
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


@pytest.mark.unit
def test_write_external_tensor_data_writes_when_present(tmp_path):
    # A program with external named data (e.g. a CudaPartitioner delegate's
    # weights) must have its .ptd written into the .pte's directory.
    prog = SimpleNamespace(
        _tensor_data={"forward": b"weights"},
        write_tensor_data_to_file=MagicMock(),
    )
    pte = tmp_path / "model.pte"
    _write_external_tensor_data(prog, str(pte))
    prog.write_tensor_data_to_file.assert_called_once_with(
        os.path.dirname(os.path.abspath(str(pte)))
    )


@pytest.mark.unit
def test_write_external_tensor_data_noop_when_empty(tmp_path):
    # TRT-only programs have empty _tensor_data (falsy) -> no .ptd written.
    prog = SimpleNamespace(
        _tensor_data={},
        write_tensor_data_to_file=MagicMock(),
    )
    _write_external_tensor_data(prog, str(tmp_path / "model.pte"))
    prog.write_tensor_data_to_file.assert_not_called()


@pytest.mark.unit
def test_write_external_tensor_data_fails_loud_without_attr(tmp_path):
    # _tensor_data always exists on a real ExecutorchProgram; it is accessed
    # directly (no getattr default) so a future rename fails loudly instead of
    # silently skipping the .ptd write and reintroducing the null-weights crash.
    prog = SimpleNamespace(write_tensor_data_to_file=MagicMock())
    with pytest.raises(AttributeError):
        _write_external_tensor_data(prog, str(tmp_path / "model.pte"))
