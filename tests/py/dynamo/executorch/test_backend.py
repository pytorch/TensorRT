import base64
from types import SimpleNamespace

import pytest

executorch = pytest.importorskip("executorch")

from torch_tensorrt.dynamo.runtime._TorchTensorRTModule import (  # noqa: E402
    DEVICE_IDX,
    ENGINE_IDX,
    INPUT_BINDING_NAMES_IDX,
    OUTPUT_BINDING_NAMES_IDX,
    REQUIRES_OUTPUT_ALLOCATOR_IDX,
    SERIALIZATION_LEN,
)
from torch_tensorrt.executorch.backend import (  # noqa: E402
    TensorRTBackend,
    _get_engine_info_from_edge_program,
)
from torch_tensorrt.executorch.serialization import (  # noqa: E402
    TENSORRT_MAGIC,
    deserialize_engine,
)


class _SchemaTarget:
    def __init__(self, name):
        self._schema = SimpleNamespace(name=name)


def _make_placeholder_node(*engine_info):
    return SimpleNamespace(
        op="call_function",
        target=_SchemaTarget("tensorrt::no_op_placeholder_for_execute_engine"),
        args=(["x"], *engine_info),
        name="trt_node",
    )


def _make_edge_program(*nodes):
    return SimpleNamespace(
        graph_module=SimpleNamespace(graph=SimpleNamespace(nodes=list(nodes))),
        constants={},
    )


@pytest.mark.unit
def test_get_engine_info_rejects_multiple_engine_nodes():
    engine_info = [""] * SERIALIZATION_LEN
    edge_program = _make_edge_program(
        _make_placeholder_node(*engine_info),
        _make_placeholder_node(*engine_info),
    )

    with pytest.raises(RuntimeError, match="exactly 1 engine node"):
        _get_engine_info_from_edge_program(edge_program)


@pytest.mark.unit
def test_preprocess_rejects_output_allocator():
    engine_info = [""] * SERIALIZATION_LEN
    engine_info[ENGINE_IDX] = base64.b64encode(b"engine").decode("utf-8")
    engine_info[REQUIRES_OUTPUT_ALLOCATOR_IDX] = "1"
    edge_program = _make_edge_program(_make_placeholder_node(*engine_info))

    with pytest.raises(RuntimeError, match="output allocator"):
        TensorRTBackend.preprocess(edge_program, [])


@pytest.mark.unit
def test_preprocess_serializes_engine_blob():
    engine_info = [""] * SERIALIZATION_LEN
    engine_info[ENGINE_IDX] = base64.b64encode(b"engine-bytes").decode("utf-8")
    engine_info[DEVICE_IDX] = "2%8%0%0%GPU"
    engine_info[INPUT_BINDING_NAMES_IDX] = "x"
    engine_info[OUTPUT_BINDING_NAMES_IDX] = "y"
    edge_program = _make_edge_program(_make_placeholder_node(*engine_info))

    result = TensorRTBackend.preprocess(edge_program, [])

    assert isinstance(result.processed_bytes, bytes)
    assert result.processed_bytes[:4] == TENSORRT_MAGIC
    engine, metadata = deserialize_engine(result.processed_bytes)
    assert engine == b"engine-bytes"
    assert metadata.device_id == 2
    assert [binding.name for binding in metadata.io_bindings] == ["x", "y"]
    assert [binding.is_input for binding in metadata.io_bindings] == [True, False]
