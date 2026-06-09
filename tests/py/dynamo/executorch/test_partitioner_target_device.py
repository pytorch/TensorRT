from types import SimpleNamespace

import pytest

executorch = pytest.importorskip("executorch.exir")

import torch  # noqa: E402
from executorch.exir.backend.compile_spec_schema import CompileSpec  # noqa: E402
from torch_tensorrt.dynamo.runtime._TorchTensorRTModule import (  # noqa: E402
    DEVICE_IDX,
    ENGINE_IDX,
    SERIALIZATION_LEN,
)
from torch_tensorrt.executorch.partitioner import (  # noqa: E402
    _TARGET_DEVICE_COMPILE_SPEC_KEY,
    TensorRTPartitioner,
)


# A realistic single-engine edge program so partition() runs the *real*
# _get_engine_info_from_edge_program / _parse_device_id path. That is what
# guards "an engine node is present and its device is extractable at partition()
# time" -- a monkeypatched extractor would not. Mirrors the mocked edge programs
# in tests/py/dynamo/executorch/test_backend.py.
class _SchemaTarget:
    def __init__(self, name):
        self._schema = SimpleNamespace(name=name)


def _engine_node(device_id):
    engine_info = [""] * SERIALIZATION_LEN
    engine_info[ENGINE_IDX] = torch.frombuffer(bytearray(b"engine"), dtype=torch.uint8)
    engine_info[DEVICE_IDX] = device_id
    return SimpleNamespace(
        op="call_function",
        target=_SchemaTarget("tensorrt::no_op_placeholder_for_execute_engine"),
        args=(["x"], *engine_info),
        name="trt_node",
    )


def _edge_program(*nodes):
    return SimpleNamespace(
        graph_module=SimpleNamespace(graph=SimpleNamespace(nodes=list(nodes))),
        constants={},
    )


class _FakeCapabilityPartitioner:
    def __init__(self, *args, **kwargs):
        pass

    def propose_partitions(self):
        return [SimpleNamespace(id=1, nodes=[SimpleNamespace(meta={})])]


@pytest.fixture(autouse=True)
def _stub_partition_internals(monkeypatch):
    # Both need a real fx GraphModule, so stub them out -- the engine-info
    # extraction under test still runs for real against the mocked node.
    monkeypatch.setattr(
        "torch_tensorrt.executorch.partitioner.CapabilityBasedPartitioner",
        _FakeCapabilityPartitioner,
    )
    monkeypatch.setattr(
        "torch_tensorrt.executorch.partitioner.tag_constant_data",
        lambda exported_program: None,
    )


def _target_device(result):
    spec = result.partition_tags["tensorrt_1"]
    for cs in spec.compile_specs:
        if cs.key == _TARGET_DEVICE_COMPILE_SPEC_KEY:
            return cs.value
    return None


@pytest.mark.unit
def test_target_device_derived_for_default_gpu():
    result = TensorRTPartitioner().partition(_edge_program(_engine_node("0")))
    assert _target_device(result) == b"cuda:0"


@pytest.mark.unit
def test_target_device_derived_for_nonzero_gpu():
    # The bug this fixes: a cuda:1 engine must not be mislabeled cuda:0.
    result = TensorRTPartitioner().partition(_edge_program(_engine_node("1")))
    assert _target_device(result) == b"cuda:1"


@pytest.mark.unit
def test_target_device_falls_back_to_cuda0_on_multiple_engines():
    # >1 engine node -> real extraction raises -> contract fallback to cuda:0.
    result = TensorRTPartitioner().partition(
        _edge_program(_engine_node("1"), _engine_node("2"))
    )
    assert _target_device(result) == b"cuda:0"


@pytest.mark.unit
def test_target_device_falls_back_to_cuda0_on_malformed_graph():
    # An unexpected graph shape makes the real extraction raise; the broadened
    # except must still fall back to cuda:0 rather than abort the export.
    bad_node = SimpleNamespace(op="call_function", target=SimpleNamespace(), name="x")
    result = TensorRTPartitioner().partition(_edge_program(bad_node))
    assert _target_device(result) == b"cuda:0"


@pytest.mark.unit
def test_explicit_target_device_used_verbatim():
    # Engine reports cuda:0, but the caller pinned cuda:3 -> the pin wins and
    # extraction is skipped entirely.
    partitioner = TensorRTPartitioner(
        compile_specs=[CompileSpec(_TARGET_DEVICE_COMPILE_SPEC_KEY, b"cuda:3")]
    )
    result = partitioner.partition(_edge_program(_engine_node("0")))
    assert _target_device(result) == b"cuda:3"
