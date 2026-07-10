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


# A realistic engine node so partition() runs the *real* per-partition
# _get_engine_info_for_node / _parse_device_id path. target_device is now
# derived from each partition's OWN engine node (so a coalesced multi-engine
# graph labels each delegate with its correct GPU), which means the engine node
# must live in the partition's node list -- a monkeypatched extractor would not
# guard that. Mirrors the mocked edge programs in
# tests/py/dynamo/executorch/test_backend.py.
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
        meta={},
    )


def _edge_program(*nodes):
    return SimpleNamespace(
        graph_module=SimpleNamespace(graph=SimpleNamespace(nodes=list(nodes))),
        constants={},
    )


class _FakeCapabilityPartitioner:
    # One partition per graph node -- each TRT engine node becomes its own
    # partition, as the real CapabilityBasedPartitioner does here -- so the
    # per-partition device resolution under test runs against the engine node
    # that actually belongs to each partition.
    def __init__(self, graph_module, *args, **kwargs):
        self._graph_module = graph_module

    def propose_partitions(self):
        return [
            SimpleNamespace(id=i, nodes=[node])
            for i, node in enumerate(self._graph_module.graph.nodes)
        ]


@pytest.fixture(autouse=True)
def _stub_partition_internals(monkeypatch):
    # The partition proposal needs a real fx GraphModule, so stub it out -- the
    # per-partition engine-info extraction under test still runs for real
    # against the mocked engine nodes.
    monkeypatch.setattr(
        "torch_tensorrt.executorch.partitioner.CapabilityBasedPartitioner",
        _FakeCapabilityPartitioner,
    )
    monkeypatch.setattr(
        "torch_tensorrt.executorch.partitioner.tag_constant_data",
        lambda exported_program: None,
    )


def _target_device(result, tag="tensorrt_0"):
    spec = result.partition_tags[tag]
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
    # A cuda:1 engine must not be mislabeled cuda:0.
    result = TensorRTPartitioner().partition(_edge_program(_engine_node("1")))
    assert _target_device(result) == b"cuda:1"


@pytest.mark.unit
def test_per_partition_devices_for_coalesced_multi_engine():
    # The fix in this PR: each partition is labeled from its OWN engine's device,
    # so a coalesced multi-engine graph is not stamped with a single
    # whole-program device.
    result = TensorRTPartitioner().partition(
        _edge_program(_engine_node("0"), _engine_node("1"))
    )
    assert _target_device(result, "tensorrt_0") == b"cuda:0"
    assert _target_device(result, "tensorrt_1") == b"cuda:1"


@pytest.mark.unit
def test_target_device_falls_back_to_cuda0_on_malformed_partition():
    # A partition whose nodes carry no extractable engine makes the real
    # extraction raise; the broadened except must fall back to cuda:0 rather
    # than abort the export.
    bad_node = SimpleNamespace(
        op="call_function", target=SimpleNamespace(), name="x", meta={}
    )
    result = TensorRTPartitioner().partition(_edge_program(bad_node))
    assert _target_device(result) == b"cuda:0"


@pytest.mark.unit
def test_target_device_falls_back_to_cuda0_when_partition_has_multiple_engines(
    monkeypatch,
):
    # A single partition holding >1 engine node is ambiguous, so device
    # resolution must fall back to cuda:0 rather than guess.
    monkeypatch.setattr(
        "torch_tensorrt.executorch.partitioner.CapabilityBasedPartitioner",
        lambda *args, **kwargs: SimpleNamespace(
            propose_partitions=lambda: [
                SimpleNamespace(id=0, nodes=[_engine_node("1"), _engine_node("2")])
            ]
        ),
    )
    result = TensorRTPartitioner().partition(
        _edge_program(_engine_node("1"), _engine_node("2"))
    )
    assert _target_device(result) == b"cuda:0"


@pytest.mark.unit
def test_explicit_target_device_used_verbatim():
    # Engine reports cuda:0, but the caller pinned cuda:3 -> the pin wins and
    # per-partition extraction is skipped entirely.
    partitioner = TensorRTPartitioner(
        compile_specs=[CompileSpec(_TARGET_DEVICE_COMPILE_SPEC_KEY, b"cuda:3")]
    )
    result = partitioner.partition(_edge_program(_engine_node("0")))
    assert _target_device(result) == b"cuda:3"
