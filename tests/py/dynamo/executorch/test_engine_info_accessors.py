import base64

import pytest

executorch = pytest.importorskip("executorch.exir")

import torch  # noqa: E402
from torch_tensorrt.dynamo.runtime._serialized_engine_layout import (  # noqa: E402
    ENGINE_IDX,
    SERIALIZATION_LEN,
)
from torch_tensorrt.executorch import _export_utils  # noqa: E402
from torch_tensorrt.executorch._export_utils import (  # noqa: E402
    get_engine_info_from_state,
)

ENGINE_BYTES = b"ENGINEBYTES"


def _record(engine_value):
    record = [""] * SERIALIZATION_LEN
    record[ENGINE_IDX] = engine_value
    return record


class RuntimeWithAccessor:
    """Engine from a runtime that provides ``serialize_metadata_only``."""

    def __init__(self):
        self.metadata_calls = 0
        self.getstate_calls = 0

    def serialize_metadata_only(self):
        self.metadata_calls += 1
        return _record("")

    def __getstate__(self):
        self.getstate_calls += 1
        return (_record("ENGINEBYTES"),)


class RuntimeWithoutAccessor:
    """Engine from a C++ library built before ``serialize_metadata_only`` existed."""

    def __init__(self):
        self.getstate_calls = 0

    def __getstate__(self):
        self.getstate_calls += 1
        return (_record("ENGINEBYTES"),)


class RuntimeWithoutTensorAccessor:
    """Engine with the metadata accessor but not the tensor one.

    Both accessors were added together, so this is not a build that exists; it is
    the shape that forces ``_resolve_engine_tensor`` down its fallback branch,
    where the engine bytes come out of a cached record rather than the accessor.
    """

    def __init__(self):
        self.metadata_calls = 0
        self.getstate_calls = 0

    def serialize_metadata_only(self):
        self.metadata_calls += 1
        return _record("")

    def __getstate__(self):
        self.getstate_calls += 1
        return (_record(base64.b64encode(ENGINE_BYTES).decode()),)


class _Program:
    """Stand-in for ExportedProgram carrying what the two passes touch."""

    def __init__(self, graph_module):
        self.graph_module = graph_module
        self.constants = {}
        self.state_dict = {}


def _program_with_engine(engine):
    """A single-``execute_engine``-node program holding ``engine`` by get_attr."""
    graph = torch.fx.Graph()
    root = torch.nn.Module()
    root.engine = engine
    engine_node = graph.get_attr("engine")
    input_node = graph.placeholder("x")
    node = graph.call_function(
        torch.ops.tensorrt.execute_engine.default, ([input_node], engine_node)
    )
    graph.output((node,))
    return _Program(torch.fx.GraphModule(root, graph)), node


@pytest.mark.unit
def test_metadata_only_skips_engine_serialization():
    engine = RuntimeWithAccessor()
    record = get_engine_info_from_state(engine, metadata_only=True)

    assert len(record) == SERIALIZATION_LEN, (
        "metadata-only record must match the full record's length, or the "
        "*_IDX constants address the wrong slots"
    )
    assert record[ENGINE_IDX] == "", "metadata-only record must not carry the engine"
    assert engine.metadata_calls == 1
    assert engine.getstate_calls == 0, "__getstate__ re-serializes; it must not run"


@pytest.mark.unit
def test_falls_back_when_runtime_lacks_the_accessor():
    """A mixed build (new Python, older C++) must stay correct, only slower."""
    engine = RuntimeWithoutAccessor()
    record = get_engine_info_from_state(engine, metadata_only=True)

    assert engine.getstate_calls == 1
    assert record[ENGINE_IDX] == "ENGINEBYTES", "fallback must return the real record"
    assert len(record) == SERIALIZATION_LEN


@pytest.mark.unit
def test_fallback_and_accessor_agree_outside_the_engine_slot():
    """The paths must differ only in ENGINE_IDX, or readers see different metadata."""
    via_accessor = get_engine_info_from_state(RuntimeWithAccessor(), metadata_only=True)
    via_fallback = get_engine_info_from_state(
        RuntimeWithoutAccessor(), metadata_only=True
    )

    for index in range(SERIALIZATION_LEN):
        if index == ENGINE_IDX:
            continue
        assert via_accessor[index] == via_fallback[index], f"slot {index} differs"


@pytest.mark.unit
def test_missing_accessor_warns_once(caplog):
    caplog.set_level("WARNING")
    _export_utils._WARNED_MISSING.clear()

    get_engine_info_from_state(RuntimeWithoutAccessor(), metadata_only=True)
    get_engine_info_from_state(RuntimeWithoutAccessor(), metadata_only=True)

    warnings = [r for r in caplog.records if "serialize_metadata_only" in r.message]
    assert len(warnings) == 1, "a silently slower fallback should announce itself once"


@pytest.mark.unit
def test_metadata_only_record_is_not_cross_served_as_engine_bytes():
    """The rewrite must not source engine bytes from validation's cached record.

    Export always validates before it rewrites, and validation reads metadata
    only, so the record ``validate_engine_program`` leaves in ``resolved`` for this
    engine has an empty ENGINE_IDX by the time the rewrite runs. The rewrite reuses
    that record for the plain-string slots but must take the engine itself from
    ``_resolve_engine_tensor``, which re-resolves the engine rather than reading a
    cached ENGINE_IDX. Serving the metadata-only record as the engine yields a
    zero-length buffer and a .pte that fails at deserialize rather than here.
    """
    engine = RuntimeWithoutTensorAccessor()
    program, node = _program_with_engine(engine)

    resolved: dict = {}
    assert _export_utils.validate_engine_program(program, resolved) == 1
    assert engine.metadata_calls == 1
    assert engine.getstate_calls == 0
    assert resolved[node.name][ENGINE_IDX] == "", (
        "validation must cache a metadata-only record, or this test does not "
        "exercise the cross-serving hazard"
    )

    _export_utils.replace_execute_engine(program, resolved)

    no_op = torch.ops.tensorrt.no_op_placeholder_for_execute_engine.default
    no_op_nodes = [n for n in program.graph_module.graph.nodes if n.target is no_op]
    assert len(no_op_nodes) == 1
    buffer_node = no_op_nodes[0].args[1 + ENGINE_IDX]
    engine_buffer = getattr(program.graph_module, buffer_node.target)

    assert engine_buffer.numpy().tobytes() == ENGINE_BYTES, (
        "the rewrite wrote a zero-length engine: it sourced bytes from the "
        "metadata-only record cached by validation rather than re-resolving them"
    )
    assert engine.getstate_calls == 1, "the engine record still must be read once"
