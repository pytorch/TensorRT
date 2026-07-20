from types import SimpleNamespace

import pytest

executorch = pytest.importorskip("executorch.exir")

import torch  # noqa: E402
from torch.export.graph_signature import InputKind  # noqa: E402
from torch_tensorrt.dynamo.runtime._TorchTensorRTModule import (  # noqa: E402
    DEVICE_IDX,
    ENGINE_IDX,
    INPUT_BINDING_NAMES_IDX,
    OUTPUT_BINDING_NAMES_IDX,
    REQUIRES_OUTPUT_ALLOCATOR_IDX,
    SERIALIZATION_LEN,
)
from torch_tensorrt.executorch.backend import (  # noqa: E402
    _get_engine_info_from_edge_program,
    TensorRTBackend,
)
from torch_tensorrt.executorch.serialization import (  # noqa: E402
    deserialize_engine,
    TENSORRT_MAGIC,
)


class _SchemaTarget:
    """Callable stand-in for a tensorrt:: op overload carrying a `_schema`.

    fx `call_function` requires a callable target; the backend only reads
    `target._schema.name`, so a no-op callable with the schema attribute is
    enough to build a realistic engine node in a real fx graph.
    """

    def __init__(self, name):
        self._schema = SimpleNamespace(name=name)
        self.__name__ = name.replace("::", "_")

    def __call__(self, *args, **kwargs):  # pragma: no cover - never executed
        raise AssertionError("engine op target is not meant to be called")


_ENGINE_OP = _SchemaTarget("tensorrt::no_op_placeholder_for_execute_engine")


def _build_edge_program(
    engine_info,
    input_names=("x",),
    binding_order=None,
):
    """Construct a real fx-graph-backed edge program for the backend.

    The engine node's first arg is the ordered list of input placeholder NODES
    (mirroring torch_tensorrt's inline_trt_modules), in `binding_order`, i.e. the
    TRT binding order. The graph placeholders themselves are laid out in
    `input_names` order, i.e. the fusion-arranged runtime arg order. The backend
    must recover the permutation from node identity, not from names.

    A bare `torch.fx.Graph` (not a GraphModule) is used so the nodes carry real
    fx identity (which is what the reorder logic keys on) without triggering
    GraphModule codegen of the fake engine op / raw engine tensor.
    """
    if binding_order is None:
        binding_order = tuple(input_names)

    graph = torch.fx.Graph()
    placeholders = {name: graph.placeholder(name) for name in input_names}
    engine_inputs = [placeholders[name] for name in binding_order]
    engine_node = graph.call_function(
        _ENGINE_OP,
        (engine_inputs, *engine_info),
    )
    graph.output((engine_node,))

    graph_signature = SimpleNamespace(
        input_specs=[
            SimpleNamespace(kind=InputKind.USER_INPUT, arg=SimpleNamespace(name=name))
            for name in input_names
        ]
    )
    return SimpleNamespace(
        graph_module=SimpleNamespace(graph=graph),
        graph_signature=graph_signature,
        constants={},
    )


def _make_multi_engine_program(engine_info):
    """An edge program whose graph contains two engine nodes (invalid)."""
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    n1 = graph.call_function(_ENGINE_OP, ([x], *engine_info))
    n2 = graph.call_function(_ENGINE_OP, ([x], *engine_info))
    graph.output((n1, n2))
    return SimpleNamespace(
        graph_module=SimpleNamespace(graph=graph),
        graph_signature=SimpleNamespace(
            input_specs=[
                SimpleNamespace(
                    kind=InputKind.USER_INPUT, arg=SimpleNamespace(name="x")
                )
            ]
        ),
        constants={},
    )


def _engine_tensor(payload: bytes) -> torch.Tensor:
    return torch.frombuffer(bytearray(payload), dtype=torch.uint8)


@pytest.mark.unit
def test_get_engine_info_rejects_multiple_engine_nodes():
    engine_info = [""] * SERIALIZATION_LEN
    engine_info[ENGINE_IDX] = _engine_tensor(b"engine")
    edge_program = _make_multi_engine_program(engine_info)

    with pytest.raises(RuntimeError, match="exactly 1 engine node"):
        _get_engine_info_from_edge_program(edge_program)


@pytest.mark.unit
def test_preprocess_rejects_output_allocator():
    engine_info = [""] * SERIALIZATION_LEN
    engine_info[ENGINE_IDX] = _engine_tensor(b"engine")
    engine_info[REQUIRES_OUTPUT_ALLOCATOR_IDX] = "1"
    edge_program = _build_edge_program(engine_info)

    with pytest.raises(RuntimeError, match="output allocator"):
        TensorRTBackend.preprocess(edge_program, [])


@pytest.mark.unit
def test_preprocess_serializes_engine_blob():
    engine_info = [""] * SERIALIZATION_LEN
    engine_info[ENGINE_IDX] = _engine_tensor(b"engine-bytes")
    engine_info[DEVICE_IDX] = "2%8%0%0%GPU"
    engine_info[INPUT_BINDING_NAMES_IDX] = "x"
    engine_info[OUTPUT_BINDING_NAMES_IDX] = "y"
    edge_program = _build_edge_program(engine_info)

    result = TensorRTBackend.preprocess(edge_program, [])

    assert isinstance(result.processed_bytes, bytes)
    assert result.processed_bytes[:4] == TENSORRT_MAGIC
    engine, metadata = deserialize_engine(result.processed_bytes)
    assert engine == b"engine-bytes"
    assert metadata.device_id == 2
    assert [binding.name for binding in metadata.io_bindings] == ["x", "y"]
    assert [binding.is_input for binding in metadata.io_bindings] == [True, False]


@pytest.mark.unit
def test_preprocess_single_input_is_identity():
    # Single-input engines have zero ordering ambiguity: the one binding maps to
    # the one placeholder regardless of name (TRT name may be semantic, the
    # runtime placeholder generic). Must never reorder or reject.
    engine_info = [""] * SERIALIZATION_LEN
    engine_info[ENGINE_IDX] = _engine_tensor(b"engine-bytes")
    engine_info[INPUT_BINDING_NAMES_IDX] = "pixel_values"
    engine_info[OUTPUT_BINDING_NAMES_IDX] = "out"
    # Graph placeholder is generic ("arg_0"); the binding name differs entirely.
    edge_program = _build_edge_program(engine_info, input_names=("arg_0",))

    _, metadata = deserialize_engine(
        TensorRTBackend.preprocess(edge_program, []).processed_bytes
    )
    assert [binding.name for binding in metadata.io_bindings] == [
        "pixel_values",
        "out",
    ]


@pytest.mark.unit
def test_preprocess_reorders_by_node_identity_not_name():
    # TRT binding order is [second, first]; the engine node feeds those input
    # placeholders in that order. The graph lays the placeholders out as
    # [first, second] (the runtime delegate arg order). The backend must emit
    # the binding names permuted to runtime order: [first, second].
    engine_info = [""] * SERIALIZATION_LEN
    engine_info[ENGINE_IDX] = _engine_tensor(b"engine-bytes")
    engine_info[INPUT_BINDING_NAMES_IDX] = "second%first"
    engine_info[OUTPUT_BINDING_NAMES_IDX] = "out"
    edge_program = _build_edge_program(
        engine_info,
        input_names=("first", "second"),
        binding_order=("second", "first"),
    )

    _, metadata = deserialize_engine(
        TensorRTBackend.preprocess(edge_program, []).processed_bytes
    )
    assert [binding.name for binding in metadata.io_bindings] == [
        "first",
        "second",
        "out",
    ]
    assert [binding.is_input for binding in metadata.io_bindings] == [
        True,
        True,
        False,
    ]


@pytest.mark.unit
def test_preprocess_generic_names_multi_input_permutation():
    # Realistic lowered case: TRT names are semantic, runtime placeholders are
    # generic arg_N and DO NOT match the TRT names. Reorder must still work via
    # node identity. Binding order [timesteps, prefix, x]; graph placeholder
    # (runtime) order is [arg_0=x, arg_1=timesteps, arg_2=prefix].
    engine_info = [""] * SERIALIZATION_LEN
    engine_info[ENGINE_IDX] = _engine_tensor(b"engine-bytes")
    engine_info[INPUT_BINDING_NAMES_IDX] = "timesteps%prefix%x"
    engine_info[OUTPUT_BINDING_NAMES_IDX] = "out"
    # runtime placeholder order:      arg_0, arg_1,     arg_2
    # semantic meaning of each:       x,     timesteps, prefix
    # engine feeds bindings in order: timesteps(arg_1), prefix(arg_2), x(arg_0)
    edge_program = _build_edge_program(
        engine_info,
        input_names=("arg_0", "arg_1", "arg_2"),
        binding_order=("arg_1", "arg_2", "arg_0"),
    )

    _, metadata = deserialize_engine(
        TensorRTBackend.preprocess(edge_program, []).processed_bytes
    )
    # Binding names are carried with their node; sorted by runtime slot the
    # result is [name-fed-by-arg_0, name-fed-by-arg_1, name-fed-by-arg_2] =
    # [x, timesteps, prefix].
    assert [binding.name for binding in metadata.io_bindings] == [
        "x",
        "timesteps",
        "prefix",
        "out",
    ]


@pytest.mark.unit
def test_preprocess_rejects_binding_count_mismatch():
    # Defensive invariant: number of binding names must equal number of engine
    # input nodes. Two names but one node -> refuse rather than corrupt.
    engine_info = [""] * SERIALIZATION_LEN
    engine_info[ENGINE_IDX] = _engine_tensor(b"engine-bytes")
    engine_info[INPUT_BINDING_NAMES_IDX] = "a%b"
    engine_info[OUTPUT_BINDING_NAMES_IDX] = "out"
    edge_program = _build_edge_program(engine_info, input_names=("arg_0",))

    with pytest.raises(ValueError, match="input binding names"):
        TensorRTBackend.preprocess(edge_program, [])


@pytest.mark.unit
def test_preprocess_rejects_non_placeholder_engine_input():
    # Defensive invariant: every engine input must be a delegate runtime
    # placeholder (activation input); weights are baked into the engine blob.
    # If an engine input is not a placeholder, its runtime arg slot is unknown,
    # so refuse rather than emit a mis-ordered binding.
    engine_info = [""] * SERIALIZATION_LEN
    engine_info[ENGINE_IDX] = _engine_tensor(b"engine-bytes")
    engine_info[INPUT_BINDING_NAMES_IDX] = "a%b"
    engine_info[OUTPUT_BINDING_NAMES_IDX] = "out"

    graph = torch.fx.Graph()
    p = graph.placeholder("arg_0")
    # Second engine input is a get_attr node, not a placeholder.
    attr = graph.get_attr("some_const")
    engine_node = graph.call_function(_ENGINE_OP, ([p, attr], *engine_info))
    graph.output((engine_node,))
    edge_program = SimpleNamespace(
        graph_module=SimpleNamespace(graph=graph),
        graph_signature=SimpleNamespace(
            input_specs=[
                SimpleNamespace(
                    kind=InputKind.USER_INPUT, arg=SimpleNamespace(name="arg_0")
                )
            ]
        ),
        constants={},
    )

    with pytest.raises(ValueError, match="not delegate runtime placeholders"):
        TensorRTBackend.preprocess(edge_program, [])


@pytest.mark.unit
def test_preprocess_preserves_output_binding_order():
    # Outputs are bound positionally by the runtime too, but their order is
    # stable by construction (getitem index order == engine output-binding
    # order), so preprocess must pass output names through unchanged, after the
    # (reordered) inputs.
    engine_info = [""] * SERIALIZATION_LEN
    engine_info[ENGINE_IDX] = _engine_tensor(b"engine-bytes")
    engine_info[INPUT_BINDING_NAMES_IDX] = "x"
    engine_info[OUTPUT_BINDING_NAMES_IDX] = "out0%out1%out2"
    edge_program = _build_edge_program(engine_info, input_names=("x",))

    _, metadata = deserialize_engine(
        TensorRTBackend.preprocess(edge_program, []).processed_bytes
    )
    assert [binding.name for binding in metadata.io_bindings] == [
        "x",
        "out0",
        "out1",
        "out2",
    ]
    assert [binding.is_input for binding in metadata.io_bindings] == [
        True,
        False,
        False,
        False,
    ]
