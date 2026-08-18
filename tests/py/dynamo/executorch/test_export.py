import importlib
from types import SimpleNamespace
from typing import NamedTuple
from unittest.mock import MagicMock

import pytest

pytest.importorskip("executorch.exir")

import torch  # noqa: E402
import torch_tensorrt  # noqa: E402
from torch._subclasses.fake_tensor import is_fake  # noqa: E402
from torch.export.graph_signature import (  # noqa: E402
    CustomObjArgument,
    ExportGraphSignature,
    InputKind,
    InputSpec,
    TensorArgument,
)
from torch_tensorrt.dynamo.runtime._TorchTensorRTModule import (  # noqa: E402
    ENGINE_IDX,
    REQUIRES_OUTPUT_ALLOCATOR_IDX,
    SERIALIZATION_LEN,
)


class FakeExportedProgram:
    pass


class FakeTensorRTPartitioner:
    def __init__(self, compile_specs):
        self.compile_specs = compile_specs


class _EngineState:
    def __init__(self, payload=b"engine-bytes", requires_output_allocator=False):
        self.info = [""] * SERIALIZATION_LEN
        self.info[ENGINE_IDX] = payload
        self.info[REQUIRES_OUTPUT_ALLOCATOR_IDX] = str(int(requires_output_allocator))

    def __getstate__(self):
        return (self.info,)


def _engine_program(*, lifted, execute_count=1, requires_output_allocator=False):
    graph = torch.fx.Graph()
    engine = _EngineState(requires_output_allocator=requires_output_allocator)
    if lifted:
        engine_node = graph.placeholder("obj_engine")
        root = torch.nn.Module()
        engine_target = "engine_fqn"
        input_specs = [
            InputSpec(
                kind=InputKind.CUSTOM_OBJ,
                arg=CustomObjArgument(name=engine_node.name, class_fqn=""),
                target=engine_target,
            )
        ]
        constants = {engine_target: engine}
    else:
        root = torch.nn.Module()
        root.engine = engine
        engine_node = graph.get_attr("engine")
        input_specs = []
        constants = {}

    input_node = graph.placeholder("x")
    input_specs.append(
        InputSpec(
            kind=InputKind.USER_INPUT,
            arg=TensorArgument(name=input_node.name),
            target=None,
        )
    )
    results = [
        graph.call_function(
            torch.ops.tensorrt.execute_engine.default,
            ([input_node], engine_node),
        )
        for _ in range(execute_count)
    ]
    graph.output(results[0] if len(results) == 1 else tuple(results))
    graph_module = torch.fx.GraphModule(root, graph)
    signature = ExportGraphSignature(input_specs=input_specs, output_specs=[])
    program = SimpleNamespace(
        graph_module=graph_module,
        graph_signature=signature,
        _graph_signature=signature,
        state_dict={},
        constants=constants,
    )
    return program, engine_node, input_node, engine


def _patch_lowering(monkeypatch, engine_counts=None):
    import executorch.exir
    import torch_tensorrt._features as features
    import torch_tensorrt.executorch as executorch_api
    import torch_tensorrt.executorch._export_utils as export_utils

    monkeypatch.setattr(
        features,
        "ENABLED_FEATURES",
        features.ENABLED_FEATURES._replace(torch_tensorrt_runtime=True),
    )
    export_module = importlib.import_module("torch_tensorrt.executorch.export")
    engine_counts = engine_counts or {}
    lower = MagicMock(return_value=object())
    monkeypatch.setattr(executorch.exir, "to_edge_transform_and_lower", lower)
    monkeypatch.setattr(executorch_api, "TensorRTPartitioner", FakeTensorRTPartitioner)
    monkeypatch.setattr(executorch_api, "get_edge_compile_config", lambda: "default")
    monkeypatch.setattr(export_module, "ExportedProgram", FakeExportedProgram)
    monkeypatch.setattr(
        export_utils,
        "validate_engine_program",
        lambda program, resolved=None: engine_counts.get(program, 1),
    )
    monkeypatch.setattr(export_utils, "stage_exported_program", lambda program: program)
    monkeypatch.setattr(
        export_utils,
        "replace_execute_engine",
        lambda program, resolved=None: ("rewritten", program),
    )
    return export_module, lower


@pytest.mark.unit
@pytest.mark.skipif(
    not torch_tensorrt.ENABLED_FEATURES.torch_tensorrt_runtime,
    reason="Torch-TensorRT runtime operators are not available",
)
@pytest.mark.parametrize("lifted", [False, True])
def test_validate_and_replace_execute_engine(lifted):
    export_utils = importlib.import_module("torch_tensorrt.executorch._export_utils")
    program, engine_node, input_node, _ = _engine_program(lifted=lifted)

    assert export_utils.validate_engine_program(program) == 1
    rewritten = export_utils.replace_execute_engine(program)

    assert rewritten is program
    nodes = list(program.graph_module.graph.nodes)
    assert not any(
        node.target is torch.ops.tensorrt.execute_engine.default for node in nodes
    )
    no_op_nodes = [
        node
        for node in nodes
        if node.target
        is torch.ops.tensorrt.no_op_placeholder_for_execute_engine.default
    ]
    assert len(no_op_nodes) == 1
    buffer_node = no_op_nodes[0].args[1 + ENGINE_IDX]
    engine_buffer = getattr(program.graph_module, buffer_node.target)
    assert engine_buffer.dtype == torch.uint8
    assert engine_buffer.device.type == "cpu"
    assert bytes(engine_buffer.tolist()) == b"engine-bytes"
    assert program.state_dict[buffer_node.target] is engine_buffer
    assert engine_node not in nodes
    assert input_node in nodes

    if lifted:
        assert program.graph_signature.inputs_to_lifted_custom_objs == {}
        assert program.constants == {}
        assert [spec.arg.name for spec in program.graph_signature.input_specs] == [
            input_node.name
        ]
    else:
        assert not hasattr(program.graph_module, "engine")


@pytest.mark.unit
@pytest.mark.skipif(
    not torch_tensorrt.ENABLED_FEATURES.torch_tensorrt_runtime,
    reason="Torch-TensorRT runtime operators are not available",
)
@pytest.mark.parametrize("lifted", [False, True])
def test_replace_execute_engine_cleans_shared_engine_after_last_use(lifted):
    export_utils = importlib.import_module("torch_tensorrt.executorch._export_utils")
    program, engine_node, input_node, _ = _engine_program(
        lifted=lifted, execute_count=2
    )

    assert export_utils.validate_engine_program(program) == 2
    export_utils.replace_execute_engine(program)

    nodes = list(program.graph_module.graph.nodes)
    no_op_nodes = [
        node
        for node in nodes
        if node.target
        is torch.ops.tensorrt.no_op_placeholder_for_execute_engine.default
    ]
    assert len(no_op_nodes) == 2
    buffer_nodes = {node.args[1 + ENGINE_IDX] for node in no_op_nodes}
    assert len(buffer_nodes) == 1
    assert set(program.state_dict) == {next(iter(buffer_nodes)).target}
    assert engine_node not in nodes
    assert input_node in nodes
    if lifted:
        assert program.graph_signature.inputs_to_lifted_custom_objs == {}
        assert program.constants == {}
    else:
        assert not hasattr(program.graph_module, "engine")


@pytest.mark.unit
@pytest.mark.skipif(
    not torch_tensorrt.ENABLED_FEATURES.torch_tensorrt_runtime,
    reason="Torch-TensorRT runtime operators are not available",
)
def test_validate_engine_program_rejects_output_allocator_without_mutation():
    export_utils = importlib.import_module("torch_tensorrt.executorch._export_utils")
    program, engine_node, _, engine = _engine_program(
        lifted=True, requires_output_allocator=True
    )
    original_nodes = list(program.graph_module.graph.nodes)
    original_specs = list(program.graph_signature.input_specs)

    with pytest.raises(RuntimeError, match="output allocator"):
        export_utils.validate_engine_program(program)

    assert list(program.graph_module.graph.nodes) == original_nodes
    assert program.graph_signature.input_specs == original_specs
    assert program.constants == {"engine_fqn": engine}
    assert engine_node.users
    assert program.state_dict == {}


@pytest.mark.unit
def test_export_returns_edge_and_forwards_all_options(monkeypatch):
    export_module, lower = _patch_lowering(monkeypatch)
    program = FakeExportedProgram()
    extra_a = object()
    extra_b = object()
    compile_spec = object()
    transform_pass = object()
    edge_config = object()
    constant_methods = {"get_vocab_size": 256}
    partitioners = [extra_a, extra_b]
    compile_specs = [compile_spec]

    result = export_module.export(
        program,
        transform_passes=[transform_pass],
        partitioners=partitioners,
        compile_specs=compile_specs,
        compile_config=edge_config,
        constant_methods=constant_methods,
        generate_etrecord=True,
    )

    assert result is lower.return_value
    assert lower.call_args.args == (("rewritten", program),)
    assert lower.call_args.kwargs["transform_passes"] == [transform_pass]
    assert lower.call_args.kwargs["compile_config"] is edge_config
    assert lower.call_args.kwargs["constant_methods"] == constant_methods
    assert lower.call_args.kwargs["generate_etrecord"] is True
    lowered_partitioners = lower.call_args.kwargs["partitioner"]
    assert isinstance(lowered_partitioners[0], FakeTensorRTPartitioner)
    assert lowered_partitioners[0].compile_specs == [compile_spec]
    assert lowered_partitioners[1:] == [extra_a, extra_b]
    assert partitioners == [extra_a, extra_b]
    assert compile_specs == [compile_spec]


@pytest.mark.unit
def test_export_preserves_independent_method_mapping(monkeypatch):
    prefill = FakeExportedProgram()
    decode = FakeExportedProgram()
    export_module, lower = _patch_lowering(monkeypatch)
    prefill_extra = object()
    decode_extra = object()
    prefill_spec = object()
    decode_spec = object()

    export_module.export(
        {"prefill": prefill, "decode": decode},
        partitioners={"prefill": [prefill_extra], "decode": [decode_extra]},
        compile_specs={"prefill": [prefill_spec], "decode": [decode_spec]},
    )

    assert lower.call_args.args == (
        {"prefill": ("rewritten", prefill), "decode": ("rewritten", decode)},
    )
    pipelines = lower.call_args.kwargs["partitioner"]
    assert pipelines["prefill"][0].compile_specs == [prefill_spec]
    assert pipelines["prefill"][1:] == [prefill_extra]
    assert pipelines["decode"][0].compile_specs == [decode_spec]
    assert pipelines["decode"][1:] == [decode_extra]


@pytest.mark.unit
def test_stage_exported_program_isolates_structure_and_shares_payloads():
    export_utils = importlib.import_module("torch_tensorrt.executorch._export_utils")

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(2))

        def forward(self, x):
            return x + self.weight

    program = torch.export.export(Model(), (torch.ones(2),))

    class Metadata(NamedTuple):
        shape: tuple[int, ...]
        dtype: torch.dtype

    first_node = next(iter(program.graph.nodes))
    first_node.meta["nested"] = {"items": ["source"]}
    first_node.meta["tensor_meta"] = Metadata((2,), torch.float32)
    program.graph_module.meta["nested"] = {"items": ["source"]}
    staged = export_utils.stage_exported_program(program)

    assert staged is not program
    assert staged.graph_module is not program.graph_module
    assert staged.graph is not program.graph
    assert staged.state_dict is not program.state_dict
    assert staged.constants is not program.constants
    assert staged.state_dict["weight"] is program.state_dict["weight"]
    assert isinstance(next(iter(staged.graph.nodes)).meta["tensor_meta"], Metadata)

    staged.state_dict["staged_only"] = torch.zeros(1)
    staged.graph_module.meta["staged_only"] = True
    staged.graph_module.meta["nested"]["items"].append("staged")
    next(iter(staged.graph.nodes)).meta["nested"]["items"].append("staged")
    assert "staged_only" not in program.state_dict
    assert "staged_only" not in program.graph_module.meta
    assert program.graph_module.meta["nested"]["items"] == ["source"]
    assert first_node.meta["nested"]["items"] == ["source"]


@pytest.mark.unit
def test_stage_exported_program_supports_dynamic_shapes():
    export_utils = importlib.import_module("torch_tensorrt.executorch._export_utils")

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(4, 4)

        def forward(self, x):
            return self.lin(x) * x.shape[0]

    batch = torch.export.Dim("batch", min=1, max=32)
    program = torch.export.export(
        Model(), (torch.randn(2, 4),), dynamic_shapes={"x": {0: batch}}
    )
    assert program.range_constraints

    staged = export_utils.stage_exported_program(program)

    # Symbolic metadata stays bound to the original ShapeEnv, so it must be shared
    # rather than copied for the staged program to remain consistent.
    # Symbolic leaves stay bound to the original ShapeEnv, so they must be shared
    # rather than copied for the staged program to remain consistent. The container
    # holding them is still copied so the caller's metadata cannot be mutated.
    source_nodes = {node.name: node for node in program.graph.nodes}
    symbolic_shared = 0
    for node in staged.graph.nodes:
        source_node = source_nodes[node.name]
        for key, source_value in source_node.meta.items():
            source_leaves = [
                leaf
                for leaf in torch.utils._pytree.tree_leaves(source_value)
                if isinstance(leaf, (torch.SymInt, torch.SymFloat, torch.SymBool))
                or (isinstance(leaf, torch.Tensor) and is_fake(leaf))
            ]
            if not source_leaves:
                continue
            staged_leaves = torch.utils._pytree.tree_leaves(node.meta[key])
            for leaf in source_leaves:
                assert any(leaf is staged_leaf for staged_leaf in staged_leaves)
            symbolic_shared += 1
    assert symbolic_shared

    assert staged.graph_module is not program.graph_module
    assert staged.state_dict["lin.weight"] is program.state_dict["lin.weight"]


@pytest.mark.unit
def test_stage_exported_program_copies_containers_holding_symbolic_leaves():
    """A container in node.meta must not be shared just because it holds a leaf.

    A multi-output op stores a list of fake tensors in meta["val"]. Sharing that
    list wholesale would let an Edge transform mutate the caller's program.
    """
    export_utils = importlib.import_module("torch_tensorrt.executorch._export_utils")

    class Model(torch.nn.Module):
        def forward(self, x):
            first, second = torch.chunk(x, 2, dim=1)
            return first + second

    program = torch.export.export(Model(), (torch.randn(8, 4),))
    container_nodes = [
        node
        for node in program.graph.nodes
        if isinstance(node.meta.get("val"), (list, tuple))
    ]
    assert container_nodes, "expected a multi-output op with a list-valued meta"

    staged = export_utils.stage_exported_program(program)
    staged_nodes = {node.name: node for node in staged.graph.nodes}

    for source_node in container_nodes:
        source_value = source_node.meta["val"]
        staged_value = staged_nodes[source_node.name].meta["val"]
        assert staged_value is not source_value
        for source_leaf, staged_leaf in zip(source_value, staged_value):
            assert staged_leaf is source_leaf


@pytest.mark.unit
def test_stage_exported_program_clones_nested_graph_modules():
    export_utils = importlib.import_module("torch_tensorrt.executorch._export_utils")

    class Leaf(torch.nn.Module):
        def forward(self, x):
            return x + 1

    root = torch.fx.Graph()
    x = root.placeholder("x")
    call = root.call_module("nested", (x,))
    root.output(call)
    nested = torch.fx.symbolic_trace(Leaf())
    graph_module = torch.fx.GraphModule({"nested": nested}, root)
    nested_node = next(iter(graph_module.nested.graph.nodes))
    nested_payload = torch.ones(1)
    nested_node.meta["nested"] = {
        "items": ["source"],
        "payload": nested_payload,
    }
    payload_memo = export_utils._payload_sharing_memo(
        SimpleNamespace(graph_module=graph_module, state_dict={}, constants={})
    )
    staged = export_utils._stage_graph_module(graph_module, payload_memo)

    assert staged.nested is not graph_module.nested
    staged.nested.meta["staged_only"] = True
    next(iter(staged.nested.graph.nodes)).meta["nested"]["items"].append("staged")
    staged_nested_meta = next(iter(staged.nested.graph.nodes)).meta["nested"]
    assert staged_nested_meta["payload"] is nested_payload
    assert "staged_only" not in graph_module.nested.meta
    assert nested_node.meta["nested"]["items"] == ["source"]


@pytest.mark.unit
def test_export_prepares_compiled_graph_module(monkeypatch):
    export_module, lower = _patch_lowering(monkeypatch)
    graph_module = torch.fx.GraphModule(torch.nn.Module(), torch.fx.Graph())
    prepared = FakeExportedProgram()
    prepare = MagicMock(return_value=prepared)
    monkeypatch.setattr(export_module, "_prepare_graph_module", prepare)
    inputs = [torch.ones(1)]

    export_module.export(graph_module, arg_inputs=inputs, retrace=False)

    prepare.assert_called_once_with(
        graph_module,
        arg_inputs=inputs,
        kwarg_inputs=None,
        dynamic_shapes=None,
        retrace=False,
    )
    assert lower.call_args.args == (("rewritten", prepared),)


@pytest.mark.unit
@pytest.mark.parametrize("input_option", ["inputs", "arg_inputs"])
def test_export_forwards_input_alias(monkeypatch, input_option):
    export_module, _ = _patch_lowering(monkeypatch)
    graph_module = torch.fx.GraphModule(torch.nn.Module(), torch.fx.Graph())
    prepared = FakeExportedProgram()
    prepare = MagicMock(return_value=prepared)
    monkeypatch.setattr(export_module, "_prepare_graph_module", prepare)
    example_inputs = [torch.ones(1)]

    export_module.export(graph_module, **{input_option: example_inputs}, retrace=False)

    prepare.assert_called_once_with(
        graph_module,
        arg_inputs=example_inputs,
        kwarg_inputs=None,
        dynamic_shapes=None,
        retrace=False,
    )


@pytest.mark.unit
def test_export_rejects_inputs_and_arg_inputs(monkeypatch):
    export_module, lower = _patch_lowering(monkeypatch)
    graph_module = torch.fx.GraphModule(torch.nn.Module(), torch.fx.Graph())

    with pytest.raises(ValueError, match="mutually exclusive"):
        export_module.export(
            graph_module,
            inputs=[torch.ones(1)],
            arg_inputs=[torch.ones(1)],
        )

    lower.assert_not_called()


@pytest.mark.unit
def test_export_rejects_non_linux_platform(monkeypatch):
    export_module, lower = _patch_lowering(monkeypatch)
    monkeypatch.setattr(export_module.platform, "system", lambda: "Windows")
    graph_module = torch.fx.GraphModule(torch.nn.Module(), torch.fx.Graph())

    with pytest.raises(ValueError, match="only supported on Linux"):
        export_module.export(graph_module)

    lower.assert_not_called()


@pytest.mark.unit
def test_prepare_graph_module_preserves_tensor_keyword_inputs(monkeypatch):
    export_module = importlib.import_module("torch_tensorrt.executorch.export")
    graph_module = torch.fx.GraphModule(torch.nn.Module(), torch.fx.Graph())
    exported = FakeExportedProgram()
    export_graph_module = MagicMock(return_value=exported)
    keyword_tensor = torch.ones(2, device="cuda")
    monkeypatch.setattr("torch_tensorrt.dynamo._exporter.export", export_graph_module)
    monkeypatch.setattr(
        "torch_tensorrt.dynamo._defaults.default_device",
        lambda: torch.device("cuda"),
    )

    assert (
        export_module._prepare_graph_module(
            graph_module,
            arg_inputs=(),
            kwarg_inputs={"mask": keyword_tensor},
            dynamic_shapes=None,
            retrace=True,
        )
        is exported
    )
    assert (
        export_graph_module.call_args.kwargs["kwarg_inputs"]["mask"] is keyword_tensor
    )


@pytest.mark.unit
def test_export_allows_zero_engine_program(monkeypatch):
    program = FakeExportedProgram()
    export_module, lower = _patch_lowering(monkeypatch, {program: 0})

    assert export_module.export(program) is lower.return_value
    lowered_partitioners = lower.call_args.kwargs["partitioner"]
    assert len(lowered_partitioners) == 1
    assert isinstance(lowered_partitioners[0], FakeTensorRTPartitioner)


@pytest.mark.unit
def test_export_rejects_duplicate_program_identity_before_validation(monkeypatch):
    program = FakeExportedProgram()
    export_module, lower = _patch_lowering(monkeypatch)
    validate = MagicMock(return_value=1)
    monkeypatch.setattr(
        "torch_tensorrt.executorch._export_utils.validate_engine_program", validate
    )

    with pytest.raises(ValueError, match="same ExportedProgram object"):
        export_module.export({"prefill": program, "decode": program})

    validate.assert_not_called()
    lower.assert_not_called()


@pytest.mark.unit
def test_export_validates_all_methods_before_rewriting(monkeypatch):
    prefill = FakeExportedProgram()
    decode = FakeExportedProgram()
    export_module, lower = _patch_lowering(monkeypatch)
    validate = MagicMock(side_effect=[1, RuntimeError("decode is invalid")])
    replace = MagicMock()
    monkeypatch.setattr(
        "torch_tensorrt.executorch._export_utils.validate_engine_program", validate
    )
    monkeypatch.setattr(
        "torch_tensorrt.executorch._export_utils.replace_execute_engine", replace
    )

    with pytest.raises(RuntimeError, match="decode is invalid"):
        export_module.export({"prefill": prefill, "decode": decode})

    replace.assert_not_called()
    lower.assert_not_called()


@pytest.mark.unit
def test_export_rewrite_failure_leaves_sources_unchanged(monkeypatch):
    import executorch.exir
    import torch_tensorrt.executorch as executorch_api
    import torch_tensorrt.executorch._export_utils as export_utils

    class Model(torch.nn.Module):
        def forward(self, x):
            return x + 1

    prefill = torch.export.export(Model(), (torch.ones(1),))
    decode = torch.export.export(Model(), (torch.ones(1),))
    original_codes = {
        "prefill": prefill.graph_module.code,
        "decode": decode.graph_module.code,
    }
    original_state_keys = {
        "prefill": set(prefill.state_dict),
        "decode": set(decode.state_dict),
    }

    monkeypatch.setattr(
        export_utils, "validate_engine_program", lambda program, resolved=None: 1
    )
    monkeypatch.setattr(executorch_api, "TensorRTPartitioner", FakeTensorRTPartitioner)
    monkeypatch.setattr(executorch_api, "get_edge_compile_config", lambda: "default")
    lower = MagicMock()
    monkeypatch.setattr(executorch.exir, "to_edge_transform_and_lower", lower)
    call_count = 0

    def fail_second_rewrite(program, resolved=None):
        nonlocal call_count
        call_count += 1
        program.state_dict["staged_only"] = torch.zeros(1)
        program.graph_module.meta["staged_only"] = True
        if call_count == 2:
            raise RuntimeError("rewrite failed")
        return program

    monkeypatch.setattr(export_utils, "replace_execute_engine", fail_second_rewrite)

    with pytest.raises(RuntimeError, match="rewrite failed"):
        executorch_api.export({"prefill": prefill, "decode": decode})

    assert prefill.graph_module.code == original_codes["prefill"]
    assert decode.graph_module.code == original_codes["decode"]
    assert set(prefill.state_dict) == original_state_keys["prefill"]
    assert set(decode.state_dict) == original_state_keys["decode"]
    assert "staged_only" not in prefill.graph_module.meta
    assert "staged_only" not in decode.graph_module.meta
    lower.assert_not_called()


@pytest.mark.unit
def test_export_lowering_failure_leaves_source_unchanged(monkeypatch):
    import executorch.exir
    import torch_tensorrt.executorch as executorch_api
    import torch_tensorrt.executorch._export_utils as export_utils

    class Model(torch.nn.Module):
        def forward(self, x):
            return x + 1

    source = torch.export.export(Model(), (torch.ones(1),))
    original_code = source.graph_module.code
    original_state_keys = set(source.state_dict)

    monkeypatch.setattr(
        export_utils, "validate_engine_program", lambda program, resolved=None: 1
    )
    monkeypatch.setattr(executorch_api, "TensorRTPartitioner", FakeTensorRTPartitioner)
    monkeypatch.setattr(executorch_api, "get_edge_compile_config", lambda: "default")

    def mutate_staged_program(program, resolved=None):
        program.state_dict["staged_only"] = torch.zeros(1)
        program.graph_module.meta["staged_only"] = True
        return program

    monkeypatch.setattr(export_utils, "replace_execute_engine", mutate_staged_program)
    monkeypatch.setattr(
        executorch.exir,
        "to_edge_transform_and_lower",
        MagicMock(side_effect=RuntimeError("lowering failed")),
    )

    with pytest.raises(RuntimeError, match="lowering failed"):
        executorch_api.export(source)

    assert source.graph_module.code == original_code
    assert set(source.state_dict) == original_state_keys
    assert "staged_only" not in source.graph_module.meta


@pytest.mark.unit
def test_export_accepts_false_retrace_for_exported_program(monkeypatch):
    export_module, lower = _patch_lowering(monkeypatch)
    program = FakeExportedProgram()

    export_module.export(program, retrace=False)

    assert lower.call_args.args == (("rewritten", program),)


@pytest.mark.unit
def test_export_rejects_true_retrace_for_exported_program(monkeypatch):
    export_module, lower = _patch_lowering(monkeypatch)
    program = FakeExportedProgram()

    with pytest.raises(ValueError, match="already-exported program"):
        export_module.export(program, retrace=True)

    lower.assert_not_called()


@pytest.mark.unit
def test_export_normalizes_none_per_method_values(monkeypatch):
    export_module, lower = _patch_lowering(monkeypatch)
    prefill = FakeExportedProgram()
    decode = FakeExportedProgram()

    export_module.export(
        {"prefill": prefill, "decode": decode},
        partitioners={"prefill": None},
        compile_specs={"decode": None},
    )

    pipelines = lower.call_args.kwargs["partitioner"]
    assert len(pipelines["prefill"]) == 1
    assert len(pipelines["decode"]) == 1
    assert pipelines["prefill"][0].compile_specs == []
    assert pipelines["decode"][0].compile_specs == []


@pytest.mark.unit
def test_export_rejects_shared_partitioners_across_methods(monkeypatch):
    """One partitioner instance cannot serve several methods.

    ExecuTorch backends bake the method name into the delegation spec built in
    the partitioner constructor, so a shared instance would tag every method
    with the first method's name.
    """
    export_module, lower = _patch_lowering(monkeypatch)

    with pytest.raises(ValueError, match="reuses the same"):
        export_module.export(
            {"prefill": FakeExportedProgram(), "decode": FakeExportedProgram()},
            partitioners=[object()],
        )

    lower.assert_not_called()


@pytest.mark.unit
def test_export_rejects_shared_partitioner_inside_mapping(monkeypatch):
    """The same instance listed under two method keys is rejected too."""
    export_module, lower = _patch_lowering(monkeypatch)
    shared = object()

    with pytest.raises(ValueError, match="reuses the same"):
        export_module.export(
            {"prefill": FakeExportedProgram(), "decode": FakeExportedProgram()},
            partitioners={"prefill": [shared], "decode": [shared]},
        )

    lower.assert_not_called()


@pytest.mark.unit
def test_export_allows_shared_partitioners_for_single_method(monkeypatch):
    """A flat sequence stays valid when there is only one method to tag."""
    export_module, lower = _patch_lowering(monkeypatch)
    partitioner = object()

    export_module.export(FakeExportedProgram(), partitioners=[partitioner])

    pipeline = lower.call_args.kwargs["partitioner"]
    assert pipeline[-1] is partitioner


@pytest.mark.unit
def test_export_accepts_per_method_partitioner_instances(monkeypatch):
    """Each method keeps the partitioner instance it was given."""
    export_module, lower = _patch_lowering(monkeypatch)
    prefill_partitioner = object()
    decode_partitioner = object()

    export_module.export(
        {"prefill": FakeExportedProgram(), "decode": FakeExportedProgram()},
        partitioners={
            "prefill": [prefill_partitioner],
            "decode": [decode_partitioner],
        },
    )

    pipelines = lower.call_args.kwargs["partitioner"]
    assert pipelines["prefill"][-1] is prefill_partitioner
    assert pipelines["decode"][-1] is decode_partitioner


@pytest.mark.unit
def test_export_rejects_invalid_constant_method_names(monkeypatch):
    """Constant-method names are baked into the .pte, so reject unusable ones."""
    export_module, lower = _patch_lowering(monkeypatch)

    for bad_name in ("not an identifier", "1_leading_digit", ""):
        with pytest.raises(ValueError, match="valid Python identifiers"):
            export_module.export(FakeExportedProgram(), constant_methods={bad_name: 1})

    with pytest.raises(ValueError, match="valid Python identifiers"):
        export_module.export(FakeExportedProgram(), constant_methods={42: 1})

    lower.assert_not_called()


@pytest.mark.unit
def test_export_accepts_valid_constant_method_names(monkeypatch):
    export_module, lower = _patch_lowering(monkeypatch)

    export_module.export(
        FakeExportedProgram(), constant_methods={"get_vocab_size": 256}
    )

    assert lower.call_args.kwargs["constant_methods"] == {"get_vocab_size": 256}


@pytest.mark.unit
def test_export_normalizes_mapping_transform_passes_to_dict(monkeypatch):
    """ExecuTorch dispatches per-method passes on isinstance(passes, dict).

    A Mapping that is not a dict would silently run no passes at all, so it must
    be normalized before being forwarded.
    """
    from collections.abc import Mapping as AbcMapping

    class CustomMapping(AbcMapping):
        def __init__(self, data):
            self._data = data

        def __getitem__(self, key):
            return self._data[key]

        def __iter__(self):
            return iter(self._data)

        def __len__(self):
            return len(self._data)

    export_module, lower = _patch_lowering(monkeypatch)
    passes = ["a-pass"]

    export_module.export(
        {"prefill": FakeExportedProgram(), "decode": FakeExportedProgram()},
        transform_passes=CustomMapping({"prefill": passes}),
    )

    forwarded = lower.call_args.kwargs["transform_passes"]
    assert type(forwarded) is dict
    assert forwarded == {"prefill": passes}


@pytest.mark.unit
@pytest.mark.parametrize("container_type", [list, tuple])
def test_prepare_graph_module_infers_nested_dynamic_shapes(monkeypatch, container_type):
    export_module = importlib.import_module("torch_tensorrt.executorch.export")

    class NestedModule(torch.nn.Module):
        def forward(self, nested):
            return nested[0]

    graph_module = torch.fx.symbolic_trace(NestedModule())
    exported = FakeExportedProgram()
    export_graph_module = MagicMock(return_value=exported)
    dynamic_input = torch_tensorrt.Input(
        min_shape=(1, 2),
        opt_shape=(2, 2),
        max_shape=(4, 2),
        name="nested",
    )
    monkeypatch.setattr("torch_tensorrt.dynamo._exporter.export", export_graph_module)

    assert (
        export_module._prepare_graph_module(
            graph_module,
            arg_inputs=(container_type([dynamic_input]),),
            kwarg_inputs={},
            dynamic_shapes=None,
            retrace=False,
        )
        is exported
    )
    nested_shapes = export_graph_module.call_args.kwargs["dynamic_shapes"]["nested"]
    assert isinstance(nested_shapes, container_type)
    assert len(nested_shapes) == 1
    assert set(nested_shapes[0]) == {0}


@pytest.mark.unit
def test_prepare_graph_module_preserves_shared_dynamic_dimensions(monkeypatch):
    export_module = importlib.import_module("torch_tensorrt.executorch.export")

    class SharedBatchModule(torch.nn.Module):
        def forward(self, left, right):
            return left + right

    graph_module = torch.fx.symbolic_trace(SharedBatchModule())
    exported = FakeExportedProgram()
    export_graph_module = MagicMock(return_value=exported)
    monkeypatch.setattr("torch_tensorrt.dynamo._exporter.export", export_graph_module)
    inputs = tuple(
        torch_tensorrt.Input(
            min_shape=(1, 2),
            opt_shape=(2, 2),
            max_shape=(4, 2),
            name=name,
            shared_dims={0: "batch"},
        )
        for name in ("left", "right")
    )

    assert (
        export_module._prepare_graph_module(
            graph_module,
            arg_inputs=inputs,
            kwarg_inputs={},
            dynamic_shapes=None,
            retrace=False,
        )
        is exported
    )
    dynamic_shapes = export_graph_module.call_args.kwargs["dynamic_shapes"]
    assert dynamic_shapes["left"][0] is dynamic_shapes["right"][0]


@pytest.mark.unit
def test_prepare_graph_module_requires_shapes_for_mixed_dynamic_inputs(monkeypatch):
    export_module = importlib.import_module("torch_tensorrt.executorch.export")

    class MixedModule(torch.nn.Module):
        def forward(self, dynamic, static):
            return dynamic + static

    graph_module = torch.fx.symbolic_trace(MixedModule())
    dynamic_input = torch_tensorrt.Input(
        min_shape=(1,), opt_shape=(2,), max_shape=(4,), name="dynamic"
    )
    static_input = torch.ones(2)

    with pytest.raises(ValueError, match="require explicit dynamic_shapes"):
        export_module._prepare_graph_module(
            graph_module,
            arg_inputs=(dynamic_input, static_input),
            kwarg_inputs={},
            dynamic_shapes=None,
            retrace=False,
        )

    exported = FakeExportedProgram()
    export_graph_module = MagicMock(return_value=exported)
    monkeypatch.setattr("torch_tensorrt.dynamo._exporter.export", export_graph_module)
    explicit_shapes = ({0: torch.export.Dim("batch", min=1, max=4)}, None)
    assert (
        export_module._prepare_graph_module(
            graph_module,
            arg_inputs=(dynamic_input, static_input),
            kwarg_inputs={},
            dynamic_shapes=explicit_shapes,
            retrace=False,
        )
        is exported
    )
    assert export_graph_module.call_args.kwargs["dynamic_shapes"] is explicit_shapes


@pytest.mark.unit
def test_prepare_graph_module_does_not_infer_shapes_without_inputs(monkeypatch):
    export_module = importlib.import_module("torch_tensorrt.executorch.export")
    graph_module = torch.fx.GraphModule(torch.nn.Module(), torch.fx.Graph())
    exported = FakeExportedProgram()
    export_graph_module = MagicMock(return_value=exported)
    infer_args = MagicMock()
    infer_kwargs = MagicMock()
    monkeypatch.setattr("torch_tensorrt.dynamo._exporter.export", export_graph_module)
    monkeypatch.setattr(
        "torch_tensorrt.dynamo._tracer.get_dynamic_shapes_args", infer_args
    )
    monkeypatch.setattr(
        "torch_tensorrt.dynamo._tracer.get_dynamic_shapes_kwargs", infer_kwargs
    )

    assert (
        export_module._prepare_graph_module(
            graph_module,
            arg_inputs=(),
            kwarg_inputs={},
            dynamic_shapes=None,
            retrace=False,
        )
        is exported
    )
    infer_args.assert_not_called()
    infer_kwargs.assert_not_called()
    assert export_graph_module.call_args.kwargs["dynamic_shapes"] is None


@pytest.mark.unit
def test_export_rejects_ambiguous_sources_and_options(monkeypatch):
    export_module, _ = _patch_lowering(monkeypatch)
    program = FakeExportedProgram()

    with pytest.raises(ValueError, match="already-exported program"):
        export_module.export(program, arg_inputs=[torch.ones(1)])
    with pytest.raises(TypeError, match="Compile nn.Module inputs"):
        export_module.export(torch.nn.Linear(1, 1))
    with pytest.raises(ValueError, match="unknown methods"):
        export_module.export({"forward": program}, partitioners={"missing": [object()]})
    with pytest.raises(ValueError, match="collide"):
        export_module.export({"forward": program}, constant_methods={"forward": 1})


@pytest.mark.unit
@pytest.mark.parametrize("option_name", ["partitioners", "compile_specs"])
def test_export_rejects_string_option_sequences(monkeypatch, option_name):
    export_module, _ = _patch_lowering(monkeypatch)
    program = FakeExportedProgram()

    with pytest.raises(TypeError, match=option_name):
        export_module.export(program, **{option_name: "invalid"})

    with pytest.raises(TypeError, match=option_name):
        export_module.export(
            {"forward": program}, **{option_name: {"forward": "invalid"}}
        )


@pytest.mark.unit
def test_prepare_graph_module_rejects_argument_mismatch(monkeypatch):
    export_module = importlib.import_module("torch_tensorrt.executorch.export")

    class OneInput(torch.nn.Module):
        def forward(self, x):
            return x

    graph_module = torch.fx.symbolic_trace(OneInput())
    with pytest.raises(TypeError):
        export_module._prepare_graph_module(
            graph_module,
            arg_inputs=(
                torch_tensorrt.Input((1,), name="x"),
                torch_tensorrt.Input((1,), name="extra"),
            ),
            kwarg_inputs={},
            dynamic_shapes=None,
            retrace=False,
        )


@pytest.mark.unit
def test_export_warns_for_multiple_engines(monkeypatch, caplog):
    program = FakeExportedProgram()
    export_module, _ = _patch_lowering(monkeypatch, {program: 2})

    export_module.export(program)
    assert "contains 2 TRT engines" in caplog.text


@pytest.mark.unit
def test_validate_hands_resolved_engines_to_the_rewrite(monkeypatch):
    """One engine is resolved once, even though staging returns a new program.

    Resolving an engine serializes it and base64 encodes the result, so resolving the
    same engine in both validation and the rewrite would double that cost.
    """
    export_utils = importlib.import_module("torch_tensorrt.executorch._export_utils")
    calls = []

    def counting_get_engine_info_from_state(engine_obj):
        calls.append(engine_obj)
        return ["0", "dev", "nm", "BYTES", "in", "out", "", "", "", "", ""]

    monkeypatch.setattr(
        export_utils, "get_engine_info_from_state", counting_get_engine_info_from_state
    )
    monkeypatch.setattr(
        export_utils, "validate_engine_info", lambda info, node_name=None: None
    )

    def make_program():
        engine_node = SimpleNamespace(
            op="get_attr", target="engine_attr", name="engine_attr"
        )
        node = SimpleNamespace(
            op="call_function",
            name="eng_node",
            target=torch.ops.tensorrt.execute_engine.default,
            args=(["in"], engine_node),
        )
        graph_module = SimpleNamespace(
            graph=SimpleNamespace(nodes=[node]), engine_attr=object()
        )
        return SimpleNamespace(graph_module=graph_module, constants={})

    resolved: dict = {}
    assert export_utils.validate_engine_program(make_program(), resolved) == 1
    assert len(calls) == 1

    # Staging builds a new ExportedProgram, so the rewrite has to find the earlier work
    # by node name rather than by object identity.
    assert resolved["eng_node"] is not None
    assert len(calls) == 1
