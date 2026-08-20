"""Export-side coverage for caller-owned KV-cache buffer mutations.

Both retrace modes must surface an engine's aliased KV outputs as graph-level
BUFFER_MUTATIONs so the ExecuTorch delegate keeps them as caller-owned mutable
buffers instead of freezing them:

  * retrace=False (legacy exporter): ``inline_trt_modules`` exposes them at
    transform time (guarded by ``expose_aliased_mutations``), then
    ``create_trt_exp_program`` declares the specs.
  * retrace=True (torch.export): ``torch.export`` truncates the aliased outputs
    at the fx boundary, so ``_declare_aliased_kv_mutations_on_ep`` re-declares
    them on the exported program before lowering.
"""

import operator
from types import SimpleNamespace

import pytest
import torch
from torch._export.verifier import Verifier
from torch.export.exported_program import (
    InputKind,
    InputSpec,
    OutputKind,
    OutputSpec,
    TensorArgument,
)
from torch_tensorrt.dynamo import _exporter as E

# A real ExportedProgram carries these, so the stubs below do too -- the pass has to
# hand them on rather than let them reset to their defaults.
_EXAMPLE_INPUTS = ((torch.randn(2),), {})


@pytest.mark.unit
@pytest.mark.parametrize("use_legacy, expected_expose", [(True, True), (False, False)])
def test_export_exposes_aliased_mutations_only_for_legacy_exporter(
    monkeypatch, use_legacy, expected_expose
):
    """The transform-time KV exposure runs on the retrace=False (legacy) path
    only; retrace=True defers to the post-export declaration pass, so it must not
    perturb the user outputs at transform time.
    """
    captured = {}

    def fake_transform(gm, cross_compile_module=False, expose_aliased_mutations=True):
        captured["expose"] = expose_aliased_mutations
        return gm

    monkeypatch.setattr(E, "transform", fake_transform)
    monkeypatch.setattr(E, "create_trt_exp_program", lambda *a, **k: "legacy-ep")
    monkeypatch.setattr(torch.export, "export", lambda *a, **k: "retrace-ep")

    result = E.export(torch.nn.Module(), use_legacy_exporter=use_legacy)

    assert captured["expose"] is expected_expose
    assert result == ("legacy-ep" if use_legacy else "retrace-ep")


@pytest.mark.unit
def test_declare_aliased_kv_mutations_is_noop_without_engines():
    """With no execute_engine node carrying aliased I/O, the pass returns the
    exported program unchanged (same object)."""
    pytest.importorskip("executorch.exir")
    g = torch.fx.Graph()
    x = g.placeholder("x")
    g.output((x,))
    gm = torch.fx.GraphModule(torch.nn.Module(), g)

    ep = SimpleNamespace(
        graph_module=gm,
        graph_signature=SimpleNamespace(inputs_to_buffers={}, output_specs=[]),
    )
    assert E._declare_aliased_kv_mutations_on_ep(ep) is ep


@pytest.mark.unit
def test_declare_aliased_kv_mutations_is_idempotent(monkeypatch):
    """Running on a program whose mutation is already declared must be a no-op.

    Exposure is decided both by the exporter (the legacy one declares at transform
    time) and by save()'s per-format branch, so this pass can receive a program that
    already carries the spec. Declaring again appends a second BUFFER_MUTATION for
    the same buffer, which fails the ExportedProgram verifier's output ordering.
    """
    pytest.importorskip("executorch.exir")
    import torch_tensorrt.dynamo.runtime._serialized_engine_layout as L
    import torch_tensorrt.dynamo.runtime._TorchTensorRTModule as M
    import torch_tensorrt.executorch.backend as B

    exec_target = torch.ops.tensorrt.execute_engine.default

    g = torch.fx.Graph()
    b_k_0 = g.placeholder("b_k_0")
    tokens = g.placeholder("tokens")
    engine = g.placeholder("engine")
    eng = g.call_function(exec_target, ([b_k_0, tokens], engine))
    user_out = g.call_function(operator.getitem, (eng, 0))
    kv_out = g.call_function(operator.getitem, (eng, 1))
    g.output((kv_out, user_out))

    buf_val = torch.zeros(2, 2)
    out_val = torch.zeros(1)
    b_k_0.meta["val"] = buf_val
    tokens.meta["val"] = torch.zeros(1)
    engine.meta["val"] = None
    eng.meta["val"] = [out_val, buf_val]
    user_out.meta["val"] = out_val
    kv_out.meta["val"] = buf_val
    gm = torch.fx.GraphModule(torch.nn.Module(), g)

    # k_0 already declared -- what the legacy exporter leaves behind.
    sig = SimpleNamespace(
        inputs_to_buffers={"b_k_0": "k_0"},
        input_specs=[
            InputSpec(InputKind.BUFFER, TensorArgument(name="b_k_0"), "k_0", True),
        ],
        output_specs=[
            OutputSpec(
                OutputKind.BUFFER_MUTATION, TensorArgument(name=kv_out.name), "k_0"
            ),
            OutputSpec(OutputKind.USER_OUTPUT, TensorArgument(name="user_out"), None),
        ],
    )
    ep = SimpleNamespace(
        graph_module=gm,
        graph_signature=sig,
        state_dict={},
        range_constraints={},
        module_call_graph=[],
        constants={},
        example_inputs=_EXAMPLE_INPUTS,
        verifiers=[Verifier],
    )

    info = ["x"] * (L.ALIASED_IO_IDX + 1)
    info[L.INPUT_BINDING_NAMES_IDX] = "IN"
    info[L.OUTPUT_BINDING_NAMES_IDX] = "OUT"
    monkeypatch.setattr(B, "_get_engine_info_for_node", lambda ep_, n: info)
    monkeypatch.setattr(
        M, "deserialize_aliased_io", lambda s: {"out_k": ("k_in", "kv_cache_update")}
    )
    monkeypatch.setattr(
        L,
        "deserialize_binding_names",
        lambda s: ["k_in", "tokens"] if s == "IN" else ["user_out", "out_k"],
    )

    def _must_not_rebuild(**kwargs):
        raise AssertionError(
            "the pass rebuilt the program even though k_0 was already declared"
        )

    monkeypatch.setattr(E, "ExportedProgram", _must_not_rebuild)

    assert E._declare_aliased_kv_mutations_on_ep(ep) is ep
    assert [spec.kind for spec in sig.output_specs] == [
        OutputKind.BUFFER_MUTATION,
        OutputKind.USER_OUTPUT,
    ]


@pytest.mark.unit
def test_declare_aliased_kv_mutations_declares_buffer_mutation(monkeypatch):
    """An engine whose aliased KV output is dropped from meta['val'] gets that
    output surfaced as a getitem and declared a BUFFER_MUTATION of the aliased
    input's buffer, ordered before the user outputs (verifier requirement)."""
    pytest.importorskip("executorch.exir")
    import torch_tensorrt.dynamo.runtime._serialized_engine_layout as L
    import torch_tensorrt.dynamo.runtime._TorchTensorRTModule as M
    import torch_tensorrt.executorch.backend as B

    exec_target = torch.ops.tensorrt.execute_engine.default

    # b_k_0 (KV buffer) + tokens feed the engine; meta['val'] covers only the one
    # user output -- the aliased KV output ("out_k") is truncated at the boundary.
    g = torch.fx.Graph()
    b_k_0 = g.placeholder("b_k_0")
    tokens = g.placeholder("tokens")
    engine = g.placeholder("engine")
    eng = g.call_function(exec_target, ([b_k_0, tokens], engine))
    user_out = g.call_function(operator.getitem, (eng, 0))
    g.output((user_out,))

    buf_val = torch.zeros(2, 2)
    out_val = torch.zeros(1)
    b_k_0.meta["val"] = buf_val
    tokens.meta["val"] = torch.zeros(1)
    engine.meta["val"] = None
    eng.meta["val"] = [out_val]
    user_out.meta["val"] = out_val
    gm = torch.fx.GraphModule(torch.nn.Module(), g)

    sig = SimpleNamespace(
        inputs_to_buffers={"b_k_0": "k_0"},
        input_specs=[
            InputSpec(InputKind.BUFFER, TensorArgument(name="b_k_0"), "k_0", True),
        ],
        output_specs=[
            OutputSpec(OutputKind.USER_OUTPUT, TensorArgument(name="user_out"), None),
        ],
    )
    ep = SimpleNamespace(
        graph_module=gm,
        graph_signature=sig,
        state_dict={},
        range_constraints={},
        module_call_graph=[],
        constants={},
        example_inputs=_EXAMPLE_INPUTS,
        verifiers=[Verifier],
    )

    info = ["x"] * (L.ALIASED_IO_IDX + 1)
    info[L.INPUT_BINDING_NAMES_IDX] = "IN"
    info[L.OUTPUT_BINDING_NAMES_IDX] = "OUT"
    monkeypatch.setattr(B, "_get_engine_info_for_node", lambda ep_, n: info)
    monkeypatch.setattr(
        M, "deserialize_aliased_io", lambda s: {"out_k": ("k_in", "kv_cache_update")}
    )
    monkeypatch.setattr(
        L,
        "deserialize_binding_names",
        lambda s: ["k_in", "tokens"] if s == "IN" else ["user_out", "out_k"],
    )

    captured = {}

    class _CapturingEP:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(E, "ExportedProgram", _CapturingEP)

    E._declare_aliased_kv_mutations_on_ep(ep)

    new_specs = captured["graph_signature"].output_specs
    # BUFFER_MUTATION for k_0 is declared first, ahead of the user output.
    assert len(new_specs) == 2
    assert new_specs[0].kind == OutputKind.BUFFER_MUTATION
    assert new_specs[0].target == "k_0"
    assert new_specs[1].kind == OutputKind.USER_OUTPUT

    # The mutation getitem is prepended to the graph output (mutations first).
    out_node = next(n for n in gm.graph.nodes if n.op == "output")
    assert out_node.args[0][0].name == new_specs[0].arg.name
    assert out_node.args[0][1] is user_out

    # The engine's meta['val'] is extended to cover the truncated aliased output.
    assert len(eng.meta["val"]) == 2

    # A rewrite replaces only the graph and signature; every other field of the
    # source program has to come through untouched.
    for field in (
        "state_dict",
        "range_constraints",
        "module_call_graph",
        "constants",
        "example_inputs",
        "verifiers",
    ):
        assert captured[field] is getattr(ep, field)


@pytest.mark.unit
@pytest.mark.parametrize("output_format", ["exported_program", "executorch"])
@pytest.mark.parametrize("use_legacy", [True, False])
def test_save_declares_aliased_mutations_without_retrace(
    monkeypatch, tmp_path, output_format, use_legacy
):
    """retrace=False must declare the aliased KV mutations as well.

    Only the legacy exporter exposes them at transform time, so with
    use_legacy_exporter=False nothing declares them and the saved program omits an
    update the engine performs. The pass skips buffers already declared, so save()
    can run it for either exporter.
    """
    pytest.importorskip("executorch.exir")
    import torch_tensorrt
    from torch_tensorrt import _compile as C
    from torch_tensorrt.dynamo import _exporter as E

    sentinel = object()
    declared = []

    def _declare(ep, **kwargs):
        declared.append(ep)
        return ep

    monkeypatch.setattr(E, "export", lambda *a, **k: sentinel)
    monkeypatch.setattr(E, "_declare_aliased_kv_mutations_on_ep", _declare)
    monkeypatch.setattr(C, "_normalize_engine_constants_to_python", lambda ep: None)
    monkeypatch.setattr(C, "_save_as_executorch", lambda *a, **k: None)
    monkeypatch.setattr(torch.export, "save", lambda *a, **k: None)

    g = torch.fx.Graph()
    g.output((g.placeholder("x"),))
    gm = torch.fx.GraphModule(torch.nn.Module(), g)

    torch_tensorrt.save(
        gm,
        str(tmp_path / "out.pte"),
        output_format=output_format,
        retrace=False,
        use_legacy_exporter=use_legacy,
    )

    assert declared == [sentinel]


@pytest.mark.unit
def test_declare_aliased_kv_mutations_declares_copyback(monkeypatch):
    """retrace=True: a trailing copy-back output (a non-KV mutable buffer with no
    engine aliasing) is reclassified from USER_OUTPUT to BUFFER_MUTATION of its
    buffer and ordered ahead of the user outputs."""
    pytest.importorskip("executorch.exir")

    # No execute_engine node -> the pure copy-back case (num_copyback drives the
    # pass). ``user_out`` is a real return; ``state_new`` is the copy-back new value
    # that lift_mutated_buffers appended as the trailing output.
    g = torch.fx.Graph()
    x = g.placeholder("x")
    state_in = g.placeholder("state_in")
    user_out = g.call_function(torch.add, (x, x))
    state_new = g.call_function(torch.add, (state_in, x))
    g.output((user_out, state_new))
    gm = torch.fx.GraphModule(torch.nn.Module(), g)

    sig = SimpleNamespace(
        inputs_to_buffers={"state_in": "state_0"},
        input_specs=[
            InputSpec(
                InputKind.BUFFER, TensorArgument(name="state_in"), "state_0", True
            ),
            InputSpec(InputKind.USER_INPUT, TensorArgument(name="x"), None),
        ],
        output_specs=[
            OutputSpec(OutputKind.USER_OUTPUT, TensorArgument(name="user_out"), None),
            OutputSpec(OutputKind.USER_OUTPUT, TensorArgument(name="state_new"), None),
        ],
    )
    ep = SimpleNamespace(
        graph_module=gm,
        graph_signature=sig,
        state_dict={},
        range_constraints={},
        module_call_graph=[],
        constants={},
        example_inputs=_EXAMPLE_INPUTS,
        verifiers=[Verifier],
    )

    captured = {}

    class _CapturingEP:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(E, "ExportedProgram", _CapturingEP)

    E._declare_aliased_kv_mutations_on_ep(ep, copyback_buffers=["state_0"])

    new_specs = captured["graph_signature"].output_specs
    # The trailing copy-back output becomes a BUFFER_MUTATION of state_0, first.
    assert len(new_specs) == 2
    assert new_specs[0].kind == OutputKind.BUFFER_MUTATION
    assert new_specs[0].target == "state_0"
    assert new_specs[0].arg.name == state_new.name
    assert new_specs[1].kind == OutputKind.USER_OUTPUT

    # Mutation is prepended to the graph output; the user output follows.
    out_node = next(n for n in gm.graph.nodes if n.op == "output")
    assert out_node.args[0][0] is state_new
    assert out_node.args[0][1] is user_out


@pytest.mark.unit
def test_declare_aliased_kv_mutations_skips_already_declared_copyback(monkeypatch):
    """A buffer torch.export already declared BUFFER_MUTATION is not declared twice,
    and its trailing value still leaves the user outputs.

    torch.export moves a mutation it recognises to the front of the outputs. The
    trailing value lift appended for that same buffer is internal plumbing, so it must
    be detached either way, otherwise it stays a user output and the saved model gains
    a return it never had."""
    pytest.importorskip("executorch.exir")

    g = torch.fx.Graph()
    x = g.placeholder("x")
    state_in = g.placeholder("state_in")
    state_new = g.call_function(torch.add, (state_in, x))
    user_out = g.call_function(torch.add, (x, x))
    # torch.export's own mutation output first, then the user output, then the
    # trailing copy-back value lift appended for the same buffer.
    g.output((state_new, user_out, state_new))
    gm = torch.fx.GraphModule(torch.nn.Module(), g)

    sig = SimpleNamespace(
        inputs_to_buffers={"state_in": "state_0"},
        input_specs=[
            InputSpec(
                InputKind.BUFFER, TensorArgument(name="state_in"), "state_0", True
            ),
            InputSpec(InputKind.USER_INPUT, TensorArgument(name="x"), None),
        ],
        output_specs=[
            OutputSpec(
                OutputKind.BUFFER_MUTATION,
                TensorArgument(name=state_new.name),
                "state_0",
            ),
            OutputSpec(OutputKind.USER_OUTPUT, TensorArgument(name="user_out"), None),
            OutputSpec(
                OutputKind.USER_OUTPUT, TensorArgument(name=state_new.name), None
            ),
        ],
    )
    ep = SimpleNamespace(
        graph_module=gm,
        graph_signature=sig,
        state_dict={},
        range_constraints={},
        module_call_graph=[],
        constants={},
        example_inputs=_EXAMPLE_INPUTS,
        verifiers=[Verifier],
    )

    captured = {}

    class _CapturingEP:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(E, "ExportedProgram", _CapturingEP)

    E._declare_aliased_kv_mutations_on_ep(ep, copyback_buffers=["state_0"])

    new_specs = captured["graph_signature"].output_specs
    # state_0 keeps its single existing mutation spec; no second one is added.
    mutations = [s for s in new_specs if s.kind == OutputKind.BUFFER_MUTATION]
    assert [s.target for s in mutations] == ["state_0"]
    # The trailing copy-back value is gone from the user outputs.
    user_outputs = [s for s in new_specs if s.kind == OutputKind.USER_OUTPUT]
    assert [s.arg.name for s in user_outputs] == ["user_out"]

    out_node = next(n for n in gm.graph.nodes if n.op == "output")
    assert list(out_node.args[0]) == [state_new, user_out]


@pytest.mark.unit
def test_declare_aliased_kv_mutations_pairs_copyback_by_position(monkeypatch):
    """With a mix of declared and undeclared buffers, each remaining copy-back value
    is paired with the buffer lift appended it for.

    lift appends the values in copyback_buffers order, so the pairing has to come from
    that full run. Dropping the declared buffers before slicing shifts the run and
    declares one buffer's value as another's mutation, which the runtime then copies
    into a buffer of a different shape."""
    pytest.importorskip("executorch.exir")

    g = torch.fx.Graph()
    x = g.placeholder("x")
    a_in = g.placeholder("a_in")
    b_in = g.placeholder("b_in")
    c_in = g.placeholder("c_in")
    a_new = g.call_function(torch.add, (a_in, x))
    b_new = g.call_function(torch.add, (b_in, x))
    c_new = g.call_function(torch.add, (c_in, x))
    user_out = g.call_function(torch.add, (x, x))
    # Only b is recognised by torch.export, so its mutation leads. The trailing run
    # is a_new, b_new, c_new, matching copyback_buffers order.
    g.output((b_new, user_out, a_new, b_new, c_new))
    gm = torch.fx.GraphModule(torch.nn.Module(), g)

    def buf_spec(name, target):
        return InputSpec(InputKind.BUFFER, TensorArgument(name=name), target, True)

    sig = SimpleNamespace(
        inputs_to_buffers={"a_in": "a", "b_in": "b", "c_in": "c"},
        input_specs=[
            buf_spec("a_in", "a"),
            buf_spec("b_in", "b"),
            buf_spec("c_in", "c"),
            InputSpec(InputKind.USER_INPUT, TensorArgument(name="x"), None),
        ],
        output_specs=[
            OutputSpec(
                OutputKind.BUFFER_MUTATION, TensorArgument(name=b_new.name), "b"
            ),
            OutputSpec(OutputKind.USER_OUTPUT, TensorArgument(name="user_out"), None),
            OutputSpec(OutputKind.USER_OUTPUT, TensorArgument(name=a_new.name), None),
            OutputSpec(OutputKind.USER_OUTPUT, TensorArgument(name=b_new.name), None),
            OutputSpec(OutputKind.USER_OUTPUT, TensorArgument(name=c_new.name), None),
        ],
    )
    ep = SimpleNamespace(
        graph_module=gm,
        graph_signature=sig,
        state_dict={},
        range_constraints={},
        module_call_graph=[],
        constants={},
        example_inputs=_EXAMPLE_INPUTS,
        verifiers=[Verifier],
    )

    captured = {}

    class _CapturingEP:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(E, "ExportedProgram", _CapturingEP)

    E._declare_aliased_kv_mutations_on_ep(ep, copyback_buffers=["a", "b", "c"])

    new_specs = captured["graph_signature"].output_specs
    mutations = [s for s in new_specs if s.kind == OutputKind.BUFFER_MUTATION]
    # Each buffer appears once, paired with the value appended for it.
    assert [(s.target, s.arg.name) for s in mutations] == [
        ("a", a_new.name),
        ("c", c_new.name),
        ("b", b_new.name),
    ]
    user_outputs = [s for s in new_specs if s.kind == OutputKind.USER_OUTPUT]
    assert [s.arg.name for s in user_outputs] == ["user_out"]


@pytest.mark.unit
def test_create_trt_exp_program_declares_copyback(monkeypatch):
    """retrace=False: create_trt_exp_program reads
    gm.meta['_copyback_mutation_buffers'], tags the trailing outputs with their
    buffer target, and emits them as BUFFER_MUTATION specs ahead of the user
    outputs."""
    pytest.importorskip("executorch.exir")

    g = torch.fx.Graph()
    x = g.placeholder("x")
    state_in = g.placeholder("state_in")
    user_out = g.call_function(torch.add, (x, x))
    state_new = g.call_function(torch.add, (state_in, x))
    g.output((user_out, state_new))
    gm = torch.fx.GraphModule(torch.nn.Module(), g)
    gm.recompile()
    gm.meta["_copyback_mutation_buffers"] = ["state_0"]

    captured = {}

    class _CapturingEP:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    # Stub the heavy tail: lift() (param/buffer lifting) and the real EP ctor.
    monkeypatch.setattr(E, "ExportedProgram", _CapturingEP)
    monkeypatch.setattr(E, "lift", lambda gm_, sig_: (gm_, sig_, {}, {}))

    E.create_trt_exp_program(gm)

    specs = captured["graph_signature"].output_specs
    assert len(specs) == 2
    assert specs[0].kind == OutputKind.BUFFER_MUTATION
    assert specs[0].target == "state_0"
    assert specs[0].arg.name == state_new.name
    assert specs[1].kind == OutputKind.USER_OUTPUT

    # The trailing output is tagged with its buffer and reordered mutation-first.
    assert state_new.meta["_kv_mutation_target"] == "state_0"
    out_node = next(n for n in gm.graph.nodes if n.op == "output")
    assert out_node.args[0][0] is state_new
    assert out_node.args[0][1] is user_out


@pytest.mark.unit
def test_create_trt_exp_program_declares_write_only_copyback_buffer():
    """A copy-back buffer with no reader must still reach ``lift`` as a get_attr.

    ``transform``'s dead-code pass drops the get_attr of a buffer nothing reads,
    and ``lift`` derives BUFFER input specs from get_attr nodes alone, so the
    BUFFER_MUTATION declared here would name a buffer the signature never
    declares. Drives the real ``lift``/``ExportedProgram`` constructor so a
    regression reproduces the verifier error instead of passing vacuously.
    """
    pytest.importorskip("executorch.exir")
    from torch._subclasses.fake_tensor import FakeTensorMode

    fake_mode = FakeTensorMode()
    with fake_mode:
        fake_x = torch.randn(2)

    g = torch.fx.Graph()
    x = g.placeholder("x")
    x.meta["val"] = fake_x
    # Neither output reads the buffer -- the write-only shape.
    user_out = g.call_function(torch.ops.aten.mul.Tensor, (x, x))
    user_out.meta["val"] = fake_x
    state_new = g.call_function(torch.ops.aten.add.Tensor, (x, x))
    state_new.meta["val"] = fake_x
    g.output((user_out, state_new))

    gm = torch.fx.GraphModule(torch.nn.Module(), g)
    gm.register_buffer("state_0", torch.zeros(2))
    gm.recompile()
    gm.meta["_copyback_mutation_buffers"] = ["state_0"]

    ep = E.create_trt_exp_program(gm, arg_inputs=(torch.randn(2),))

    buffer_inputs = [
        s.target for s in ep.graph_signature.input_specs if s.kind == InputKind.BUFFER
    ]
    mutations = [
        (s.target, s.kind)
        for s in ep.graph_signature.output_specs
        if s.kind == OutputKind.BUFFER_MUTATION
    ]
    assert buffer_inputs == ["state_0"]
    assert mutations == [("state_0", OutputKind.BUFFER_MUTATION)]


@pytest.mark.unit
def test_create_trt_exp_program_does_not_duplicate_copyback_get_attr(monkeypatch):
    """The re-add is keyed on the existing get_attr targets, so a copy-back buffer
    that still has a live reader keeps its single get_attr, and a name that is not
    registered on the module is left alone rather than producing a dangling one."""
    pytest.importorskip("executorch.exir")

    g = torch.fx.Graph()
    x = g.placeholder("x")
    state_in = g.get_attr("state_0")
    user_out = g.call_function(torch.add, (x, x))
    state_new = g.call_function(torch.add, (state_in, x))
    g.output((user_out, state_new))
    root = torch.nn.Module()
    root.register_buffer("state_0", torch.zeros(2))
    gm = torch.fx.GraphModule(root, g)
    gm.recompile()
    # "ghost" is not registered on the module, so it cannot be re-added.
    gm.meta["_copyback_mutation_buffers"] = ["state_0", "ghost"]

    monkeypatch.setattr(E, "ExportedProgram", lambda **kwargs: None)
    monkeypatch.setattr(E, "lift", lambda gm_, sig_: (gm_, sig_, {}, {}))

    E.create_trt_exp_program(gm)

    targets = sorted(n.target for n in gm.graph.nodes if n.op == "get_attr")
    assert targets == ["state_0"]


@pytest.mark.unit
@pytest.mark.parametrize("use_legacy, expect_warning", [(True, False), (False, True)])
def test_save_warns_when_copyback_cannot_be_declared(
    monkeypatch, tmp_path, caplog, use_legacy, expect_warning
):
    """retrace=False only declares copy-back through the legacy exporter, so the
    non-legacy combination drops it. Say so rather than saving a signature that
    omits the update."""
    pytest.importorskip("executorch.exir")
    import torch_tensorrt
    from torch_tensorrt import _compile as C
    from torch_tensorrt.dynamo import _exporter as E

    monkeypatch.setattr(E, "export", lambda *a, **k: object())
    monkeypatch.setattr(E, "_declare_aliased_kv_mutations_on_ep", lambda ep, **k: ep)
    monkeypatch.setattr(C, "_normalize_engine_constants_to_python", lambda ep: None)
    monkeypatch.setattr(torch.export, "save", lambda *a, **k: None)

    g = torch.fx.Graph()
    g.output((g.placeholder("x"),))
    gm = torch.fx.GraphModule(torch.nn.Module(), g)
    gm.meta["_copyback_mutation_buffers"] = ["state_0"]

    with caplog.at_level("WARNING"):
        torch_tensorrt.save(
            gm,
            str(tmp_path / "out.pt2"),
            output_format="exported_program",
            retrace=False,
            use_legacy_exporter=use_legacy,
        )

    warned = any("copy-back" in r.message for r in caplog.records)
    assert warned is expect_warning
