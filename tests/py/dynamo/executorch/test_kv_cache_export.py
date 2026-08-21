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
def test_declare_aliased_kv_mutations_reads_engine_metadata_only(monkeypatch):
    """The pass reads binding names and aliased_io and never the engine payload, so
    it must ask for the metadata-only record. Reading the full record re-serializes
    the ICudaEngine -- 0.4 s and a 67 MB transient string for a ~50 MB engine -- and
    the loop runs before the pass can tell whether any engine has aliased I/O at all,
    so a program with no aliased KV pays that per engine and then returns unchanged.
    """
    pytest.importorskip("executorch.exir")
    import torch_tensorrt.dynamo.runtime._serialized_engine_layout as L
    import torch_tensorrt.dynamo.runtime._TorchTensorRTModule as M
    import torch_tensorrt.executorch.backend as B

    exec_target = torch.ops.tensorrt.execute_engine.default

    g = torch.fx.Graph()
    tokens = g.placeholder("tokens")
    engine = g.placeholder("engine")
    eng = g.call_function(exec_target, ([tokens], engine))
    user_out = g.call_function(operator.getitem, (eng, 0))
    g.output((user_out,))

    out_val = torch.zeros(1)
    tokens.meta["val"] = torch.zeros(1)
    engine.meta["val"] = None
    eng.meta["val"] = [out_val]
    user_out.meta["val"] = out_val
    gm = torch.fx.GraphModule(torch.nn.Module(), g)

    ep = SimpleNamespace(
        graph_module=gm,
        graph_signature=SimpleNamespace(
            inputs_to_buffers={},
            input_specs=[],
            output_specs=[
                OutputSpec(
                    OutputKind.USER_OUTPUT, TensorArgument(name=user_out.name), None
                )
            ],
        ),
    )

    calls = []
    info = ["x"] * (L.ALIASED_IO_IDX + 1)

    def _spy(ep_, node, **kwargs):
        calls.append(kwargs)
        return info

    monkeypatch.setattr(B, "_get_engine_info_for_node", _spy)
    monkeypatch.setattr(M, "deserialize_aliased_io", lambda s: {})

    assert E._declare_aliased_kv_mutations_on_ep(ep) is ep
    assert calls == [{"metadata_only": True}]


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
    monkeypatch.setattr(B, "_get_engine_info_for_node", lambda ep_, n, **kw: info)
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
    monkeypatch.setattr(B, "_get_engine_info_for_node", lambda ep_, n, **kw: info)
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
    a return it never had.

    The legacy exporter's shape -- mutation declared *and* the trailing value already
    consumed -- never reaches here: ``save()`` passes no ``copyback_buffers`` on that
    path (see ``test_saved_copyback_program_reloads_and_keeps_its_user_output``)."""
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
            self.__dict__.update(kwargs)
            captured.update(kwargs)

    monkeypatch.setattr(E, "ExportedProgram", _CapturingEP)

    result = E._declare_aliased_kv_mutations_on_ep(ep, copyback_buffers=["state_0"])

    specs = result.graph_signature.output_specs
    # state_0 keeps its single existing mutation spec; no second one is added.
    mutations = [s for s in specs if s.kind == OutputKind.BUFFER_MUTATION]
    assert [s.target for s in mutations] == ["state_0"]
    # The trailing copy-back value is gone from the user outputs.
    user_outputs = [s for s in specs if s.kind == OutputKind.USER_OUTPUT]
    assert [s.arg.name for s in user_outputs] == ["user_out"]

    out_node = next(n for n in gm.graph.nodes if n.op == "output")
    assert list(out_node.args[0]) == [state_new, user_out]


@pytest.mark.unit
def test_declare_copyback_is_idempotent(monkeypatch):
    """A second run is a no-op. The copy-back slice is positional over the trailing
    outputs, so slicing twice detaches a genuine user output and the saved program then
    fails to load."""
    pytest.importorskip("executorch.exir")

    g = torch.fx.Graph()
    x = g.placeholder("x")
    state_in = g.placeholder("state_in")
    state_new = g.call_function(torch.add, (state_in, x))
    user_out = g.call_function(torch.mul, (x, x))
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
            OutputSpec(
                OutputKind.USER_OUTPUT, TensorArgument(name=user_out.name), None
            ),
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

    class _CapturingEP:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
            # A real ExportedProgram builds a *new* GraphModule and shallow-copies
            # root.meta into it, so the marker the pass sets on root before
            # construction is readable off the resulting program's graph_module --
            # which is what lets a program the pass produced be fed back through it.
            # Exposing root directly reproduces that readback. The mechanism the real
            # class relies on is the meta copy, and it is the part a torch upgrade
            # could take away.
            self.graph_module = kwargs["root"]

    monkeypatch.setattr(E, "ExportedProgram", _CapturingEP)

    first = E._declare_aliased_kv_mutations_on_ep(ep, copyback_buffers=["state_0"])
    second = E._declare_aliased_kv_mutations_on_ep(first, copyback_buffers=["state_0"])

    assert second is first
    specs = second.graph_signature.output_specs
    mutations = [s for s in specs if s.kind == OutputKind.BUFFER_MUTATION]
    assert [s.target for s in mutations] == ["state_0"]
    user_outputs = [s for s in specs if s.kind == OutputKind.USER_OUTPUT]
    assert [s.arg.name for s in user_outputs] == [user_out.name]


@pytest.mark.unit
def test_declare_copyback_runs_without_executorch_extra(monkeypatch):
    """Pins that an unimportable ``torch_tensorrt.executorch.backend`` does not suppress
    copy-back declaration: the trailing copy-back value is still reclassified as a
    BUFFER_MUTATION of its buffer.

    The graph carries an ``execute_engine`` node so the engine loop is reachable. Without
    one the loop filters every node on target anyway and the skip goes untested."""
    import sys

    # Make `from torch_tensorrt.executorch.backend import ...` raise ImportError,
    # simulating a plain install without the [executorch] extra.
    monkeypatch.setitem(sys.modules, "torch_tensorrt.executorch.backend", None)

    g = torch.fx.Graph()
    x = g.placeholder("x")
    state_in = g.placeholder("state_in")
    engine = g.placeholder("engine")
    eng = g.call_function(torch.ops.tensorrt.execute_engine.default, ([x], engine))
    eng.meta["val"] = [torch.zeros(1)]
    state_new = g.call_function(torch.add, (state_in, x))
    user_out = g.call_function(torch.mul, (x, x))
    # lift appended the copy-back value (state_new) as the trailing user output; the
    # mutation is not yet declared.
    g.output((user_out, state_new))
    gm = torch.fx.GraphModule(torch.nn.Module(), g)

    sig = SimpleNamespace(
        inputs_to_buffers={"state_in": "state_0"},
        input_specs=[
            InputSpec(
                InputKind.BUFFER, TensorArgument(name="state_in"), "state_0", True
            ),
            InputSpec(InputKind.USER_INPUT, TensorArgument(name="x"), None),
            InputSpec(InputKind.USER_INPUT, TensorArgument(name="engine"), None),
        ],
        output_specs=[
            OutputSpec(
                OutputKind.USER_OUTPUT, TensorArgument(name=user_out.name), None
            ),
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

    class _CapturingEP:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    monkeypatch.setattr(E, "ExportedProgram", _CapturingEP)

    result = E._declare_aliased_kv_mutations_on_ep(ep, copyback_buffers=["state_0"])

    specs = result.graph_signature.output_specs
    mutations = [s for s in specs if s.kind == OutputKind.BUFFER_MUTATION]
    assert [s.target for s in mutations] == ["state_0"]
    user_outputs = [s for s in specs if s.kind == OutputKind.USER_OUTPUT]
    assert [s.arg.name for s in user_outputs] == [user_out.name]


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
@pytest.mark.parametrize(
    "use_legacy, output_format, expect_warning",
    [
        (True, "exported_program", False),
        (False, "exported_program", True),
        (True, "executorch", False),
        (False, "executorch", False),
    ],
)
def test_save_warns_when_copyback_cannot_be_declared(
    monkeypatch, tmp_path, caplog, use_legacy, output_format, expect_warning
):
    """One combination of ``retrace=False`` leaves copy-back undeclared, and only it
    warns.

    Under ``exported_program`` the legacy exporter is what declares copy-back, so the
    non-legacy pairing drops it and the saved signature would omit the update
    silently. Under ``executorch`` the program goes on to
    ``torch_tensorrt.executorch.export``, which declares copy-back for every source
    shape it accepts, so neither pairing drops it -- warning there would contradict
    what the branch does.

    The two formats reach the warning through the same block, so pinning them
    together is what keeps them from drifting apart.
    """
    pytest.importorskip("executorch.exir")
    import torch_tensorrt
    from torch_tensorrt import _compile as C
    from torch_tensorrt.dynamo import _exporter as E

    monkeypatch.setattr(E, "export", lambda *a, **k: object())
    monkeypatch.setattr(E, "_declare_aliased_kv_mutations_on_ep", lambda ep, **k: ep)
    monkeypatch.setattr(C, "_normalize_engine_constants_to_python", lambda ep: None)
    monkeypatch.setattr(C, "_save_as_executorch", lambda *a, **k: None)
    monkeypatch.setattr(torch.export, "save", lambda *a, **k: None)

    g = torch.fx.Graph()
    g.output((g.placeholder("x"),))
    gm = torch.fx.GraphModule(torch.nn.Module(), g)
    gm.meta["_copyback_mutation_buffers"] = ["state_0"]

    suffix = "pte" if output_format == "executorch" else "pt2"
    with caplog.at_level("WARNING"):
        torch_tensorrt.save(
            gm,
            str(tmp_path / f"out.{suffix}"),
            output_format=output_format,
            retrace=False,
            use_legacy_exporter=use_legacy,
        )

    warned = any("copy-back" in r.message for r in caplog.records)
    assert warned is expect_warning


@pytest.mark.unit
@pytest.mark.parametrize("use_legacy", [True, False])
def test_saved_copyback_program_reloads_and_keeps_its_user_output(tmp_path, use_legacy):
    """A saved copy-back program declares the mutation, returns exactly the outputs the
    source module returned, and updates the buffer when run after a reload.

    ``create_trt_exp_program`` declares the mutation itself and consumes the trailing
    value while doing so, so re-declaring it slices a genuine user output and relabels
    it as the buffer's new contents; the saved program then fails to load."""
    pytest.importorskip("executorch.exir")

    import torch_tensorrt
    from torch_tensorrt.dynamo.lowering._buffer_lifting import (
        inline_lifted_buffers_into_gm,
        lift_mutated_buffers,
    )

    class _CopyBackModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("state", torch.zeros(4))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            self.state.copy_(self.state + x.sum())
            return x * 3.0

    dim = torch.export.Dim("d", min=2, max=64)
    ep = torch.export.export(
        _CopyBackModel(), (torch.ones(8),), dynamic_shapes={"x": {0: dim}}
    )
    gm, lifted = lift_mutated_buffers(ep.module())
    gm = inline_lifted_buffers_into_gm(gm, lifted)
    assert gm.meta["_copyback_mutation_buffers"] == ["state"]

    path = str(tmp_path / "out.pt2")
    torch_tensorrt.save(
        gm,
        path,
        output_format="exported_program",
        retrace=True,
        use_legacy_exporter=use_legacy,
        dynamic_shapes={"x": {0: dim}},
        arg_inputs=(torch.ones(8),),
    )

    loaded = torch.export.load(path)
    mutations = [
        s.target
        for s in loaded.graph_signature.output_specs
        if s.kind == OutputKind.BUFFER_MUTATION
    ]
    assert mutations == ["state"]

    module = loaded.module()
    before = dict(module.named_buffers())["state"].clone()
    outputs = torch.utils._pytree.tree_leaves(module(torch.ones(8)))
    assert len(outputs) == 1
    torch.testing.assert_close(outputs[0], torch.full((8,), 3.0))
    assert not torch.allclose(dict(module.named_buffers())["state"], before)


@pytest.mark.unit
def test_serialized_engine_rejects_copyback_buffers():
    """``lift_mutable_buffers=True`` on a buffer whose write the engine cannot alias
    raises, instead of returning an engine whose buffer never updates."""
    from torch_tensorrt.dynamo._compiler import (
        convert_exported_program_to_serialized_trt_engine,
    )

    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("state", torch.zeros(4))

        def forward(self, x):
            self.state.add_(x)
            return self.state.sum()

    exp_program = torch.export.export(M(), (torch.ones(4),))

    with pytest.raises(RuntimeError, match="cannot express the write-back") as excinfo:
        convert_exported_program_to_serialized_trt_engine(
            exp_program,
            arg_inputs=[torch.ones(4)],
            lift_mutable_buffers=True,
            min_block_size=1,
        )
    # The offending buffer has to be named, or the user cannot act on the error.
    assert "state" in str(excinfo.value)


@pytest.mark.unit
@pytest.mark.parametrize(
    "source_shape",
    ["graph_module", "graph_module_no_retrace", "exported_program", "mapping"],
)
def test_executorch_export_declares_copyback_for_every_source_shape(
    monkeypatch, source_shape
):
    """Every source shape ``torch_tensorrt.executorch.export`` accepts reaches the Edge
    program with its copy-back buffer declared as a BUFFER_MUTATION, on every method,
    and with the module's own output still there.

    Undeclared, the buffer's new value stays a trailing user output that nothing copies
    back and the buffer loads frozen. Only the legacy exporter declares the mutation
    while building the program, so some of these sources arrive already declared and
    have to come through a second declaration attempt unharmed.
    """
    pytest.importorskip("executorch.exir")
    from torch_tensorrt.dynamo.lowering._buffer_lifting import (
        inline_lifted_buffers_into_gm,
        lift_mutated_buffers,
    )
    from torch_tensorrt.executorch import export as executorch_export

    class _CopyBackModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("state", torch.zeros(4))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            self.state.copy_(self.state + x)
            return x * 3.0

    def _graph_module() -> torch.fx.GraphModule:
        """The shape ``compile()`` hands on: the mutated buffer lifted to an input, its
        new value appended as a trailing output, and its name left in the meta."""
        program = torch.export.export(_CopyBackModel(), (torch.ones(4),))
        gm, lifted = lift_mutated_buffers(program.module())
        gm = inline_lifted_buffers_into_gm(gm, lifted)
        assert gm.meta["_copyback_mutation_buffers"] == ["state"]
        return gm

    def _exported(use_legacy_exporter: bool) -> torch.export.ExportedProgram:
        return E.export(
            _graph_module(),
            arg_inputs=(torch.ones(4),),
            use_legacy_exporter=use_legacy_exporter,
        )

    # A GraphModule source is retraced against example inputs placed on the default
    # device, and this model's buffer is on the host.
    monkeypatch.setattr(
        "torch_tensorrt.dynamo._defaults.default_device", lambda: torch.device("cpu")
    )

    if source_shape == "graph_module":
        source = _graph_module()
        options = {"arg_inputs": (torch.ones(4),), "retrace": True}
        methods = ("forward",)
    elif source_shape == "graph_module_no_retrace":
        # Retracing is off by default, which routes through the legacy exporter and
        # its transform-time declaration.
        source = _graph_module()
        options = {"arg_inputs": (torch.ones(4),)}
        methods = ("forward",)
    elif source_shape == "exported_program":
        source, options, methods = _exported(False), {}, ("forward",)
    else:
        # decode comes from the legacy exporter, which declared its mutation already:
        # it is the method that must not end up describing that mutation twice.
        source = {"prefill": _exported(False), "decode": _exported(True)}
        options, methods = {}, ("prefill", "decode")

    edge_program = executorch_export(source, **options)

    for method in methods:
        specs = edge_program.exported_program(method).graph_signature.output_specs
        assert [(spec.kind, spec.target) for spec in specs] == [
            (OutputKind.BUFFER_MUTATION, "state"),
            (OutputKind.USER_OUTPUT, None),
        ]


@pytest.mark.unit
def test_executorch_export_leaves_the_source_program_intact(tmp_path):
    """A caller who hands ``torch_tensorrt.executorch.export`` their own
    ExportedProgram keeps it usable afterwards.

    The declaration pass rewrites the graph's output node in place and returns the
    matching signature on a new program, so running it on the caller's object leaves
    that object's graph and signature describing different outputs. Nothing on the
    program raises for that -- ``module()`` and calling it still work -- until the
    ExportedProgram verifier runs, so the check below saves the program as well as
    comparing it.
    """
    pytest.importorskip("executorch.exir")
    from torch_tensorrt.dynamo.lowering._buffer_lifting import (
        inline_lifted_buffers_into_gm,
        lift_mutated_buffers,
    )
    from torch_tensorrt.executorch import export as executorch_export

    class _CopyBackModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("state", torch.zeros(4))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            self.state.copy_(self.state + x)
            return x * 3.0

    program = torch.export.export(_CopyBackModel(), (torch.ones(4),))
    gm, lifted = lift_mutated_buffers(program.module())
    source = E.export(
        inline_lifted_buffers_into_gm(gm, lifted),
        arg_inputs=(torch.ones(4),),
        use_legacy_exporter=False,
    )

    def _outputs(exported_program):
        output_node = next(
            node
            for node in exported_program.graph_module.graph.nodes
            if node.op == "output"
        )
        return (
            [getattr(arg, "name", arg) for arg in output_node.args[0]],
            [
                (spec.kind, spec.target)
                for spec in exported_program.graph_signature.output_specs
            ],
        )

    before = _outputs(source)

    def _edge_output_specs(edge_program):
        specs = edge_program.exported_program("forward").graph_signature.output_specs
        return [(spec.kind, spec.target) for spec in specs]

    declared = [
        (OutputKind.BUFFER_MUTATION, "state"),
        (OutputKind.USER_OUTPUT, None),
    ]
    assert _edge_output_specs(executorch_export(source)) == declared
    assert _outputs(source) == before
    torch.export.save(source, str(tmp_path / "source.pt2"))
    # Exporting the same program twice has to give the same Edge program, which it
    # cannot if the first export consumed something from it.
    assert _edge_output_specs(executorch_export(source)) == declared
