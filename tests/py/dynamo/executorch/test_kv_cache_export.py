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
from torch.export.exported_program import (
    InputKind,
    InputSpec,
    OutputKind,
    OutputSpec,
    TensorArgument,
)
from torch_tensorrt.dynamo import _exporter as E


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
        graph_signature=SimpleNamespace(inputs_to_buffers={}),
    )
    assert E._declare_aliased_kv_mutations_on_ep(ep) is ep


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

    # The engine's meta['val'] is extended to cover the previously-dropped output.
    assert len(eng.meta["val"]) == 2
