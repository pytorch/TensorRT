"""Export-side coverage for zero-copy aliased KV buffers.

Two halves have to agree for a zero-copy ``.pte`` to be correct:

  * ``rewire_aliased_mutations_to_buffers`` declares the buffer to be its own
    mutation result, which removes ExecuTorch's copy-back and takes the aliased
    output out of the delegate.
  * ``unstage_aliased_buffers_pass`` removes the host staging copy so the
    engine's in-place write lands in the caller's buffer.

The interesting failures are all silent -- a rewired mutation whose buffer is
still staged simply never updates -- so most of what is asserted here is which
mutations are left alone, which mis-shapes raise, and that a marked buffer that
is never un-staged is caught rather than dropped.
"""

import json
import logging
import operator
import re
from types import SimpleNamespace

import pytest

pytest.importorskip("executorch.exir")

import torch  # noqa: E402
import torch_tensorrt  # noqa: E402
from executorch.exir.backend.compile_spec_schema import CompileSpec  # noqa: E402
from executorch.exir.delegate import executorch_call_delegate  # noqa: E402
from executorch.exir.schema import DeviceType  # noqa: E402
from torch.export.exported_program import (  # noqa: E402
    OutputKind,
    OutputSpec,
    TensorArgument,
)
from torch_tensorrt.executorch import _zero_copy as Z  # noqa: E402
from torch_tensorrt.executorch.backend import (  # noqa: E402
    ZERO_COPY_KV_COMPILE_SPEC_KEY,
)

# The graphs below are built around torch.ops.tensorrt.execute_engine, which only
# exists once the Torch-TensorRT runtime operator library has loaded.
pytestmark = pytest.mark.skipif(
    not torch_tensorrt.ENABLED_FEATURES.torch_tensorrt_runtime,
    reason="Torch-TensorRT runtime operators are not available",
)


def _require_real_engine():
    """Gate the tests that build a real engine, at run time rather than collection.

    A decorator ``skipif`` resolves while pytest collects, and on a remote-GPU
    runner collection happens off the GPU host, so the skip is frozen in before
    any GPU is attached. These are the only tests that put this feature on a real
    engine, so where that happens the lane stays green with that coverage gone.
    The other CUDA gates in this directory are runtime gates for the same reason.
    """
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA + TensorRT for a real engine")


def _patch_engine_metadata(monkeypatch, *, aliased_io, input_names, output_names):
    """Make every engine node report one fixed set of bindings and aliases."""
    import torch_tensorrt.dynamo.runtime._serialized_engine_layout as layout
    import torch_tensorrt.dynamo.runtime._TorchTensorRTModule as trt_module
    import torch_tensorrt.executorch._export_utils as export_utils

    info = ["x"] * (layout.ALIASED_IO_IDX + 1)
    info[layout.INPUT_BINDING_NAMES_IDX] = "IN"
    info[layout.OUTPUT_BINDING_NAMES_IDX] = "OUT"

    # The rewiring resolves engine info through _resolve_engine_info (the node is
    # still an execute_engine at this stage), so that is what to fake. The stub
    # requires metadata_only: without it the read goes through
    # TRTEngine.__getstate__ and re-serializes the whole engine to recover the
    # binding names and aliased_io, which are the only fields wanted here.
    def _fake_resolve(ep, node, *, metadata_only=False):
        assert metadata_only, "zero-copy reads binding metadata, not the engine"
        return info

    monkeypatch.setattr(export_utils, "_resolve_engine_info", _fake_resolve)
    monkeypatch.setattr(trt_module, "deserialize_aliased_io", lambda s: aliased_io)
    monkeypatch.setattr(
        layout,
        "deserialize_binding_names",
        lambda s: list(input_names) if s == "IN" else list(output_names),
    )


def _kv_program(*, mutation_value="aliased_getitem"):
    """A one-engine program: engine(k_buffer, tokens) -> (logits, k_out).

    ``mutation_value`` picks what the KV buffer's BUFFER_MUTATION is bound to:

      * ``"aliased_getitem"``: the engine's aliased output (what export
        declares for a caller-owned KV cache).
      * ``"user_getitem"``: a non-aliased engine output, the shape a copy-back
        mutation has.
      * ``"external_op"``: a value produced outside the engine.
    """
    graph = torch.fx.Graph()
    k_buffer = graph.placeholder("b_k_0")
    tokens = graph.placeholder("tokens")
    engine = graph.placeholder("engine")
    engine_call = graph.call_function(
        torch.ops.tensorrt.execute_engine.default, ([k_buffer, tokens], engine)
    )
    logits = graph.call_function(operator.getitem, (engine_call, 0))
    k_out = graph.call_function(operator.getitem, (engine_call, 1))
    mutation = {
        "aliased_getitem": k_out,
        "user_getitem": logits,
        "external_op": None,
    }[mutation_value]
    if mutation is None:
        mutation = graph.call_function(torch.add, (k_buffer, k_buffer))
    graph.output((mutation, logits))
    graph_module = torch.fx.GraphModule(torch.nn.Module(), graph)

    signature = SimpleNamespace(
        inputs_to_buffers={"b_k_0": "k_0"},
        input_specs=[],
        output_specs=[
            OutputSpec(
                OutputKind.BUFFER_MUTATION, TensorArgument(name=mutation.name), "k_0"
            ),
            OutputSpec(OutputKind.USER_OUTPUT, TensorArgument(name=logits.name), None),
        ],
    )
    program = SimpleNamespace(
        graph_module=graph_module,
        graph_signature=signature,
        _graph_signature=signature,
    )
    return program, k_buffer, k_out


@pytest.mark.unit
def test_rewire_points_the_mutation_at_its_buffer_and_marks_it(monkeypatch):
    """The aliased output is replaced by the buffer itself and then dies.

    With the mutation bound to the placeholder there is nothing for ExecuTorch
    to copy back, and with no other user the getitem leaves the graph -- which
    is what takes the aliased output out of the delegate. The elided output's
    binding name is returned so the backend can exempt exactly that one.
    """
    program, k_buffer, k_out = _kv_program()
    _patch_engine_metadata(
        monkeypatch,
        aliased_io={"out_k": ("k_in", "kv_cache_update")},
        input_names=["k_in", "tokens"],
        output_names=["logits", "out_k"],
    )

    assert Z.rewire_aliased_mutations_to_buffers(program) == ["out_k"]

    specs = program._graph_signature.output_specs
    assert specs[0].kind == OutputKind.BUFFER_MUTATION
    assert specs[0].target == "k_0"
    assert specs[0].arg.name == k_buffer.name
    output_node = program.graph_module.graph.output_node()
    assert output_node.args[0][0] is k_buffer
    assert k_out not in program.graph_module.graph.nodes
    assert k_buffer.meta["_torch_tensorrt_aliased_buffer"] is True


@pytest.mark.unit
@pytest.mark.parametrize("mutation_value", ["user_getitem", "external_op"])
def test_rewire_leaves_mutations_the_engine_does_not_alias(monkeypatch, mutation_value):
    """Only a mutation the engine satisfies in place may be rewired.

    A copy-back mutation ("user_getitem") and a mutation computed outside the
    engine ("external_op") both need their value copied into the buffer. Both
    look exactly like an aliased mutation in the graph, so the discriminator has
    to be the engine's own aliased_io -- rewiring either would delete a real
    update with no error.
    """
    program, k_buffer, _ = _kv_program(mutation_value=mutation_value)
    original_spec = program._graph_signature.output_specs[0]
    _patch_engine_metadata(
        monkeypatch,
        aliased_io={"out_k": ("k_in", "kv_cache_update")},
        input_names=["k_in", "tokens"],
        output_names=["logits", "out_k"],
    )

    assert Z.rewire_aliased_mutations_to_buffers(program) == []
    assert program._graph_signature.output_specs[0] is original_spec
    assert "_torch_tensorrt_aliased_buffer" not in k_buffer.meta


@pytest.mark.unit
@pytest.mark.parametrize("unresolvable", ["unknown-input", "index-past-the-args"])
def test_rewire_skips_an_alias_whose_input_does_not_resolve(monkeypatch, unresolvable):
    """An aliased_io entry naming an input this delegate does not take is skipped.

    Two ways it can fail to resolve: the name is not one of the engine's input
    bindings at all, or it is but its index is past the end of the delegate's
    argument list. Neither leaves a mutation that could be rewired, and
    ``_declare_aliased_kv_mutations_on_ep`` has already warned about both for the
    same engine, so both are skipped rather than reported here.
    """
    if unresolvable == "unknown-input":
        aliased_input, input_names = "not_an_input", ["k_in", "tokens"]
    else:
        # A real binding name, but the third one, while the engine node takes two
        # arguments -- so the index is past the end of the argument list.
        aliased_input, input_names = "spare", ["k_in", "tokens", "spare"]
    program, k_buffer, _ = _kv_program()
    _patch_engine_metadata(
        monkeypatch,
        aliased_io={"out_k": (aliased_input, "kv_cache_update")},
        input_names=input_names,
        output_names=["logits", "out_k"],
    )

    assert Z.rewire_aliased_mutations_to_buffers(program) == []
    assert "_torch_tensorrt_aliased_buffer" not in k_buffer.meta


def _mixed_program():
    """One engine, one method, both kinds of mutation at once.

    ``engine(b_k_0, b_state_0, tokens) -> (logits, out_k, out_state)`` where the
    engine aliases only ``out_k`` onto ``b_k_0``. ``b_state_0`` is the #4459
    shape: a mutable buffer with no aliasing available, whose new value
    ``lift_mutated_buffers`` appended as a trailing output for ExecuTorch to copy
    back. In the graph the two mutations are indistinguishable -- each is a
    ``getitem`` off the engine node whose buffer is also an engine input.
    """
    graph = torch.fx.Graph()
    k_buffer = graph.placeholder("b_k_0")
    state_buffer = graph.placeholder("b_state_0")
    tokens = graph.placeholder("tokens")
    engine = graph.placeholder("engine")
    engine_call = graph.call_function(
        torch.ops.tensorrt.execute_engine.default,
        ([k_buffer, state_buffer, tokens], engine),
    )
    logits = graph.call_function(operator.getitem, (engine_call, 0))
    k_out = graph.call_function(operator.getitem, (engine_call, 1))
    state_out = graph.call_function(operator.getitem, (engine_call, 2))
    graph.output((k_out, state_out, logits))
    graph_module = torch.fx.GraphModule(torch.nn.Module(), graph)

    signature = SimpleNamespace(
        inputs_to_buffers={"b_k_0": "k_0", "b_state_0": "state_0"},
        input_specs=[],
        output_specs=[
            OutputSpec(
                OutputKind.BUFFER_MUTATION, TensorArgument(name=k_out.name), "k_0"
            ),
            OutputSpec(
                OutputKind.BUFFER_MUTATION,
                TensorArgument(name=state_out.name),
                "state_0",
            ),
            OutputSpec(OutputKind.USER_OUTPUT, TensorArgument(name=logits.name), None),
        ],
    )
    program = SimpleNamespace(
        graph_module=graph_module,
        graph_signature=signature,
        _graph_signature=signature,
    )
    return program, k_buffer, state_buffer, k_out, state_out


@pytest.mark.unit
def test_rewire_keeps_the_copyback_in_a_method_that_also_has_an_aliased_kv(monkeypatch):
    """Zero-copy and a copy-back buffer may share one method, and must not mix.

    Rewiring the copy-back would delete a real update with no error, and refusing
    the aliased one would give up the whole feature for any model carrying a
    non-KV mutable buffer beside its cache. The engine's own aliased_io is what
    separates them: only ``out_k`` is listed, so only ``b_k_0`` is rewired and
    only its binding name is offered to the backend as elided. ``b_state_0``
    keeps its delegate output, which is the value ExecuTorch copies back.
    """
    program, k_buffer, state_buffer, k_out, state_out = _mixed_program()
    _patch_engine_metadata(
        monkeypatch,
        aliased_io={"out_k": ("k_in", "kv_cache_update")},
        input_names=["k_in", "state_in", "tokens"],
        output_names=["logits", "out_k", "out_state"],
    )

    assert Z.rewire_aliased_mutations_to_buffers(program) == ["out_k"]

    kv_spec, state_spec, _ = program._graph_signature.output_specs
    assert kv_spec.arg.name == k_buffer.name
    assert k_buffer.meta["_torch_tensorrt_aliased_buffer"] is True
    assert k_out not in program.graph_module.graph.nodes

    assert state_spec.kind == OutputKind.BUFFER_MUTATION
    assert state_spec.target == "state_0"
    assert state_spec.arg.name == state_out.name
    assert state_out in program.graph_module.graph.nodes
    assert program.graph_module.graph.output_node().args[0][1] is state_out
    # Un-staging keys on this mark, so leaving it off b_state_0 is what keeps the
    # copy-back buffer's staging copy -- the engine writes that copy and
    # ExecuTorch copies it back, exactly as without zero-copy.
    assert "_torch_tensorrt_aliased_buffer" not in state_buffer.meta


@pytest.mark.unit
def test_rewire_is_a_noop_without_aliased_io(monkeypatch):
    program, k_buffer, _ = _kv_program()
    _patch_engine_metadata(
        monkeypatch,
        aliased_io={},
        input_names=["k_in", "tokens"],
        output_names=["logits", "out_k"],
    )

    assert Z.rewire_aliased_mutations_to_buffers(program) == []
    assert "_torch_tensorrt_aliased_buffer" not in k_buffer.meta


@pytest.mark.unit
def test_rewire_rejects_an_engine_whose_every_output_is_aliased(monkeypatch):
    """A delegate with no outputs at all is not a shape anything supports.

    Nothing downstream reports it: the runtime reads elision off a single
    argument count, which a zero-output delegate satisfies, and the delegate
    itself is a pure node a later graph-wide dead-code elimination can erase.
    So the failure has to be raised here.
    """
    graph = torch.fx.Graph()
    k_buffer = graph.placeholder("b_k_0")
    engine = graph.placeholder("engine")
    engine_call = graph.call_function(
        torch.ops.tensorrt.execute_engine.default, ([k_buffer], engine)
    )
    k_out = graph.call_function(operator.getitem, (engine_call, 0))
    graph.output((k_out,))
    graph_module = torch.fx.GraphModule(torch.nn.Module(), graph)
    signature = SimpleNamespace(
        inputs_to_buffers={"b_k_0": "k_0"},
        input_specs=[],
        output_specs=[
            OutputSpec(
                OutputKind.BUFFER_MUTATION, TensorArgument(name=k_out.name), "k_0"
            )
        ],
    )
    program = SimpleNamespace(
        graph_module=graph_module,
        graph_signature=signature,
        _graph_signature=signature,
    )
    _patch_engine_metadata(
        monkeypatch,
        aliased_io={"out_k": ("k_in", "kv_cache_update")},
        input_names=["k_in"],
        output_names=["out_k"],
    )

    with pytest.raises(RuntimeError, match="no outputs at all"):
        Z.rewire_aliased_mutations_to_buffers(program)


@pytest.mark.unit
@pytest.mark.parametrize("dead_chain_length", [1, 2])
def test_rewire_rejects_an_engine_whose_only_other_output_is_dead(
    monkeypatch, dead_chain_length
):
    """A surviving-but-unread output does not keep the engine's delegate alive.

    The engine has two outputs: an aliased buffer this elides, and a second one
    whose consumers end in nothing. Counting the second as an output would let
    the check pass, and the dead-code elimination would erase the whole chain,
    leaving exactly the zero-output delegate the check exists to refuse. The
    two-link chain is the case a check reading only the engine's immediate users
    misses: that first link does have a user, so it reads as live.
    """
    graph = torch.fx.Graph()
    k_buffer = graph.placeholder("b_k_0")
    engine = graph.placeholder("engine")
    engine_call = graph.call_function(
        torch.ops.tensorrt.execute_engine.default, ([k_buffer], engine)
    )
    k_out = graph.call_function(operator.getitem, (engine_call, 0))
    dead = graph.call_function(operator.getitem, (engine_call, 1))
    for _ in range(dead_chain_length - 1):
        dead = graph.call_function(torch.add, (dead, dead))
    graph.output((k_out,))
    graph_module = torch.fx.GraphModule(torch.nn.Module(), graph)
    signature = SimpleNamespace(
        inputs_to_buffers={"b_k_0": "k_0"},
        input_specs=[],
        output_specs=[
            OutputSpec(
                OutputKind.BUFFER_MUTATION, TensorArgument(name=k_out.name), "k_0"
            )
        ],
    )
    program = SimpleNamespace(
        graph_module=graph_module,
        graph_signature=signature,
        _graph_signature=signature,
    )
    _patch_engine_metadata(
        monkeypatch,
        aliased_io={"out_k": ("k_in", "kv_cache_update")},
        input_names=["k_in"],
        output_names=["out_k", "out_dead"],
    )

    with pytest.raises(RuntimeError, match="no outputs at all"):
        Z.rewire_aliased_mutations_to_buffers(program)


def _zero_copy_specs(*names):
    """What ``TensorRTPartitioner`` stamps on the delegate whose engine elided.

    The value is the list of aliased output binding names, one per buffer the
    engine writes in place, which is what the count checks read.
    """
    return [
        CompileSpec(
            ZERO_COPY_KV_COMPILE_SPEC_KEY,
            json.dumps(list(names or ("out_k",))).encode(),
        )
    ]


def _staged_delegate_graph(
    *, backend_id="TensorRTBackend", device=DeviceType.CUDA, compile_specs=None
):
    """A lowered graph: delegate(lowered, _h2d_copy(k_buffer), _h2d_copy(tokens))."""
    graph = torch.fx.Graph()
    k_buffer = graph.placeholder("b_k_0")
    tokens = graph.placeholder("tokens")
    lowered = graph.get_attr("lowered_module_0")
    h2d = torch.ops.et_copy._h2d_copy.default
    staged_k = graph.call_function(h2d, (k_buffer,))
    staged_tokens = graph.call_function(h2d, (tokens,))
    delegate = graph.call_function(
        executorch_call_delegate, (lowered, staged_k, staged_tokens)
    )
    graph.output((delegate,))

    root = torch.nn.Module()
    root.lowered_module_0 = SimpleNamespace(
        backend_id=backend_id, compile_specs=compile_specs
    )
    graph_module = torch.fx.GraphModule(root, graph)

    for node, spec_device in (
        (k_buffer, DeviceType.CPU),
        (tokens, DeviceType.CPU),
        (staged_k, device),
        (staged_tokens, device),
    ):
        node.meta["spec"] = SimpleNamespace(device=spec_device, device_index=3)
    return graph_module, k_buffer, staged_k, delegate


@pytest.mark.unit
def test_unstage_feeds_the_buffer_straight_to_the_delegate():
    """The marked buffer replaces its staging copy and moves to the device.

    Moving the spec is not cosmetic: memory planning reads it, and a buffer left
    in a host arena is somewhere the engine cannot write.
    """
    graph_module, k_buffer, staged_k, delegate = _staged_delegate_graph(
        compile_specs=_zero_copy_specs()
    )
    k_buffer.meta["_torch_tensorrt_aliased_buffer"] = True

    assert Z._unstage_aliased_buffers(graph_module) == 1

    assert delegate.args[1] is k_buffer
    assert k_buffer.meta["spec"].device == DeviceType.CUDA
    assert k_buffer.meta["spec"].device_index == 3
    # The other input is an ordinary one and keeps its staging copy.
    assert delegate.args[2] is not None
    assert delegate.args[2].target is torch.ops.et_copy._h2d_copy.default
    # The orphaned staging is erased, but only it -- the other staging survives.
    assert staged_k not in graph_module.graph.nodes


@pytest.mark.unit
def test_unstage_keeps_staging_for_an_unmarked_buffer():
    graph_module, k_buffer, staged_k, delegate = _staged_delegate_graph()

    assert Z._unstage_aliased_buffers(graph_module) == 0
    assert delegate.args[1] is staged_k
    assert k_buffer.meta["spec"].device == DeviceType.CPU


def _direct_delegate_graph(*, compile_specs=None, device=DeviceType.CUDA):
    """A lowered graph with no staging at all: delegate(lowered, k_buffer).

    The shape the un-staging pass itself leaves behind, and so the shape its own
    second run is handed: the marked buffer is the delegate's argument outright
    and there is no ``_h2d_copy`` to remove.

    ``device`` is what the buffer's own spec asks for, which is only half of
    where memory planning ends up putting it. The other half is the
    ``enable_non_cpu_memory_planning`` the program is finalized with; the graph
    does not record it and the pass is told separately.
    """
    graph = torch.fx.Graph()
    k_buffer = graph.placeholder("b_k_0")
    lowered = graph.get_attr("lowered_module_0")
    delegate = graph.call_function(executorch_call_delegate, (lowered, k_buffer))
    graph.output((k_buffer, delegate))
    root = torch.nn.Module()
    root.lowered_module_0 = SimpleNamespace(
        backend_id="TensorRTBackend", compile_specs=compile_specs
    )
    graph_module = torch.fx.GraphModule(root, graph)
    k_buffer.meta["spec"] = SimpleNamespace(device=device, device_index=0)
    k_buffer.meta["_torch_tensorrt_aliased_buffer"] = True
    return graph_module, k_buffer, delegate


@pytest.mark.unit
def test_unstage_accepts_a_buffer_that_never_had_a_staging_copy():
    """A marked buffer already handed straight to its delegate needs no work.

    What the pass has to leave behind is a marked buffer that is a delegate
    argument planned in device memory; removing a staging copy is only the usual
    route there. Keying success on having removed one instead rejects this
    program, which is already in the shape zero-copy wants -- and it is the shape
    the pass's own second run sees.
    """
    graph_module, k_buffer, delegate = _direct_delegate_graph(
        compile_specs=[CompileSpec(ZERO_COPY_KV_COMPILE_SPEC_KEY, b"[]")]
    )

    assert Z._unstage_aliased_buffers(graph_module) == 0

    assert delegate.args[1] is k_buffer
    assert k_buffer.meta["spec"].device == DeviceType.CUDA


@pytest.mark.unit
@pytest.mark.parametrize("route", ["direct", "staged"])
def test_unstage_refuses_a_marked_buffer_whose_only_delegate_is_unstamped(route):
    """The two halves of one post-condition have to answer one graph the same way.

    A TensorRT delegate carrying no zero-copy compile spec is one whose engine
    elided no aliased output, so a marked buffer reaching it says nothing about
    the engine that does write it in place -- that engine's write is still going
    to a staging copy nothing reads back. :func:`check_zero_copy_kv` counts only
    stamped delegates over the finalized program, on either route the buffer
    took, so both routes are pinned here.
    """
    stamped = [CompileSpec(ZERO_COPY_KV_COMPILE_SPEC_KEY, b"[]")]
    if route == "direct":
        graph_module, _, _ = _direct_delegate_graph(compile_specs=None)
    else:
        graph_module, k_buffer, _, _ = _staged_delegate_graph(compile_specs=None)
        k_buffer.meta["_torch_tensorrt_aliased_buffer"] = True

    with pytest.raises(RuntimeError, match="declaring zero-copy KV"):
        Z._unstage_aliased_buffers(graph_module)

    # The same graph with the delegate stamped is accepted, so what the refusal
    # reads is the missing stamp and not something else about these graphs.
    if route == "direct":
        graph_module, _, _ = _direct_delegate_graph(compile_specs=stamped)
        assert Z._unstage_aliased_buffers(graph_module) == 0
    else:
        graph_module, k_buffer, _, _ = _staged_delegate_graph(compile_specs=stamped)
        k_buffer.meta["_torch_tensorrt_aliased_buffer"] = True
        assert Z._unstage_aliased_buffers(graph_module) == 1


@pytest.mark.unit
@pytest.mark.parametrize("spec", ["absent", "host"])
def test_unstage_refuses_a_direct_buffer_that_is_not_on_the_device(spec):
    """Reaching the delegate directly is not enough; the buffer has to be there.

    A marked buffer whose own spec asks for the host is planned in a host arena,
    and the engine cannot write a host pointer in place. Accepting it on the
    strength of the mark and the delegate edge alone writes a ``.pte`` whose
    every ``execute()`` fails on the alias-target guard. A buffer with no spec at
    all is the other half of the same refusal, and means the pass is running
    somewhere the specs do not exist yet.
    """
    graph_module, k_buffer, _ = _direct_delegate_graph(
        device=DeviceType.CPU if spec == "host" else DeviceType.CUDA
    )
    if spec == "absent":
        del k_buffer.meta["spec"]
    expected = (
        "carries no TensorSpec" if spec == "absent" else "its TensorSpec asks for"
    )

    with pytest.raises(RuntimeError, match=expected):
        Z._unstage_aliased_buffers(graph_module)


@pytest.mark.unit
@pytest.mark.parametrize("shape", ["direct", "staged"])
def test_unstage_refuses_a_marked_buffer_under_host_only_memory_planning(shape):
    """A CUDA spec does not mean CUDA memory when planning ignores spec devices.

    ``enable_non_cpu_memory_planning=False`` plans every tensor into the one host
    arena whatever its ``TensorSpec`` says, so no marked buffer can be written in
    place under it. Both graph shapes are covered because the pass reaches them
    by different branches -- the buffer that already is a delegate argument, and
    the one whose staging copy the pass removes -- and the refusal belongs to
    neither, so it is checked once for the whole graph.
    """
    if shape == "direct":
        graph_module, k_buffer, _ = _direct_delegate_graph()
        assert k_buffer.meta["spec"].device == DeviceType.CUDA, (
            "this shape only stands for the planning-mode hazard while the "
            "buffer's spec is CUDA: what the refusal below exists for is a CUDA "
            "spec that still lands in the host arena, and a host spec is refused "
            "under any planning mode"
        )
    else:
        graph_module, k_buffer, _, _ = _staged_delegate_graph()
        k_buffer.meta["_torch_tensorrt_aliased_buffer"] = True

    with pytest.raises(RuntimeError, match="enable_non_cpu_memory_planning=False"):
        Z._unstage_aliased_buffers(graph_module, device_memory_planning=False)


@pytest.mark.unit
def test_zero_copy_backend_config_carries_the_planning_mode_into_the_pass():
    """The refusal is only reachable if the config's flag actually gets there.

    ``zero_copy_backend_config`` is the only place that sees the
    ``ExecutorchBackendConfig``, so building the pass without reading
    ``enable_non_cpu_memory_planning`` off it leaves the check above unreachable
    from any real finalization.
    """
    from executorch.exir import ExecutorchBackendConfig

    config = Z.zero_copy_backend_config(
        ExecutorchBackendConfig(enable_non_cpu_memory_planning=False)
    )
    graph_module, _, _ = _direct_delegate_graph()

    with pytest.raises(RuntimeError, match="enable_non_cpu_memory_planning=False"):
        config.to_out_var_pass(graph_module)


@pytest.mark.unit
def test_zero_copy_backend_config_reads_the_planning_mode_when_the_pass_runs():
    """The flag has to be read where it is in effect, not captured when it is set.

    ``ExecutorchBackendConfig`` is a plain mutable dataclass and the config
    returned here is the one the caller hands to ``to_executorch``, so the value
    memory planning uses is whatever the field holds by then. A pass that froze
    the field when the config was built disagrees with the finalizer in both
    directions, and the first of those writes a ``.pte`` whose caches are planned
    in the host arena and whose every ``execute()`` fails.
    """
    from executorch.exir import ExecutorchBackendConfig

    turned_off = Z.zero_copy_backend_config()
    turned_off.enable_non_cpu_memory_planning = False
    graph_module, _, _ = _direct_delegate_graph(compile_specs=_zero_copy_specs())
    with pytest.raises(RuntimeError, match="enable_non_cpu_memory_planning=False"):
        turned_off.to_out_var_pass(graph_module)

    turned_on = Z.zero_copy_backend_config(
        ExecutorchBackendConfig(enable_non_cpu_memory_planning=False)
    )
    turned_on.enable_non_cpu_memory_planning = True
    graph_module, _, _ = _direct_delegate_graph(compile_specs=_zero_copy_specs())
    turned_on.to_out_var_pass(graph_module)


@pytest.mark.unit
def test_zero_copy_backend_config_does_not_refuse_a_planner_the_flag_never_reaches():
    """The flag is only a ground to refuse on where ExecuTorch delivers it.

    ``to_executorch`` does not pass ``enable_non_cpu_memory_planning`` to the
    memory planner, it assigns it -- and only onto a planner that already has an
    attribute of that name. A caller-supplied planner without one, which is what
    the user guide tells people to bring for a cache shared between prefill and
    decode, never sees the field, so where the caches land is that planner's own
    business and ``False`` does not mean the host arena. Refusing on it there
    turns the field into a trap that blocks a configuration that would have
    worked; nor need the mistake be silent -- ``check_zero_copy_kv`` reads the
    arena that planner actually chose, which is why the documented path ends in
    that call.
    """
    from executorch.exir import ExecutorchBackendConfig

    def a_planner_of_ones_own(graph_module):
        raise AssertionError("memory planning does not run in this test")

    assert not hasattr(a_planner_of_ones_own, "enable_non_cpu_memory_planning")
    config = Z.zero_copy_backend_config(
        ExecutorchBackendConfig(
            enable_non_cpu_memory_planning=False,
            memory_planning_pass=a_planner_of_ones_own,
        )
    )
    graph_module, _, _ = _direct_delegate_graph(compile_specs=_zero_copy_specs())

    config.to_out_var_pass(graph_module)

    # The same field, with a planner ExecuTorch does hand it to, is refused --
    # so what is carried above is the planner and not the flag.
    stock = Z.zero_copy_backend_config(
        ExecutorchBackendConfig(enable_non_cpu_memory_planning=False)
    )
    assert hasattr(stock.memory_planning_pass, "enable_non_cpu_memory_planning")
    graph_module, _, _ = _direct_delegate_graph(compile_specs=_zero_copy_specs())
    with pytest.raises(RuntimeError, match="enable_non_cpu_memory_planning=False"):
        stock.to_out_var_pass(graph_module)


@pytest.mark.unit
def test_zero_copy_backend_config_rebuilt_over_a_derived_config_reads_it():
    """Deriving a config with ``dataclasses.replace`` is the case that needs it.

    The flag is a bool, copied by value; the pass is copied by reference. So a
    config derived from the one this returns carries a pass still reading the
    original, and the first half below is that known gap, pinned: turning
    planning off on the derived config alone is not refused by the inherited
    pass. It need not be silent either -- ``check_zero_copy_kv`` reads the arena
    memory planning chose and refuses the program it produces, for any method
    holding a host tensor to give the shared arena away, which is why the
    documented path ends in that call -- and the remedy the docstring gives is
    the second half: call the function again on the derived config and the pass
    it builds is bound to that one.
    """
    import dataclasses

    derived = dataclasses.replace(
        Z.zero_copy_backend_config(), enable_non_cpu_memory_planning=False
    )
    graph_module, _, _ = _direct_delegate_graph(compile_specs=_zero_copy_specs())
    derived.to_out_var_pass(graph_module)

    rebuilt = Z.zero_copy_backend_config(derived)
    graph_module, _, _ = _direct_delegate_graph(compile_specs=_zero_copy_specs())
    with pytest.raises(RuntimeError, match="enable_non_cpu_memory_planning=False"):
        rebuilt.to_out_var_pass(graph_module)


@pytest.mark.unit
def test_unstage_runs_a_second_time_without_raising():
    """Installing the pass twice is redundant rather than an error.

    ``save(zero_copy_kv=True)`` installs it, so a caller who also passes
    ``zero_copy_backend_config()`` as ``backend_config`` gets two of them. The
    second run finds the buffer already wired to the delegate and returns
    without un-staging anything.
    """
    graph_module, k_buffer, _, delegate = _staged_delegate_graph(
        compile_specs=_zero_copy_specs()
    )
    k_buffer.meta["_torch_tensorrt_aliased_buffer"] = True
    assert Z._unstage_aliased_buffers(graph_module) == 1

    assert Z._unstage_aliased_buffers(graph_module) == 0
    assert delegate.args[1] is k_buffer


@pytest.mark.unit
def test_unstage_raises_for_a_marked_buffer_on_another_backends_delegate():
    """Only a TensorRT engine promises the in-place write, so a marked buffer
    routed to another backend's delegate is never un-staged -- and because
    export has already dropped its copy-back, that is a broken program, not a
    silent no-op."""
    graph_module, k_buffer, staged_k, delegate = _staged_delegate_graph(
        backend_id="CudaBackend"
    )
    k_buffer.meta["_torch_tensorrt_aliased_buffer"] = True

    with pytest.raises(
        RuntimeError, match="no TensorRT delegate declaring zero-copy KV"
    ):
        Z._unstage_aliased_buffers(graph_module)


@pytest.mark.unit
def test_unstage_raises_when_a_marked_buffer_is_never_unstaged():
    """A marked buffer that reaches no TensorRT delegate at all must raise, not
    return 0. Its copy-back is already gone, so leaving it staged would silently
    discard every update."""
    graph = torch.fx.Graph()
    k_buffer = graph.placeholder("b_k_0")
    graph.output((k_buffer,))
    graph_module = torch.fx.GraphModule(torch.nn.Module(), graph)
    k_buffer.meta["spec"] = SimpleNamespace(device=DeviceType.CPU, device_index=0)
    k_buffer.meta["_torch_tensorrt_aliased_buffer"] = True

    with pytest.raises(
        RuntimeError, match="no TensorRT delegate declaring zero-copy KV"
    ):
        Z._unstage_aliased_buffers(graph_module)


@pytest.mark.unit
def test_unstage_raises_when_a_zero_copy_delegate_unstaged_nothing():
    """A TensorRT delegate that declares zero-copy but had no buffer un-staged
    (its mark did not survive to this pass) is unambiguously broken and must
    raise, naming the delegate."""
    graph_module, k_buffer, staged_k, delegate = _staged_delegate_graph(
        compile_specs=[CompileSpec(ZERO_COPY_KV_COMPILE_SPEC_KEY, b"[]")]
    )
    # k_buffer deliberately left unmarked: nothing gets un-staged for the delegate.

    with pytest.raises(RuntimeError, match="declares zero-copy KV"):
        Z._unstage_aliased_buffers(graph_module)


@pytest.mark.unit
def test_unstage_raises_when_a_zero_copy_delegate_unstaged_only_some():
    """One surviving mark must not stand in for the ones that were lost.

    The delegate's spec names both aliased outputs it elided, so it has to take
    two buffers written in place. Only one still carries the mark, and demanding
    merely one would un-stage that one, raise nothing, and leave the second cache
    wired through the staging copy whose contents are discarded -- with its
    copy-back already gone, a silently frozen cache.
    """
    graph = torch.fx.Graph()
    k_buffer = graph.placeholder("b_k_0")
    v_buffer = graph.placeholder("b_v_0")
    h2d = torch.ops.et_copy._h2d_copy.default
    staged_k = graph.call_function(h2d, (k_buffer,))
    staged_v = graph.call_function(h2d, (v_buffer,))
    lowered = graph.get_attr("lowered_module_0")
    delegate = graph.call_function(
        executorch_call_delegate, (lowered, staged_k, staged_v)
    )
    graph.output((k_buffer, v_buffer, delegate))
    root = torch.nn.Module()
    root.lowered_module_0 = SimpleNamespace(
        backend_id="TensorRTBackend", compile_specs=_zero_copy_specs("out_k", "out_v")
    )
    graph_module = torch.fx.GraphModule(root, graph)
    for node, dev in (
        (k_buffer, DeviceType.CPU),
        (v_buffer, DeviceType.CPU),
        (staged_k, DeviceType.CUDA),
        (staged_v, DeviceType.CUDA),
    ):
        node.meta["spec"] = SimpleNamespace(device=dev, device_index=0)
    # v_buffer's mark did not survive lowering; k_buffer's did.
    k_buffer.meta["_torch_tensorrt_aliased_buffer"] = True

    with pytest.raises(RuntimeError, match="takes only 1 of the 2 buffers"):
        Z._unstage_aliased_buffers(graph_module)


@pytest.mark.unit
def test_unstage_refuses_a_buffer_another_backend_stages_on_the_same_gpu():
    """A second backend's staging copy on the *same* GPU is refused, not kept.

    The other backend's ``_h2d_copy`` outlives this pass and goes on reading the
    buffer as its source, but the move has just put the buffer in device memory.
    ``_h2d_copy_out`` requires a host source and fails ``InvalidArgument`` on a
    device one, so leaving the copy in place produces a program that does not
    run. Same GPU is what makes this shape distinct: the device and index both
    match, so neither of ``_device_placement_is_safe``'s spec comparisons rejects it.
    """
    graph = torch.fx.Graph()
    k_buffer = graph.placeholder("b_k_0")
    h2d = torch.ops.et_copy._h2d_copy.default
    staged_trt = graph.call_function(h2d, (k_buffer,))
    staged_other = graph.call_function(h2d, (k_buffer,))
    lowered_trt = graph.get_attr("lowered_module_0")
    lowered_other = graph.get_attr("lowered_module_1")
    delegate_trt = graph.call_function(
        executorch_call_delegate, (lowered_trt, staged_trt)
    )
    delegate_other = graph.call_function(
        executorch_call_delegate, (lowered_other, staged_other)
    )
    graph.output((k_buffer, delegate_trt, delegate_other))
    root = torch.nn.Module()
    root.lowered_module_0 = SimpleNamespace(
        backend_id="TensorRTBackend", compile_specs=None
    )
    root.lowered_module_1 = SimpleNamespace(
        backend_id="CudaBackend", compile_specs=None
    )
    graph_module = torch.fx.GraphModule(root, graph)
    k_buffer.meta["spec"] = SimpleNamespace(device=DeviceType.CPU, device_index=0)
    staged_trt.meta["spec"] = SimpleNamespace(device=DeviceType.CUDA, device_index=0)
    staged_other.meta["spec"] = SimpleNamespace(device=DeviceType.CUDA, device_index=0)
    k_buffer.meta["_torch_tensorrt_aliased_buffer"] = True

    with pytest.raises(RuntimeError, match="consumer this pass leaves in place"):
        Z._unstage_aliased_buffers(graph_module)

    # Refused before anything moved: the buffer is where the other backend's
    # staging copy expects to read it.
    assert k_buffer.meta["spec"].device == DeviceType.CPU
    assert delegate_trt.args[1] is staged_trt
    assert delegate_other.args[1] is staged_other


@pytest.mark.unit
def test_unstage_refuses_a_left_behind_staging_of_a_buffer_already_on_the_device():
    """The same refusal when the buffer's spec already names the engine's GPU.

    Nothing moves in this shape, so a check asked only about the move skips it
    entirely -- and then the pass rewires the delegate anyway and hands the
    engine a buffer whose other consumer, another backend's ``_h2d_copy``, reads
    device memory as a host source and fails ``InvalidArgument`` on every call.
    What the pass has to establish is where the buffer ends up, which is the same
    place either way, so the question is asked either way.
    """
    graph = torch.fx.Graph()
    k_buffer = graph.placeholder("b_k_0")
    h2d = torch.ops.et_copy._h2d_copy.default
    staged_trt = graph.call_function(h2d, (k_buffer,))
    staged_other = graph.call_function(h2d, (k_buffer,))
    lowered_trt = graph.get_attr("lowered_module_0")
    lowered_other = graph.get_attr("lowered_module_1")
    delegate_trt = graph.call_function(
        executorch_call_delegate, (lowered_trt, staged_trt)
    )
    delegate_other = graph.call_function(
        executorch_call_delegate, (lowered_other, staged_other)
    )
    graph.output((k_buffer, delegate_trt, delegate_other))
    root = torch.nn.Module()
    root.lowered_module_0 = SimpleNamespace(
        backend_id="TensorRTBackend", compile_specs=None
    )
    root.lowered_module_1 = SimpleNamespace(
        backend_id="CudaBackend", compile_specs=None
    )
    graph_module = torch.fx.GraphModule(root, graph)
    # The buffer is already where the staging copy would have put it, which is
    # the one thing that separates this from the test above.
    k_buffer.meta["spec"] = SimpleNamespace(device=DeviceType.CUDA, device_index=0)
    staged_trt.meta["spec"] = SimpleNamespace(device=DeviceType.CUDA, device_index=0)
    staged_other.meta["spec"] = SimpleNamespace(device=DeviceType.CUDA, device_index=0)
    k_buffer.meta["_torch_tensorrt_aliased_buffer"] = True

    with pytest.raises(RuntimeError, match="consumer this pass leaves in place"):
        Z._unstage_aliased_buffers(graph_module)

    assert delegate_trt.args[1] is staged_trt


@pytest.mark.unit
def test_unstage_moves_a_buffer_two_tensorrt_delegates_stage_from():
    """Two TensorRT delegates staging one cache is not a surviving consumer.

    Both staging copies go, so what is left reading the buffer is two delegates
    taking it directly, which is the shape this pass exists to produce. The walk
    reaches the second one after the first has already been rewired, so a
    surviving-consumer test that did not allow a rewired delegate would refuse
    the program the pass had just built.
    """
    graph = torch.fx.Graph()
    k_buffer = graph.placeholder("b_k_0")
    h2d = torch.ops.et_copy._h2d_copy.default
    staged_first = graph.call_function(h2d, (k_buffer,))
    staged_second = graph.call_function(h2d, (k_buffer,))
    lowered_first = graph.get_attr("lowered_module_0")
    lowered_second = graph.get_attr("lowered_module_1")
    first = graph.call_function(executorch_call_delegate, (lowered_first, staged_first))
    second = graph.call_function(
        executorch_call_delegate, (lowered_second, staged_second)
    )
    graph.output((k_buffer, first, second))
    root = torch.nn.Module()
    for name in ("lowered_module_0", "lowered_module_1"):
        setattr(
            root,
            name,
            SimpleNamespace(
                backend_id="TensorRTBackend", compile_specs=_zero_copy_specs()
            ),
        )
    graph_module = torch.fx.GraphModule(root, graph)
    k_buffer.meta["spec"] = SimpleNamespace(device=DeviceType.CPU, device_index=0)
    for staged in (staged_first, staged_second):
        staged.meta["spec"] = SimpleNamespace(device=DeviceType.CUDA, device_index=0)
    k_buffer.meta["_torch_tensorrt_aliased_buffer"] = True

    assert Z._unstage_aliased_buffers(graph_module) == 2
    assert first.args[1] is k_buffer
    assert second.args[1] is k_buffer


@pytest.mark.unit
def test_unstage_raises_when_the_staging_copy_is_not_on_cuda():
    """Following the staging to the CPU would put the buffer out of the engine's
    reach, and the copy-back that would have saved it is already gone."""
    graph_module, k_buffer, _, _ = _staged_delegate_graph(device=DeviceType.CPU)
    k_buffer.meta["_torch_tensorrt_aliased_buffer"] = True

    with pytest.raises(RuntimeError, match="not.*CUDA"):
        Z._unstage_aliased_buffers(graph_module)


@pytest.mark.unit
@pytest.mark.parametrize("missing", ["staging-copy", "buffer"])
def test_unstage_raises_when_either_side_of_the_move_has_no_spec(missing):
    """The move reads a spec on both nodes, and the message names the bare one.

    One condition covers both, so a message that always blamed the staging copy
    would send someone whose copy has a spec to look at the wrong node.
    """
    graph_module, k_buffer, staged_k, _ = _staged_delegate_graph()
    k_buffer.meta["_torch_tensorrt_aliased_buffer"] = True
    bare, kind = (
        (staged_k, "staging copy")
        if missing == "staging-copy"
        else (k_buffer, "buffer placeholder")
    )
    del bare.meta["spec"]

    # Anchored on the node kind as well as the name: the buffer's name appears
    # again later in the message, so a loose pattern would match either wording.
    with pytest.raises(
        RuntimeError, match=re.escape(f"no TensorSpec on the {kind} '{bare.name}'")
    ):
        Z._unstage_aliased_buffers(graph_module)


@pytest.mark.unit
def test_unstage_allows_a_buffer_that_is_also_its_mutation_output():
    """The zero-copy shape itself: the marked buffer is both the delegate's
    staged input and its own BUFFER_MUTATION graph output. The output-node
    reference carries no device of its own, so the device move must be allowed --
    the real lowered KV graph looks exactly like this."""
    graph = torch.fx.Graph()
    k_buffer = graph.placeholder("b_k_0")
    lowered = graph.get_attr("lowered_module_0")
    h2d = torch.ops.et_copy._h2d_copy.default
    staged_k = graph.call_function(h2d, (k_buffer,))
    delegate = graph.call_function(executorch_call_delegate, (lowered, staged_k))
    graph.output((k_buffer, delegate))
    root = torch.nn.Module()
    root.lowered_module_0 = SimpleNamespace(
        backend_id="TensorRTBackend", compile_specs=_zero_copy_specs()
    )
    graph_module = torch.fx.GraphModule(root, graph)
    k_buffer.meta["spec"] = SimpleNamespace(device=DeviceType.CPU, device_index=3)
    staged_k.meta["spec"] = SimpleNamespace(device=DeviceType.CUDA, device_index=3)
    k_buffer.meta["_torch_tensorrt_aliased_buffer"] = True

    assert Z._unstage_aliased_buffers(graph_module) == 1
    assert delegate.args[1] is k_buffer
    assert k_buffer.meta["spec"].device == DeviceType.CUDA


@pytest.mark.unit
@pytest.mark.parametrize("route", ["staged", "direct"])
def test_unstage_refuses_a_buffer_a_surviving_consumer_reads(route):
    """The surviving-consumer refusal belongs to the placement, not to the move.

    A buffer already reaching its delegate directly is in the same position as
    one this pass un-stages: it is planned in the engine's device memory, and a
    consumer that reads it as a host source fails ``InvalidArgument`` on the
    first call. Asking only on the branch that removes a staging copy would let
    the identical graph through by the other door -- including the graph this
    pass's own first run produces, which its second run reads directly.
    """
    graph = torch.fx.Graph()
    k_buffer = graph.placeholder("b_k_0")
    lowered = graph.get_attr("lowered_module_0")
    h2d = torch.ops.et_copy._h2d_copy.default
    delegate_arg = (
        graph.call_function(h2d, (k_buffer,)) if route == "staged" else k_buffer
    )
    # Another backend's host copy of the same buffer, which nothing removes.
    foreign = graph.call_function(h2d, (k_buffer,))
    lowered_other = graph.get_attr("lowered_module_1")
    other = graph.call_function(executorch_call_delegate, (lowered_other, foreign))
    delegate = graph.call_function(executorch_call_delegate, (lowered, delegate_arg))
    graph.output((k_buffer, delegate, other))
    root = torch.nn.Module()
    root.lowered_module_0 = SimpleNamespace(
        backend_id="TensorRTBackend", compile_specs=_zero_copy_specs()
    )
    root.lowered_module_1 = SimpleNamespace(
        backend_id="CudaBackend", compile_specs=None
    )
    graph_module = torch.fx.GraphModule(root, graph)
    k_buffer.meta["spec"] = SimpleNamespace(
        device=DeviceType.CPU if route == "staged" else DeviceType.CUDA,
        device_index=0,
    )
    if route == "staged":
        delegate_arg.meta["spec"] = SimpleNamespace(
            device=DeviceType.CUDA, device_index=0
        )
    # On the GPU the engine is not on, so it is the placement and not merely the
    # presence of a second reader that this refuses.
    foreign.meta["spec"] = SimpleNamespace(device=DeviceType.CUDA, device_index=1)
    k_buffer.meta["_torch_tensorrt_aliased_buffer"] = True

    with pytest.raises(RuntimeError, match="does not survive|leaves in place"):
        Z._unstage_aliased_buffers(graph_module)


@pytest.mark.unit
def test_unstage_refuses_to_move_a_shared_buffer():
    """A buffer read by a consumer other than its TensorRT delegate staging
    cannot have its device moved -- that would silently retarget the other
    consumer too, exactly what ExecuTorch's PropagateDevicePass rejects.
    """
    graph = torch.fx.Graph()
    k_buffer = graph.placeholder("b_k_0")
    lowered = graph.get_attr("lowered_module_0")
    h2d = torch.ops.et_copy._h2d_copy.default
    staged_k = graph.call_function(h2d, (k_buffer,))
    other = graph.call_function(torch.add, (k_buffer, k_buffer))
    delegate = graph.call_function(executorch_call_delegate, (lowered, staged_k))
    graph.output((delegate, other))
    root = torch.nn.Module()
    root.lowered_module_0 = SimpleNamespace(
        backend_id="TensorRTBackend", compile_specs=None
    )
    graph_module = torch.fx.GraphModule(root, graph)
    k_buffer.meta["spec"] = SimpleNamespace(device=DeviceType.CPU, device_index=3)
    staged_k.meta["spec"] = SimpleNamespace(device=DeviceType.CUDA, device_index=3)
    k_buffer.meta["_torch_tensorrt_aliased_buffer"] = True

    with pytest.raises(RuntimeError, match="consumer this pass leaves in place"):
        Z._unstage_aliased_buffers(graph_module)


@pytest.mark.unit
def test_unstage_refuses_to_move_a_buffer_staged_to_two_gpus():
    """One buffer staged to two TensorRT delegates on *different* GPUs cannot be
    un-staged for either: a spec carries one device index, so whichever engine
    lost the race would be handed an address on the other's GPU. ``spec.device``
    is only CUDA/CPU, so it is the device-index comparison in
    ``_device_placement_is_safe`` that refuses the first delegate here -- both
    stagings feed a TensorRT delegate, which is what separates this from the
    two-backends shapes and leaves the index the only comparison that can catch
    it.

    Which is why the refusal alone would not pin it. Were the index comparison
    gone, delegate 0 would be un-staged and the direct-consumer branch would then
    refuse delegate 1, and this test would still see a RuntimeError. What it
    checks is that nothing moved: the buffer is still on the host and delegate 0
    still reads its staging copy.
    """
    graph = torch.fx.Graph()
    k_buffer = graph.placeholder("b_k_0")
    h2d = torch.ops.et_copy._h2d_copy.default
    staged_0 = graph.call_function(h2d, (k_buffer,))
    staged_1 = graph.call_function(h2d, (k_buffer,))
    lowered_0 = graph.get_attr("lowered_module_0")
    lowered_1 = graph.get_attr("lowered_module_1")
    delegate_0 = graph.call_function(executorch_call_delegate, (lowered_0, staged_0))
    delegate_1 = graph.call_function(executorch_call_delegate, (lowered_1, staged_1))
    graph.output((k_buffer, delegate_0, delegate_1))
    root = torch.nn.Module()
    root.lowered_module_0 = SimpleNamespace(
        backend_id="TensorRTBackend", compile_specs=None
    )
    root.lowered_module_1 = SimpleNamespace(
        backend_id="TensorRTBackend", compile_specs=None
    )
    graph_module = torch.fx.GraphModule(root, graph)
    k_buffer.meta["spec"] = SimpleNamespace(device=DeviceType.CPU, device_index=0)
    staged_0.meta["spec"] = SimpleNamespace(device=DeviceType.CUDA, device_index=0)
    staged_1.meta["spec"] = SimpleNamespace(device=DeviceType.CUDA, device_index=1)
    k_buffer.meta["_torch_tensorrt_aliased_buffer"] = True

    with pytest.raises(RuntimeError, match="consumer this pass leaves in place"):
        Z._unstage_aliased_buffers(graph_module)

    assert k_buffer.meta["spec"].device == DeviceType.CPU
    assert delegate_0.args[1] is staged_0


@pytest.mark.unit
def test_unstage_refuses_to_move_a_buffer_a_second_consumer_stages_to_the_host():
    """The device-*type* half of ``_device_placement_is_safe``'s spec comparison.

    Its sibling, the device index, is pinned by
    ``test_unstage_refuses_to_move_a_buffer_staged_to_two_gpus``. Here the second
    consumer is another ``_h2d_copy`` that stays on the host, so the indices
    agree and only the type comparison separates the two: deleting it accepts
    this graph, leaving a copy that reads a buffer the move has put on the GPU.
    """
    graph = torch.fx.Graph()
    k_buffer = graph.placeholder("b_k_0")
    h2d = torch.ops.et_copy._h2d_copy.default
    staged_trt = graph.call_function(h2d, (k_buffer,))
    staged_host = graph.call_function(h2d, (k_buffer,))
    lowered_trt = graph.get_attr("lowered_module_0")
    lowered_other = graph.get_attr("lowered_module_1")
    delegate_trt = graph.call_function(
        executorch_call_delegate, (lowered_trt, staged_trt)
    )
    delegate_other = graph.call_function(
        executorch_call_delegate, (lowered_other, staged_host)
    )
    graph.output((k_buffer, delegate_trt, delegate_other))
    root = torch.nn.Module()
    root.lowered_module_0 = SimpleNamespace(
        backend_id="TensorRTBackend", compile_specs=None
    )
    root.lowered_module_1 = SimpleNamespace(
        backend_id="TensorRTBackend", compile_specs=None
    )
    graph_module = torch.fx.GraphModule(root, graph)
    k_buffer.meta["spec"] = SimpleNamespace(device=DeviceType.CPU, device_index=0)
    staged_trt.meta["spec"] = SimpleNamespace(device=DeviceType.CUDA, device_index=0)
    staged_host.meta["spec"] = SimpleNamespace(device=DeviceType.CPU, device_index=0)
    k_buffer.meta["_torch_tensorrt_aliased_buffer"] = True

    with pytest.raises(RuntimeError, match="consumer this pass leaves in place"):
        Z._unstage_aliased_buffers(graph_module)

    assert k_buffer.meta["spec"].device == DeviceType.CPU
    assert delegate_trt.args[1] is staged_trt


@pytest.mark.unit
def test_unstage_refuses_a_buffer_another_backend_stages_to_a_different_gpu():
    """A marked buffer staged to a TensorRT delegate on cuda:0 and to a
    *non*-TensorRT delegate on cuda:1 cannot be moved either.

    Un-staging skips the other backend's delegate, so its staging copy keeps
    reading the buffer while staging it to cuda:1, and re-homing the buffer onto
    the TensorRT engine's cuda:0 would move the source of that read to the wrong
    GPU. Two of ``_device_placement_is_safe``'s comparisons refuse this shape -- the
    device index, which runs first, and the surviving copy's non-TensorRT user
    -- so this test does not discriminate between them. The index comparison on
    its own is pinned by
    ``test_unstage_refuses_to_move_a_buffer_staged_to_two_gpus``, where every
    staging does feed a TensorRT delegate.
    """
    graph = torch.fx.Graph()
    k_buffer = graph.placeholder("b_k_0")
    h2d = torch.ops.et_copy._h2d_copy.default
    staged_trt = graph.call_function(h2d, (k_buffer,))
    staged_other = graph.call_function(h2d, (k_buffer,))
    lowered_trt = graph.get_attr("lowered_module_0")
    lowered_other = graph.get_attr("lowered_module_1")
    delegate_trt = graph.call_function(
        executorch_call_delegate, (lowered_trt, staged_trt)
    )
    delegate_other = graph.call_function(
        executorch_call_delegate, (lowered_other, staged_other)
    )
    graph.output((k_buffer, delegate_trt, delegate_other))
    root = torch.nn.Module()
    root.lowered_module_0 = SimpleNamespace(
        backend_id="TensorRTBackend", compile_specs=None
    )
    root.lowered_module_1 = SimpleNamespace(
        backend_id="CudaBackend", compile_specs=None
    )
    graph_module = torch.fx.GraphModule(root, graph)
    k_buffer.meta["spec"] = SimpleNamespace(device=DeviceType.CPU, device_index=0)
    staged_trt.meta["spec"] = SimpleNamespace(device=DeviceType.CUDA, device_index=0)
    staged_other.meta["spec"] = SimpleNamespace(device=DeviceType.CUDA, device_index=1)
    k_buffer.meta["_torch_tensorrt_aliased_buffer"] = True

    with pytest.raises(RuntimeError, match="consumer this pass leaves in place"):
        Z._unstage_aliased_buffers(graph_module)


@pytest.mark.unit
def test_unstage_refuses_to_rehome_a_buffer_already_on_another_gpu():
    """A buffer already resident on cuda:0 is not re-homed to a second TensorRT
    delegate's cuda:1.

    Whether the move needs checking at all is decided by comparing the buffer's
    device *and index* against the staging copy's. Comparing the device alone
    would call this buffer already placed -- both ends are CUDA -- skip the
    check, and overwrite the index with the second engine's, leaving the first
    engine holding an address on the other GPU.
    """
    graph = torch.fx.Graph()
    k_buffer = graph.placeholder("b_k_0")
    h2d = torch.ops.et_copy._h2d_copy.default
    staged_0 = graph.call_function(h2d, (k_buffer,))
    staged_1 = graph.call_function(h2d, (k_buffer,))
    lowered_0 = graph.get_attr("lowered_module_0")
    lowered_1 = graph.get_attr("lowered_module_1")
    delegate_0 = graph.call_function(executorch_call_delegate, (lowered_0, staged_0))
    delegate_1 = graph.call_function(executorch_call_delegate, (lowered_1, staged_1))
    graph.output((k_buffer, delegate_0, delegate_1))
    root = torch.nn.Module()
    root.lowered_module_0 = SimpleNamespace(
        backend_id="TensorRTBackend", compile_specs=None
    )
    root.lowered_module_1 = SimpleNamespace(
        backend_id="TensorRTBackend", compile_specs=None
    )
    graph_module = torch.fx.GraphModule(root, graph)
    k_buffer.meta["spec"] = SimpleNamespace(device=DeviceType.CUDA, device_index=0)
    staged_0.meta["spec"] = SimpleNamespace(device=DeviceType.CUDA, device_index=0)
    staged_1.meta["spec"] = SimpleNamespace(device=DeviceType.CUDA, device_index=1)
    k_buffer.meta["_torch_tensorrt_aliased_buffer"] = True

    with pytest.raises(RuntimeError, match="consumer this pass leaves in place"):
        Z._unstage_aliased_buffers(graph_module)
    assert k_buffer.meta["spec"].device_index == 0


@pytest.mark.unit
def test_zero_copy_backend_config_keeps_the_callers_config():
    """It composes onto a config rather than replacing one: a caller finalizing
    a zero-copy program still needs their own memory planning and passes."""
    from executorch.exir import ExecutorchBackendConfig

    inner = object()
    base = ExecutorchBackendConfig(to_out_var_pass=inner, emit_stacktrace=True)

    config = torch_tensorrt.executorch.zero_copy_backend_config(base)

    assert config.emit_stacktrace is True
    assert config.memory_planning_pass is base.memory_planning_pass
    assert config.to_out_var_pass is not inner
    # The caller's to_out_var_pass is not dropped, it is run after the un-staging.
    graph_module, k_buffer, _, delegate = _staged_delegate_graph(
        compile_specs=_zero_copy_specs()
    )
    k_buffer.meta["_torch_tensorrt_aliased_buffer"] = True
    seen = []
    base = ExecutorchBackendConfig(to_out_var_pass=lambda gm: seen.append(gm))
    torch_tensorrt.executorch.zero_copy_backend_config(base).to_out_var_pass.call(
        graph_module
    )
    assert seen == [graph_module]
    assert delegate.args[1] is k_buffer


def _finalized_program(forward=None, **methods):
    """The shape ``check_zero_copy_kv`` reads: to_executorch()'s return value.

    One positional graph module makes a single-method ``forward`` program; the
    keywords name a method each. ``exported_program`` defaults to ``forward``
    and raises ``KeyError`` on a method the program does not have, like
    ``ExecutorchProgramManager``'s -- which is what a program with no ``forward``
    does to a caller that never asked for one.
    """
    if forward is not None:
        methods = {"forward": forward, **methods}
    return SimpleNamespace(
        methods=set(methods),
        exported_program=lambda method_name="forward": SimpleNamespace(
            graph_module=methods[method_name]
        ),
    )


CUDA_ARENA = 2


HOST_ARENA = 1


def _planned(
    graph_module,
    *,
    arena=CUDA_ARENA,
    on_device=True,
    host_tensor_arena=None,
    device_type=DeviceType.CUDA,
    device_index=None,
):
    """Add what memory planning leaves behind, which the checker reads.

    The graphs above are built for the passes that run before planning, so they
    carry no ``mem_id`` and the module records no arena devices. Planning assigns
    both. ``on_device=False`` drops the arena-device record, which is what
    ``enable_non_cpu_memory_planning=False`` leaves -- and also what a
    caller-supplied planner that does not go through ``apply_algo`` leaves, that
    being the only thing that writes it. What separates those two is where the
    program's *host* tensors ended up: host-only planning puts every tensor in
    one bucket, so they share the buffer's arena, while a device-aware planner
    keeps them apart.
    ``host_tensor_arena`` is where the CPU-spec tensors go, and it is also put on
    the output node's spec, which is where a real finalized program carries one.
    Leaving it unset gives the two shapes above: a separate arena when the
    program records devices, and the buffer's own when it does not.
    ``device_index`` is the GPU the arena is recorded for; unset, it is the one
    the graph's own CUDA specs ask for, which is what a planner that honoured
    them would record. Passing a different one is the multi-GPU mistake.
    """
    from executorch.exir.schema import NonConstBufferDevice

    if host_tensor_arena is None:
        host_tensor_arena = HOST_ARENA if on_device else arena
    asked_for = []
    for node in graph_module.graph.nodes:
        spec = node.meta.get("spec")
        if node.op == "placeholder" and spec is not None:
            spec.mem_id = arena if spec.device == DeviceType.CUDA else host_tensor_arena
            if spec.device == DeviceType.CUDA:
                asked_for.append(spec.device_index)
        if node.op == "output":
            node.meta["spec"] = [
                SimpleNamespace(
                    device=DeviceType.CPU, device_index=0, mem_id=host_tensor_arena
                )
            ]
    if on_device:
        graph_module.meta["non_const_buffer_device"] = [
            NonConstBufferDevice(
                buffer_idx=arena,
                device_type=device_type,
                device_index=(
                    next(iter(asked_for), 0) if device_index is None else device_index
                ),
            )
        ]
    return graph_module


def _unstaged_graph():
    """A graph whose marked buffer already reaches its delegate directly."""
    graph_module, k_buffer, _, _ = _staged_delegate_graph(
        compile_specs=_zero_copy_specs()
    )
    k_buffer.meta["_torch_tensorrt_aliased_buffer"] = True
    Z._unstage_aliased_buffers(graph_module)
    return _planned(graph_module)


@pytest.mark.unit
def test_check_zero_copy_kv_accepts_an_unstaged_buffer():
    Z.check_zero_copy_kv(_finalized_program(_unstaged_graph()))


@pytest.mark.unit
def test_check_zero_copy_kv_rejects_a_still_staged_buffer():
    """The shape a program finalized without zero_copy_backend_config has: the
    buffer is marked, so export dropped its copy-back, but it still reaches the
    delegate through a staging copy the engine's write is thrown away with."""
    graph_module, k_buffer, _, _ = _staged_delegate_graph(
        compile_specs=_zero_copy_specs()
    )
    k_buffer.meta["_torch_tensorrt_aliased_buffer"] = True

    with pytest.raises(RuntimeError, match="do not reach the TensorRT delegate"):
        Z.check_zero_copy_kv(_finalized_program(_planned(graph_module)))


@pytest.mark.unit
def test_check_zero_copy_kv_accepts_a_buffer_that_never_had_a_staging_copy():
    """The checker and the un-staging pass read the same post-condition.

    A program the pass has already un-staged hands the buffer straight to the
    delegate with no staging copy left. The pass accepts that shape
    (``test_unstage_accepts_a_buffer_that_never_had_a_staging_copy``) and so must
    this: the two disagreeing is what would let one path refuse a program the
    other calls correct.
    """
    graph_module, _, _ = _direct_delegate_graph(compile_specs=_zero_copy_specs())

    Z.check_zero_copy_kv(_finalized_program(_planned(graph_module)))


@pytest.mark.unit
def test_check_zero_copy_kv_rejects_a_direct_buffer_planned_on_the_host():
    """Reaching the delegate directly is only half of it; placement is the rest.

    Finalizing with ``enable_non_cpu_memory_planning=False`` and no zero-copy
    config gives exactly this: no staging copy is inserted, so the wiring is what
    zero-copy wants, and every tensor still lands in the one host arena. Nothing
    else refuses it -- the un-staging pass never ran -- and the engine is handed a
    host pointer it cannot write, so the ``.pte`` fails its first ``execute()``.
    The graph is the accepted one above with only the arena changed, so the
    refusal can come from nothing else. The program records no arena devices --
    host-only planning writes none -- so what identifies the arena as the host's
    is that the program's host tensors are in it too.
    """
    graph_module, _, _ = _direct_delegate_graph(compile_specs=_zero_copy_specs())

    with pytest.raises(RuntimeError, match="also holds the program's host tensors"):
        Z.check_zero_copy_kv(
            _finalized_program(
                _planned(graph_module, arena=HOST_ARENA, on_device=False)
            )
        )


@pytest.mark.unit
def test_check_zero_copy_kv_rejects_an_arena_no_planner_recorded():
    """An unrecorded arena is what the runtime reads as the host, whoever planned it.

    ``apply_algo`` is the only thing in ExecuTorch that writes
    ``non_const_buffer_device``, and ``to_executorch`` takes any callable as
    ``memory_planning_pass``, so a caller-supplied planner can put the cache in
    device memory and leave the ``.pte`` saying nothing. It is refused all the
    same: ``MethodMeta::memory_planned_buffer_device`` answers ``CPU`` for an
    arena with no entry, so the runner backs it with host memory and the engine
    fails the alias-target guard on the first call.

    This is the host-arena test's graph with the host tensors moved out of the
    buffer's arena, so the two refusals are told apart by which of them fires:
    the arena here is not one the program's host tensors are in, and the message
    says the record is absent rather than accusing the planner of putting the
    cache among the host tensors.
    """
    graph_module, _, _ = _direct_delegate_graph(compile_specs=_zero_copy_specs())

    with pytest.raises(RuntimeError, match="records no CUDA arena at all"):
        Z.check_zero_copy_kv(
            _finalized_program(
                _planned(
                    graph_module,
                    arena=CUDA_ARENA,
                    on_device=False,
                    host_tensor_arena=HOST_ARENA,
                )
            )
        )


@pytest.mark.unit
def test_check_zero_copy_kv_rejects_an_unplanned_buffer_without_blaming_the_planner():
    """A buffer with no ``mem_id`` was not planned, which is not the host arena.

    A planner that excludes mutable buffers, or one built with graph-input
    allocation off, leaves the cache with no ``mem_id`` at all. Nothing then
    says where it lives, so it is refused -- but reporting it among the host
    tensors would tell the caller their planner made a placement it never made,
    and the two are separated here by the message. The control is the same
    program with the cache in the recorded CUDA arena, which is accepted, so the
    refusal is the missing ``mem_id`` and nothing else about this graph.
    """
    graph_module, k_buffer, _ = _direct_delegate_graph(compile_specs=_zero_copy_specs())
    program = _finalized_program(_planned(graph_module, host_tensor_arena=HOST_ARENA))
    Z.check_zero_copy_kv(program)

    del k_buffer.meta["spec"].mem_id
    with pytest.raises(RuntimeError, match="carries no mem_id") as raised:
        Z.check_zero_copy_kv(program)
    assert "host tensors" not in str(raised.value)


@pytest.mark.unit
def test_check_zero_copy_kv_rejects_an_arena_the_program_records_as_non_cuda():
    """When the record *is* present it is read, and only its CUDA entries count.

    Nothing in ExecuTorch emits a CPU entry today -- the builder filters them --
    so this pins the filter against a program that carries one, hand-built or
    from a future planner, rather than against the stock one.
    """
    graph_module, _, _ = _direct_delegate_graph(compile_specs=_zero_copy_specs())

    with pytest.raises(RuntimeError, match="does not record as CUDA"):
        Z.check_zero_copy_kv(
            _finalized_program(
                _planned(
                    graph_module,
                    device_type=DeviceType.CPU,
                    host_tensor_arena=HOST_ARENA,
                )
            )
        )


@pytest.mark.unit
def test_check_zero_copy_kv_rejects_an_arena_recorded_for_another_gpu():
    """A CUDA arena is not enough; it has to be the GPU the cache asks for.

    The runtime allocates the cache out of the arena the program records, so an
    arena recorded for another device hands the engine an address on a GPU it is
    not running on -- which fails exactly as a host pointer does, and which the
    device *type* on its own cannot tell from a correct program.
    """
    graph_module, k_buffer, _ = _direct_delegate_graph(compile_specs=_zero_copy_specs())
    assert k_buffer.meta["spec"].device_index == 0

    with pytest.raises(RuntimeError, match="asks for cuda:0 and was planned"):
        Z.check_zero_copy_kv(_finalized_program(_planned(graph_module, device_index=1)))


@pytest.mark.unit
def test_check_zero_copy_kv_counts_one_buffer_in_two_slots_once():
    """Two argument slots holding one buffer are one cache, not two.

    The delegate's spec names two elided aliased outputs, so it owes two caches
    written in place, and it has one. Counting slots satisfies that count with
    the same buffer twice -- which is the arrangement the count exists to catch,
    a delegate whose second mark did not survive. The un-staging pass counts the
    same way, so the two are pinned together: they must not start disagreeing
    about one graph.
    """
    graph = torch.fx.Graph()
    k_buffer = graph.placeholder("b_k_0")
    lowered = graph.get_attr("lowered_module_0")
    delegate = graph.call_function(
        executorch_call_delegate, (lowered, k_buffer, k_buffer)
    )
    graph.output((k_buffer, delegate))
    root = torch.nn.Module()
    root.lowered_module_0 = SimpleNamespace(
        backend_id="TensorRTBackend", compile_specs=_zero_copy_specs("out_k", "out_v")
    )
    graph_module = torch.fx.GraphModule(root, graph)
    k_buffer.meta["spec"] = SimpleNamespace(device=DeviceType.CUDA, device_index=0)
    k_buffer.meta["_torch_tensorrt_aliased_buffer"] = True

    with pytest.raises(RuntimeError, match="takes only 1 of the 2 buffers"):
        Z._unstage_aliased_buffers(graph_module)
    with pytest.raises(RuntimeError, match="takes 1 marked buffer"):
        Z.check_zero_copy_kv(_finalized_program(_planned(graph_module)))


@pytest.mark.unit
def test_check_zero_copy_kv_rejects_a_buffer_staged_at_its_own_delegate():
    """An unstamped TensorRT delegate must not stand in for the one that elided.

    The engine whose aliased output was elided -- the one carrying the zero-copy
    spec -- still reads a staging copy, so its write is discarded and the cache
    never updates. Another TensorRT engine happens to read the same buffer
    directly, which says nothing about that write. Taking the union over every
    TensorRT delegate accepts this program; narrowing to the stamped ones is what
    refuses it here, and ``..._a_second_zero_copy_delegate_standing_in`` is the
    case where both are stamped and that narrowing is not enough.
    """
    graph = torch.fx.Graph()
    k_buffer = graph.placeholder("b_k_0")
    h2d = torch.ops.et_copy._h2d_copy.default
    staged_k = graph.call_function(h2d, (k_buffer,))
    lowered_kv = graph.get_attr("lowered_module_0")
    lowered_plain = graph.get_attr("lowered_module_1")
    delegate_kv = graph.call_function(executorch_call_delegate, (lowered_kv, staged_k))
    delegate_plain = graph.call_function(
        executorch_call_delegate, (lowered_plain, k_buffer)
    )
    graph.output((k_buffer, delegate_kv, delegate_plain))
    root = torch.nn.Module()
    root.lowered_module_0 = SimpleNamespace(
        backend_id="TensorRTBackend", compile_specs=_zero_copy_specs()
    )
    root.lowered_module_1 = SimpleNamespace(
        backend_id="TensorRTBackend", compile_specs=None
    )
    graph_module = torch.fx.GraphModule(root, graph)
    k_buffer.meta["spec"] = SimpleNamespace(device=DeviceType.CUDA, device_index=0)
    k_buffer.meta["_torch_tensorrt_aliased_buffer"] = True

    with pytest.raises(RuntimeError, match="do not reach the TensorRT delegate"):
        Z.check_zero_copy_kv(_finalized_program(_planned(graph_module)))


def _two_zero_copy_delegate_graph(*, crossed, first_specs=None):
    """Two stamped TensorRT delegates in one method, one elided output each.

    ``crossed`` gives the second delegate both caches and leaves the first
    reading a staging copy of the one it elided, which is the shape a check that
    reads the stamped delegates as one set cannot see: every marked buffer does
    reach a stamped delegate, just not the one whose write was removed.
    ``first_specs`` overrides what the first delegate's spec claims it elided.
    """
    graph = torch.fx.Graph()
    k_buffer = graph.placeholder("b_k_0")
    v_buffer = graph.placeholder("b_v_0")
    lowered_k = graph.get_attr("lowered_module_0")
    lowered_v = graph.get_attr("lowered_module_1")
    if crossed:
        staged_k = graph.call_function(torch.ops.et_copy._h2d_copy.default, (k_buffer,))
        k_args, v_args = (lowered_k, staged_k), (lowered_v, k_buffer, v_buffer)
    else:
        k_args, v_args = (lowered_k, k_buffer), (lowered_v, v_buffer)
    delegate_k = graph.call_function(executorch_call_delegate, k_args)
    delegate_v = graph.call_function(executorch_call_delegate, v_args)
    graph.output((k_buffer, v_buffer, delegate_k, delegate_v))
    root = torch.nn.Module()
    root.lowered_module_0 = SimpleNamespace(
        backend_id="TensorRTBackend",
        compile_specs=(
            _zero_copy_specs("out_k") if first_specs is None else first_specs
        ),
    )
    root.lowered_module_1 = SimpleNamespace(
        backend_id="TensorRTBackend", compile_specs=_zero_copy_specs("out_v")
    )
    graph_module = torch.fx.GraphModule(root, graph)
    for buffer in (k_buffer, v_buffer):
        buffer.meta["spec"] = SimpleNamespace(device=DeviceType.CUDA, device_index=0)
        buffer.meta["_torch_tensorrt_aliased_buffer"] = True
    return _planned(graph_module)


@pytest.mark.unit
def test_check_zero_copy_kv_rejects_a_second_zero_copy_delegate_standing_in():
    """A stamped delegate is counted against its own spec, not pooled with the rest.

    Both delegates here declare zero-copy KV, so narrowing by the spec does not
    separate them, and both marked buffers reach one of the two directly -- so
    the still-staged refusal has nothing to say. What is wrong is which delegate
    took which: ``lowered_module_0`` elided ``out_k`` and reads a staging copy of
    the cache that write was removed for, so that cache never updates. The
    matched half is the same graph with each delegate holding the cache it
    elided, and it is accepted, so the refusal can come only from the crossing.
    """
    Z.check_zero_copy_kv(
        _finalized_program(_two_zero_copy_delegate_graph(crossed=False))
    )

    with pytest.raises(RuntimeError, match="takes 0 marked buffer"):
        Z.check_zero_copy_kv(
            _finalized_program(_two_zero_copy_delegate_graph(crossed=True))
        )


@pytest.mark.unit
def test_check_zero_copy_kv_rejects_a_stamped_delegate_with_no_names_to_count():
    """A spec listing no name still says its engine elided an aliased output.

    Only a delegate whose own engine had one is stamped, so the count it cannot
    read off the spec falls back to at least one -- the same fallback
    ``_unstage_aliased_buffers`` makes. Reading "no names" as "no buffers owed"
    would accept the crossing above whenever the partitioner's list is empty or
    unreadable.
    """
    graph_module = _two_zero_copy_delegate_graph(
        crossed=True,
        first_specs=[CompileSpec(ZERO_COPY_KV_COMPILE_SPEC_KEY, b"[]")],
    )

    with pytest.raises(RuntimeError, match="must take at least one marked buffer"):
        Z.check_zero_copy_kv(_finalized_program(graph_module))


@pytest.mark.unit
def test_check_zero_copy_kv_rejects_a_buffer_only_another_backend_takes():
    """Another backend's delegate taking the buffer directly is not zero-copy.

    The mark is on this buffer because a TensorRT engine writes it in place, and
    that engine here is still reading a staging copy whose contents are thrown
    away. Both delegates carry the zero-copy spec, so the backend is the only
    thing separating them: counting any backend's delegate passes this program.
    """
    graph = torch.fx.Graph()
    k_buffer = graph.placeholder("b_k_0")
    h2d = torch.ops.et_copy._h2d_copy.default
    staged_k = graph.call_function(h2d, (k_buffer,))
    lowered_trt = graph.get_attr("lowered_module_0")
    lowered_other = graph.get_attr("lowered_module_1")
    delegate_trt = graph.call_function(
        executorch_call_delegate, (lowered_trt, staged_k)
    )
    delegate_other = graph.call_function(
        executorch_call_delegate, (lowered_other, k_buffer)
    )
    graph.output((k_buffer, delegate_trt, delegate_other))
    root = torch.nn.Module()
    root.lowered_module_0 = SimpleNamespace(
        backend_id="TensorRTBackend", compile_specs=_zero_copy_specs()
    )
    root.lowered_module_1 = SimpleNamespace(
        backend_id="CudaBackend", compile_specs=_zero_copy_specs()
    )
    graph_module = torch.fx.GraphModule(root, graph)
    k_buffer.meta["spec"] = SimpleNamespace(device=DeviceType.CUDA, device_index=0)
    k_buffer.meta["_torch_tensorrt_aliased_buffer"] = True

    with pytest.raises(RuntimeError, match="do not reach the TensorRT delegate"):
        Z.check_zero_copy_kv(_finalized_program(_planned(graph_module)))


@pytest.mark.unit
def test_check_zero_copy_kv_rejects_a_program_with_nothing_marked():
    """zero_copy_kv=True on a model with no engine-aliased buffer only warns, so
    the .pte that comes out is an ordinary staged one. Refuse it rather than let
    a caller who asked for zero-copy ship a program that never got it."""
    graph_module, _, _, _ = _staged_delegate_graph()

    with pytest.raises(RuntimeError, match="marked for in-place update"):
        Z.check_zero_copy_kv(_finalized_program(graph_module))


@pytest.mark.unit
def test_check_zero_copy_kv_accepts_a_program_with_no_forward_method():
    """The shape the user guide's zero-copy example exports: prefill and decode,
    no ``forward``. Reading the default method would raise KeyError naming a
    method the caller never asked for. A method that rewired nothing of its own
    is not an error either, so only ``decode`` here carries a marked buffer."""
    unmarked, _, _, _ = _staged_delegate_graph()

    Z.check_zero_copy_kv(_finalized_program(prefill=unmarked, decode=_unstaged_graph()))


@pytest.mark.unit
def test_check_zero_copy_kv_catches_a_method_other_than_forward():
    """The silent case: ``forward`` got zero-copy and ``decode`` degenerated to
    staged. Stopping at ``forward`` would write a .pte whose decode cache never
    updates, so the failure has to name the method that lost it."""
    staged, k_buffer, _, _ = _staged_delegate_graph()
    k_buffer.meta["_torch_tensorrt_aliased_buffer"] = True

    with pytest.raises(RuntimeError, match="in method 'decode'"):
        Z.check_zero_copy_kv(_finalized_program(_unstaged_graph(), decode=staged))


@pytest.mark.unit
def test_check_zero_copy_kv_rejects_a_multi_method_program_with_nothing_marked():
    """Nothing marked anywhere is about the program, not about one method: a
    method with no aliased buffer mutation is an error only when no other method
    has one, and the failure lists every method it looked in."""
    first, _, _, _ = _staged_delegate_graph()
    second, _, _, _ = _staged_delegate_graph()

    with pytest.raises(RuntimeError, match=r"\(decode, prefill\)"):
        Z.check_zero_copy_kv(_finalized_program(prefill=first, decode=second))


def _stamped_delegate_with_no_mark():
    """A stamped zero-copy delegate whose buffer lost its mark.

    Nothing but the compile spec is left recording that this engine's aliased
    output was elided, the mark being the only other witness to it.
    """
    graph_module, k_buffer, _ = _direct_delegate_graph(
        compile_specs=_zero_copy_specs("out_k")
    )
    del k_buffer.meta["_torch_tensorrt_aliased_buffer"]
    return _planned(graph_module)


@pytest.mark.unit
def test_check_zero_copy_kv_rejects_a_stamped_delegate_whose_method_lost_its_mark():
    """A method is read for both records, not passed over on the marks alone.

    ``_unstage_aliased_buffers`` refuses this graph -- the spec says an aliased
    output was elided and no marked buffer arrives to hold it -- so a check that
    passed it would call correct a method the pass calls broken. Both halves
    matter: the two-method program pins that a marked method does not vouch for
    an unmarked one, and the single-method program pins the message, which would
    otherwise be the program-wide "probably not exported with zero_copy_kv=True"
    on a program whose own compile spec says it was.
    """
    with pytest.raises(RuntimeError, match=r"takes 0 marked buffer"):
        Z.check_zero_copy_kv(
            _finalized_program(
                prefill=_unstaged_graph(), decode=_stamped_delegate_with_no_mark()
            )
        )

    with pytest.raises(RuntimeError, match=r"takes 0 marked buffer"):
        Z.check_zero_copy_kv(
            _finalized_program(decode=_stamped_delegate_with_no_mark())
        )


@pytest.mark.unit
def test_zero_copy_backend_config_defaults_to_executorch_defaults():
    """Called with no config it starts from ExecuTorch's defaults, and the one
    field it replaces is to_out_var_pass, wrapped in the un-staging pass."""
    from executorch.exir import ExecutorchBackendConfig

    config = torch_tensorrt.executorch.zero_copy_backend_config()

    defaults = ExecutorchBackendConfig()
    # The un-staging pass specifically, not merely "some object that is not the
    # default" -- which is all any wrapper would have to be. Compared by type
    # rather than by name, since the name is not what makes two configs from
    # this builder carry the same pass.
    assert isinstance(config.to_out_var_pass, Z._UnstageThenToOutVar)
    assert type(config.memory_planning_pass) is type(defaults.memory_planning_pass)
    assert type(config.sym_shape_eval_pass) is type(defaults.sym_shape_eval_pass)
    assert config.emit_stacktrace == defaults.emit_stacktrace


@pytest.mark.unit
@pytest.mark.parametrize(
    "skip",
    [True, {"decode": True}, {"prefill": False, "decode": True}, {"decode": False}],
    ids=["bool", "dict-one-true", "dict-one-of-two-true", "dict-all-false"],
)
def test_zero_copy_backend_config_refuses_skip_h2d_for_method_inputs(skip):
    """The one option that cannot be carried through, refused wherever it is on.

    ``skip_h2d_for_method_inputs`` is ExecuTorch's own un-staging of method
    inputs and it demands each placeholder it un-stages have exactly one user. A
    rewired cache has two -- the delegate, and the graph output it is its own
    mutation result for -- so ``PropagateDevicePass`` raises on every zero-copy
    graph. Preserving the option hands back a config that cannot finalize at
    all, which is a failure a long way from the line that caused it.

    ``dict-all-false`` is the case that makes the refusal key on truthiness
    rather than on ``True``: that pass is handed the field whole and only tests
    it for truth, so it reads a dict of ``False`` as on for every method, and
    finalizing such a config raises with ``placeholder 'b_k_cache' to have
    exactly one user``. ``False`` and ``{}`` are read as off there and are
    carried, in ``..._carries_skip_h2d_left_falsy``.

    The ids name what each value *is*, not who it applies to: that pass never
    resolves this field per method, so every dict here is on for every method
    whatever key it carries.
    """
    from executorch.exir import ExecutorchBackendConfig
    from executorch.exir.passes.propagate_device_config import PropagateDeviceConfig

    base = ExecutorchBackendConfig(
        propagate_device_config=PropagateDeviceConfig(skip_h2d_for_method_inputs=skip)
    )

    with pytest.raises(ValueError, match="skip_h2d_for_method_inputs"):
        Z.zero_copy_backend_config(base)


@pytest.mark.unit
def test_zero_copy_backend_config_refuses_skip_h2d_in_a_per_method_config():
    """``propagate_device_config`` is itself one config or a dict of them.

    ExecuTorch resolves a dict of ``PropagateDeviceConfig`` by method name
    (``_program.py``, ``edge_to_executorch_passes``) and hands the chosen one's
    ``skip_h2d_for_method_inputs`` to the pass, so the option reaches
    ``PropagateDevicePass`` from here exactly as it does from the single-config
    form. Reading only the single form leaves that route open: measured, such a
    config finalizes into the same ``exactly one user`` failure.
    """
    from executorch.exir import ExecutorchBackendConfig
    from executorch.exir.passes.propagate_device_config import PropagateDeviceConfig

    base = ExecutorchBackendConfig(
        propagate_device_config={
            "prefill": PropagateDeviceConfig(),
            "decode": PropagateDeviceConfig(skip_h2d_for_method_inputs=True),
        }
    )

    with pytest.raises(ValueError, match="skip_h2d_for_method_inputs for decode"):
        Z.zero_copy_backend_config(base)


@pytest.mark.unit
@pytest.mark.parametrize("skip", [False, {}], ids=["bool", "empty-dict"])
def test_zero_copy_backend_config_carries_skip_h2d_left_falsy(skip):
    """A field left falsy *to PropagateDevicePass* is carried through.

    That pass only tests the field for truth, so what it reads as off is exactly
    ``False`` and the empty dict -- and those two are what this carries. A
    non-empty dict of ``False`` is not one of them; it is refused, in
    ``..._refuses_skip_h2d_for_method_inputs[dict-all-false]``.
    """
    from executorch.exir import ExecutorchBackendConfig
    from executorch.exir.passes.propagate_device_config import PropagateDeviceConfig

    base = ExecutorchBackendConfig(
        propagate_device_config=PropagateDeviceConfig(skip_h2d_for_method_inputs=skip)
    )

    config = Z.zero_copy_backend_config(base)

    assert config.propagate_device_config.skip_h2d_for_method_inputs == skip


@pytest.mark.unit
def test_unstage_pass_runs_the_inner_pass_after_unstaging():
    """A caller's own to_out_var_pass has to survive being composed with."""
    graph_module, k_buffer, _, delegate = _staged_delegate_graph(
        compile_specs=_zero_copy_specs()
    )
    k_buffer.meta["_torch_tensorrt_aliased_buffer"] = True
    seen = []

    def inner(gm):
        # The un-staging is already done by the time the inner pass sees the graph.
        seen.append(delegate.args[1] is k_buffer)
        return "inner-result"

    result = Z.unstage_aliased_buffers_pass(inner).call(graph_module)

    assert seen == [True]
    assert result == "inner-result"


# --------------------------------------------------------------------------
# Multi-delegate: a method that lowers to two TensorRT engines -- one with an
# aliased+elided KV buffer, one plain-compute engine with none. The zero-copy
# CompileSpec is appended once, to a single TensorRTPartitioner, and the
# partitioner must stamp it onto ONLY the delegate whose own engine had an
# aliased output elided. Stamped partition-wide instead, the plain delegate
# declares zero-copy while un-staging nothing, and the un-staging cross-check
# then rejects an otherwise-correct program.
# --------------------------------------------------------------------------


def _no_op_engine_node(
    graph, input_nodes, *, aliased_io, input_names, output_names, engine=""
):
    """A no_op_placeholder_for_execute_engine node with inlined engine info.

    Mirrors what replace_execute_engine() produces before partitioning: args are
    ``(input_list, *engine_info)`` with the binding names and aliased_io in their
    serialized wire form, so the partitioner's real per-engine resolution
    (_resolve_engine_info / _aliased_inputs_by_output_index) runs unmocked.

    ``engine`` is the serialized-plan slot, which the partitioner never reads.
    Give it bytes to hand the same node to ``TensorRTBackend.preprocess``, which
    does.
    """
    from torch_tensorrt.dynamo.runtime._serialized_engine_layout import (
        ALIASED_IO_IDX,
        DEVICE_IDX,
        ENGINE_IDX,
        INPUT_BINDING_NAMES_IDX,
        OUTPUT_BINDING_NAMES_IDX,
        SERIALIZATION_LEN,
        SERIALIZED_ENGINE_BINDING_DELIM,
    )
    from torch_tensorrt.dynamo.runtime._TorchTensorRTModule import serialize_aliased_io

    info = [""] * SERIALIZATION_LEN
    info[ENGINE_IDX] = engine
    info[DEVICE_IDX] = "0"
    info[INPUT_BINDING_NAMES_IDX] = SERIALIZED_ENGINE_BINDING_DELIM.join(input_names)
    info[OUTPUT_BINDING_NAMES_IDX] = SERIALIZED_ENGINE_BINDING_DELIM.join(output_names)
    info[ALIASED_IO_IDX] = serialize_aliased_io(aliased_io)
    return graph.call_function(
        torch.ops.tensorrt.no_op_placeholder_for_execute_engine.default,
        (list(input_nodes), *info),
    )


def _two_engine_program():
    """engine_a(k_buffer, tokens) -> (logits, out_k[aliased]); engine_b(x) -> (y).

    ``k_buffer`` carries ``_torch_tensorrt_aliased_buffer`` (rewiring already
    ran); engine_a aliases its ``out_k`` output onto it, engine_b aliases nothing.
    """
    graph = torch.fx.Graph()
    k_buffer = graph.placeholder("b_k_0")
    tokens = graph.placeholder("tokens")
    x = graph.placeholder("x")
    engine_a = _no_op_engine_node(
        graph,
        [k_buffer, tokens],
        aliased_io={"out_k": ("k_in", "kv_cache_update")},
        input_names=["k_in", "tokens"],
        output_names=["logits", "out_k"],
    )
    engine_b = _no_op_engine_node(
        graph,
        [x],
        aliased_io={},
        input_names=["x_in"],
        output_names=["y"],
    )
    graph.output((engine_a, engine_b))
    k_buffer.meta["_torch_tensorrt_aliased_buffer"] = True
    graph_module = torch.fx.GraphModule(torch.nn.Module(), graph)
    program = SimpleNamespace(
        graph_module=graph_module,
        graph_signature=SimpleNamespace(buffers_to_mutate={}, inputs_to_buffers={}),
        constants={},
    )
    return program, engine_a, engine_b


def _partition_two_engines(program, engine_a, engine_b, monkeypatch):
    """Run the real TensorRTPartitioner, one partition per engine node."""
    from torch_tensorrt.executorch.backend import _serialize_elided_output_names
    from torch_tensorrt.executorch.partitioner import TensorRTPartitioner

    class _FakeCap:
        def __init__(self, graph_module, *args, **kwargs):
            self._engines = [engine_a, engine_b]

        def propose_partitions(self):
            return [
                SimpleNamespace(id=i, nodes=[node])
                for i, node in enumerate(self._engines)
            ]

    monkeypatch.setattr(
        "torch_tensorrt.executorch.partitioner.CapabilityBasedPartitioner", _FakeCap
    )
    monkeypatch.setattr(
        "torch_tensorrt.executorch.partitioner.tag_constant_data",
        lambda exported_program: None,
    )
    # Appended once, method-wide -- exactly how export() builds the partitioner.
    partitioner = TensorRTPartitioner(
        compile_specs=[
            CompileSpec(
                ZERO_COPY_KV_COMPILE_SPEC_KEY,
                _serialize_elided_output_names(["out_k"]),
            )
        ]
    )
    return partitioner.partition(program)


def _zero_copy_names(compile_specs):
    from torch_tensorrt.executorch.backend import _elided_output_names

    return _elided_output_names(compile_specs)


@pytest.mark.unit
def test_partition_stamps_zero_copy_only_on_the_kv_delegate(monkeypatch):
    """The KV delegate carries the zero-copy spec naming its own elided binding,
    and the plain-compute delegate carries no zero-copy spec at all.

    The method-wide spec the partitioner is constructed with must not reach every
    partition: the names it holds are the method's, and only this engine's own
    aliased_io says which of them are its.
    """
    program, engine_a, engine_b = _two_engine_program()
    result = _partition_two_engines(program, engine_a, engine_b, monkeypatch)

    kv_specs = result.partition_tags["tensorrt_0"].compile_specs
    plain_specs = result.partition_tags["tensorrt_1"].compile_specs
    assert _zero_copy_names(kv_specs) == {"out_k"}
    assert _zero_copy_names(plain_specs) is None


@pytest.mark.unit
def test_multi_delegate_zero_copy_lowers_without_false_raise(monkeypatch):
    """A correct two-delegate zero-copy program survives the whole pipeline: run
    the real partitioner, build the lowered two-delegate graph from the specs it
    produced, and un-stage.

    The KV buffer is un-staged and the plain delegate is left alone. A plain
    delegate stamped zero-copy would instead make _unstage_aliased_buffers raise
    "declares zero-copy KV ... but takes no buffer marked for in-place update"
    over a program that is correct.
    """
    program, engine_a, engine_b = _two_engine_program()
    result = _partition_two_engines(program, engine_a, engine_b, monkeypatch)
    kv_specs = result.partition_tags["tensorrt_0"].compile_specs
    plain_specs = result.partition_tags["tensorrt_1"].compile_specs

    graph = torch.fx.Graph()
    k_buffer = graph.placeholder("b_k_0")
    x = graph.placeholder("x")
    h2d = torch.ops.et_copy._h2d_copy.default
    staged_k = graph.call_function(h2d, (k_buffer,))
    staged_x = graph.call_function(h2d, (x,))
    kv_lowered = graph.get_attr("lowered_module_0")
    plain_lowered = graph.get_attr("lowered_module_1")
    kv_delegate = graph.call_function(executorch_call_delegate, (kv_lowered, staged_k))
    plain_delegate = graph.call_function(
        executorch_call_delegate, (plain_lowered, staged_x)
    )
    graph.output((k_buffer, kv_delegate, plain_delegate))
    root = torch.nn.Module()
    root.lowered_module_0 = SimpleNamespace(
        backend_id="TensorRTBackend", compile_specs=kv_specs
    )
    root.lowered_module_1 = SimpleNamespace(
        backend_id="TensorRTBackend", compile_specs=plain_specs
    )
    graph_module = torch.fx.GraphModule(root, graph)
    for node, dev in (
        (k_buffer, DeviceType.CPU),
        (x, DeviceType.CPU),
        (staged_k, DeviceType.CUDA),
        (staged_x, DeviceType.CUDA),
    ):
        node.meta["spec"] = SimpleNamespace(device=dev, device_index=3)
    k_buffer.meta["_torch_tensorrt_aliased_buffer"] = True

    assert Z._unstage_aliased_buffers(graph_module) == 1
    assert kv_delegate.args[1] is k_buffer
    # The plain delegate keeps its staging and is never demanded to un-stage.
    assert plain_delegate.args[1] is staged_x


@pytest.mark.unit
def test_unstage_raises_when_the_plain_delegate_is_wrongly_stamped():
    """The other side of the per-partition stamping, in isolation: a delegate
    that carries the zero-copy spec and un-stages nothing must raise, whatever
    put the spec there. Narrowing which delegates get stamped must not weaken
    this -- it is the lost-update guard for the KV delegate too.
    """
    zero_copy_spec = [CompileSpec(ZERO_COPY_KV_COMPILE_SPEC_KEY, b'["out_k"]')]
    graph = torch.fx.Graph()
    k_buffer = graph.placeholder("b_k_0")
    x = graph.placeholder("x")
    h2d = torch.ops.et_copy._h2d_copy.default
    staged_k = graph.call_function(h2d, (k_buffer,))
    staged_x = graph.call_function(h2d, (x,))
    kv_lowered = graph.get_attr("lowered_module_0")
    plain_lowered = graph.get_attr("lowered_module_1")
    kv_delegate = graph.call_function(executorch_call_delegate, (kv_lowered, staged_k))
    plain_delegate = graph.call_function(
        executorch_call_delegate, (plain_lowered, staged_x)
    )
    graph.output((k_buffer, kv_delegate, plain_delegate))
    root = torch.nn.Module()
    # Both delegates wrongly carry the spec -- the shape that per-engine stamping
    # in TensorRTPartitioner exists to prevent.
    root.lowered_module_0 = SimpleNamespace(
        backend_id="TensorRTBackend", compile_specs=list(zero_copy_spec)
    )
    root.lowered_module_1 = SimpleNamespace(
        backend_id="TensorRTBackend", compile_specs=list(zero_copy_spec)
    )
    graph_module = torch.fx.GraphModule(root, graph)
    for node, dev in (
        (k_buffer, DeviceType.CPU),
        (x, DeviceType.CPU),
        (staged_k, DeviceType.CUDA),
        (staged_x, DeviceType.CUDA),
    ):
        node.meta["spec"] = SimpleNamespace(device=dev, device_index=3)
    k_buffer.meta["_torch_tensorrt_aliased_buffer"] = True

    with pytest.raises(RuntimeError, match="declares zero-copy KV"):
        Z._unstage_aliased_buffers(graph_module)


# --------------------------------------------------------------------------
# Single-engine, on the same engine-node helper: the same per-engine derivation
# also has to narrow *within* one engine, from every aliased output down to the
# ones whose aliased input is a buffer export rewired.
# --------------------------------------------------------------------------


@pytest.mark.unit
def test_a_mixed_alias_engine_derives_the_narrower_set_and_is_then_refused():
    """One engine, two aliased outputs, one marked buffer: the narrower elidable
    set is right, and an engine that needs it cannot be lowered.

    An aliased output whose input is not a buffer export rewired -- a user alias,
    which nothing rewires and whose placeholder therefore carries no
    ``_torch_tensorrt_aliased_buffer`` -- is still a delegate output. Deriving the
    elidable set from the engine's aliased_io alone would exempt it too, and the
    backend would then accept a delegate that dropped a mutation nothing writes
    back.

    The derivation is right and the engine is still unusable, because the runtime
    reads elision off one argument count and subtracts the engine's *whole*
    aliased-output count. A .pte written for this shape loads and then fails
    every ``execute()`` with an argument-count error, so ``preprocess`` refuses
    it where the export can still be re-run.
    """
    from torch_tensorrt.executorch._zero_copy import _aliased_inputs_by_output_index
    from torch_tensorrt.executorch.backend import (
        TensorRTBackend,
        _serialize_elided_output_names,
    )
    from torch_tensorrt.executorch.partitioner import TensorRTPartitioner

    graph = torch.fx.Graph()
    k_buffer = graph.placeholder("b_k_0")
    user_alias = graph.placeholder("u")
    engine = _no_op_engine_node(
        graph,
        [k_buffer, user_alias],
        aliased_io={
            "out_k": ("k_in", "kv_cache_update"),
            "out_u": ("u_in", "user"),
        },
        input_names=["k_in", "u_in"],
        output_names=["out_k", "out_u"],
        engine=b"engine-bytes",
    )
    # out_k has already left the partition with the mutation that was rewired
    # onto the buffer, so the delegate returns the one surviving binding.
    graph.output((engine,))
    k_buffer.meta["_torch_tensorrt_aliased_buffer"] = True
    graph_module = torch.fx.GraphModule(torch.nn.Module(), graph)
    program = SimpleNamespace(
        graph_module=graph_module,
        graph_signature=SimpleNamespace(buffers_to_mutate={}, inputs_to_buffers={}),
        constants={},
    )
    partition = SimpleNamespace(id=0, nodes=[engine])

    # Both outputs are aliased, or the narrowing below would have nothing to do.
    assert set(_aliased_inputs_by_output_index(program, engine)) == {0, 1}

    # The spec deliberately names the output the derivation must NOT pick, so a
    # result of {"out_k"} can only have come from the engine's aliased_io and the
    # marks on its inputs.
    partitioner = TensorRTPartitioner(
        compile_specs=[CompileSpec(ZERO_COPY_KV_COMPILE_SPEC_KEY, b'["out_u"]')]
    )
    elided = partitioner._partition_elided_output_names(program, partition)
    assert elided == {"out_k"}

    with pytest.raises(ValueError, match="Partial elision is not expressible"):
        TensorRTBackend.preprocess(
            program,
            [
                CompileSpec(
                    ZERO_COPY_KV_COMPILE_SPEC_KEY,
                    _serialize_elided_output_names(elided),
                )
            ],
        )


def _one_aliased_engine_partition(marked):
    """One engine with one aliased output, its buffer input marked or not."""
    from torch_tensorrt.executorch.partitioner import TensorRTPartitioner

    graph = torch.fx.Graph()
    k_buffer = graph.placeholder("b_k_0")
    engine = _no_op_engine_node(
        graph,
        [k_buffer],
        aliased_io={"out_k": ("k_in", "kv_cache_update")},
        input_names=["k_in"],
        output_names=["out_k"],
    )
    graph.output((engine,))
    if marked:
        k_buffer.meta["_torch_tensorrt_aliased_buffer"] = True
    program = SimpleNamespace(
        graph_module=torch.fx.GraphModule(torch.nn.Module(), graph),
        graph_signature=SimpleNamespace(buffers_to_mutate={}, inputs_to_buffers={}),
        constants={},
    )
    return TensorRTPartitioner(), program, SimpleNamespace(id=0, nodes=[engine])


def _make_engine_info_unreadable(monkeypatch):
    def _boom(*args, **kwargs):
        raise RuntimeError("engine record unreadable")

    monkeypatch.setattr(
        "torch_tensorrt.executorch.partitioner._get_engine_info_for_node", _boom
    )


@pytest.mark.unit
def test_unreadable_engine_propagates_when_a_buffer_was_rewired(monkeypatch):
    """A method with a rewired buffer must not fall back to eliding nothing.

    The aliased outputs of a rewired buffer left the graph before partitioning,
    so an empty set stamps no delegate and the export dies further down blaming a
    lost aliased-buffer mark -- a failure that names neither this partition nor
    the record that would not read.
    """
    partitioner, program, partition = _one_aliased_engine_partition(marked=True)
    _make_engine_info_unreadable(monkeypatch)

    with pytest.raises(RuntimeError, match="engine record unreadable"):
        partitioner._partition_elided_output_names(program, partition)


@pytest.mark.unit
def test_unreadable_engine_elides_nothing_when_no_buffer_was_rewired(
    monkeypatch, caplog
):
    """With nothing rewired the delegate really does carry every binding.

    The pair with the test above is the whole point of the handler: the same
    failure is survivable here and not there, and only the graph says which.
    """
    partitioner, program, partition = _one_aliased_engine_partition(marked=False)
    _make_engine_info_unreadable(monkeypatch)

    with caplog.at_level(
        logging.WARNING, logger="torch_tensorrt.executorch.partitioner"
    ):
        assert partitioner._partition_elided_output_names(program, partition) == set()
    assert "could not resolve elided outputs" in caplog.text


@pytest.mark.unit
def test_a_partition_holding_two_engines_elides_nothing():
    """Elision is derived from one engine's aliased_io, so two is not answerable.

    ``TensorRTBackend.preprocess`` refuses a multi-engine partition outright, but
    this runs first, and guessing here would stamp one engine's binding names
    onto a delegate that also carries another's.
    """
    from torch_tensorrt.executorch.partitioner import TensorRTPartitioner

    program, engine_a, engine_b = _two_engine_program()
    both = SimpleNamespace(id=0, nodes=[engine_a, engine_b])

    partitioner = TensorRTPartitioner()
    assert partitioner._partition_elided_output_names(program, both) == set()
    # The same partitioner does answer for engine_a alone, so the empty set above
    # is the two-engine shape and not a graph that had nothing to elide.
    single = SimpleNamespace(id=0, nodes=[engine_a])
    assert partitioner._partition_elided_output_names(program, single) == {"out_k"}


# --------------------------------------------------------------------------
# GPU integration: the mark set during rewiring must survive real lowering, or
# the un-staging pass has nothing to act on and every KV update is lost. Only a
# real export exercises that -- the stub graphs above set the mark by hand.
# --------------------------------------------------------------------------
VOCAB = 64
DIM = 32
HEADS = 2
HEAD_DIM = 16
MAX_LEN = 16


class _KVDecodeStep(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = torch.nn.Embedding(VOCAB, DIM)
        self.pos_embed = torch.nn.Embedding(MAX_LEN, DIM)
        self.q = torch.nn.Linear(DIM, HEADS * HEAD_DIM, bias=False)
        self.k = torch.nn.Linear(DIM, HEADS * HEAD_DIM, bias=False)
        self.v = torch.nn.Linear(DIM, HEADS * HEAD_DIM, bias=False)
        self.o = torch.nn.Linear(HEADS * HEAD_DIM, DIM, bias=False)
        self.lm = torch.nn.Linear(DIM, VOCAB, bias=False)
        self.register_buffer("k_cache", torch.zeros(1, HEADS, MAX_LEN, HEAD_DIM))
        self.register_buffer("v_cache", torch.zeros(1, HEADS, MAX_LEN, HEAD_DIM))

    def forward(self, tokens: torch.Tensor, input_pos: torch.Tensor) -> torch.Tensor:
        pos_idx = input_pos.reshape(-1)
        pos = input_pos.reshape(())
        x = self.embed(tokens) + self.pos_embed(input_pos.reshape(1, 1))

        def split_heads(proj: torch.Tensor) -> torch.Tensor:
            return proj.view(1, 1, HEADS, HEAD_DIM).transpose(1, 2)

        q = split_heads(self.q(x))
        k = split_heads(self.k(x))
        v = split_heads(self.v(x))
        self.k_cache.index_copy_(2, pos_idx, k)
        self.v_cache.index_copy_(2, pos_idx, v)
        scores = (q @ self.k_cache.transpose(-1, -2)) / (HEAD_DIM**0.5)
        allowed = torch.arange(MAX_LEN, device=x.device) <= pos
        bias = torch.where(
            allowed,
            torch.zeros((), dtype=x.dtype, device=x.device),
            torch.full((), torch.finfo(x.dtype).min, dtype=x.dtype, device=x.device),
        )
        attn = torch.softmax(scores + bias.view(1, 1, 1, MAX_LEN), dim=-1)
        out = (attn @ self.v_cache).transpose(1, 2).reshape(1, 1, HEADS * HEAD_DIM)
        return self.lm(self.o(out))


@pytest.mark.parametrize("generate_etrecord", [False, True], ids=["plain", "etrecord"])
def test_aliased_buffer_mark_survives_real_lowering(generate_etrecord):
    """After a real export(..., zero_copy_kv=True), the KV buffer placeholder in
    the lowered edge program still carries ``_torch_tensorrt_aliased_buffer`` --
    the token the to_out_var_pass keys the un-staging on.

    ``generate_etrecord=True`` is covered because it makes ExecuTorch deep copy
    the whole program, and the mark rides on node meta. Losing it there would not
    raise here: it surfaces later as the un-staging pass finding a marked buffer
    it never un-staged, by which point the connection to this option is gone.
    """
    _require_real_engine()
    with torch.no_grad():
        torch.manual_seed(0)
        model = _KVDecodeStep().eval().cuda()
        tokens = torch.zeros(1, 1, dtype=torch.long).cuda()
        input_pos = torch.tensor([0], dtype=torch.long).cuda()

        exported_program = torch.export.export(model, (tokens, input_pos))
        trt_gm = torch_tensorrt.dynamo.compile(
            exported_program,
            arg_inputs=(tokens, input_pos),
            min_block_size=1,
            truncate_double=True,
        )
        edge = torch_tensorrt.executorch.export(
            trt_gm,
            arg_inputs=(tokens, input_pos),
            retrace=False,
            zero_copy_kv=True,
            generate_etrecord=generate_etrecord,
        )

    ep = edge.exported_program()
    marked = {
        ep.graph_signature.inputs_to_buffers[node.name]
        for node in ep.graph_module.graph.nodes
        if node.op == "placeholder" and node.meta.get("_torch_tensorrt_aliased_buffer")
    }
    # Both, by name. The model registers two caches, so asserting the list is
    # non-empty would pass on partial marker loss -- which is the very shape the
    # per-delegate count check elsewhere in this feature exists for.
    assert marked == {"k_cache", "v_cache"}


def test_finalizing_a_real_export_with_executorch_defaults_is_caught_only_by_the_check():
    """The documented two-call path, with the second call left as the default.

    ``to_executorch()`` with ExecuTorch's own defaults runs no un-staging, and
    export has already removed the copy-back, so what it produces is a whole
    ``.pte`` whose caches never update -- wrong output for a KV cache. This pins
    both halves of that: finalization *succeeds*, because ``to_executorch`` is
    ExecuTorch's own and nothing in this library is between the caller and it,
    and ``check_zero_copy_kv`` on the result is what says so -- on a program
    that really was lowered and really was finalized, which is the shape the
    stub-graph tests of that check cannot build.
    """
    _require_real_engine()
    with torch.no_grad():
        torch.manual_seed(0)
        model = _KVDecodeStep().eval().cuda()
        tokens = torch.zeros(1, 1, dtype=torch.long).cuda()
        input_pos = torch.tensor([0], dtype=torch.long).cuda()

        exported_program = torch.export.export(model, (tokens, input_pos))
        trt_gm = torch_tensorrt.dynamo.compile(
            exported_program,
            arg_inputs=(tokens, input_pos),
            min_block_size=1,
            truncate_double=True,
        )
        edge = torch_tensorrt.executorch.export(
            trt_gm,
            arg_inputs=(tokens, input_pos),
            retrace=False,
            zero_copy_kv=True,
        )

    program = edge.to_executorch()

    with pytest.raises(RuntimeError, match="do not reach the TensorRT delegate"):
        torch_tensorrt.executorch.check_zero_copy_kv(program)


class _MixedDecodeStep(torch.nn.Module):
    """A decode step with an engine-aliased KV cache and a copy-back buffer.

    ``k_cache``/``v_cache`` are written by ``index_copy_`` on the sequence axis,
    which the converter turns into an aliased engine binding. ``conv_state`` is a
    ring shift -- a whole-buffer rewrite with no position to alias on -- so
    ``lift_mutated_buffers`` records it in ``_copyback_mutation_buffers`` and its
    new value comes back as a trailing delegate output instead.
    """

    def __init__(self) -> None:
        super().__init__()
        self.embed = torch.nn.Embedding(VOCAB, DIM)
        self.q = torch.nn.Linear(DIM, HEADS * HEAD_DIM, bias=False)
        self.k = torch.nn.Linear(DIM, HEADS * HEAD_DIM, bias=False)
        self.v = torch.nn.Linear(DIM, HEADS * HEAD_DIM, bias=False)
        self.o = torch.nn.Linear(HEADS * HEAD_DIM, DIM, bias=False)
        self.lm = torch.nn.Linear(DIM, VOCAB, bias=False)
        self.register_buffer("k_cache", torch.zeros(1, HEADS, MAX_LEN, HEAD_DIM))
        self.register_buffer("v_cache", torch.zeros(1, HEADS, MAX_LEN, HEAD_DIM))
        self.register_buffer("conv_state", torch.zeros(1, DIM, 4))

    def forward(self, tokens: torch.Tensor, input_pos: torch.Tensor) -> torch.Tensor:
        pos_idx = input_pos.reshape(-1)
        pos = input_pos.reshape(())
        x = self.embed(tokens)

        shifted = torch.cat([self.conv_state[:, :, 1:], x.reshape(1, DIM, 1)], dim=2)
        self.conv_state.copy_(shifted)
        x = x + self.conv_state.sum(dim=2).reshape(1, 1, DIM)

        def split_heads(proj: torch.Tensor) -> torch.Tensor:
            return proj.view(1, 1, HEADS, HEAD_DIM).transpose(1, 2)

        q = split_heads(self.q(x))
        k = split_heads(self.k(x))
        v = split_heads(self.v(x))
        self.k_cache.index_copy_(2, pos_idx, k)
        self.v_cache.index_copy_(2, pos_idx, v)
        scores = (q @ self.k_cache.transpose(-1, -2)) / (HEAD_DIM**0.5)
        allowed = torch.arange(MAX_LEN, device=x.device) <= pos
        bias = torch.where(
            allowed,
            torch.zeros((), dtype=x.dtype, device=x.device),
            torch.full((), torch.finfo(x.dtype).min, dtype=x.dtype, device=x.device),
        )
        attn = torch.softmax(scores + bias.view(1, 1, 1, MAX_LEN), dim=-1)
        out = (attn @ self.v_cache).transpose(1, 2).reshape(1, 1, HEADS * HEAD_DIM)
        return self.lm(self.o(out))


def _real_delegates(graph_module):
    return [
        node
        for node in graph_module.graph.nodes
        if node.op == "call_function" and node.target is executorch_call_delegate
    ]


def _lowered_module(graph_module, delegate):
    return getattr(graph_module, delegate.args[0].target)


def _assert_marked_buffers_reach_the_engine_unstaged(program):
    """Every marked buffer is a direct argument of a TensorRT delegate.

    Finalizing a zero-copy program without raising is a weak signal, because both
    of the completeness raises at the end of ``_unstage_aliased_buffers`` fire off
    its own bookkeeping: a pass that records each un-staging and then leaves the
    argument pointing at the staging copy still satisfies them. Only the graph says
    whether the rewiring happened, and getting it wrong is silent -- the engine
    writes per-call scratch that is discarded and the cache never updates.

    The library's own check runs first, on the whole program. It is the stronger
    of the two -- it also requires the delegate to carry the zero-copy compile
    spec and the buffer to be planned somewhere the engine can write it, neither
    of which the assertions below read -- and it is the only place the container
    API it reads, ``methods`` and ``exported_program(name)``, meets a real
    ``ExecutorchProgramManager``: every other test of it builds the program
    itself, so an upstream rename would leave those green and break
    ``save(zero_copy_kv=True)`` for every caller. What follows is kept because it
    reads the graph rather than the marks, which is what says the rewiring
    actually happened.
    """
    torch_tensorrt.executorch.check_zero_copy_kv(program)
    graph_module = program.exported_program().graph_module
    marked = [
        node
        for node in graph_module.graph.nodes
        if node.op == "placeholder" and node.meta.get("_torch_tensorrt_aliased_buffer")
    ]
    assert marked, "no buffer was marked for in-place update"
    reached = {
        arg
        for node in _real_delegates(graph_module)
        if _lowered_module(graph_module, node).backend_id == "TensorRTBackend"
        for arg in node.args[1:]
        if isinstance(arg, torch.fx.Node)
    }
    for node in marked:
        assert node in reached, (
            f"buffer '{node.name}' is marked for in-place update but is not a "
            "direct argument of any TensorRT delegate -- it either still reaches "
            "one through a staging copy, or reaches none at all. Either way "
            "nothing writes the caller's buffer and the cache never updates"
        )


class _StubProgram:
    """The three attributes the reorder and upstream's write-back pass read.

    ``graph_signature`` is a property over ``_graph_signature`` because that is
    how ``ExportedProgram`` exposes it, and the reorder replaces the signature by
    assigning the private name.
    """

    def __init__(self, graph_module, signature):
        self.graph_module = graph_module
        self._graph_signature = signature

    @property
    def graph_signature(self):
        return self._graph_signature

    @property
    def graph(self):
        return self.graph_module.graph


def _two_mutation_program(first_value_is_inplace):
    """A two-mutation program in the shape the write-back pass sees.

    Slot 0 mutates ``first``; its value is either an in-place op on its own
    buffer -- which upstream reads as needing no copy -- or an ordinary
    functional result, which does. Slot 1 is always an ordinary copy-back on
    ``cb``. Nothing here carries the zero-copy mark, so a reorder keyed on that
    mark cannot see slot 0 at all.
    """
    from torch.export.exported_program import ExportGraphSignature
    from torch.export.graph_signature import InputKind, InputSpec

    graph = torch.fx.Graph()
    b_first = graph.placeholder("b_first")
    b_cb = graph.placeholder("b_cb")
    x = graph.placeholder("x")
    first_value = (
        graph.call_function(torch.ops.aten.add_.Tensor, (b_first, x))
        if first_value_is_inplace
        else graph.call_function(torch.ops.aten.add.Tensor, (b_first, x))
    )
    cb_value = graph.call_function(torch.ops.aten.add.Tensor, (b_cb, x))
    user = graph.call_function(torch.ops.aten.mul.Tensor, (x, x))
    graph.output((first_value, cb_value, user))
    graph_module = torch.fx.GraphModule(torch.nn.Module(), graph)
    signature = ExportGraphSignature(
        input_specs=[
            InputSpec(InputKind.BUFFER, TensorArgument("b_first"), "first", False),
            InputSpec(InputKind.BUFFER, TensorArgument("b_cb"), "cb", False),
            InputSpec(InputKind.USER_INPUT, TensorArgument("x"), None),
        ],
        output_specs=[
            OutputSpec(
                OutputKind.BUFFER_MUTATION, TensorArgument(first_value.name), "first"
            ),
            OutputSpec(OutputKind.BUFFER_MUTATION, TensorArgument(cb_value.name), "cb"),
            OutputSpec(OutputKind.USER_OUTPUT, TensorArgument(user.name), None),
        ],
    )
    return _StubProgram(graph_module, signature)


@pytest.mark.unit
@pytest.mark.parametrize("reorder", [False, True], ids=["without", "with"])
def test_reorder_moves_an_inplace_mutation_this_feature_did_not_create(reorder):
    """The reorder keys on upstream's predicate, not on this feature's own mark.

    Slot 0 is in-place in the graph and carries no zero-copy mark, so upstream
    inserts no copy for it exactly as for a rewired cache, and a reorder that
    asked "did zero-copy rewire this?" instead of "will upstream copy this?"
    would leave the pair crossed while reporting that it moved nothing. A
    mutation ``reinplace_pass`` rewrites is *not* this case: that pass runs after
    the reorder, so such a mutation is still ordinary when the predicate is asked
    and no reorder here can pre-empt it -- see ``order_copyback_mutations_first``.
    """
    from executorch.exir.passes.insert_write_back_for_buffers_pass import (
        insert_write_back_for_buffers_pass,
    )

    program = _two_mutation_program(first_value_is_inplace=True)
    moved = Z.order_copyback_mutations_first(program) if reorder else 0
    _, signature = insert_write_back_for_buffers_pass(program)
    value_of = {buffer: value for value, buffer in signature.buffers_to_mutate.items()}

    if not reorder:
        # Pin the defect too, so the assertions below cannot pass vacuously.
        assert value_of["first"].startswith("copy_")
        return
    assert moved == 2
    assert not value_of["first"].startswith("copy_"), (
        "'first' is mutated in place, so upstream inserts no copy for it and its "
        f"finalized value must not be one; got {value_of['first']!r}"
    )
    assert value_of["cb"].startswith("copy_"), (
        "'cb' is copied back, so its finalized value is the copy upstream "
        f"inserted; got {value_of['cb']!r}"
    )


@pytest.mark.unit
def test_reorder_leaves_an_already_correct_order_alone():
    """Two copy-back mutations need no move, and the function says so."""
    program = _two_mutation_program(first_value_is_inplace=False)
    before = [spec.arg.name for spec in program.graph_signature.output_specs]

    assert Z.order_copyback_mutations_first(program) == 0
    assert [spec.arg.name for spec in program.graph_signature.output_specs] == before


@pytest.mark.unit
def test_reorder_treats_an_unlifted_mutation_target_as_needing_no_copy():
    """A mutation upstream cannot resolve to a lifted input gets no copy.

    ``insert_write_back_for_buffers_pass`` only copies mutations whose target is
    in the map it builds from the input specs, so one that is not belongs with
    the copy-free mutations however its value was produced -- and its value here
    is an ordinary functional result, which is what the lineage test reads as
    needing a copy. Asking only the lineage test would leave this slot ahead of
    the real copy-back and cross the finalized pairing.
    """
    from torch.export.exported_program import ExportGraphSignature
    from torch.export.graph_signature import InputKind, InputSpec

    graph = torch.fx.Graph()
    b_cb = graph.placeholder("b_cb")
    x = graph.placeholder("x")
    unlifted_value = graph.call_function(torch.ops.aten.add.Tensor, (x, x))
    cb_value = graph.call_function(torch.ops.aten.add.Tensor, (b_cb, x))
    graph.output((unlifted_value, cb_value))
    program = _StubProgram(
        torch.fx.GraphModule(torch.nn.Module(), graph),
        ExportGraphSignature(
            input_specs=[
                InputSpec(InputKind.BUFFER, TensorArgument("b_cb"), "cb", False),
                InputSpec(InputKind.USER_INPUT, TensorArgument("x"), None),
            ],
            output_specs=[
                # No input spec targets "ghost", so it is not in the lifted map.
                OutputSpec(
                    OutputKind.BUFFER_MUTATION,
                    TensorArgument(unlifted_value.name),
                    "ghost",
                ),
                OutputSpec(
                    OutputKind.BUFFER_MUTATION, TensorArgument(cb_value.name), "cb"
                ),
            ],
        ),
    )

    assert Z.order_copyback_mutations_first(program) == 2
    assert [spec.target for spec in program.graph_signature.output_specs] == [
        "cb",
        "ghost",
    ]


@pytest.mark.unit
def test_reorder_groups_a_non_node_mutation_value_with_the_copies():
    """A mutation slot holding a literal is ordered as though it were copied.

    Upstream reads a non-Node value as needing a copy and then raises walking it,
    so putting it anywhere else would make this reorder the thing that raises and
    hide the program upstream is actually complaining about.
    """
    from torch.export.exported_program import ExportGraphSignature
    from torch.export.graph_signature import ConstantArgument, InputKind, InputSpec

    graph = torch.fx.Graph()
    b_first = graph.placeholder("b_first")
    b_cb = graph.placeholder("b_cb")
    x = graph.placeholder("x")
    inplace_value = graph.call_function(torch.ops.aten.add_.Tensor, (b_first, x))
    graph.output((inplace_value, 7))
    program = _StubProgram(
        torch.fx.GraphModule(torch.nn.Module(), graph),
        ExportGraphSignature(
            input_specs=[
                InputSpec(InputKind.BUFFER, TensorArgument("b_first"), "first", False),
                InputSpec(InputKind.BUFFER, TensorArgument("b_cb"), "cb", False),
                InputSpec(InputKind.USER_INPUT, TensorArgument("x"), None),
            ],
            output_specs=[
                OutputSpec(
                    OutputKind.BUFFER_MUTATION,
                    TensorArgument(inplace_value.name),
                    "first",
                ),
                OutputSpec(
                    OutputKind.BUFFER_MUTATION,
                    ConstantArgument(name="literal", value=7),
                    "cb",
                ),
            ],
        ),
    )

    assert Z.order_copyback_mutations_first(program) == 2
    assert [spec.target for spec in program.graph_signature.output_specs] == [
        "cb",
        "first",
    ]
    assert program.graph_module.graph.output_node().args[0][0] == 7


def _assert_each_mutation_names_its_own_value(program):
    """The finalized signature pairs every mutated buffer with its own new value.

    ExecuTorch finalizes the mutations by inserting a copy for each one whose
    value is not already reached from a buffer placeholder through in-place ops,
    moving those copies to the front of the output tuple, and then reassigning
    the mutation specs' arguments by position. Rewiring a cache to its own
    placeholder takes it out of that leading run, and so does any other
    mutation upstream reads as in-place, so a method that mixes the two kinds
    comes out of finalization with each buffer named against another buffer's
    value unless the mutations were declared in the order the pass assumes.
    Nothing inside the finalizer reads the pairing, so the ``.pte`` is written
    either way -- what reads it is
    anyone inspecting the program, and the eager call path that copies mutated
    values back into the state dict in this order.
    """
    signature = program.exported_program().graph_signature
    placeholder_of = {fqn: name for name, fqn in signature.inputs_to_buffers.items()}
    mutated = {
        spec.target: spec.arg.name
        for spec in signature.output_specs
        if spec.kind == OutputKind.BUFFER_MUTATION
    }
    assert set(mutated) == {"k_cache", "v_cache", "conv_state"}
    for cache in ("k_cache", "v_cache"):
        assert mutated[cache] == placeholder_of[cache], (
            f"the finalized signature gives {cache} the value "
            f"{mutated[cache]!r}, but zero-copy left that cache as its own "
            "mutation result, so its value is its own placeholder "
            f"{placeholder_of[cache]!r}"
        )
    assert mutated["conv_state"] not in placeholder_of.values(), (
        "the finalized signature gives conv_state a buffer placeholder as its "
        f"value ({mutated['conv_state']!r}); it is copied back, so its value is "
        "the copy ExecuTorch inserted"
    )


@pytest.mark.parametrize("retrace", [False, True], ids=["legacy", "retrace"])
def test_zero_copy_kv_keeps_a_copyback_buffer_in_the_same_method(retrace):
    """A real method holding both kinds of mutable buffer exports and keeps both.

    The KV caches end up bound to their own placeholders -- no value for
    ExecuTorch to copy, which is the zero copy -- while ``conv_state`` stays bound
    to a delegate output, which is the value ExecuTorch copies back into it.
    Losing that distinction in either direction is silent wrong output, so it is
    pinned on a real engine rather than a stub: the aliased_io the discriminator
    reads is produced by the converter, not by this test.

    Both exporters are covered because they reach that distinction by different
    routes, and only one of them is the ``save()`` default. The legacy exporter
    declares all three mutations while it inlines the engines, so
    ``_declare_aliased_kv_mutations_on_ep`` finds nothing left to do and the
    discriminator never runs. Under ``retrace=True`` the retraced program arrives
    with no mutations declared at all -- torch.export drops the aliased outputs at
    the fx boundary and leaves the copy-back value as a plain return -- so that
    pass is what separates the two kinds, by reading each engine's ``aliased_io``.
    """
    _require_real_engine()
    with torch.no_grad():
        torch.manual_seed(0)
        model = _MixedDecodeStep().eval().cuda()
        tokens = torch.zeros(1, 1, dtype=torch.long).cuda()
        input_pos = torch.tensor([0], dtype=torch.long).cuda()

        exported_program = torch.export.export(model, (tokens, input_pos))
        trt_gm = torch_tensorrt.dynamo.compile(
            exported_program,
            arg_inputs=(tokens, input_pos),
            min_block_size=1,
            truncate_double=True,
        )
        assert trt_gm.meta.get("_copyback_mutation_buffers") == ["conv_state"], (
            "the model no longer produces a copy-back buffer, so this test would "
            "pass without exercising the combination it exists for"
        )
        edge = torch_tensorrt.executorch.export(
            trt_gm,
            arg_inputs=(tokens, input_pos),
            retrace=retrace,
            zero_copy_kv=True,
        )

    ep = edge.exported_program()
    output_args = list(ep.graph_module.graph.output_node().args[0])
    bound = {
        spec.target: value
        for spec, value in zip(ep.graph_signature.output_specs, output_args)
        if spec.kind == OutputKind.BUFFER_MUTATION
    }
    assert set(bound) == {"k_cache", "v_cache", "conv_state"}
    for name in ("k_cache", "v_cache"):
        assert bound[name].op == "placeholder", (
            f"{name} is still satisfied by a delegate output, so ExecuTorch will "
            "copy it back and zero-copy bought nothing"
        )
        assert bound[name].meta.get("_torch_tensorrt_aliased_buffer") is True
    assert bound["conv_state"].op == "call_function", (
        "conv_state was rewired to its own placeholder, which deletes the "
        "copy-back of a buffer no engine writes in place -- a lost update"
    )
    assert "_torch_tensorrt_aliased_buffer" not in bound["conv_state"].meta

    # Everything above is the export half. The staging the other half removes does
    # not exist until PropagateDevicePass runs inside to_executorch, so this is the
    # earliest point at which the caches can be seen reaching the engine directly.
    # Composed onto a caller's own config, the optional form the user guide
    # describes. It is the only shape under which a preserved field can make the
    # returned config unfinalizable; a no-argument zero_copy_backend_config()
    # starts from the defaults and so has nothing to preserve.
    from executorch.exir import ExecutorchBackendConfig

    program = edge.to_executorch(
        config=torch_tensorrt.executorch.zero_copy_backend_config(
            ExecutorchBackendConfig(extract_delegate_segments=False)
        )
    )
    _assert_marked_buffers_reach_the_engine_unstaged(program)
    _assert_each_mutation_names_its_own_value(program)


class _SplitRolesDecodeStep(_MixedDecodeStep):
    """The same two buffer kinds, but on two different TensorRT engines.

    ``torch.sinh`` is pinned out of TensorRT by the test, so the attention half
    -- which holds the engine-aliased caches -- and the ``conv_state`` half end up
    in separate partitions. Only the first engine has aliased outputs, and the
    copy-back output rides on the second.
    """

    def forward(self, tokens: torch.Tensor, input_pos: torch.Tensor) -> torch.Tensor:
        pos_idx = input_pos.reshape(-1)
        pos = input_pos.reshape(())
        x = self.embed(tokens)

        def split_heads(proj: torch.Tensor) -> torch.Tensor:
            return proj.view(1, 1, HEADS, HEAD_DIM).transpose(1, 2)

        q = split_heads(self.q(x))
        k = split_heads(self.k(x))
        v = split_heads(self.v(x))
        self.k_cache.index_copy_(2, pos_idx, k)
        self.v_cache.index_copy_(2, pos_idx, v)
        scores = (q @ self.k_cache.transpose(-1, -2)) / (HEAD_DIM**0.5)
        allowed = torch.arange(MAX_LEN, device=x.device) <= pos
        bias = torch.where(
            allowed,
            torch.zeros((), dtype=x.dtype, device=x.device),
            torch.full((), torch.finfo(x.dtype).min, dtype=x.dtype, device=x.device),
        )
        attn = torch.softmax(scores + bias.view(1, 1, 1, MAX_LEN), dim=-1)
        out = (attn @ self.v_cache).transpose(1, 2).reshape(1, 1, HEADS * HEAD_DIM)

        h = torch.sinh(self.o(out) + x)
        shifted = torch.cat([self.conv_state[:, :, 1:], h.reshape(1, DIM, 1)], dim=2)
        self.conv_state.copy_(shifted)
        return self.lm(h + self.conv_state.sum(dim=2).reshape(1, 1, DIM))


@pytest.mark.parametrize("retrace", [False, True], ids=["legacy", "retrace"])
def test_zero_copy_kv_with_the_copyback_on_a_second_delegate(retrace):
    """Two TensorRT delegates, one with the aliased caches and one with the copy-back.

    This is the shape ``_delegate_declares_zero_copy`` reasons about: the
    partitioner must stamp ``zero_copy_kv`` on the KV delegate only, or
    ``_unstage_aliased_buffers``'s cross-check demands an aliased buffer from the
    plain compute delegate and the export dies. Finalizing here is the assertion:
    a wrongly stamped delegate raises inside ``to_executorch``.

    Under ``retrace=True`` this is also the only shape where
    ``_declare_aliased_kv_mutations_on_ep`` has to pick the aliased engine out of
    several: it scans every ``execute_engine`` node and skips the ones whose
    ``aliased_io`` is empty, and the copy-back value it detaches comes off a
    different engine than the caches it declares. The legacy exporter declares all
    of that while inlining, so that scan runs only on this parameter.
    """
    _require_real_engine()
    with torch.no_grad():
        torch.manual_seed(0)
        model = _SplitRolesDecodeStep().eval().cuda()
        tokens = torch.zeros(1, 1, dtype=torch.long).cuda()
        input_pos = torch.tensor([0], dtype=torch.long).cuda()

        exported_program = torch.export.export(model, (tokens, input_pos))
        trt_gm = torch_tensorrt.dynamo.compile(
            exported_program,
            arg_inputs=(tokens, input_pos),
            min_block_size=1,
            truncate_double=True,
            torch_executed_ops={"torch.ops.aten.sinh.default"},
        )
        aliased_per_engine = [
            bool(getattr(sub, "aliased_io", None)) for _, sub in trt_gm.named_children()
        ]
        assert len(aliased_per_engine) > 1 and sum(aliased_per_engine) == 1, (
            "the model no longer lowers to several engines with the aliasing on "
            f"exactly one of them ({aliased_per_engine}), so it does not exercise "
            "the multi-delegate split this test exists for"
        )
        assert trt_gm.meta.get("_copyback_mutation_buffers") == ["conv_state"]

        edge = torch_tensorrt.executorch.export(
            trt_gm,
            arg_inputs=(tokens, input_pos),
            retrace=retrace,
            zero_copy_kv=True,
        )

    ep = edge.exported_program()
    graph_module = ep.graph_module
    output_args = list(graph_module.graph.output_node().args[0])
    bound = {
        spec.target: value
        for spec, value in zip(ep.graph_signature.output_specs, output_args)
        if spec.kind == OutputKind.BUFFER_MUTATION
    }
    assert set(bound) == {"k_cache", "v_cache", "conv_state"}
    for name in ("k_cache", "v_cache"):
        assert bound[name].op == "placeholder"
        assert bound[name].meta.get("_torch_tensorrt_aliased_buffer") is True
    assert bound["conv_state"].target is operator.getitem

    delegates = _real_delegates(graph_module)
    assert len(delegates) > 1
    kv_delegate = next(
        node
        for node in delegates
        if any(
            isinstance(arg, torch.fx.Node)
            and arg.meta.get("_torch_tensorrt_aliased_buffer")
            for arg in node.args[1:]
        )
    )
    copyback_delegate = bound["conv_state"].args[0]
    assert copyback_delegate in delegates
    assert copyback_delegate is not kv_delegate, (
        "the copy-back landed on the same delegate as the aliased caches, so this "
        "test is running the single-delegate shape again"
    )

    stamped = [
        node
        for node in delegates
        if any(
            spec.key == ZERO_COPY_KV_COMPILE_SPEC_KEY
            for spec in _lowered_module(graph_module, node).compile_specs
        )
    ]
    assert stamped == [kv_delegate], (
        "the zero-copy spec must sit on the delegate whose engine lost an output "
        "and on no other; a plain compute delegate carrying it is asked for an "
        "aliased buffer it never had"
    )

    # The un-staging cross-check runs here, not above -- and so does the
    # un-staging itself, which only the finalized graph shows.
    program = edge.to_executorch(
        config=torch_tensorrt.executorch.zero_copy_backend_config()
    )
    _assert_marked_buffers_reach_the_engine_unstaged(program)


@pytest.mark.parametrize("retrace", [False, True], ids=["legacy", "retrace"])
def test_zero_copy_kv_beside_an_executorch_cuda_delegate(retrace):
    """An aliased KV cache in a method that also holds an ExecuTorch CUDA delegate.

    ``erfinv`` has no TensorRT converter, so with a ``CudaPartitioner`` catch-all
    the method lowers to TensorRT, CudaBackend and TensorRT delegates in sequence.
    The un-staging must reach into the TensorRT delegate only. What is asserted
    here is the reachable half -- that only the KV TensorRT delegate is stamped,
    and that the caches reach it un-staged with a CUDA delegate in the middle.
    That the gate itself refuses a marked buffer on another backend is pinned by
    ``test_unstage_raises_for_a_marked_buffer_on_another_backends_delegate``.
    """
    _require_real_engine()
    cuda_backend = pytest.importorskip("executorch.backends.cuda.cuda_backend")
    cuda_partitioner = pytest.importorskip("executorch.backends.cuda.cuda_partitioner")

    class _CudaNeighbourDecodeStep(_KVDecodeStep):
        def __init__(self):
            super().__init__()
            # A TensorRT-supported op AFTER erfinv, so the CUDA delegate is
            # sandwiched between two TensorRT ones. Ending on erfinv would leave
            # the method with a single TensorRT delegate, and the assertion below
            # that only the KV delegate carries the zero-copy spec would then hold
            # whatever the partitioner did.
            self.tail = torch.nn.Linear(VOCAB, VOCAB, bias=False)

        def forward(self, tokens, input_pos):
            h = super().forward(tokens, input_pos)
            return self.tail(torch.erfinv(torch.tanh(h)))

    with torch.no_grad():
        torch.manual_seed(0)
        model = _CudaNeighbourDecodeStep().eval().cuda()
        tokens = torch.zeros(1, 1, dtype=torch.long).cuda()
        input_pos = torch.tensor([0], dtype=torch.long).cuda()

        exported_program = torch.export.export(model, (tokens, input_pos))
        trt_gm = torch_tensorrt.dynamo.compile(
            exported_program,
            arg_inputs=(tokens, input_pos),
            min_block_size=1,
            truncate_double=True,
        )
        edge = torch_tensorrt.executorch.export(
            trt_gm,
            arg_inputs=(tokens, input_pos),
            retrace=retrace,
            zero_copy_kv=True,
            partitioners=[
                cuda_partitioner.CudaPartitioner(
                    [
                        cuda_backend.CudaBackend.generate_method_name_compile_spec(
                            "forward"
                        )
                    ]
                )
            ],
        )

    graph_module = edge.exported_program().graph_module
    delegates = _real_delegates(graph_module)
    backends = {
        node: _lowered_module(graph_module, node).backend_id for node in delegates
    }
    assert sorted(backends.values()) == [
        "CudaBackend",
        "TensorRTBackend",
        "TensorRTBackend",
    ], (
        f"the method no longer lowers to TensorRT/CudaBackend/TensorRT "
        f"({sorted(backends.values())}), so it does not cover the sandwiched "
        "CUDA delegate this test exists for"
    )

    def _marked_args(node):
        return [
            arg.name
            for arg in node.args[1:]
            if isinstance(arg, torch.fx.Node)
            and arg.meta.get("_torch_tensorrt_aliased_buffer")
        ]

    kv_delegates = [node for node in delegates if _marked_args(node)]
    assert len(kv_delegates) == 1 and backends[kv_delegates[0]] == "TensorRTBackend", (
        "the aliased buffers must reach exactly one TensorRT delegate; "
        f"got {[(backends[n], _marked_args(n)) for n in kv_delegates]}"
    )

    stamped = [
        node
        for node in delegates
        if any(
            spec.key == ZERO_COPY_KV_COMPILE_SPEC_KEY
            for spec in _lowered_module(graph_module, node).compile_specs
        )
    ]
    assert stamped == kv_delegates, (
        "only the delegate whose engine lost an aliased output may carry the "
        "zero-copy spec; the CUDA delegate and the trailing TensorRT one had no "
        f"aliased buffer, yet {[backends[n] for n in stamped]} are stamped"
    )

    program = edge.to_executorch(
        config=torch_tensorrt.executorch.zero_copy_backend_config()
    )
    _assert_marked_buffers_reach_the_engine_unstaged(program)


# --------------------------------------------------------------------------
# save() path: unlike the direct export()+to_executorch() contract -- two paired
# calls the caller must not forget, and nothing enforces -- torch_tensorrt.save()
# owns both steps, so a single zero_copy_kv=True must hand export() the opt-in,
# install the finalization config before to_executorch(), and read the finalized
# program back through check_zero_copy_kv. These are CPU-only: the TensorRT
# lowering and the ExecuTorch finalization are stubbed so the wiring is checked
# without a GPU. The passes they invoke have their own coverage above; a real
# end-to-end run is exercised by kv_cache_decode_check on GPU.
# --------------------------------------------------------------------------
def _trivial_exported_program():
    """A tiny CPU ExportedProgram -- enough for save() to reach _save_as_executorch.

    It carries no execute_engine node, so the retrace=True KV-declaration pass is
    a no-op on it and the stubs below stand in for the real lowering.
    """

    class _Add(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x + 1

    return torch.export.export(_Add(), (torch.randn(3),))


def _install_save_stubs(monkeypatch, *, wrap_config=True):
    """Stub the executorch lowering that save() drives and record how it is called.

    Returns a namespace capturing the kwargs export() received, the arguments
    zero_copy_backend_config() was wrapped with, the config finally handed to
    to_executorch(), and the programs check_zero_copy_kv() was given. When
    ``wrap_config`` is False the real zero_copy_backend_config runs, so the
    recorded config is the genuine one.
    """
    import torch_tensorrt._compile as compile_module
    import torch_tensorrt.executorch as executorch_api

    monkeypatch.setattr(
        compile_module,
        "ENABLED_FEATURES",
        compile_module.ENABLED_FEATURES._replace(torch_tensorrt_runtime=True),
    )

    calls = SimpleNamespace(
        export_kwargs=None,
        wrap_args=[],
        to_executorch_config="unset",
        program=None,
        checked=[],
    )

    def _to_executorch(config=None):
        calls.to_executorch_config = config
        calls.program = SimpleNamespace(
            _tensor_data=None, write_to_file=lambda f: f.write(b"stub-pte")
        )
        return calls.program

    edge = SimpleNamespace(to_executorch=_to_executorch)

    # The stub program has no graph, so the real check cannot read it; what these
    # tests pin is that save() runs it, on the finalized program, before writing.
    monkeypatch.setattr(
        executorch_api,
        "check_zero_copy_kv",
        lambda program: calls.checked.append(program),
    )

    def _export(exp_program, **kwargs):
        calls.export_kwargs = kwargs
        return edge

    monkeypatch.setattr(executorch_api, "export", _export)

    if wrap_config:
        wrapped = object()

        def _wrap(config=None):
            calls.wrap_args.append(config)
            return wrapped

        monkeypatch.setattr(executorch_api, "zero_copy_backend_config", _wrap)
        calls.wrapped_sentinel = wrapped
    else:
        real_wrap = executorch_api.zero_copy_backend_config

        def _wrap(config=None):
            calls.wrap_args.append(config)
            return real_wrap(config)

        monkeypatch.setattr(executorch_api, "zero_copy_backend_config", _wrap)

    return calls


@pytest.mark.unit
def test_save_zero_copy_kv_true_threads_flag_and_installs_config(monkeypatch, tmp_path):
    """save(zero_copy_kv=True, backend_config=cfg) opts export() in and wraps the
    caller's config exactly once, forwarding the wrapped one to to_executorch()."""
    calls = _install_save_stubs(monkeypatch)
    user_cfg = object()

    torch_tensorrt.save(
        _trivial_exported_program(),
        str(tmp_path / "model.pte"),
        output_format="executorch",
        zero_copy_kv=True,
        backend_config=user_cfg,
    )

    assert calls.export_kwargs["zero_copy_kv"] is True
    # The user's config is wrapped once (preserving their fields), not double-wrapped.
    assert calls.wrap_args == [user_cfg]
    assert calls.to_executorch_config is calls.wrapped_sentinel
    assert calls.checked == [calls.program]


@pytest.mark.unit
def test_save_zero_copy_kv_true_wraps_defaults_without_a_config(monkeypatch, tmp_path):
    """With no backend_config, zero_copy_backend_config(None) starts from ET
    defaults; the finalization config is still installed."""
    calls = _install_save_stubs(monkeypatch)

    torch_tensorrt.save(
        _trivial_exported_program(),
        str(tmp_path / "model.pte"),
        output_format="executorch",
        zero_copy_kv=True,
    )

    assert calls.export_kwargs["zero_copy_kv"] is True
    assert calls.wrap_args == [None]
    assert calls.to_executorch_config is calls.wrapped_sentinel


@pytest.mark.unit
def test_save_zero_copy_kv_true_installs_the_real_unstaging_pass(monkeypatch, tmp_path):
    """End of the wiring with the real config builder: the config reaching
    to_executorch() carries the un-staging to_out_var_pass, not ET's default."""
    calls = _install_save_stubs(monkeypatch, wrap_config=False)

    torch_tensorrt.save(
        _trivial_exported_program(),
        str(tmp_path / "model.pte"),
        output_format="executorch",
        zero_copy_kv=True,
    )

    assert calls.wrap_args == [None]
    assert (
        type(calls.to_executorch_config.to_out_var_pass).__name__
        == "_UnstageThenToOutVar"
    )


@pytest.mark.unit
def test_save_refuses_skip_h2d_before_it_compiles_anything(monkeypatch, tmp_path):
    """The one refusal that reads nothing but the config fires before the compile.

    Reached only through ``zero_copy_backend_config``, it lands after export has
    partitioned the graph and built every engine, so a caller who set a field
    that was never going to be allowed pays the whole compile to find out.
    ``save`` already validates the weight-streaming budget up front for that
    reason; this is the same rule applied to the same kind of field.
    """
    from executorch.exir import ExecutorchBackendConfig
    from executorch.exir.passes.propagate_device_config import PropagateDeviceConfig

    calls = _install_save_stubs(monkeypatch)

    with pytest.raises(ValueError, match="skip_h2d_for_method_inputs"):
        torch_tensorrt.save(
            _trivial_exported_program(),
            str(tmp_path / "model.pte"),
            output_format="executorch",
            zero_copy_kv=True,
            backend_config=ExecutorchBackendConfig(
                propagate_device_config=PropagateDeviceConfig(
                    skip_h2d_for_method_inputs=True
                )
            ),
        )

    # Refused before the lowering ran, which is the whole of this: the same
    # ValueError comes out either way.
    assert calls.export_kwargs is None


@pytest.mark.unit
def test_save_defaults_leave_kv_staged(monkeypatch, tmp_path):
    """Default save() (zero_copy_kv omitted) never wraps the config, so the KV
    buffer keeps its staging and its copy-back: the caller's config reaches
    to_executorch() untouched and export() is told zero_copy_kv=False."""
    calls = _install_save_stubs(monkeypatch)
    user_cfg = object()

    torch_tensorrt.save(
        _trivial_exported_program(),
        str(tmp_path / "model.pte"),
        output_format="executorch",
        backend_config=user_cfg,
    )

    assert calls.export_kwargs["zero_copy_kv"] is False
    assert calls.wrap_args == []
    assert calls.to_executorch_config is user_cfg
    assert calls.checked == []


@pytest.mark.unit
@pytest.mark.parametrize(
    "retrace", [False, True], ids=["retrace-false", "retrace-true"]
)
def test_save_forwards_zero_copy_kv_from_a_graph_module(monkeypatch, tmp_path, retrace):
    """A compiled module is a GraphModule, and save reaches ExecuTorch by a
    different branch for each value of ``retrace``.

    The tests above hand save an ``ExportedProgram``, which is the third branch.
    Dropping the option from either of these two leaves a caller who asked for
    zero-copy with an ordinary staged ``.pte`` and no error, since export is
    never told and so nothing is rewired for the checker to miss.
    """
    calls = _install_save_stubs(monkeypatch)

    class _Add(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x + 1

    torch_tensorrt.save(
        torch.fx.symbolic_trace(_Add()),
        str(tmp_path / "model.pte"),
        output_format="executorch",
        arg_inputs=[torch.randn(3)],
        retrace=retrace,
        # Neither branch's default exporter reads a plain traced GraphModule: the
        # legacy one wants the engine-node shape a real compile produces.
        use_legacy_exporter=False,
        zero_copy_kv=True,
    )

    assert calls.export_kwargs["zero_copy_kv"] is True
    assert calls.to_executorch_config is calls.wrapped_sentinel
    assert calls.checked == [calls.program]
