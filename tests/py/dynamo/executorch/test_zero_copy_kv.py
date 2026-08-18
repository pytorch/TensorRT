"""Export-side coverage for zero-copy aliased KV buffers.

Two halves have to agree for a zero-copy ``.pte`` to be correct:

  * ``rewire_aliased_mutations_to_buffers`` declares the buffer to be its own
    mutation result, which removes ExecuTorch's copy-back and takes the aliased
    output out of the delegate.
  * ``unstage_aliased_buffers_pass`` removes the host staging copy so the
    engine's in-place write lands in the caller's buffer.

The interesting failures are all silent -- a rewired mutation whose buffer is
still staged simply never updates -- so most of what is asserted here is which
mutations are left alone and which mis-shapes raise.
"""

import operator
from types import SimpleNamespace

import pytest

pytest.importorskip("executorch.exir")

import torch  # noqa: E402
import torch_tensorrt  # noqa: E402
from executorch.exir.delegate import executorch_call_delegate  # noqa: E402
from executorch.exir.schema import DeviceType  # noqa: E402
from torch.export.exported_program import (  # noqa: E402
    OutputKind,
    OutputSpec,
    TensorArgument,
)
from torch_tensorrt.executorch import _zero_copy as Z  # noqa: E402

# The graphs below are built around torch.ops.tensorrt.execute_engine, which only
# exists once the Torch-TensorRT runtime operator library has loaded.
pytestmark = pytest.mark.skipif(
    not torch_tensorrt.ENABLED_FEATURES.torch_tensorrt_runtime,
    reason="Torch-TensorRT runtime operators are not available",
)


def _patch_engine_metadata(monkeypatch, *, aliased_io, input_names, output_names):
    """Make every engine node report one fixed set of bindings and aliases."""
    import torch_tensorrt.dynamo.runtime._serialized_engine_layout as layout
    import torch_tensorrt.dynamo.runtime._TorchTensorRTModule as trt_module
    import torch_tensorrt.executorch.backend as backend

    info = ["x"] * (layout.ALIASED_IO_IDX + 1)
    info[layout.INPUT_BINDING_NAMES_IDX] = "IN"
    info[layout.OUTPUT_BINDING_NAMES_IDX] = "OUT"
    monkeypatch.setattr(backend, "_get_engine_info_for_node", lambda ep, node: info)
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
    is what takes the aliased output out of the delegate.
    """
    program, k_buffer, k_out = _kv_program()
    _patch_engine_metadata(
        monkeypatch,
        aliased_io={"out_k": ("k_in", "kv_cache_update")},
        input_names=["k_in", "tokens"],
        output_names=["logits", "out_k"],
    )

    assert Z.rewire_aliased_mutations_to_buffers(program) == 1

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

    assert Z.rewire_aliased_mutations_to_buffers(program) == 0
    assert program._graph_signature.output_specs[0] is original_spec
    assert "_torch_tensorrt_aliased_buffer" not in k_buffer.meta


@pytest.mark.unit
def test_rewire_is_a_noop_without_aliased_io(monkeypatch):
    program, k_buffer, _ = _kv_program()
    _patch_engine_metadata(
        monkeypatch,
        aliased_io={},
        input_names=["k_in", "tokens"],
        output_names=["logits", "out_k"],
    )

    assert Z.rewire_aliased_mutations_to_buffers(program) == 0
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


def _staged_delegate_graph(*, backend_id="TensorRTBackend", device=DeviceType.CUDA):
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
    root.lowered_module_0 = SimpleNamespace(backend_id=backend_id)
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
    graph_module, k_buffer, staged_k, delegate = _staged_delegate_graph()
    k_buffer.meta["_torch_tensorrt_aliased_buffer"] = True

    assert Z._unstage_aliased_buffers(graph_module) == 1

    assert delegate.args[1] is k_buffer
    assert k_buffer.meta["spec"].device == DeviceType.CUDA
    assert k_buffer.meta["spec"].device_index == 3
    # The other input is an ordinary one and keeps its staging copy.
    assert delegate.args[2] is not None
    assert delegate.args[2].target is torch.ops.et_copy._h2d_copy.default


@pytest.mark.unit
def test_unstage_keeps_staging_for_an_unmarked_buffer():
    graph_module, k_buffer, staged_k, delegate = _staged_delegate_graph()

    assert Z._unstage_aliased_buffers(graph_module) == 0
    assert delegate.args[1] is staged_k
    assert k_buffer.meta["spec"].device == DeviceType.CPU


@pytest.mark.unit
def test_unstage_ignores_another_backends_delegate():
    """Only a TensorRT engine promises the in-place write."""
    graph_module, k_buffer, staged_k, delegate = _staged_delegate_graph(
        backend_id="CudaBackend"
    )
    k_buffer.meta["_torch_tensorrt_aliased_buffer"] = True

    assert Z._unstage_aliased_buffers(graph_module) == 0
    assert delegate.args[1] is staged_k


@pytest.mark.unit
def test_unstage_raises_when_the_staging_copy_is_not_on_cuda():
    """Following the staging to the CPU would put the buffer out of the engine's
    reach, and the copy-back that would have saved it is already gone."""
    graph_module, k_buffer, _, _ = _staged_delegate_graph(device=DeviceType.CPU)
    k_buffer.meta["_torch_tensorrt_aliased_buffer"] = True

    with pytest.raises(RuntimeError, match="not.*CUDA"):
        Z._unstage_aliased_buffers(graph_module)


@pytest.mark.unit
def test_unstage_raises_when_the_staging_copy_has_no_spec():
    graph_module, k_buffer, staged_k, _ = _staged_delegate_graph()
    k_buffer.meta["_torch_tensorrt_aliased_buffer"] = True
    del staged_k.meta["spec"]

    with pytest.raises(RuntimeError, match="no TensorSpec"):
        Z._unstage_aliased_buffers(graph_module)


@pytest.mark.unit
def test_unstage_pass_runs_the_inner_pass_after_unstaging():
    """A caller's own to_out_var_pass has to survive being composed with."""
    graph_module, k_buffer, _, delegate = _staged_delegate_graph()
    k_buffer.meta["_torch_tensorrt_aliased_buffer"] = True
    seen = []

    def inner(gm):
        # The un-staging is already done by the time the inner pass sees the graph.
        seen.append(delegate.args[1] is k_buffer)
        return "inner-result"

    result = Z.unstage_aliased_buffers_pass(inner).call(graph_module)

    assert seen == [True]
    assert result == "inner-result"
